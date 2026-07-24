import json

import torch
# torch.multiprocessing.set_sharing_strategy('file_system')

import math
import pickle
import gc
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import logging

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import List, Optional, cast
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import Batch
from torch_geometric.loader import ImbalancedSampler
from pytorch_lightning import seed_everything
import torch.nn.functional as F
from timm.optim.lars import Lars

from src.common_utils import get_arg_parser, filter_warnings, init_log
from src.vocabulary import Vocabulary
from src.torch_data.datamodules import SliceDataModule
from src.torch_data.samplers import BalancedBinaryBatchSampler
from src.models.swav_vd import GraphSwAVVD
from src.models.modules.losses import SupConLoss, InfoNCE, OrthogonalProjectionLoss
from src.swav.assignment_protocols import sinkhorn, uot_sinkhorn_gpu

contrastive_options = {
    "supcon": SupConLoss,
    "simclr": SupConLoss,
    "info_nce": InfoNCE
}

assignment_functions = {
    "sinkhorn": sinkhorn,
    "uot_sinkhorn": uot_sinkhorn_gpu
}

@dataclass
class TrainingContext:
    config: DictConfig
    device: torch.device
    projection_criterion: Optional[OrthogonalProjectionLoss]
    contrastive_criterion: Optional[object]
    use_amp: bool
    writer: Optional[SummaryWriter] = None
    ce_losses: List[float] = field(default_factory=list)
    proj_losses: List[float] = field(default_factory=list)
    reg_losses: List[float] = field(default_factory=list)
    swav_losses: List[float] = field(default_factory=list)
    contrast_losses: List[float] = field(default_factory=list)
    attention_distribution_losses: List[float] = field(default_factory=list)

gnn_name_map = {
    "gcn": "GCN",
    "gin": "GIN",
    "gine": "GINE",
    "ggnn": "GGNN",
    "gatv2": "GATv2",
    "gated": "Gated",
    "st": "ST"
}

def is_lambda_enabled(ctx: TrainingContext, lambda_name: str) -> bool:
    return float(ctx.config.hyper_parameters.lambdas.get(lambda_name, 0.0)) > 0.0

def is_contrastive_enabled(ctx: TrainingContext) -> bool:
    return (
        bool(ctx.config.swav.contrastive.get("enabled", True))
        and is_lambda_enabled(ctx, "contrastive")
    )

loss_metric_to_lambda = {
    "ce": "classification",
    "proj": "projection",
    "reg": "regularization",
    "swav": "swav",
    "contrast": "contrastive",
    "attention_distribution": "attention_distribution",
}

def log_training_step_metrics(
        ctx: TrainingContext,
        optimizer,
        metrics,
        training_step: int):
    if ctx.writer is None:
        return

    log_every_n_steps = max(
        1,
        int(ctx.config.hyper_parameters.get("log_every_n_steps", 50)),
    )
    if training_step % log_every_n_steps != 0:
        return

    for metric_name, lambda_name in loss_metric_to_lambda.items():
        raw_value = metrics[metric_name]
        weight = float(
            ctx.config.hyper_parameters.lambdas.get(lambda_name, 0.0)
        )
        ctx.writer.add_scalar(
            f"train_step/loss/raw/{metric_name}",
            raw_value,
            training_step,
        )
        ctx.writer.add_scalar(
            f"train_step/loss/weighted/{metric_name}",
            weight * raw_value,
            training_step,
        )

    ctx.writer.add_scalar(
        "train_step/loss/total",
        metrics["total"],
        training_step,
    )
    ctx.writer.add_scalar(
        "train_step/optimization/learning_rate",
        optimizer.param_groups[0]["lr"],
        training_step,
    )

def log_training_epoch_metrics(
        ctx: TrainingContext,
        epoch: int,
        epoch_metrics):
    if ctx.writer is None:
        return

    tensorboard_epoch = epoch + 1
    total_loss = 0.0
    for metric_name, lambda_name in loss_metric_to_lambda.items():
        raw_value = epoch_metrics[metric_name]
        weight = float(
            ctx.config.hyper_parameters.lambdas.get(lambda_name, 0.0)
        )
        total_loss += weight * raw_value
        ctx.writer.add_scalar(
            f"train_epoch/loss/raw/{metric_name}",
            raw_value,
            tensorboard_epoch,
        )
        ctx.writer.add_scalar(
            f"train_epoch/loss/weighted/{metric_name}",
            weight * raw_value,
            tensorboard_epoch,
        )
    ctx.writer.add_scalar(
        "train_epoch/loss/total",
        total_loss,
        tensorboard_epoch,
    )
    ctx.writer.flush()

def log_validation_metrics(
        ctx: TrainingContext,
        epoch: int,
        eval_stats):
    if ctx.writer is None:
        return

    tensorboard_epoch = epoch + 1
    for metric_name, value in eval_stats.items():
        ctx.writer.add_scalar(
            f"validation/{metric_name}",
            value,
            tensorboard_epoch,
        )
    ctx.writer.flush()

def validate_training_views(ctx: TrainingContext, num_views: int):
    if num_views < 2:
        raise ValueError(f"SwAV loss requires at least 2 augmented views, got {num_views}.")
    if is_contrastive_enabled(ctx) and num_views != 2:
        raise ValueError(
            "Contrastive loss currently expects exactly 2 augmented views, "
            f"got {num_views}."
        )

def forward_model(
        ctx: TrainingContext,
        model,
        graphs,
        forward_fn=None,
        **forward_kwargs):
    if forward_fn is None:
        forward_fn = model
    return forward_fn(
        graphs.to(ctx.device, non_blocking=True),
        **forward_kwargs,
    )

def get_train_batch_size_for_epoch(ctx: TrainingContext, epoch: int) -> int:
    batch_sizes = ctx.config.hyper_parameters.batch_sizes
    if len(batch_sizes) == 0:
        raise ValueError("config.hyper_parameters.batch_sizes must contain at least one value.")
    return int(batch_sizes[min(epoch, len(batch_sizes) - 1)])

def get_train_steps_for_epoch(ctx: TrainingContext, epoch: int, num_samples: int) -> int:
    batch_size = get_train_batch_size_for_epoch(ctx, epoch)
    return math.ceil(num_samples / batch_size)


def get_total_train_steps(ctx: TrainingContext, num_samples: int) -> int:
    n_epochs = int(ctx.config.hyper_parameters.n_epochs)
    return sum(
        get_train_steps_for_epoch(ctx, epoch, num_samples)
        for epoch in range(n_epochs)
    )


def get_warmup_train_steps(ctx: TrainingContext, num_samples: int) -> int:
    n_epochs = int(ctx.config.hyper_parameters.n_epochs)
    warmup_epochs = min(int(ctx.config.swav.warmup_epochs), n_epochs)
    return sum(
        get_train_steps_for_epoch(ctx, epoch, num_samples)
        for epoch in range(warmup_epochs)
    )


def build_lr_scheduler(ctx: TrainingContext, optimizer, num_samples: int):
    warmup_steps = get_warmup_train_steps(ctx, num_samples)
    total_steps = get_total_train_steps(ctx, num_samples)
    cosine_steps = total_steps - warmup_steps

    base_lr = float(ctx.config.swav.base_lr)
    start_warmup = float(ctx.config.swav.start_warmup)
    final_lr = float(ctx.config.swav.final_lr)
    lr_normalizer = base_lr if base_lr != 0.0 else 1.0

    def lr_lambda(step: int):
        step = min(step, max(total_steps - 1, 0))

        if warmup_steps > 0 and step < warmup_steps:
            if warmup_steps == 1:
                lr = start_warmup
            else:
                progress = step / (warmup_steps - 1)
                lr = start_warmup + progress * (base_lr - start_warmup)
        elif cosine_steps > 0:
            t = step - warmup_steps
            lr = final_lr + 0.5 * (base_lr - final_lr) * (
                1 + math.cos(math.pi * t / cosine_steps)
            )
        else:
            lr = base_lr

        return lr / lr_normalizer

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def load_train_dataset_stats(train_slices, train_stats_filepath: Path):
    if not train_stats_filepath.exists():
        logging.info(f"{train_stats_filepath} not found. Retrieving labels...")
        ys = []
        for slice_path in tqdm(train_slices, desc=f"Slice files"):
            with Path(slice_path).open("rb") as rbfi:
                slice_graph: nx.DiGraph = pickle.load(rbfi)
                ys.append(int(slice_graph.graph["label"]))

        neg_cnt = ys.count(0)
        pos_cnt = ys.count(1)
        majority_cnt = max(neg_cnt, pos_cnt)
        dataset_stats = {
            "ys": ys,
            "sampler_num_samples": majority_cnt * 2,
            "neg_cnt": neg_cnt,
            "pos_cnt": pos_cnt,
        }
        logging.info(f"Successfully retrieved {len(ys)} labels. Writing all stats to {train_stats_filepath}...")
        with train_stats_filepath.open("w") as wfi:
            json.dump(dataset_stats, wfi, indent=2)
        return dataset_stats

    logging.info(f"Reading dataset stats from {train_stats_filepath}...")
    with train_stats_filepath.open("r") as rfi:
        dataset_stats = json.load(rfi)
    logging.info(f"Completed. Retrieved stats.")
    return dataset_stats

def build_training_sampling(config: DictConfig, dataset_root: Path):
    train_slices_filepath = dataset_root / config.train_slices_filename
    logging.info(f"Loading training slice paths list from {train_slices_filepath}...")
    with train_slices_filepath.open("r") as rfi:
        train_slices = json.load(rfi)
    logging.info(f"Completed. Loaded {len(train_slices)} slices.")

    use_imbalanced_sampler = bool(config.hyper_parameters.get("use_imbalanced_sampler", False))
    use_balanced_batch_sampler = bool(config.hyper_parameters.get("use_balanced_batch_sampler", False))
    if use_imbalanced_sampler and use_balanced_batch_sampler:
        raise ValueError("use_imbalanced_sampler and use_balanced_batch_sampler cannot both be enabled.")
    if use_balanced_batch_sampler:
        for batch_size in config.hyper_parameters.batch_sizes:
            if int(batch_size) < 2 or int(batch_size) % 2 != 0:
                raise ValueError(
                    "use_balanced_batch_sampler requires all train batch sizes "
                    f"to be even and >= 2, got {batch_size}."
                )

    num_samples = len(train_slices)
    sampler = None
    train_batch_sampler_factory = None
    if use_imbalanced_sampler or use_balanced_batch_sampler:
        train_stats_filepath = dataset_root / config.train_stats_filename
        dataset_stats = load_train_dataset_stats(train_slices, train_stats_filepath)
        num_samples = int(dataset_stats["sampler_num_samples"])

        if use_imbalanced_sampler:
            logging.info("Using ImbalancedSampler for training data loader.")
            sampler = ImbalancedSampler(
                torch.tensor(dataset_stats["ys"], dtype=torch.long),
                num_samples=num_samples,
            )
        else:
            logging.info("Using balanced batch sampler for training data loader.")
            def train_batch_sampler_factory(batch_size):
                return BalancedBinaryBatchSampler(
                    labels=dataset_stats["ys"],
                    batch_size=batch_size,
                    seed=config.seed,
                )

    return (
        num_samples,
        sampler,
        train_batch_sampler_factory,
        use_imbalanced_sampler,
        use_balanced_batch_sampler,
    )

def build_all_views_batch(batched_graph):
    graphs = batched_graph.graphs.to_data_list()
    augmented_views = [
        view
        for view_batch in batched_graph.augmented_views
        for view in view_batch.to_data_list()
    ]
    return Batch.from_data_list(graphs + augmented_views)

def compute_training_loss(ctx: TrainingContext, model, batched_graph):
    num_views = len(batched_graph.augmented_views)
    validate_training_views(ctx, num_views)
    batch_size = batched_graph.sz

    labels = batched_graph.labels.to(ctx.device, non_blocking=True)
    combined_graphs = getattr(batched_graph, "all_views", None)
    if combined_graphs is None:
        combined_graphs = build_all_views_batch(batched_graph)

    if is_lambda_enabled(ctx, "attention_distribution"):
        (
            logits_all,
            activations_all,
            graph_encodings_all,
            _,
            output_all,
            attention_distribution_loss,
        ) = forward_model(
            ctx,
            model,
            combined_graphs,
            forward_fn=model.forward_training_views,
            batch_size=batch_size,
            num_views=num_views,
        )
    else:
        logits_all, activations_all, graph_encodings_all, _, output_all = (
            forward_model(ctx, model, combined_graphs)
        )
        attention_distribution_loss = logits_all.new_zeros(())

    logits = logits_all[:batch_size]
    activations = activations_all[:batch_size]
    anchor_graph_encodings = graph_encodings_all[:batch_size]
    graph_encodings = graph_encodings_all[batch_size:].chunk(num_views)
    output = output_all[batch_size:].chunk(num_views)

    zero_loss = logits.new_zeros(())

    ce_loss = F.cross_entropy(logits, labels)

    projection_loss = zero_loss
    if is_lambda_enabled(ctx, "projection"):
        if ctx.projection_criterion is None:
            raise RuntimeError("projection_criterion is not initialized.")
        projection_loss = ctx.projection_criterion(activations, labels)

    regularization_loss = zero_loss
    if is_lambda_enabled(ctx, "regularization"):
        regularization_loss = torch.norm(anchor_graph_encodings, dim=-1).mean() + torch.norm(activations, dim=-1).mean()

    swav_loss = logits.new_zeros(())
    if is_lambda_enabled(ctx, "swav"):
        views_for_assign = [view_id for view_id in ctx.config.swav.views_for_assign if view_id < num_views]
        if not views_for_assign:
            raise ValueError(f"No valid SwAV assignment views for {num_views} augmented views.")
        for view_id in views_for_assign:
            with torch.no_grad():
                with torch.amp.autocast(ctx.device.type, enabled=False):
                    out = output[view_id].detach().float()
                    q = assignment_functions[ctx.config.swav.assignment_protocol](out)
            subloss = logits.new_zeros(())
            for v in np.delete(np.arange(num_views), view_id):
                x = output[v] / ctx.config.swav.temperature
                subloss = subloss - torch.mean(torch.sum(q * F.log_softmax(x, dim=-1), dim=-1))
            swav_loss = swav_loss + subloss / (num_views - 1)
        swav_loss = swav_loss / len(views_for_assign)

    contrastive_loss = logits.new_zeros(())
    if is_contrastive_enabled(ctx):
        if ctx.contrastive_criterion is None:
            raise RuntimeError("contrastive_criterion is not initialized.")
        if ctx.config.swav.contrastive.criterion == "info_nce":
            contrastive_loss = (ctx.contrastive_criterion(anchor_graph_encodings, graph_encodings[0]) + ctx.contrastive_criterion(anchor_graph_encodings, graph_encodings[1])) / 2.0
        elif ctx.config.swav.contrastive.criterion == "supcon":
            contrastive_loss = ctx.contrastive_criterion(torch.stack([anchor_graph_encodings, graph_encodings[0], graph_encodings[1]], dim=1), labels=labels)
        elif ctx.config.swav.contrastive.criterion == "simclr":
            contrastive_loss = ctx.contrastive_criterion(torch.stack([anchor_graph_encodings, graph_encodings[0], graph_encodings[1]], dim=1))
        else:
            raise ValueError(f"Unsupported contrastive criterion: {ctx.config.swav.contrastive.criterion}")

    loss = (
        ctx.config.hyper_parameters.lambdas.classification * ce_loss
        + ctx.config.hyper_parameters.lambdas.projection * projection_loss
        + ctx.config.hyper_parameters.lambdas.regularization * regularization_loss
        + ctx.config.hyper_parameters.lambdas.swav * swav_loss
        + ctx.config.hyper_parameters.lambdas.contrastive * contrastive_loss
        + ctx.config.hyper_parameters.lambdas.attention_distribution
        * attention_distribution_loss
    )

    metrics = {
        "total": loss.item(),
        "ce": ce_loss.item(),
        "proj": projection_loss.item(),
        "reg": regularization_loss.item(),
        "swav": swav_loss.item(),
        "contrast": contrastive_loss.item(),
        "attention_distribution": attention_distribution_loss.item(),
    }
    return loss, metrics

def run_training_step(ctx: TrainingContext, model, optimizer, scaler, batched_graph, training_step: int):
    optimizer.zero_grad(set_to_none=True)

    with torch.amp.autocast(ctx.device.type, enabled=ctx.use_amp):
        loss, metrics = compute_training_loss(ctx, model, batched_graph)
    scaler.scale(loss).backward()
    # cancel gradients for the prototypes
    if training_step < ctx.config.swav.freeze_prototypes_niters:
        for name, p in model.named_parameters():
            if "swav_prototypes" not in name:
                continue
            p.grad = None

    scaler.step(optimizer)
    scaler.update()
    del loss
    return metrics

def train(ctx: TrainingContext, train_loader, model, optimizer, scaler, lr_scheduler, epoch, training_step: int):
    logging.info(f"Epoch {epoch + 1}")
    model.train()
    progress_bar = tqdm(train_loader, desc="Training")

    epoch_ce_losses = []
    epoch_proj_losses = []
    epoch_reg_losses = []
    epoch_swav_losses = []
    epoch_contrast_losses = []
    epoch_attention_distribution_losses = []

    for batched_graph in progress_bar:
        with torch.no_grad():
            w = model.swav_prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            model.swav_prototypes.weight.copy_(w)

        batch_losses = run_training_step(
            ctx,
            model,
            optimizer,
            scaler,
            batched_graph,
            training_step,
        )
        log_training_step_metrics(
            ctx,
            optimizer,
            batch_losses,
            training_step,
        )
        lr_scheduler.step()
        training_step += 1

        epoch_ce_losses.append(batch_losses["ce"])
        epoch_proj_losses.append(batch_losses["proj"])
        epoch_reg_losses.append(batch_losses["reg"])
        epoch_swav_losses.append(batch_losses["swav"])
        epoch_contrast_losses.append(batch_losses["contrast"])
        epoch_attention_distribution_losses.append(
            batch_losses["attention_distribution"]
        )

        progress_bar.set_postfix({
            "ce": batch_losses["ce"],
            'proj': batch_losses["proj"],
            'reg': batch_losses["reg"],
            'swav': batch_losses["swav"],
            'contrast': batch_losses["contrast"],
            'attn_dist': batch_losses["attention_distribution"],
        })
        del batched_graph, batch_losses

    logging.info(
        f"Epoch {epoch + 1} - Cross-Entropy Loss: "
        f"{np.mean(epoch_ce_losses):.4f}, Projection Loss: "
        f"{np.mean(epoch_proj_losses):.4f}, Regularization Loss: "
        f"{np.mean(epoch_reg_losses):.4f}, SwAV Loss: "
        f"{np.mean(epoch_swav_losses):.4f}, Contrastive Loss: "
        f"{np.mean(epoch_contrast_losses):.4f}, Attention Distribution "
        f"Loss: {np.mean(epoch_attention_distribution_losses):.4f}"
    )
    ctx.ce_losses.append(float(np.mean(epoch_ce_losses)))
    ctx.proj_losses.append(float(np.mean(epoch_proj_losses)))
    ctx.reg_losses.append(float(np.mean(epoch_reg_losses)))
    ctx.swav_losses.append(float(np.mean(epoch_swav_losses)))
    ctx.contrast_losses.append(float(np.mean(epoch_contrast_losses)))
    ctx.attention_distribution_losses.append(
        float(np.mean(epoch_attention_distribution_losses))
    )
    log_training_epoch_metrics(
        ctx,
        epoch,
        {
            "ce": ctx.ce_losses[-1],
            "proj": ctx.proj_losses[-1],
            "reg": ctx.reg_losses[-1],
            "swav": ctx.swav_losses[-1],
            "contrast": ctx.contrast_losses[-1],
            "attention_distribution": (
                ctx.attention_distribution_losses[-1]
            ),
        },
    )

    del train_loader
    return training_step

def evaluate(ctx: TrainingContext, model, val_loader):
    progress_bar = tqdm(val_loader, desc="Validation")
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in progress_bar:
            labels = batch.labels.to(ctx.device, non_blocking=True)
            logits = forward_model(
                ctx,
                model,
                batch.graphs,
                forward_fn=model.forward_logits,
            )
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            _, preds = logits.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            del batch, labels, logits, preds, loss
    
    stats = {
        "eval_loss": total_loss / len(val_loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds)
    }
    
    del val_loader, all_preds, all_labels
    
    return stats

def test(ctx: TrainingContext, model, test_loader):
    progress_bar = tqdm(test_loader, desc="Testing")
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in progress_bar:
            labels = batch.labels.to(ctx.device, non_blocking=True)
            logits = forward_model(
                ctx,
                model,
                batch.graphs,
                forward_fn=model.forward_logits,
            )
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            _, preds = logits.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            del batch, labels, logits, preds, loss
    
    stats = {
        "test_loss": total_loss / len(test_loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds)
    }
    
    return stats

def require_checkpoint(checkpoint_path: Path, description: str):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{description} checkpoint not found at {checkpoint_path}. "
            "Run training first, or use --skip_training only when the checkpoint already exists."
        )

def load_config_from_args(args) -> DictConfig:
    config = cast(DictConfig, OmegaConf.load(args.config))

    if args.no_cl:
        OmegaConf.update(config, "swav.contrastive.enabled", False, force_add=True)
    if args.exclude_NNs:
        config.exclude_NNs = True
    if args.use_lr_warmup:
        config.hyper_parameters.use_warmup_lr = True
    if args.no_cl_warmup:
        config.hyper_parameters.contrastive_warmup_epochs = 0
    if args.train_batch_size is not None:
        if args.train_batch_size < 1:
            raise ValueError("--train_batch_size must be >= 1")
        config.hyper_parameters.batch_sizes = [
            args.train_batch_size
            for _ in config.hyper_parameters.batch_sizes
        ]
    if args.test_batch_size is not None:
        if args.test_batch_size < 1:
            raise ValueError("--test_batch_size must be >= 1")
        config.hyper_parameters.test_batch_size = args.test_batch_size

    return config

def log_cli_compatibility_warnings(args):
    if args.exclude_NNs:
        logging.warning(
            "--exclude_NNs is accepted for compatibility but this SwAV training script "
            "does not construct NN contrastive pairs; it only affects checkpoint naming."
        )
    if args.use_lr_warmup:
        logging.warning(
            "--use_lr_warmup sets hyper_parameters.use_warmup_lr, but this script uses "
            "the SwAV LR schedule controlled by swav.warmup_epochs."
        )
    if args.no_cl_warmup:
        logging.warning(
            "--no_cl_warmup sets hyper_parameters.contrastive_warmup_epochs to 0, but "
            "contrastive warmup is not implemented in this SwAV training loop."
        )

def get_dataset_root(config: DictConfig, args) -> Path:
    if args.use_temp_data:
        return Path(config.temp_root)
    return Path(config.data_folder) / config.dataset.name

def build_runtime_context(config: DictConfig):
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(config.hyper_parameters.get("use_amp", False)) and runtime_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logging.info(f"Device: {runtime_device}")
    logging.info(f"AMP enabled: {use_amp}")
    ctx = TrainingContext(
        config=config,
        device=runtime_device,
        projection_criterion=None,
        contrastive_criterion=None,
        use_amp=use_amp,
    )
    return ctx, scaler

def load_vocab_and_model(ctx: TrainingContext, dataset_root: Path):
    vocab = Vocabulary.from_w2v(str(dataset_root / "w2v.wv"))
    vocab_size = vocab.get_vocab_size()
    pad_idx = vocab.get_pad_id()

    logging.info("Building model...")
    model = GraphSwAVVD(ctx.config, vocab, vocab_size, pad_idx).to(ctx.device)
    logging.info("Model building completed.")

    return vocab, model


def add_swav_arguments(arg_parser):
    arg_parser.add_argument("--skip_training", action='store_true', help="Skip training phase")
    arg_parser.add_argument("--no_cl", action='store_true', help="Disable contrastive learning")
    arg_parser.add_argument("--exclude_NNs", action='store_true', help="Exclude NN pairs if contrastive pair filtering is used")
    arg_parser.add_argument("--use_lr_warmup", action='store_true', help="Enable learning-rate warmup in the config")
    arg_parser.add_argument("--no_cl_warmup", action='store_true', help="Disable contrastive learning warmup in the config")
    arg_parser.add_argument("--train_batch_size", type=int, default=None,
                            help="Override all configured training batch sizes.")
    arg_parser.add_argument("--test_batch_size", type=int, default=None,
                            help="Override validation and test batch size.")


def build_optimizer(config: DictConfig, model):
    # Using LARS optimizer as per the SwAV paper.
    return Lars(
        model.parameters(),
        lr=config.swav.base_lr,
        momentum=0.9,
        weight_decay=config.swav.wd
    )


def initialize_loss_criteria(ctx: TrainingContext):
    if is_contrastive_enabled(ctx):
        ctx.contrastive_criterion = contrastive_options[ctx.config.swav.contrastive.criterion](
            temperature=ctx.config.swav.contrastive.temperature
        )
    ctx.projection_criterion = OrthogonalProjectionLoss()


def build_checkpoint_paths(
        ctx: TrainingContext,
        model,
        use_imbalanced_sampler: bool,
        use_balanced_batch_sampler: bool,
):
    config = ctx.config
    dataset_name = Path(config.dataset.name).name
    gnn_name = gnn_name_map[config.gnn.name]
    nn_text = "ExcludeNN" if config.exclude_NNs else "IncludeNN"
    cl_warmup_text = "CLWarmup" if config.hyper_parameters.contrastive_warmup_epochs > 0 else "NoCLWarmup"
    contrastive_text = "NoContrastive" if not is_contrastive_enabled(ctx) else config.swav.contrastive.criterion
    do_swav = "NoSwAV" if config.hyper_parameters.lambdas.swav == 0.0 else "DoSwAV"
    gnn_attention_only = "GNNAttentionOnly" if config.gnn.attention_only else "GNNWithPooling"
    use_edge_attr = "WithEdgeAttr" if config.gnn.use_edge_attr else "NoEdgeAttr"
    attention_distribution_text = (
        "AttentionDistribution-"
        f"{float(ctx.config.hyper_parameters.lambdas.attention_distribution):g}"
        if is_lambda_enabled(ctx, "attention_distribution")
        else "NoAttentionDistribution"
    )

    if use_balanced_batch_sampler:
        sampler_text = "WithBalancedBatchSampler"
    elif use_imbalanced_sampler:
        sampler_text = "WithImbalancedSampler"
    else:
        sampler_text = "NoSampler"

    checkpoint_dir = (
        Path(config.model_save_dir)
        / "graph_swav_classification"
        / dataset_name
        / gnn_name
        / nn_text
        / cl_warmup_text
        / do_swav
        / contrastive_text
        / gnn_attention_only
        / use_edge_attr
        / attention_distribution_text
        / sampler_text
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_name = model.__class__.__name__
    best_f1_directory = checkpoint_dir / "best_f1"
    best_f1_directory.mkdir(parents=True, exist_ok=True)
    best_f1_checkpoint_path = best_f1_directory / f"{model_name}.ckpt"

    best_loss_directory = checkpoint_dir / "best_loss"
    best_loss_directory.mkdir(parents=True, exist_ok=True)
    best_loss_checkpoint_path = best_loss_directory / f"{model_name}.ckpt"

    logging.info(f"Checkpoint directory: {checkpoint_dir}")
    return checkpoint_dir, best_f1_checkpoint_path, best_loss_checkpoint_path


def release_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def train_and_evaluate_epoch(
        ctx: TrainingContext,
        data_module: SliceDataModule,
        model,
        optimizer,
        scaler,
        lr_scheduler,
        epoch: int,
        training_step: int,
):
    data_module.set_train_batch_size(get_train_batch_size_for_epoch(ctx, epoch))
    train_loader = data_module.train_dataloader()
    training_step = train(
        ctx,
        train_loader,
        model,
        optimizer,
        scaler,
        lr_scheduler,
        epoch,
        training_step,
    )

    eval_stats = evaluate(ctx, model, data_module.val_dataloader())
    release_memory()

    cache_info = data_module.get_cache_info()
    if 'train' in cache_info:
        print(f"Cache Info: {cache_info['train']}")

    return eval_stats, training_step


def update_best_checkpoints(
        model,
        eval_stats,
        best_val_f1: float,
        best_val_loss: float,
        best_f1_checkpoint_path: Path,
        best_loss_checkpoint_path: Path,
):
    if eval_stats["f1"] > best_val_f1:
        best_val_f1 = eval_stats["f1"]
        torch.save(model.state_dict(), best_f1_checkpoint_path)
        logging.info(f"New best model saved with F1: {best_val_f1:.4f}")

    if eval_stats["eval_loss"] < best_val_loss:
        best_val_loss = eval_stats["eval_loss"]
        torch.save(model.state_dict(), best_loss_checkpoint_path)
        logging.info(f"New best model saved with Loss: {best_val_loss:.4f}")

    return best_val_f1, best_val_loss


def run_training_epochs(
        ctx: TrainingContext,
        data_module: SliceDataModule,
        model,
        optimizer,
        scaler,
        lr_scheduler,
        best_f1_checkpoint_path: Path,
        best_loss_checkpoint_path: Path,
):
    best_val_f1 = -1.0
    best_val_loss = float('inf')
    training_step = 0

    for epoch in range(ctx.config.hyper_parameters.n_epochs):
        eval_stats, training_step = train_and_evaluate_epoch(
            ctx,
            data_module,
            model,
            optimizer,
            scaler,
            lr_scheduler,
            epoch,
            training_step,
        )
        log_validation_metrics(ctx, epoch, eval_stats)
        best_val_f1, best_val_loss = update_best_checkpoints(
            model,
            eval_stats,
            best_val_f1,
            best_val_loss,
            best_f1_checkpoint_path,
            best_loss_checkpoint_path,
        )


def plot_training_losses(ctx: TrainingContext, checkpoint_dir: Path):
    plt.plot(ctx.ce_losses, label='Cross-Entropy Loss')
    plt.plot(ctx.proj_losses, label='Projection Loss')
    plt.plot(ctx.reg_losses, label='Regularization Loss')
    plt.plot(ctx.swav_losses, label='SwAV Loss')
    plt.plot(ctx.contrast_losses, label='Contrastive Loss')
    plt.plot(
        ctx.attention_distribution_losses,
        label='Attention Distribution Loss',
    )
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Training Loss Convergence')
    plt.savefig(checkpoint_dir / 'training_losses.png', bbox_inches='tight')
    plt.close()

def initialize_tensorboard(
        ctx: TrainingContext,
        checkpoint_dir: Path):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_log_dir = (
        checkpoint_dir
        / "tensorboard"
        / f"{timestamp}-seed-{ctx.config.seed}"
    )
    ctx.writer = SummaryWriter(
        log_dir=str(tensorboard_log_dir),
        flush_secs=10,
    )
    ctx.writer.add_text(
        "run/configuration",
        f"```yaml\n{OmegaConf.to_yaml(ctx.config)}\n```",
        0,
    )
    ctx.writer.flush()
    logging.info(f"TensorBoard log directory: {tensorboard_log_dir}")

def close_tensorboard(ctx: TrainingContext):
    if ctx.writer is None:
        return
    ctx.writer.close()
    ctx.writer = None


def run_test_and_save(
        ctx: TrainingContext,
        model,
        data_module: SliceDataModule,
        checkpoint_dir: Path,
        output_filename: str,
        log_message: str,
        checkpoint_path: Optional[Path] = None,
        checkpoint_description: Optional[str] = None,
        clear_cache: bool = False,
):
    logging.info(log_message)
    if checkpoint_path is not None:
        require_checkpoint(checkpoint_path, checkpoint_description or log_message)
        model.load_state_dict(torch.load(checkpoint_path, map_location=ctx.device))

    test_stats = test(ctx, model, data_module.test_dataloader())
    if clear_cache:
        data_module.clear_cache()
        release_memory()

    logging.info(f"Test Stats: {test_stats}")
    logging.info("Testing completed.")
    with (checkpoint_dir / output_filename).open("w") as wfi:
        json.dump(test_stats, wfi, indent=4)


def run_final_testing(
        ctx: TrainingContext,
        model,
        data_module: SliceDataModule,
        checkpoint_dir: Path,
        best_f1_checkpoint_path: Path,
        best_loss_checkpoint_path: Path,
        include_current_model: bool,
):
    if include_current_model:
        run_test_and_save(
            ctx,
            model,
            data_module,
            checkpoint_dir,
            "test_statistics_epoch100.json",
            "Testing model after training...",
        )

    run_test_and_save(
        ctx,
        model,
        data_module,
        checkpoint_dir,
        "test_statistics_best_val_f1.json",
        "Testing model with best validation F1...",
        checkpoint_path=best_f1_checkpoint_path,
        checkpoint_description="Best validation F1",
        clear_cache=True,
    )

    run_test_and_save(
        ctx,
        model,
        data_module,
        checkpoint_dir,
        "test_statistics_best_val_loss.json",
        "Testing model with best validation loss...",
        checkpoint_path=best_loss_checkpoint_path,
        checkpoint_description="Best validation loss",
        clear_cache=True,
    )


def main(config: DictConfig, args):
    seed_everything(config.seed, workers=True)
    init_log(Path(__file__).stem)
    log_cli_compatibility_warnings(args)

    dataset_root = get_dataset_root(config, args)
    ctx, scaler = build_runtime_context(config)
    vocab, model = load_vocab_and_model(ctx, dataset_root)

    (
        num_samples,
        sampler,
        train_batch_sampler_factory,
        use_imbalanced_sampler,
        use_balanced_batch_sampler,
    ) = build_training_sampling(config, dataset_root)

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_lr_scheduler(ctx, optimizer, num_samples)
    initialize_loss_criteria(ctx)

    (
        checkpoint_dir,
        best_f1_checkpoint_path,
        best_loss_checkpoint_path,
    ) = build_checkpoint_paths(
        ctx,
        model,
        use_imbalanced_sampler,
        use_balanced_batch_sampler,
    )

    logging.info("Loading data module...")
    data_module = SliceDataModule(
        config,
        vocab,
        get_train_batch_size_for_epoch(ctx, 0),
        train_sampler=sampler,
        train_batch_sampler_factory=train_batch_sampler_factory,
        use_temp_data=args.use_temp_data,
    )
    logging.info("Data module loading completed.")

    if not args.skip_training:
        initialize_tensorboard(ctx, checkpoint_dir)
        try:
            run_training_epochs(
                ctx,
                data_module,
                model,
                optimizer,
                scaler,
                lr_scheduler,
                best_f1_checkpoint_path,
                best_loss_checkpoint_path,
            )
            plot_training_losses(ctx, checkpoint_dir)
        finally:
            close_tensorboard(ctx)

    run_final_testing(
        ctx,
        model,
        data_module,
        checkpoint_dir,
        best_f1_checkpoint_path,
        best_loss_checkpoint_path,
        include_current_model=not args.skip_training,
    )

    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()


if __name__ == "__main__":
    filter_warnings()
    arg_parser = get_arg_parser()
    add_swav_arguments(arg_parser)
    args = arg_parser.parse_args()
    config = load_config_from_args(args)
    main(config, args)
