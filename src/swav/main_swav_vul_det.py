import os
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

from multiprocessing import cpu_count
from os.path import join, splitext, basename, exists
from omegaconf import DictConfig, OmegaConf
from typing import cast
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

contrastive_criterion = None
projection_criterion = None
vocab = None
config = None
device = None

ce_losses = []
proj_losses = []
reg_losses = []
swav_losses = []
contrast_losses = []

contrastive_options = {
    "supcon": SupConLoss,
    "simclr": SupConLoss,
    "info_nce": InfoNCE
}

assignment_functions = {
    "sinkhorn": sinkhorn,
    "uot_sinkhorn": uot_sinkhorn_gpu
}

gnn_name_map = {
    "gcn": "GCN",
    "gin": "GIN",
    "gine": "GINE",
    "ggnn": "GGNN",
    "gatv2": "GATv2",
    "gated": "Gated",
    "st": "ST"
}

def is_lambda_enabled(lambda_name: str) -> bool:
    return float(config.hyper_parameters.lambdas.get(lambda_name, 0.0)) > 0.0

def is_contrastive_enabled() -> bool:
    return (
        bool(config.swav.contrastive.get("enabled", True))
        and is_lambda_enabled("contrastive")
    )

def validate_training_views(num_views: int):
    if num_views < 2:
        raise ValueError(f"SwAV loss requires at least 2 augmented views, got {num_views}.")
    if is_contrastive_enabled() and num_views != 2:
        raise ValueError(
            "Contrastive loss currently expects exactly 2 augmented views, "
            f"got {num_views}."
        )

def forward_model(
        model,
        graphs,
        forward_fn=None):
    if forward_fn is None:
        forward_fn = model
    return forward_fn(graphs.to(device, non_blocking=True))

def get_train_batch_size_for_epoch(epoch: int) -> int:
    batch_sizes = config.hyper_parameters.batch_sizes
    if len(batch_sizes) == 0:
        raise ValueError("config.hyper_parameters.batch_sizes must contain at least one value.")
    return int(batch_sizes[min(epoch, len(batch_sizes) - 1)])

def get_train_steps_for_epoch(epoch: int, num_samples: int) -> int:
    batch_size = get_train_batch_size_for_epoch(epoch)
    return math.ceil(num_samples / batch_size)

def get_epoch_start_steps(num_samples: int):
    epoch_start_steps = []
    cumulative_steps = 0
    for epoch in range(config.hyper_parameters.n_epochs):
        epoch_start_steps.append(cumulative_steps)
        cumulative_steps += get_train_steps_for_epoch(epoch, num_samples)
    return epoch_start_steps

def build_lr_schedule(num_samples: int):
    n_epochs = int(config.hyper_parameters.n_epochs)
    warmup_epochs = min(int(config.swav.warmup_epochs), n_epochs)
    warmup_steps = sum(
        get_train_steps_for_epoch(epoch, num_samples)
        for epoch in range(warmup_epochs)
    )
    cosine_steps = sum(
        get_train_steps_for_epoch(epoch, num_samples)
        for epoch in range(warmup_epochs, n_epochs)
    )

    warmup_lr_schedule = np.linspace(
        config.swav.start_warmup,
        config.swav.base_lr,
        warmup_steps,
    )
    if cosine_steps == 0:
        return warmup_lr_schedule

    iters = np.arange(cosine_steps)
    cosine_lr_schedule = np.array([
        config.swav.final_lr
        + 0.5 * (config.swav.base_lr - config.swav.final_lr)
        * (1 + math.cos(math.pi * t / cosine_steps))
        for t in iters
    ])
    return np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

def load_train_dataset_stats(train_slices, train_stats_filepath):
    if not exists(train_stats_filepath):
        logging.info(f"{train_stats_filepath} not found. Retrieving labels...")
        ys = []
        for slice_path in tqdm(train_slices, desc=f"Slice files"):
            with open(slice_path, "rb") as rbfi:
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
        with open(train_stats_filepath, "w") as wfi:
            json.dump(dataset_stats, wfi, indent=2)
        return dataset_stats

    logging.info(f"Reading dataset stats from {train_stats_filepath}...")
    with open(train_stats_filepath, "r") as rfi:
        dataset_stats = json.load(rfi)
    logging.info(f"Completed. Retrieved stats.")
    return dataset_stats

def build_all_views_batch(batched_graph):
    graphs = batched_graph.graphs.to_data_list()
    augmented_views = [
        view
        for view_batch in batched_graph.augmented_views
        for view in view_batch.to_data_list()
    ]
    return Batch.from_data_list(graphs + augmented_views)

def compute_training_loss(model, batched_graph):
    num_views = len(batched_graph.augmented_views)
    validate_training_views(num_views)

    labels = batched_graph.labels.to(device, non_blocking=True)
    combined_graphs = getattr(batched_graph, "all_views", None)
    if combined_graphs is None:
        combined_graphs = build_all_views_batch(batched_graph)
    logits_all, activations_all, graph_encodings_all, _, output_all = forward_model(
        model,
        combined_graphs,
    )
    batch_size = batched_graph.sz
    logits = logits_all[:batch_size]
    activations = activations_all[:batch_size]
    anchor_graph_encodings = graph_encodings_all[:batch_size]
    graph_encodings = graph_encodings_all[batch_size:].chunk(num_views)
    output = output_all[batch_size:].chunk(num_views)

    zero_loss = logits.new_zeros(())

    ce_loss = F.cross_entropy(logits, labels)

    projection_loss = zero_loss
    if is_lambda_enabled("projection"):
        if projection_criterion is None:
            raise RuntimeError("projection_criterion is not initialized.")
        projection_loss = projection_criterion(activations, labels)

    regularization_loss = zero_loss
    if is_lambda_enabled("regularization"):
        regularization_loss = torch.norm(anchor_graph_encodings, dim=-1).mean() + torch.norm(activations, dim=-1).mean()

    swav_loss = logits.new_zeros(())
    if is_lambda_enabled("swav"):
        views_for_assign = [view_id for view_id in config.swav.views_for_assign if view_id < num_views]
        if not views_for_assign:
            raise ValueError(f"No valid SwAV assignment views for {num_views} augmented views.")
        for view_id in views_for_assign:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    out = output[view_id].detach().float()
                    q = assignment_functions[config.swav.assignment_protocol](out)
            subloss = logits.new_zeros(())
            for v in np.delete(np.arange(num_views), view_id):
                x = output[v] / config.swav.temperature
                subloss = subloss - torch.mean(torch.sum(q * F.log_softmax(x, dim=-1), dim=-1))
            swav_loss = swav_loss + subloss / (num_views - 1)
        swav_loss = swav_loss / len(views_for_assign)

    contrastive_loss = logits.new_zeros(())
    if is_contrastive_enabled():
        if contrastive_criterion is None:
            raise RuntimeError("contrastive_criterion is not initialized.")
        if config.swav.contrastive.criterion == "info_nce":
            contrastive_loss = (contrastive_criterion(anchor_graph_encodings, graph_encodings[0]) + contrastive_criterion(anchor_graph_encodings, graph_encodings[1])) / 2.0
        elif config.swav.contrastive.criterion == "supcon":
            contrastive_loss = contrastive_criterion(torch.stack([anchor_graph_encodings, graph_encodings[0], graph_encodings[1]], dim=1), labels=labels)
        elif config.swav.contrastive.criterion == "simclr":
            contrastive_loss = contrastive_criterion(torch.stack([anchor_graph_encodings, graph_encodings[0], graph_encodings[1]], dim=1))
        else:
            raise ValueError(f"Unsupported contrastive criterion: {config.swav.contrastive.criterion}")

    loss = (
        config.hyper_parameters.lambdas.classification * ce_loss
        + config.hyper_parameters.lambdas.projection * projection_loss
        + config.hyper_parameters.lambdas.regularization * regularization_loss
        + config.hyper_parameters.lambdas.swav * swav_loss
        + config.hyper_parameters.lambdas.contrastive * contrastive_loss
    )

    metrics = {
        "ce": ce_loss.item(),
        "proj": projection_loss.item(),
        "reg": regularization_loss.item(),
        "swav": swav_loss.item(),
        "contrast": contrastive_loss.item(),
    }
    return loss, metrics

def run_training_step(model, optimizer, scaler, batched_graph, iteration, use_amp):
    optimizer.zero_grad(set_to_none=True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        loss, metrics = compute_training_loss(model, batched_graph)
    scaler.scale(loss).backward()
    # cancel gradients for the prototypes
    if iteration < config.swav.freeze_prototypes_niters:
        for name, p in model.named_parameters():
            if "swav_prototypes" not in name:
                continue
            p.grad = None

    scaler.step(optimizer)
    scaler.update()
    del loss
    return metrics

def train(train_loader, model, optimizer, scaler, epoch, lr_schedule, epoch_start_step, use_amp):
    logging.info(f"Epoch {epoch + 1}")
    model.train()
    progress_bar = tqdm(train_loader, desc="Training")

    epoch_ce_losses = []
    epoch_proj_losses = []
    epoch_reg_losses = []
    epoch_swav_losses = []
    epoch_contrast_losses = []

    for it, batched_graph in enumerate(progress_bar):
        # update learning rate
        iteration = epoch_start_step + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        with torch.no_grad():
            w = model.swav_prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            model.swav_prototypes.weight.copy_(w)

        batch_losses = run_training_step(
            model,
            optimizer,
            scaler,
            batched_graph,
            iteration,
            use_amp,
        )

        epoch_ce_losses.append(batch_losses["ce"])
        epoch_proj_losses.append(batch_losses["proj"])
        epoch_reg_losses.append(batch_losses["reg"])
        epoch_swav_losses.append(batch_losses["swav"])
        epoch_contrast_losses.append(batch_losses["contrast"])

        progress_bar.set_postfix({
            "ce": batch_losses["ce"],
            'proj': batch_losses["proj"],
            'reg': batch_losses["reg"],
            'swav': batch_losses["swav"],
            'contrast': batch_losses["contrast"]
        })
        del batched_graph, batch_losses

    logging.info(f"Epoch {epoch + 1} - Cross-Entropy Loss: {np.mean(epoch_ce_losses):.4f}, Projection Loss: {np.mean(epoch_proj_losses):.4f}, Regularization Loss: {np.mean(epoch_reg_losses):.4f}, SwAV Loss: {np.mean(epoch_swav_losses):.4f}, Contrastive Loss: {np.mean(epoch_contrast_losses):.4f}")
    ce_losses.append(float(np.mean(epoch_ce_losses)))
    proj_losses.append(float(np.mean(epoch_proj_losses)))
    reg_losses.append(float(np.mean(epoch_reg_losses)))
    swav_losses.append(float(np.mean(epoch_swav_losses)))
    contrast_losses.append(float(np.mean(epoch_contrast_losses)))

    del train_loader

def evaluate(model, val_loader):
    progress_bar = tqdm(val_loader, desc="Validation")
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in progress_bar:
            labels = batch.labels.to(device, non_blocking=True)
            logits = forward_model(
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

def test(model, test_loader):
    progress_bar = tqdm(test_loader, desc="Testing")
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in progress_bar:
            labels = batch.labels.to(device, non_blocking=True)
            logits = forward_model(
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

def require_checkpoint(checkpoint_path: str, description: str):
    if not exists(checkpoint_path):
        raise FileNotFoundError(
            f"{description} checkpoint not found at {checkpoint_path}. "
            "Run training first, or use --skip_training only when the checkpoint already exists."
        )

if __name__ == "__main__":
    filter_warnings()
    arg_parser = get_arg_parser()
    arg_parser.add_argument("--skip_training", action='store_true', help="Skip training phase")
    arg_parser.add_argument("--no_cl", action='store_true', help="Disable contrastive learning")
    arg_parser.add_argument("--exclude_NNs", action='store_true', help="Exclude NN pairs if contrastive pair filtering is used")
    arg_parser.add_argument("--use_lr_warmup", action='store_true', help="Enable learning-rate warmup in the config")
    arg_parser.add_argument("--no_cl_warmup", action='store_true', help="Disable contrastive learning warmup in the config")
    arg_parser.add_argument("--train_batch_size", type=int, default=None,
                            help="Override all configured training batch sizes.")
    arg_parser.add_argument("--test_batch_size", type=int, default=None,
                            help="Override validation and test batch size.")

    args = arg_parser.parse_args()

    config = cast(DictConfig, OmegaConf.load(args.config))
    seed_everything(config.seed, workers=True)

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
    init_log(splitext(basename(__file__))[0])

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

    if config.num_workers != -1:
        USE_CPU = min(config.num_workers, cpu_count())
    else:
        USE_CPU = cpu_count()

    dataset_root = join(config.data_folder, config.dataset.name)
    if args.use_temp_data:
        dataset_root = config.temp_root
    
    vocab = Vocabulary.from_w2v(join(dataset_root, "w2v.wv"))
    vocab_size = vocab.get_vocab_size()
    pad_idx = vocab.get_pad_id()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = bool(config.hyper_parameters.get("use_amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    logging.info(f"Device: {device}")
    logging.info(f"AMP enabled: {use_amp}")

    logging.info("Building model...")
    model = GraphSwAVVD(config, vocab, vocab_size, pad_idx).to(device)
    logging.info("Model building completed.")

    # if config.dataset.name == "BigVul":
    #     train_slices_filepath = join(dataset_root, f"{config.dataset.version}_{config.train_slices_filename}")
    # else:
    train_slices_filepath = join(dataset_root, config.train_slices_filename)
    logging.info(f"Loading training slice paths list from {train_slices_filepath}...")
    with open(train_slices_filepath, "r") as rfi:
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
        train_stats_filepath = join(dataset_root, config.train_stats_filename)
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

    # optimizer = torch.optim.AdamW([{
    #             "params": p
    #         } for p in model.parameters()], config.hyper_parameters.learning_rate)
    
    # Using LARS optimizer as per the SwAV paper
    optimizer = Lars(
        model.parameters(),
        lr=config.swav.base_lr,
        momentum=0.9,
        weight_decay=config.swav.wd
    )
    lr_schedule = build_lr_schedule(num_samples)
    epoch_start_steps = get_epoch_start_steps(num_samples)

    contrastive_criterion = None
    if is_contrastive_enabled():
        contrastive_criterion = contrastive_options[config.swav.contrastive.criterion](temperature=config.swav.contrastive.temperature)
    projection_criterion = OrthogonalProjectionLoss()

    proj_losses = []
    ce_losses = []
    swav_losses = []
    contrast_losses = []

    dataset_name = basename(config.dataset.name)
    # if dataset_name == "BigVul":
    #     dataset_name = join(dataset_name, config.dataset.version)
    gnn_name = gnn_name_map[config.gnn.name]
    nn_text = "ExcludeNN" if config.exclude_NNs else "IncludeNN"
    cl_warmup_text = "CLWarmup" if config.hyper_parameters.contrastive_warmup_epochs > 0 else "NoCLWarmup"
    contrastive_text = "NoContrastive" if not is_contrastive_enabled() else config.swav.contrastive.criterion
    do_swav = "NoSwAV" if config.hyper_parameters.lambdas.swav == 0.0 else "DoSwAV"
    gnn_attention_only = "GNNAttentionOnly" if config.gnn.attention_only else "GNNWithPooling"
    use_edge_attr = "WithEdgeAttr" if config.gnn.use_edge_attr else "NoEdgeAttr"
    if use_balanced_batch_sampler:
        sampler_text = "WithBalancedBatchSampler"
    elif use_imbalanced_sampler:
        sampler_text = "WithImbalancedSampler"
    else:
        sampler_text = "NoSampler"
    checkpoint_dir = join(config.model_save_dir, "graph_swav_classification", dataset_name, gnn_name, nn_text, cl_warmup_text, do_swav, contrastive_text, gnn_attention_only, use_edge_attr, sampler_text)
    if not exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    model_name = model.__class__.__name__
    best_f1_directory = join(checkpoint_dir, "best_f1")
    if not exists(best_f1_directory):
        os.makedirs(best_f1_directory, exist_ok=True)
    best_f1_checkpoint_path = join(best_f1_directory, f"{model_name}.ckpt")
    best_loss_directory = join(checkpoint_dir, "best_loss")
    if not exists(best_loss_directory):
        os.makedirs(best_loss_directory, exist_ok=True)
    best_loss_checkpoint_path = join(best_loss_directory, f"{model_name}.ckpt")
    logging.info(f"Checkpoint directory: {checkpoint_dir}")
    
    best_val_f1 = -1.0
    best_val_loss = float('inf')
    logging.info("Loading data module...")
    data_module = SliceDataModule(
        config,
        vocab,
        get_train_batch_size_for_epoch(0),
        train_sampler=sampler,
        train_batch_sampler_factory=train_batch_sampler_factory,
        use_temp_data=args.use_temp_data,
    )
    logging.info("Data module loading completed.")
    if not args.skip_training:
        for epoch in range(config.hyper_parameters.n_epochs):
            data_module.set_train_batch_size(get_train_batch_size_for_epoch(epoch))
            train_loader = data_module.train_dataloader()
            train(
                train_loader,
                model,
                optimizer,
                scaler,
                epoch,
                lr_schedule,
                epoch_start_steps[epoch],
                use_amp,
            )
            eval_stats = evaluate(model, data_module.val_dataloader())
            if eval_stats["f1"] > best_val_f1:
                best_val_f1 = eval_stats["f1"]
                torch.save(model.state_dict(), best_f1_checkpoint_path)
                logging.info(f"New best model saved with F1: {best_val_f1:.4f}")
            if eval_stats["eval_loss"] < best_val_loss:
                best_val_loss = eval_stats["eval_loss"]
                torch.save(model.state_dict(), best_loss_checkpoint_path)
                logging.info(f"New best model saved with Loss: {best_val_loss:.4f}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

            cache_info = data_module.get_cache_info()
            if 'train' in cache_info:
                print(f"Cache Info: {cache_info['train']}")

        plt.plot(ce_losses, label='Cross-Entropy Loss')
        plt.plot(proj_losses, label='Projection Loss')
        plt.plot(reg_losses, label='Regularization Loss')
        plt.plot(swav_losses, label='SwAV Loss')
        plt.plot(contrast_losses, label='Contrastive Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.title('Training Loss Convergence')
        plt.savefig(join(checkpoint_dir, 'training_losses.png'), bbox_inches='tight')
        plt.close()

        logging.info("Testing model after training...")
        test_stats = test(model, data_module.test_dataloader())
        logging.info(f"Test Stats: {test_stats}")
        logging.info("Testing completed.")

        with open(join(checkpoint_dir, "test_statistics_epoch100.json"), "w") as wfi:
            json.dump(test_stats, wfi, indent=4)
    
    logging.info("Testing model with best validation F1...")
    require_checkpoint(best_f1_checkpoint_path, "Best validation F1")
    model.load_state_dict(torch.load(best_f1_checkpoint_path, map_location=device))
    test_stats = test(model, data_module.test_dataloader())
    data_module.clear_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()
    logging.info(f"Test Stats: {test_stats}")
    logging.info("Testing completed.")
    with open(join(checkpoint_dir, "test_statistics_best_val_f1.json"), "w") as wfi:
        json.dump(test_stats, wfi, indent=4)
    
    logging.info("Testing model with best validation loss...")
    require_checkpoint(best_loss_checkpoint_path, "Best validation loss")
    model.load_state_dict(torch.load(best_loss_checkpoint_path, map_location=device))
    test_stats = test(model, data_module.test_dataloader())
    
    data_module.clear_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    logging.info(f"Test Stats: {test_stats}")
    logging.info("Testing completed.")
    with open(join(checkpoint_dir, "test_statistics_best_val_loss.json"), "w") as wfi:
        json.dump(test_stats, wfi, indent=4)
    
    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()
