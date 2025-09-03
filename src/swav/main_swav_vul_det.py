import os
import json
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

import logging

from multiprocessing import cpu_count
from os.path import join, splitext, basename, exists
from omegaconf import DictConfig, OmegaConf
from typing import cast
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pytorch_lightning import seed_everything
import torch.nn.functional as F
from timm.optim.lars import Lars

from src.common_utils import get_arg_parser, filter_warnings, init_log
from src.vocabulary import Vocabulary
from src.torch_data.datamodules import SliceDataModule
from src.models.swav_vd import GraphSwAVVD
from src.models.modules.losses import SupConLoss, InfoNCEContrastiveLoss, OrthogonalProjectionLoss
from src.swav.assignment_protocols import sinkhorn, uot_sinkhorn_gpu
from src.swav.graph_augmentations import generate_SF_augmentations

contrastive_criterion = None
projection_criterion = None
vocab = None
config = None
device = None

proj_losses = []
ce_losses = []
swav_losses = []
contrast_losses = []

contrastive_options = {
    "supcon": SupConLoss,
    "info_nce": InfoNCEContrastiveLoss
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

def train(train_loader, model, optimizer, epoch, lr_schedule):
    logging.info(f"Epoch {epoch + 1}")
    model.train()
    progress_bar = tqdm(train_loader, desc="Training")

    epoch_proj_losses = []
    epoch_ce_losses = []
    epoch_swav_losses = []
    epoch_contrast_losses = []

    for it, batched_graph in enumerate(progress_bar):
        # update learning rate
        iteration = epoch * len(train_loader) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[iteration]

        with torch.no_grad():
            w = model.swav_prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            model.swav_prototypes.weight.copy_(w)
        
        labels = batched_graph.labels.to(device)
        graphs = batched_graph.graphs.to(device)
        logits, activations, anchor_graph_encodings, _, _ = model(graphs)
        projection_loss = projection_criterion(activations, labels)
        epoch_proj_losses.append(projection_loss.item())
        proj_losses.append(projection_loss.item())

        ce_loss = F.cross_entropy(logits, labels)
        epoch_ce_losses.append(ce_loss.item())
        ce_losses.append(ce_loss.item())
        
        inputs = generate_SF_augmentations(batched_graph, vocab, config.dataset.token.max_parts)
        _, _, graph_encodings, _, output = zip(*(model(inp.graphs.to(device)) for inp in inputs))

        swav_loss = 0
        for view_id in config.swav.views_for_assign:
            with torch.no_grad():
                out = output[view_id].detach()
                q = assignment_functions[config.swav.assignment_protocol](out)
            subloss = 0
            for v in np.delete(np.arange(np.sum(config.swav.nmb_views)), view_id):
                x = output[v] / config.swav.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=-1), dim=-1))
            swav_loss += subloss / (np.sum(config.swav.nmb_views) - 1)
        swav_loss /= len(config.swav.views_for_assign)
        epoch_swav_losses.append(swav_loss.item())
        swav_losses.append(swav_loss.item())

        h1, h2 = F.normalize(graph_encodings[0], dim=-1), F.normalize(graph_encodings[1], dim=-1)
        if config.swav.contrastive.criterion == "info_nce":
            contrastive_loss = contrastive_criterion(h1, h2)
        elif config.swav.contrastive.criterion == "supcon":
            contrastive_loss = contrastive_criterion(torch.stack([F.normalize(anchor_graph_encodings, dim=-1), h1, h2], dim=1), labels=labels)
        epoch_contrast_losses.append(contrastive_loss.item())
        contrast_losses.append(contrastive_loss.item())

        progress_bar.set_postfix({
            'proj': projection_loss.item(),
            "ce": ce_loss.item(),
            'swav': swav_loss.item(),
            'contrast': contrastive_loss.item()
        })

        loss = (
            ce_loss
            + config.hyper_parameters.projection_weight_factor * projection_loss
            + config.swav.weight_factor * swav_loss
            + config.swav.weight_factor * config.swav.contrastive.lambda_h * contrastive_loss
        )

        optimizer.zero_grad()
        loss.backward()
        # cancel gradients for the prototypes
        if iteration < config.swav.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "swav_prototypes" not in name:
                    continue
                p.grad = None

        optimizer.step()

    logging.info(f"Epoch {epoch + 1} - Cross-Entropy Loss: {np.mean(epoch_ce_losses):.4f}, Projection Loss: {np.mean(epoch_proj_losses):.4f}, SwAV Loss: {np.mean(epoch_swav_losses):.4f}, Contrastive Loss: {np.mean(epoch_contrast_losses):.4f}")

def eval(model, val_loader):
    progress_bar = tqdm(val_loader, desc="Validation")
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in progress_bar:
            labels = batch.labels.to(device)
            graphs = batch.graphs.to(device)
            logits, _, _, _, _ = model(graphs)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            _, preds = logits.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    stats = {
        "eval_loss": total_loss / len(val_loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds)
    }
    
    return stats

def test(model, test_loader):
    progress_bar = tqdm(test_loader, desc="Testing")
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in progress_bar:
            labels = batch.labels.to(device)
            graphs = batch.graphs.to(device)
            logits, _, _, _, _ = model(graphs)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            _, preds = logits.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    stats = {
        "test_loss": total_loss / len(test_loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds)
    }
    
    return stats

if __name__ == "__main__":
    filter_warnings()
    arg_parser = get_arg_parser()
    arg_parser.add_argument("--no_cl", action='store_true', help="Use contrastive learning")
    arg_parser.add_argument("--exclude_NNs", action='store_true', help="Exclude NN pairs during contrastive learning")
    arg_parser.add_argument("--use_lr_warmup", action='store_true', help="Exclude Learning Rate warmup")
    arg_parser.add_argument("--no_cl_warmup", action='store_true', help="Use contrastive learning warmup")

    args = arg_parser.parse_args()

    config = cast(DictConfig, OmegaConf.load(args.config))
    seed_everything(config.seed, workers=True)

    if args.exclude_NNs:
        config.exclude_NNs = True
    if args.use_lr_warmup:
        config.hyper_parameters.use_warmup_lr = True
    if args.no_cl_warmup:
        config.hyper_parameters.contrastive_warmup_epochs = 0

    init_log(splitext(basename(__file__))[0])

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

    logging.info("Loading data module...")
    data_module = SliceDataModule(config, vocab, use_temp_data=args.use_temp_data)
    logging.info("Data module loading completed.")

    logging.info("Building model...")
    model = GraphSwAVVD(config, vocab, vocab_size, pad_idx).to(device)
    logging.info("Model building completed.")

    train_loader = data_module.train_dataloader()

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
    warmup_lr_schedule = np.linspace(config.swav.start_warmup, config.swav.base_lr, len(train_loader) * config.swav.warmup_epochs)
    iters = np.arange(len(train_loader) * (config.hyper_parameters.n_epochs - config.swav.warmup_epochs))
    cosine_lr_schedule = np.array([config.swav.final_lr + 0.5 * (config.swav.base_lr - config.swav.final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader) * (config.hyper_parameters.n_epochs - config.swav.warmup_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    contrastive_criterion = contrastive_options[config.swav.contrastive.criterion](temperature=config.swav.contrastive.temperature)
    projection_criterion = OrthogonalProjectionLoss()

    proj_losses = []
    ce_losses = []
    swav_losses = []
    contrast_losses = []

    dataset_name = basename(config.dataset.name)
    gnn_name = gnn_name_map[config.gnn.name]
    nn_text = "ExcludeNN" if config.exclude_NNs else "IncludeNN"
    cl_warmup_text = "CLWarmup" if config.hyper_parameters.contrastive_warmup_epochs > 0 else "NoCLWarmup"
    checkpoint_dir = join(config.model_save_dir, "graph_swav_classification", dataset_name, gnn_name, nn_text, cl_warmup_text)
    if not exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    model_name = model.__class__.__name__
    checkpoint_path = join(checkpoint_dir, f"{model_name}.ckpt")
    
    best_val_f1 = 0.0
    for epoch in range(config.hyper_parameters.n_epochs):
        train(train_loader, model, optimizer, epoch, lr_schedule)
        eval_stats = eval(model, data_module.val_dataloader())
        if eval_stats["f1"] <= best_val_f1:
            continue
        best_val_f1 = eval_stats["f1"]
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"New best model saved with F1: {best_val_f1:.4f}")

    plt.plot(proj_losses, label='Projection Loss')
    plt.plot(ce_losses, label='Cross-Entropy Loss')
    plt.plot(swav_losses, label='SwAV Loss')
    plt.plot(contrast_losses, label='Contrastive Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Training Loss Convergence')
    plt.savefig(join(dataset_root, 'training_losses.png'), bbox_inches='tight')
    plt.close()

    logging.info("Testing model...")
    test_stats = test(model, data_module.test_dataloader())
    logging.info(f"Test Stats: {test_stats}")
    logging.info("Testing completed.")

    with open(join(dataset_root, "test_statistics.json"), "w") as wfi:
        json.dump(test_stats, wfi, indent=4)
    
    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()