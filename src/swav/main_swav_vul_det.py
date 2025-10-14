import os
import json
import torch
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
from imblearn.under_sampling import ClusterCentroids

from torch_geometric.loader import DataLoader, ImbalancedSampler
from pytorch_lightning import seed_everything
import torch.nn.functional as F
from timm.optim.lars import Lars

from src.common_utils import get_arg_parser, filter_warnings, init_log
from src.vocabulary import Vocabulary
from src.torch_data.datasets import SliceDataset
from src.torch_data.datamodules import SliceDataModule
from src.models.swav_vd import GraphSwAVVD
from src.models.modules.losses import SupConLoss, InfoNCE, OrthogonalProjectionLoss
from src.swav.assignment_protocols import sinkhorn, uot_sinkhorn_gpu
from src.swav.graph_augmentations import generate_SF_augmentations

contrastive_criterion = None
projection_criterion = None
vocab = None
config = None
device = None
resampling_criterion = None

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

def train(train_loader, model, optimizer, epoch, lr_schedule):
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
        
        ce_loss = F.cross_entropy(logits, labels)
        epoch_ce_losses.append(ce_loss.item())

        activations_resampled = torch.empty((0, activations.shape[1]), dtype=torch.float32)

        # if torch.unique(labels).size(0) > 1:
        #     activations_resampled, labels_resampled = resampling_criterion.fit_resample(activations.detach().cpu().numpy(), labels.detach().cpu().numpy())
        if activations_resampled.shape[0] > 2:
            activations_resampled = torch.tensor(activations_resampled, dtype=torch.float32).to(device)
            labels_resampled = torch.tensor(labels_resampled, dtype=torch.long).to(device)
            projection_loss = projection_criterion(activations_resampled, labels_resampled)
        else:
            projection_loss = projection_criterion(activations, labels)
        epoch_proj_losses.append(projection_loss.item())

        regularization_loss = torch.norm(anchor_graph_encodings, dim=-1).mean() + torch.norm(activations, dim=-1).mean()
        epoch_reg_losses.append(regularization_loss.item())
        
        inputs = batched_graph.augmented_views
        _, _, graph_encodings, _, output = zip(*(model(inp.to(device)) for inp in inputs))

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

        if config.swav.contrastive.criterion == "info_nce":
            contrastive_loss = (contrastive_criterion(anchor_graph_encodings, graph_encodings[0]) + contrastive_criterion(anchor_graph_encodings, graph_encodings[1])) / 2.0
        elif config.swav.contrastive.criterion == "supcon":
            contrastive_loss = contrastive_criterion(torch.stack([anchor_graph_encodings, graph_encodings[0], graph_encodings[1]], dim=1), labels=labels)
        elif config.swav.contrastive.criterion == "simclr":
            contrastive_loss = contrastive_criterion(torch.stack([anchor_graph_encodings, graph_encodings[0], graph_encodings[1]], dim=1))
        epoch_contrast_losses.append(contrastive_loss.item())

        progress_bar.set_postfix({
            "ce": ce_loss.item(),
            'proj': projection_loss.item(),
            'reg': regularization_loss.item(),
            'swav': swav_loss.item(),
            'contrast': contrastive_loss.item()
        })

        loss = (
            config.hyper_parameters.lambdas.classification * ce_loss
            + config.hyper_parameters.lambdas.projection * projection_loss
            + config.hyper_parameters.lambdas.regularization * regularization_loss
            + config.hyper_parameters.lambdas.swav * swav_loss
            + config.hyper_parameters.lambdas.contrastive * contrastive_loss
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

        del batched_graph, labels, graphs, logits, activations, activations_resampled, anchor_graph_encodings
        del inputs, graph_encodings, output
        del ce_loss, projection_loss, regularization_loss
        del swav_loss, contrastive_loss, loss

        if (it % 10) == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    logging.info(f"Epoch {epoch + 1} - Cross-Entropy Loss: {np.mean(epoch_ce_losses):.4f}, Projection Loss: {np.mean(epoch_proj_losses):.4f}, Regularization Loss: {np.mean(epoch_reg_losses):.4f}, SwAV Loss: {np.mean(epoch_swav_losses):.4f}, Contrastive Loss: {np.mean(epoch_contrast_losses):.4f}")
    ce_losses.append(float(np.mean(epoch_ce_losses)))
    proj_losses.append(float(np.mean(epoch_proj_losses)))
    reg_losses.append(float(np.mean(epoch_reg_losses)))
    swav_losses.append(float(np.mean(epoch_swav_losses)))
    contrast_losses.append(float(np.mean(epoch_contrast_losses)))

    del train_loader

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

            del batch, labels, graphs, logits, preds, loss
    
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
            labels = batch.labels.to(device)
            graphs = batch.graphs.to(device)
            logits, _, _, _, _ = model(graphs)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            _, preds = logits.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            del batch, labels, graphs, logits, preds, loss
    
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

    logging.info("Building model...")
    model = GraphSwAVVD(config, vocab, vocab_size, pad_idx).to(device)
    logging.info("Model building completed.")

    if config.dataset.name == "BigVul":
        train_slices_filepath = join(dataset_root, f"{config.dataset.version}_{config.train_slices_filename}")
    else:
        train_slices_filepath = join(dataset_root, config.train_slices_filename)
    logging.info(f"Loading training slice paths list from {train_slices_filepath}...")
    with open(train_slices_filepath, "r") as rfi:
        train_slices = json.load(rfi)
    logging.info(f"Completed. Loaded {len(train_slices)} slices.")

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
    train_loader_lens = [math.ceil(len(train_slices) / batch_size) for batch_size in config.hyper_parameters.batch_sizes]
    warmup_lr_schedule = np.linspace(config.swav.start_warmup, config.swav.base_lr, sum([train_loader_lens[i // 10] for i in range(config.swav.warmup_epochs)]))
    iters = np.arange(sum([train_loader_lens[i // 10] for i in range(config.swav.warmup_epochs, config.hyper_parameters.n_epochs)]))
    cosine_lr_schedule = np.array([config.swav.final_lr + 0.5 * (config.swav.base_lr - config.swav.final_lr) * (1 + \
                            math.cos(math.pi * t / (sum([train_loader_lens[i // 10] for i in range(config.swav.warmup_epochs, config.hyper_parameters.n_epochs)])))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    contrastive_criterion = contrastive_options[config.swav.contrastive.criterion](temperature=config.swav.contrastive.temperature)
    projection_criterion = OrthogonalProjectionLoss()
    resampling_criterion = ClusterCentroids(sampling_strategy='auto', random_state=42)

    proj_losses = []
    ce_losses = []
    swav_losses = []
    contrast_losses = []

    dataset_name = basename(config.dataset.name)
    if dataset_name == "BigVul":
        dataset_name = join(dataset_name, config.dataset.version)
    gnn_name = gnn_name_map[config.gnn.name]
    nn_text = "ExcludeNN" if config.exclude_NNs else "IncludeNN"
    cl_warmup_text = "CLWarmup" if config.hyper_parameters.contrastive_warmup_epochs > 0 else "NoCLWarmup"
    contrastive_text = "NoContrastive" if config.hyper_parameters.lambdas.contrastive == 0.0 else config.swav.contrastive.criterion
    do_swav = "NoSwAV" if config.hyper_parameters.lambdas.swav == 0.0 else "DoSwAV"
    gnn_attention_only = "GNNAttentionOnly" if config.gnn.attention_only else "GNNWithPooling"
    use_edge_attr = "WithEdgeAttr" if config.gnn.use_edge_attr else "NoEdgeAttr"
    use_imbalanced_sampler = "WithImbalancedSampler" if config.hyper_parameters.use_imbalanced_sampler else "NoImbalancedSampler"
    checkpoint_dir = join(config.model_save_dir, "graph_swav_classification", dataset_name, gnn_name, nn_text, cl_warmup_text, do_swav, contrastive_text, gnn_attention_only, use_edge_attr, use_imbalanced_sampler)
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
    
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    logging.info("Loading data module...")
    sampler = None
    if config.hyper_parameters.use_imbalanced_sampler:
        logging.info("Using Imbalanced Sampler for training data loader.")
        ys = []
        for slice_path in tqdm(train_slices, desc=f"Slice files"):
            with open(slice_path, "rb") as rbfi:
                slice_graph: nx.DiGraph = pickle.load(rbfi)
                ys.append(slice_graph.graph["label"])
        neg_cnt = ys.count(0)
        pos_cnt = len(ys) - neg_cnt
        majority_cnt = max(neg_cnt, pos_cnt)
        sampler = ImbalancedSampler(torch.tensor(ys, dtype=torch.long), num_samples=majority_cnt*2)
    data_module = SliceDataModule(config, vocab, config.hyper_parameters.batch_sizes[0], train_sampler=sampler, use_temp_data=args.use_temp_data)
    logging.info("Data module loading completed.")
    for epoch in range(config.hyper_parameters.n_epochs):
        train_loader = data_module.train_dataloader()
        train(train_loader, model, optimizer, epoch, lr_schedule)
        data_module.set_train_batch_size(config.hyper_parameters.batch_sizes[min((epoch + 1) // 10, len(config.hyper_parameters.batch_sizes) - 1)])
        eval_stats = eval(model, data_module.val_dataloader())
        if eval_stats["f1"] > best_val_f1:
            best_val_f1 = eval_stats["f1"]
            torch.save(model.state_dict(), best_f1_checkpoint_path)
            logging.info(f"New best model saved with F1: {best_val_f1:.4f}")
        if eval_stats["eval_loss"] < best_val_loss:
            best_val_loss = eval_stats["eval_loss"]
            torch.save(model.state_dict(), best_loss_checkpoint_path)
            logging.info(f"New best model saved with Loss: {best_val_loss:.4f}")
        data_module.clear_cache()
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