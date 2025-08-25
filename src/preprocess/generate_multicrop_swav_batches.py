import random
import os
import json
import torch
import ot
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


import logging

from multiprocessing import cpu_count
from omegaconf import DictConfig, OmegaConf
from typing import List, cast
from tqdm import tqdm
from math import ceil
from os.path import join, isdir, exists
from collections import defaultdict

from pytorch_lightning import seed_everything

from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_node, dropout_edge, k_hop_subgraph
import torch.nn.functional as F

from src.common_utils import get_arg_parser
from src.vocabulary import Vocabulary
from src.torch_data.graphs import SliceGraph
from src.torch_data.samples import SliceGraphSample, SliceGraphBatch

from src.models.modules.gnns import GraphSwAVModel
from src.models.modules.losses import InfoNCEContrastiveLoss

def init_log():
    LOG_DIR = "logs"
    if not isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logging.basicConfig(
        handlers=[
            logging.FileHandler(join(LOG_DIR, "generate_multicrop_swav_batches.log")),
            logging.StreamHandler()
        ],
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("=========New session=========")
    logging.info(f"Logging dir: {LOG_DIR}")

def random_node_drop(batch: Batch, p: float = 0.1) -> Batch:
    """
    Apply random node drop per graph in the batch—but only when
    at least one node remains. Otherwise keep original graph.
    """
    all_perturbed_graphs = []
    for graph_data in batch.to_data_list():
        num_nodes = graph_data.num_nodes
        edge_index, edge_mask, node_mask = dropout_node(
            graph_data.edge_index,
            p=p,
            num_nodes=num_nodes,
            training=True,
            relabel_nodes=True
        )
        # If all nodes dropped => skip augmentation
        if node_mask.sum().item() < 1:
            all_perturbed_graphs.append(graph_data)
            continue

        x = graph_data.x[node_mask]
        edge_attr = None
        if graph_data.edge_attr is not None:
            edge_attr = graph_data.edge_attr[edge_mask]
        new_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        all_perturbed_graphs.append(new_data)

    return Batch.from_data_list(all_perturbed_graphs)

def random_edge_drop(batch: Batch, p: float = 0.1) -> Batch:
    """Randomly drop edges from each graph in the batch with probability p."""
    edge_index, edge_mask = dropout_edge(batch.edge_index, p=p, training=True, force_undirected=False)
    new_batch = batch.clone()
    new_batch.edge_index = edge_index
    if batch.edge_attr is not None:
        new_batch.edge_attr = new_batch.edge_attr[edge_mask]
    return new_batch

def attribute_mask(batch: Batch, mask_id: int, mask_rate: float = 0.1, node_feature_name='x') -> Batch:
    """Randomly mask node features in the batch with probability mask_rate."""
    new_batch = batch.clone()
    mask = torch.rand(new_batch.num_nodes, device=new_batch.x.device) < mask_rate
    new_batch.x = new_batch.x.clone()
    new_batch.x[mask] = mask_id  # or any masking value
    return new_batch

def augment(batch: Batch, mask_id: int) -> Batch:
    """Compose augmentations to produce a random view of the batch."""
    batch_aug = random_node_drop(batch, p=0.1)
    batch_aug = random_edge_drop(batch_aug, p=0.1)
    batch_aug = attribute_mask(batch_aug, mask_id, mask_rate=0.1)
    return batch_aug

def subgraph_crop(batch: Batch, num_hops: int = 2, ratio: float = 0.5) -> Batch:
    all_subgraphs = []
    for graph_data in batch.to_data_list():
        num_nodes = graph_data.num_nodes
        keep = max(1, int(num_nodes * ratio))
        seed = torch.randperm(num_nodes)[:keep]
        subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(seed, num_hops, graph_data.edge_index, relabel_nodes=True)
        subgraph = Data(x=graph_data.x[subset],
                   edge_index=edge_index_sub,
                   edge_attr=graph_data.edge_attr[edge_mask])
        all_subgraphs.append(subgraph)

    return Batch.from_data_list(all_subgraphs)

def augment_multicrop(batch: Batch,
                      mask_id: int,
                      n_local_views: int = 2) -> List[Batch]:
    """
    Generate multiple views per batch for SwAV-style multi-crop:
    - 2 global augmented views
    - n_local_views subgraph-cropped views
    """
    views = []
    # Two global augmented views
    views.append(augment(batch, mask_id=mask_id))  # global view 1
    views.append(augment(batch, mask_id=mask_id))  # global view 2

    # Local subgraph views
    for _ in range(n_local_views):
        crop = subgraph_crop(batch)
        crop_aug = augment(crop, mask_id=mask_id)
        views.append(crop_aug)
    return views

def sinkhorn(out, n_iters=3, epsilon=0.05):
    Q = torch.exp(out / epsilon).t()
    Q /= Q.sum()
    for _ in range(n_iters):
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= Q.sum(dim=0, keepdim=True)
    return (Q / Q.sum(dim=0, keepdim=True)).t()

def uot_sinkhorn_gpu(scores: torch.Tensor,
                     epsilon=0.05,
                     rho=0.5,
                     n_iters=5):
    """
    GPU-only, batched unbalanced Sinkhorn-like updates.
    scores: [B, K] = z @ prototypes
    Returns Q: [B, K] soft assignment distributions.
    """
    Q = torch.exp(scores / epsilon)  # non-negative weights [B,K]
    for _ in range(n_iters):
        # Row update with relaxation: softly approach uniform mass
        row_sum = Q.sum(dim=1, keepdim=True)
        Q = Q / (row_sum + rho)
        # Column update with relaxation
        col_sum = Q.sum(dim=0, keepdim=True)
        Q = Q / (col_sum + rho)
    # Normalize rows to sum to one (softmax-like)
    Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-8)
    return Q

def get_merged_clusters(cluster_ids: List[int], prototypes, min_size: int = 2) -> List[int]:
    """
    Merge small clusters into larger ones.
    """
    N = prototypes.size(1)
    with torch.no_grad():
        proto_sim = prototypes.T @ prototypes

    cluster_ids_tensor = torch.tensor(cluster_ids, dtype=torch.int64)
    all_ids = torch.arange(N, device=cluster_ids_tensor.device)

    unique, counts = torch.unique(cluster_ids_tensor, return_counts=True)
    small_clusters = unique[counts < min_size].tolist()

    while len(small_clusters) > 0:
        for cid in small_clusters:
            c_proto_sym = proto_sim[cid]
            c_proto_sym[cid] = float('-inf')  # Exclude self-similarity
            non_empty_proto_mask = torch.isin(all_ids, unique)
            c_proto_sym[~non_empty_proto_mask] = float('-inf')  # Exclude empty clusters
            best_match = c_proto_sym.argmax().item()
            cluster_ids_tensor[cluster_ids_tensor == cid] = best_match
            unique, counts = torch.unique(cluster_ids_tensor, return_counts=True)
            small_clusters = unique[counts < min_size].tolist()
            if len(small_clusters) == 0:
                break
    
    return cluster_ids_tensor.tolist()

if __name__ == "__main__":
    arg_parser = get_arg_parser()
    arg_parser.add_argument("--do_train", action="store_true", help="Enable training; if not set, use pretrained model.")
    args = arg_parser.parse_args()
    init_log()

    config = cast(DictConfig, OmegaConf.load(args.config))
    if config.num_workers != -1:
        USE_CPU = min(config.num_workers, cpu_count())
    else:
        USE_CPU = cpu_count()

    dataset_root = join(config.data_folder, config.dataset.name)
    if args.use_temp_data:
        dataset_root = config.temp_root
    
    seed_everything(config.seed, workers=True)

    max_len = config.dataset.token.max_parts
    vocab = Vocabulary.from_w2v(config.gnn.w2v_path)
    vocab_size = vocab.get_vocab_size()
    pad_idx = vocab.get_pad_id()
    
    train_slices_filepath = join(dataset_root, config.train_slices_filename)
    logging.info(f"Loading training slice paths list from {train_slices_filepath}...")
    with open(train_slices_filepath, "r") as rfi:
        train_slices = json.load(rfi)
    logging.info(f"Completed. Loaded {len(train_slices)} slices.")

    logging.info(f"Going over {len(train_slices)} files...")

    sample_list = []
    for slice_path in tqdm(train_slices, desc=f"Slice files"):
        slice_graph = SliceGraph(slice_path)
        sample_list.append(SliceGraphSample(graph=slice_graph.to_torch_graph(vocab, max_len),
                            label=slice_graph.label))
    logging.info(f"Completed. Loaded {len(sample_list)} samples.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSwAVModel(config, vocab, vocab_size, pad_idx).to(device)
    optimizer = torch.optim.AdamW([{
                "params": p
            } for p in model.parameters()], config.swav.learning_rate)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #             optimizer,
    #             lr_lambda=lambda epoch: config.hyper_parameters.decay_gamma
    #                                 ** epoch)

    K = max(1024, ceil(len(train_slices) / config.hyper_parameters.batch_size))
    D = config.gnn.projection_dim
    prototypes = torch.nn.Parameter(torch.randn(D, K, device=device))
    logging.info(f"Initialized prototypes with shape {prototypes.shape}.")
    with torch.no_grad():
        prototypes.data = F.normalize(prototypes.data, dim=0)

    BATCH_SIZE = 256
    
    contrastive_criterion = InfoNCEContrastiveLoss(temperature=config.swav.contrastive.temperature)

    swav_losses = []
    contrast_losses = []

    if args.do_train:
        for epoch in range(config.swav.n_epochs):
            logging.info(f"Epoch {epoch + 1}")
            freeze_proto = (epoch == 0)
            if freeze_proto:
                optimizer = torch.optim.AdamW([{
                    "params": p
                } for p in model.parameters()], config.hyper_parameters.learning_rate)
            else:
                optimizer = torch.optim.AdamW([{
                    "params": p
                } for p in list(model.parameters()) + [prototypes]], config.hyper_parameters.learning_rate)
            model.train()
            random.shuffle(sample_list)
            progress_bar = tqdm(range(0, len(sample_list), BATCH_SIZE))

            epoch_swav_losses = []
            epoch_contrast_losses = []

            for i in progress_bar:
                batched_graph = SliceGraphBatch(sample_list[i:i + BATCH_SIZE])
                batched_graph = batched_graph.graphs.to(device)
                views = augment_multicrop(batched_graph, mask_id=vocab.get_unk_id(), n_local_views=4)
                zs, features = zip(*(model(v) for v in views))
                scores = [z @ prototypes for z in zs]
                with torch.no_grad():
                    # qs = [sinkhorn(s) for s in scores]
                    qs = [uot_sinkhorn_gpu(s) for s in scores]
                # ps = [F.softmax(s / config.swav.temperature, dim=1) for s in scores]

                swav_loss = 0
                global_idxs = [0, 1]
                for i in global_idxs:
                    subloss = 0
                    for j in range(len(zs)):
                        if j == i:
                            continue
                        x = scores[i] / config.swav.temperature
                        subloss -= torch.mean(torch.sum(qs[j] * F.log_softmax(x, dim=-1), dim=-1))
                    swav_loss += subloss / (len(zs) - 1)
                swav_loss /= len(global_idxs)

                h1, h2 = F.normalize(features[0], dim=-1), F.normalize(features[1], dim=-1)
                contrastive_loss = contrastive_criterion(h1, h2)
                
                loss = swav_loss + config.swav.contrastive.lambda_h * contrastive_loss

                epoch_swav_losses.append(swav_loss.item())
                epoch_contrast_losses.append(contrastive_loss.item())

                swav_losses.append(swav_loss.item())
                contrast_losses.append(contrastive_loss.item())
                progress_bar.set_postfix({
                    'swav': swav_loss.item(),
                    'contrast': contrastive_loss.item()
                })

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    prototypes.data = F.normalize(prototypes.data, dim=0)
            # scheduler.step()
            logging.info(f"Epoch {epoch + 1} - SwAV Loss: {np.mean(epoch_swav_losses):.4f}, Contrastive Loss: {np.mean(epoch_contrast_losses):.4f}")

        plt.plot(swav_losses, label='SwAV Loss')
        plt.plot(contrast_losses, label='Contrastive Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.title('Training Loss Convergence')
        plt.savefig(join(dataset_root, 'training_losses.png'), bbox_inches='tight')
        plt.close()
    else:
        logging.info("Skipping training. Using pretrained model and prototypes.")
        model_save_path = join(dataset_root, config.swav.model_save_path)
        if not exists(model_save_path):
            raise FileNotFoundError(f"Model save path {model_save_path} does not exist.")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        logging.info(f"Model loaded from {model_save_path}")

        prototypes_save_path = join(dataset_root, config.swav.prototypes_save_path)
        if not exists(prototypes_save_path):
            raise FileNotFoundError(f"Prototypes save path {prototypes_save_path} does not exist.")
        prototypes.data = torch.load(prototypes_save_path, map_location=device)
        logging.info(f"Prototypes loaded from {prototypes_save_path}")

    logging.info("Generating cluster IDs...")
    progress_bar = tqdm(range(0, len(sample_list), BATCH_SIZE))
    model.eval()
    all_cluster_ids = []
    for i in progress_bar:
        batched_graph = SliceGraphBatch(sample_list[i:i + BATCH_SIZE])
        batched_graph = batched_graph.graphs.to(device)
        global_view = augment(batched_graph, mask_id=vocab.get_unk_id())
        with torch.no_grad():
            z, _ = model(global_view)
            scores = z @ prototypes
            q = uot_sinkhorn_gpu(scores)
            cluster_ids = q.argmax(dim=1)
            all_cluster_ids.extend(cluster_ids.cpu().numpy().tolist())
    logging.info("Obtained cluster IDs. Merging small clusters...")
    all_cluster_ids = get_merged_clusters(all_cluster_ids, prototypes, min_size=config.hyper_parameters.batch_size)
    
    logging.info("Finalized cluster IDs. Generating groups...")
    cluster_id_maps = defaultdict(list)
    for idx, cluster_id in enumerate(tqdm(all_cluster_ids)):
        cluster_id_maps[f"{cluster_id}"].append(idx)
    
    swav_multicrop_batches = [indices for indices in cluster_id_maps.values()]
    swav_batches_filepath = join(dataset_root, config.swav_batches_filename)
    logging.info(f"Generated {len(swav_multicrop_batches)} groups. Saving to {swav_batches_filepath}...")
    with open(swav_batches_filepath, "w") as wfo:
        json.dump(swav_multicrop_batches, wfo)

    logging.info(f"Saving model...")
    model_save_path = join(dataset_root, config.swav.model_save_path)
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")
    logging.info(f"Saving prototypes...")
    prototypes_save_path = join(dataset_root, config.swav.prototypes_save_path)
    torch.save(prototypes.data, prototypes_save_path)
    logging.info(f"Prototypes saved to {prototypes_save_path}")

    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()