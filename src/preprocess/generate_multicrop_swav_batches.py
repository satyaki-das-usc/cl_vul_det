import random
import os
import json
import torch
import networkx as nx

import logging

from omegaconf import DictConfig, OmegaConf
from typing import List, cast
from tqdm import tqdm
from math import ceil
from os.path import join, isdir
from collections import defaultdict

from pytorch_lightning import seed_everything

from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_node, dropout_edge, k_hop_subgraph
import torch.nn.functional as F

from src.vocabulary import Vocabulary
from src.torch_data.graphs import SliceGraph
from src.torch_data.samples import SliceGraphSample, SliceGraphBatch

from src.models.modules.gnns import GraphSwAVModel

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
    """Randomly drop nodes from each graph in the batch with probability p."""
    edge_index, edge_mask, node_mask = dropout_node(batch.edge_index, p=p, num_nodes=batch.num_nodes, training=True, relabel_nodes=True)
    # Filter node features and batch mapping according to the mask
    x = batch.x[node_mask]
    batch_vec = batch.batch[node_mask]

    # Filter edge attributes if necessary
    edge_attr = None
    if batch.edge_attr is not None:
        edge_attr = batch.edge_attr[edge_mask]

    # Rebuild Data and then Batch for correct consistency
    new_data = Data(x=x, edge_index=edge_index,
                    edge_attr=edge_attr, batch=batch_vec)

    return Batch.from_data_list([new_data])

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

if __name__ == "__main__":
    config = cast(DictConfig, OmegaConf.load("configs/dwk.yaml"))
    max_len = config.dataset.token.max_parts
    dataset_root = config.data_folder
    config.gnn.w2v_path = join(dataset_root, "w2v.wv")
    seed_everything(config.seed, workers=True)

    init_log()

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
            } for p in model.parameters()], config.hyper_parameters.learning_rate)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #             optimizer,
    #             lr_lambda=lambda epoch: config.hyper_parameters.decay_gamma
    #                                 ** epoch)

    K = max(1024, ceil(len(train_slices) / config.hyper_parameters.batch_size))
    D = config.gnn.projection_dim
    prototypes = torch.nn.Parameter(torch.randn(D, K, device=device))
    with torch.no_grad():
        prototypes.data = F.normalize(prototypes.data, dim=0)

    BATCH_SIZE = 512
    
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

        for i in progress_bar:
            batched_graph = SliceGraphBatch(sample_list[i:i + BATCH_SIZE])
            batched_graph = batched_graph.graphs.to(device)
            views = augment_multicrop(batched_graph, mask_id=vocab.get_unk_id(), n_local_views=4)
            zs, features = zip(*(model(v) for v in views))
            scores = [z @ prototypes for z in zs]
            with torch.no_grad():
                qs = [sinkhorn(s) for s in scores]
            ps = [F.softmax(s / config.swav.temperature, dim=1) for s in scores]

            loss = 0
            global_idxs = [0, 1]
            V = len(zs) - 2
            for i in global_idxs:
                for j in range(len(zs)):
                    if j == i: continue
                    loss += -(qs[j] * ps[i].log()).sum(dim=1).mean()
            loss = loss / (len(global_idxs) * (1 + V))

            progress_bar.set_postfix({'loss': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                prototypes.data = F.normalize(prototypes.data, dim=0)
        # scheduler.step()
    
    logging.info("Generating cluster IDs...")
    progress_bar = tqdm(range(0, len(sample_list), BATCH_SIZE))
    model.eval()
    all_cluster_ids = []
    for i in progress_bar:
        batched_graph = SliceGraphBatch(sample_list[i:i + BATCH_SIZE])
        batched_graph = batched_graph.graphs.to(device)
        with torch.no_grad():
            z, _ = model(batched_graph)
            scores = z @ prototypes
            q = sinkhorn(scores)
            cluster_ids = q.argmax(dim=1)
            all_cluster_ids.extend(cluster_ids.cpu().numpy().tolist())
    logging.info("Obtained cluster IDs. Generating groups...")
    cluster_id_maps = defaultdict(list)
    for idx, cluster_id in enumerate(tqdm(all_cluster_ids)):
        cluster_id_maps[f"{cluster_id}"].append(idx)
    
    swav_multicrop_batches = [indices for indices in cluster_id_maps.values()]
    swav_batches_filepath = join(dataset_root, config.swav_batches_filename)
    logging.info(f"Generated {len(swav_multicrop_batches)} groups. Saving to {swav_batches_filepath}...")
    with open(swav_batches_filepath, "w") as wfo:
        json.dump(swav_multicrop_batches, wfo)

    logging.info(f"Saving model...")
    torch.save(model.state_dict(), config.swav.model_save_path)
    logging.info(f"Model saved to {config.swav.model_save_path}")
    logging.info(f"Saving prototypes...")
    torch.save(prototypes.data, config.swav.prototypes_save_path)
    logging.info(f"Prototypes saved to {config.swav.prototypes_save_path}")

    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()