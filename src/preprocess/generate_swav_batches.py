import os
import json
import torch
import random

import logging

import torch.nn.functional as F
import matplotlib.pyplot as plt

from multiprocessing import cpu_count
from os.path import join, isdir
from math import ceil
from omegaconf import DictConfig, OmegaConf
from typing import cast
from collections import defaultdict

from sklearn.manifold import TSNE
from tqdm import tqdm
from pytorch_lightning import seed_everything

from src.common_utils import get_arg_parser, filter_warnings
from src.vocabulary import Vocabulary

from src.torch_data.graphs import SliceGraph
from src.torch_data.samples import SliceGraphSample, SliceGraphBatch

from src.models.modules.gnns import GraphSwAVModel

max_len = 16

def init_log():
    LOG_DIR = "logs"
    if not isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logging.basicConfig(
        handlers=[
            logging.FileHandler(join(LOG_DIR, "generate_swav_batches.log")),
            logging.StreamHandler()
        ],
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("=========New session=========")
    logging.info(f"Logging dir: {LOG_DIR}")

def soft_cross_entropy(preds, targets):
    log_probs = F.log_softmax(preds, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()

def sinkhorn_knopp(Q, n_iters=3, epsilon=0.2):
    eps = 1e-8
    Q = torch.exp(Q / epsilon).T
    Q /= (Q.sum() + eps)

    K, B = Q.shape
    for _ in range(n_iters):
        Q /= torch.sum(Q, dim=1, keepdim=True) + eps
        Q /= K
        
        Q /= torch.sum(Q, dim=0, keepdim=True) + eps
        Q /= B

    Q *= B
    return Q.T

def soft_intra_cluster_dissimilarity(activations, soft_assignments):
    activations = F.normalize(activations, dim=1)
    sim_matrix = activations @ activations.T
    dissim_matrix = 1 - sim_matrix

    co_assign_probs = soft_assignments @ soft_assignments.T 

    loss = (co_assign_probs * dissim_matrix).mean()
    return loss

def prototype_dissimilarity_loss(prototypes, margin=0.5):
    sim_matrix = prototypes @ prototypes.T
    identity = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
    off_diag_sim = sim_matrix * (1 - identity)

    # Penalize similarity above the margin
    # loss = torch.clamp(off_diag_sim - margin, min=0).mean()
    return off_diag_sim.mean()
    return loss

# def centroid_dissimilarity_loss(centroids, invalid_mask, margin=0.5):
#     centroids = F.normalize(centroids, dim=1)
#     sim_matrix = centroids @ centroids.T
#     identity = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
#     off_diag_sim = sim_matrix * (1 - identity)

#     # Penalize similarity above the margin
#     # loss = torch.clamp(off_diag_sim - margin, min=0).mean()
#     return off_diag_sim.mean()

def is_adequate_batch(labels, min_vul=3, min_nonvul=1):
    vul_count = (labels == 1).sum().item()
    nonvul_count = (labels == 0).sum().item()
    
    return vul_count >= min_vul and nonvul_count >= min_nonvul

def compute_cluster_centroids(activations, cluster_assignments, num_clusters):
    D = activations.size(1)
    centroids = torch.zeros(num_clusters, D, device=activations.device)
    counts = torch.zeros(num_clusters, device=activations.device)

    for i in range(num_clusters):
        mask = (cluster_assignments == i)
        if mask.any():
            centroids[i] = activations[mask].mean(dim=0)
            counts[i] = mask.sum()

    invalid_mask = (counts == 0 )
    
    return centroids, invalid_mask

def get_centroid_merged_clusters(centroids, invalid_mask, inadequate_clusters, all_labels, cluster_to_indices, config):
    centroids = F.normalize(centroids, dim=1)
    with torch.no_grad():
        similarity_matrix = centroids @ centroids.T

        similarity_matrix[invalid_mask, :] = float('inf')
        similarity_matrix[:, invalid_mask] = float('inf')
    
    merged_clusters = dict()
    for cluster_id in tqdm(inadequate_clusters):
        sim_row = similarity_matrix[cluster_id].clone()

        sorted_indices = torch.argsort(sim_row, descending=True)
        for target_cid in sorted_indices:
            if target_cid.item() == cluster_id:
                continue
            if f"{target_cid.item()}" not in cluster_to_indices:
                continue
            if target_cid.item() in inadequate_clusters:
                continue
            merged_labels = torch.cat((all_labels[cluster_to_indices[f"{cluster_id}"]], all_labels[cluster_to_indices[f"{target_cid.item()}"]]))
            if not is_adequate_batch(merged_labels, min_vul=config.min_vul_per_batch, min_nonvul=config.min_nonvul_per_batch):
                continue
            merged_clusters[cluster_id] = target_cid.item()
            break
    
    return merged_clusters

def get_prototype_merged_clusters(prototypes, inadequate_clusters, all_labels, cluster_to_indices, config):
    with torch.no_grad():
        prototypes = F.normalize(model.cluster_head.weight, dim=1)
        similarity_matrix = prototypes @ prototypes.T
    
    merged_clusters = dict()

    for cluster_id in tqdm(inadequate_clusters):
        sim_row = similarity_matrix[cluster_id].clone()

        sorted_indices = torch.argsort(sim_row, descending=True)
        for target_cid in sorted_indices:
            if target_cid.item() == cluster_id:
                continue
            if f"{target_cid.item()}" not in cluster_to_indices:
                continue
            if target_cid.item() in inadequate_clusters:
                continue
            merged_labels = torch.cat((all_labels[cluster_to_indices[f"{cluster_id}"]], all_labels[cluster_to_indices[f"{target_cid.item()}"]]))
            if not is_adequate_batch(merged_labels, min_vul=config.min_vul_per_batch, min_nonvul=config.min_nonvul_per_batch):
                continue
            merged_clusters[cluster_id] = target_cid.item()
            break
    
    return merged_clusters

if __name__ == "__main__":
    filter_warnings()
    arg_parser = get_arg_parser()
    arg_parser.add_argument(
        "--merge_type",
        type=str,
        default="centroid",
        choices=["prototype", "centroid"],
        help="Type of cluster merging to use: 'prototype' or 'centroid'"
    )
    args = arg_parser.parse_args()

    config = cast(DictConfig, OmegaConf.load(args.config))
    seed_everything(config.seed, workers=True)

    init_log()

    if config.num_workers != -1:
        USE_CPU = min(config.num_workers, cpu_count())
    else:
        USE_CPU = cpu_count()

    if args.use_temp_data:
        dataset_root = config.temp_root
    else:
        dataset_root = config.data_folder
    
    max_len = config.dataset.token.max_parts
    vocab = Vocabulary.from_w2v(join(dataset_root, "w2v.wv"))
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

    BATCH_SIZE = 512
    num_clusters = ceil(len(sample_list) / config.hyper_parameters.batch_size)
    in_channels = sample_list[0].graph.x.shape[1]
    edge_dim = sample_list[0].graph.edge_attr.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSwAVModel(config, vocab, vocab_size, pad_idx, num_clusters=num_clusters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(config.swav.n_epochs):
        logging.info(f"Epoch {epoch + 1}")
        model.train()
        progress_bar = tqdm(range(0, len(sample_list), BATCH_SIZE))
        random.shuffle(sample_list)
        # progress_bar.set_description(f"Loss: {torch.nan}")
        epoch_loss = [0.0]
        epoch_vul_rowwise_entropy = [0.0]
        epoch_vul_usage_entropy = [0.0]
        for i in progress_bar:
            batched_graph = SliceGraphBatch(sample_list[i:i + BATCH_SIZE])
            labels = batched_graph.labels.to(device)
            batched_graph = batched_graph.graphs.to(device)

            outs, activations = model(batched_graph)
        
            with torch.no_grad():
                logits = outs.detach() / config.swav.sinkhorn.temperature

                logits = logits - logits.max(dim=1, keepdim=True)[0]
                sinkhorn_targets = sinkhorn_knopp(logits, n_iters=config.swav.sinkhorn.n_iters, epsilon=config.swav.sinkhorn.epsilon)
                
                vul_mask = (labels == 1)
                nonvul_mask = (labels == 0)
                
                vul_logits = logits[vul_mask]
                nonvul_logits = logits[nonvul_mask]
                
                vul_sinkhorn_targets = sinkhorn_targets[vul_mask]
                nonvul_sinkhorn_targets = sinkhorn_targets[nonvul_mask]
                
            vul_rowwise_entropy = -torch.sum(vul_sinkhorn_targets * torch.log(vul_sinkhorn_targets + 1e-8), dim=1).mean()
            nonvul_rowwise_entropy = -torch.sum(nonvul_sinkhorn_targets * torch.log(nonvul_sinkhorn_targets + 1e-8), dim=1).mean()
            
            vul_usage = vul_sinkhorn_targets.sum(dim=0)
            vul_usage = vul_usage / vul_usage.sum()
            vul_usage_entropy = -torch.sum(vul_usage * torch.log(vul_usage + 1e-8))
            
            nonvul_usage = nonvul_sinkhorn_targets.sum(dim=0)
            nonvul_usage = nonvul_usage / nonvul_usage.sum()
            nonvul_usage_entropy = -torch.sum(nonvul_usage * torch.log(nonvul_usage + 1e-8))

            vul_outs = outs[vul_mask]
            nonvul_outs = outs[nonvul_mask]

            intra_cluster_dissim_loss = soft_intra_cluster_dissimilarity(activations, sinkhorn_targets)
            
            if config.swav.use_prototype_loss:
                proto_loss = prototype_dissimilarity_loss(F.normalize(model.cluster_head.weight, dim=1), margin=config.swav.margin)
                alpha = 0.25
                beta = 0.24
                gamma = 0.27
                delta = 0.24
            else:
                proto_loss = torch.tensor(0.0, device=device)
                alpha = 0.33
                beta = 0.35
                gamma = 0.32
                delta = 0.0

            loss = (
                (soft_cross_entropy(vul_outs, vul_sinkhorn_targets) * alpha)
                + (soft_cross_entropy(nonvul_outs, nonvul_sinkhorn_targets) * beta)
                + gamma * intra_cluster_dissim_loss
                + delta * proto_loss
                - config.swav.vul_beta * vul_rowwise_entropy
                - config.swav.nonvul_beta * nonvul_rowwise_entropy
                - config.swav.vul_gamma * vul_usage_entropy
                - config.swav.nonvul_gamma * nonvul_usage_entropy
            )

            # print(f"Loss: {loss.item()}, Row-wise Entropy: {rowwise_entropy.item()}")
            # progress_bar.set_description(f"Loss: {loss.item():.2f}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            epoch_vul_rowwise_entropy.append(vul_rowwise_entropy.item())
            epoch_vul_usage_entropy.append(vul_usage_entropy.item())
            progress_bar.set_postfix(
                loss=loss.item(),
                intra_cluster_dissim_loss=intra_cluster_dissim_loss.item(),
                # proto_loss=proto_loss.item()
            )
        
        logging.info(f"Loss: {sum(epoch_loss) / len(epoch_loss)}, Row-wise Entropy: {sum(epoch_vul_rowwise_entropy) / len(epoch_vul_rowwise_entropy)}, Usage Entropy: {sum(epoch_vul_usage_entropy) / len(epoch_vul_usage_entropy)}")
    
    logging.info("It's evaluating time!")

    progress_bar = tqdm(range(0, len(sample_list), BATCH_SIZE))
    model.eval()

    all_outs = torch.empty((0, num_clusters), device=device)
    all_activations = torch.empty((0, config.gnn.hidden_size), device=device)
    all_labels = torch.empty((0,), device=device)
    with torch.no_grad():
        for i in progress_bar:
            batched_graph = SliceGraphBatch(sample_list[i:i + BATCH_SIZE])
            labels = batched_graph.labels.to(device)
            all_labels = torch.cat((all_labels, labels.detach()), dim=0)
            batched_graph = batched_graph.graphs.to(device)

            outs, activations = model(batched_graph)
            all_outs = torch.cat((all_outs, outs.detach()), dim=0)
            all_activations = torch.cat((all_activations, activations.detach()), dim=0)
        
        all_logits = all_outs.detach() / config.swav.sinkhorn.temperature
        all_logits = all_logits - all_logits.max(dim=1, keepdim=True)[0]
        all_sinkhorn_targets = sinkhorn_knopp(all_logits, n_iters=config.swav.sinkhorn.n_iters, epsilon=config.swav.sinkhorn.epsilon)
        cluster_assignments = all_sinkhorn_targets.argmax(dim=1).cpu()

        vul_mask = (all_labels == 1)
        nonvul_mask = (all_labels == 0)
        
        vul_logits = all_logits[vul_mask]
        nonvul_logits = all_logits[nonvul_mask]
        
        vul_sinkhorn_targets = all_sinkhorn_targets[vul_mask]
        nonvul_sinkhorn_targets = all_sinkhorn_targets[nonvul_mask]
    
        vul_rowwise_entropy = -torch.sum(vul_sinkhorn_targets * torch.log(vul_sinkhorn_targets + 1e-8), dim=1).mean()
        nonvul_rowwise_entropy = -torch.sum(nonvul_sinkhorn_targets * torch.log(nonvul_sinkhorn_targets + 1e-8), dim=1).mean()
        
        vul_usage = vul_sinkhorn_targets.sum(dim=0)
        vul_usage = vul_usage / vul_usage.sum()
        vul_usage_entropy = -torch.sum(vul_usage * torch.log(vul_usage + 1e-8))
        
        nonvul_usage = nonvul_sinkhorn_targets.sum(dim=0)
        nonvul_usage = nonvul_usage / nonvul_usage.sum()
        nonvul_usage_entropy = -torch.sum(nonvul_usage * torch.log(nonvul_usage + 1e-8))

    vul_cluster_usage = vul_sinkhorn_targets.sum(dim=0)
    print("Min / Max / Mean vul cluster usage:", vul_cluster_usage.min().item(), vul_cluster_usage.max().item(), vul_cluster_usage.mean().item())

    nonvul_cluster_usage = nonvul_sinkhorn_targets.sum(dim=0)
    print("Min / Max / Mean nonvul cluster usage:", nonvul_cluster_usage.min().item(), nonvul_cluster_usage.max().item(), nonvul_cluster_usage.mean().item())

    vul_std = vul_cluster_usage.std().item()
    nonvul_std = nonvul_cluster_usage.std().item()
    logging.info(f"Evaluation completed. Vul std: {vul_std}, Non-vul std: {nonvul_std}")
    logging.info(f"Vul Row-wise Entropy: {vul_rowwise_entropy.item()}, Non-vul Row-wise Entropy: {nonvul_rowwise_entropy.item()}")
    logging.info(f"Vul Usage Entropy: {vul_usage_entropy.item()}, Non-vul Usage Entropy: {nonvul_usage_entropy.item()}")
    
    logging.info(f"Saving model to {config.swav.model_save_path}...")
    torch.save(model.state_dict(), config.swav.model_save_path)
    with torch.no_grad():
        logging.info("Visualizing clusters...")

        all_cluster_ids = cluster_assignments.unique().cpu().tolist()
        selected_clusters = random.sample(all_cluster_ids, 5)

        selected_indices = []
        for cid in selected_clusters:
            cluster_mask = (cluster_assignments == cid).nonzero(as_tuple=True)[0]
            if len(cluster_mask) >= 64:
                chosen = cluster_mask[torch.randperm(len(cluster_mask))[:64]]
            else:
                chosen = cluster_mask
            selected_indices.append(chosen)

        selected_indices = torch.cat(selected_indices)
        selected_activations = all_activations[selected_indices]
        selected_assignments = cluster_assignments[selected_indices]
        selected_labels = all_labels[selected_indices]

        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca', random_state=42)
        reduced = tsne.fit_transform(selected_activations.cpu().numpy())

        # Plot
        plt.figure(figsize=(10, 8))
        for cid in selected_clusters:
            mask = (selected_assignments == cid).cpu().numpy()
            plt.scatter(reduced[mask, 0], reduced[mask, 1], label=f'Cluster {cid}', s=20)
        plt.legend()
        plt.title("t-SNE of 5 Random Clusters (64 samples each)")
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    logging.info(f"Generating minibatches from cluster assignments...")
    cluster_to_indices = defaultdict(list)

    for idx, cluster_id in tqdm(enumerate(cluster_assignments), total=len(cluster_assignments)):
        cluster_to_indices[f"{cluster_id.item()}"].append(idx)
    
    logging.info(f"Found {len(cluster_to_indices)} clusters.")
    
    inadequate_clusters = [int(cluster_id) for cluster_id, indices in cluster_to_indices.items() if not is_adequate_batch(all_labels[indices], min_vul=config.min_vul_per_batch, min_nonvul=config.min_nonvul_per_batch)]

    logging.info(f"{len(inadequate_clusters)} inadequate clusters found. Merging them with adequate clusters...")

    with torch.no_grad():
        prototypes = F.normalize(model.cluster_head.weight, dim=1)
        similarity_matrix = prototypes @ prototypes.T
    
    if args.merge_type == "prototype":
        logging.info("Using prototype-based merging...")
        merged_clusters = get_prototype_merged_clusters(prototypes, inadequate_clusters, all_labels, cluster_to_indices, config)
    elif args.merge_type == "centroid":
        logging.info("Using centroid-based merging...")
        centroids, invalid_mask = compute_cluster_centroids(all_activations, cluster_assignments, num_clusters)
        merged_clusters = get_centroid_merged_clusters(centroids, invalid_mask, inadequate_clusters, all_labels, cluster_to_indices, config)
    else:
        raise ValueError(f"Unknown merge type: {args.merge_type}")
    
    merged_assignments = cluster_assignments.clone()
    for old_cid, new_cid in merged_clusters.items():
        assert old_cid in inadequate_clusters, f"Old cluster ID {old_cid} not found in inadequate clusters"
        assert old_cid != new_cid, f"Old and new cluster IDs are the same: {old_cid}"
        assert new_cid not in inadequate_clusters, f"New cluster ID {new_cid} is inadequate"
        merged_assignments[cluster_assignments == old_cid] = new_cid
    
    final_cluster_to_indices = defaultdict(list)
    for idx, cluster_id in enumerate(merged_assignments.tolist()):
        final_cluster_to_indices[cluster_id].append(idx)

    minibatches = [indices for cluster_id, indices in sorted(final_cluster_to_indices.items())]
    swav_batches_filepath = join(dataset_root, config.swav_batches_filename)
    logging.info(f"Minibatch generation completed. Saving to {swav_batches_filepath}...")
    with open(swav_batches_filepath, "w") as wfi:
        json.dump(minibatches, wfi, indent=2)
    
    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()