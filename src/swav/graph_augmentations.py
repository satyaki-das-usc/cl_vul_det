import torch

from typing import List

from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_node, dropout_edge, k_hop_subgraph

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
                      nmb_views: List[int]) -> List[Batch]:
    """
    Generate multiple views per batch for SwAV-style multi-crop:
    - 2 global augmented views
    - n_local_views subgraph-cropped views
    """
    views = []
    # Global augmented views
    for _ in range(nmb_views[0]):
        views.append(augment(batch, mask_id=mask_id))

    # Local subgraph views
    for _ in range(nmb_views[1]):
        crop = subgraph_crop(batch)
        crop_aug = augment(crop, mask_id=mask_id)
        views.append(crop_aug)
    return views

if __name__ == "__main__":
    pass