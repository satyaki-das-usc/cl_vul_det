import torch
import pickle
import networkx as nx
import wordninja as wn

from typing import List

from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_node, dropout_edge, k_hop_subgraph

from src.vocabulary import Vocabulary
from src.torch_data.graphs import SliceGraph
from src.torch_data.samples import SliceGraphSample, SliceGraphBatch

operators3 = {'<<=', '>>='}
operators2 = {
    '->', '++', '--', '!~', '<<', '>>', '<=', '>=', '==', '!=', '&&', '||',
    '+=', '-=', '*=', '/=', '%=', '&=', '^=', '|='
}
operators1 = {
    '(', ')', '[', ']', '.', '+', '-', '*', '&', '/', '%', '<', '>', '^', '|',
    '=', ',', '?', ':', ';', '{', '}', '!', '~'
}

global_augmentation1 = "printf("");\n"
global_augmentation2 = ["if(0 == 1)", "return;\n"]

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

def custom_tokenize_code_line(line: str, subtoken: bool):
        """
        transform a string of code line into list of tokens

        Args:
            line: code line
            subtoken: whether to split into subtokens

        Returns:

        """
        tmp, w = [], []
        i = 0
        while i < len(line):
            # Ignore spaces and combine previously collected chars to form words
            if line[i] == ' ':
                tmp.append(''.join(w).strip())
                tmp.append(line[i].strip())
                w = []
                i += 1
            # Check operators and append to final list
            elif line[i:i + 3] in operators3:
                tmp.append(''.join(w).strip())
                tmp.append(line[i:i + 3].strip())
                w = []
                i += 3
            elif line[i:i + 2] in operators2:
                tmp.append(''.join(w).strip())
                tmp.append(line[i:i + 2].strip())
                w = []
                i += 2
            elif line[i] in operators1:
                tmp.append(''.join(w).strip())
                tmp.append(line[i].strip())
                w = []
                i += 1
            # Character appended to word list
            else:
                w.append(line[i])
                i += 1
        if (len(w) != 0):
            tmp.append(''.join(w).strip())
            w = []
        # Filter out irrelevant strings
        tmp = list(filter(lambda c: (c != '' and c != ' '), tmp))
        # split subtoken
        res = list()
        if (subtoken):
            for token in tmp:
                res.extend(wn.split(token))
        else:
            res = tmp
        return res
        
        # def tokenize(self, text):
        #     return text.split(self.delimiter)

        # def detokenize(self, tokens):
        #     return self.delimiter.join(tokens)

def tokenize_code_line(line: str, subtoken: bool, tokenizer=None):
    if tokenizer is not None:
        return tokenizer.tokenize(line.strip())
    return custom_tokenize_code_line(line, subtoken)

def get_slice_graph(slice_path: str) -> nx.DiGraph:
    with open(slice_path, "rb") as rbfi:
            slice_graph: nx.DiGraph = pickle.load(rbfi)
    return slice_graph

def generate_node_set_augmentation(slice_graph: nx.DiGraph, vocab: Vocabulary, max_len: int) -> SliceGraphSample:
    # Create a copy of the original graph
    augmented_graph = slice_graph.copy()
    aug_node = f"glob_aug"
    augmented_graph.add_node(aug_node)
    augmented_graph.nodes[aug_node]["sym_code"] = global_augmentation1
    augmented_graph.nodes[aug_node]["code_sym_token"] = tokenize_code_line(global_augmentation1, subtoken=False)
    augmented_graph = SliceGraph(None, augmented_graph)
    augmented_graph = SliceGraphSample(graph=augmented_graph.to_torch_graph(vocab, max_len),
                        label=augmented_graph.label, slice_path=None)

    return augmented_graph

def generate_edge_set_augmentation(slice_graph: nx.DiGraph, vocab: Vocabulary, max_len: int) -> SliceGraphSample:
    # Create a copy of the original graph
    augmented_graph = slice_graph.copy()
    cond_node = f"cond_aug"
    augmented_graph.add_node(cond_node)
    augmented_graph.nodes[cond_node]["sym_code"] = global_augmentation2[0]
    augmented_graph.nodes[cond_node]["code_sym_token"] = tokenize_code_line(global_augmentation2[0], subtoken=False)
    stmt_node = f"stmt_node"
    augmented_graph.add_node(stmt_node)
    augmented_graph.nodes[stmt_node]["sym_code"] = global_augmentation2[1]
    augmented_graph.nodes[stmt_node]["code_sym_token"] = tokenize_code_line(global_augmentation2[1], subtoken=False)
    new_edges = [(cond_node, n, {"label": "CONTROLS", "direction": "forward"}) for n in augmented_graph.nodes if n != cond_node]
    augmented_graph.add_edges_from(new_edges)
    augmented_graph = SliceGraph(None, augmented_graph)
    augmented_graph = SliceGraphSample(graph=augmented_graph.to_torch_graph(vocab, max_len),
                        label=augmented_graph.label, slice_path=None)

    return augmented_graph

def generate_SF_augmentations(batched_graph: SliceGraphBatch, vocab: Vocabulary, max_len: int):
    # Apply augmentations to the input graph
    views = []
    glob1_views = []
    glob2_views = []
    control_edges_views = []
    dd_edges_views = []
    post_dom_edges_views = []

    slice_graph_list = [get_slice_graph(slice_path) for slice_path in batched_graph.slice_paths]

    for slice_graph in slice_graph_list:
        glob1_views.append(generate_node_set_augmentation(slice_graph))
        glob2_views.append(generate_edge_set_augmentation(slice_graph))

        control_edges = []
        dd_edges = []
        post_dom_edges = []

        for u, v, edge_data in slice_graph.edges(data=True):
            if edge_data["label"] == "CONTROLS":
                control_edges.append((u, v, edge_data))
            elif edge_data["label"] == "REACHES":
                dd_edges.append((u, v, edge_data))
            # elif edge_data["label"] == "POST_DOM":
            #     post_dom_edges.append((u, v, edge_data))

        # if len(control_edges) > 0:
        control_edge_view = slice_graph.copy()
        control_edge_view.remove_edges_from(dd_edges + post_dom_edges)
        isolated = [n for n in control_edge_view.nodes() if control_edge_view.degree(n) == 0]
        control_edge_view.remove_nodes_from(isolated)
        control_edge_view = SliceGraph(None, control_edge_view)
        control_edge_view = SliceGraphSample(graph=control_edge_view.to_torch_graph(vocab, max_len),
                            label=control_edge_view.label, slice_path=None)
        control_edges_views.append(control_edge_view)
        
        # if len(dd_edges) > 0:
        dd_edge_view = slice_graph.copy()
        dd_edge_view.remove_edges_from(control_edges + post_dom_edges)
        isolated = [n for n in dd_edge_view.nodes() if dd_edge_view.degree(n) == 0]
        dd_edge_view.remove_nodes_from(isolated)
        dd_edge_view = SliceGraph(None, dd_edge_view)
        dd_edge_view = SliceGraphSample(graph=dd_edge_view.to_torch_graph(vocab, max_len),
                            label=dd_edge_view.label, slice_path=None)
        dd_edges_views.append(dd_edge_view)
        
        # if len(post_dom_edges) > 0:
        post_dom_edge_view = slice_graph.copy()
        post_dom_edge_view.remove_edges_from(control_edges + dd_edges)
        isolated = [n for n in post_dom_edge_view.nodes() if post_dom_edge_view.degree(n) == 0]
        post_dom_edge_view.remove_nodes_from(isolated)
        post_dom_edge_view = SliceGraph(None, post_dom_edge_view)
        post_dom_edge_view = SliceGraphSample(graph=post_dom_edge_view.to_torch_graph(vocab, max_len),
                            label=post_dom_edge_view.label, slice_path=None)
        post_dom_edges_views.append(post_dom_edge_view)

    views.append(SliceGraphBatch(glob1_views))
    views.append(SliceGraphBatch(glob2_views))
    views.append(SliceGraphBatch(control_edges_views))
    views.append(SliceGraphBatch(dd_edges_views))

    return views

if __name__ == "__main__":
    pass