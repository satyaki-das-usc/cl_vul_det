import torch
import pickle
import random
import networkx as nx
import wordninja as wn

from typing import Callable, List
from copy import deepcopy

from torch_geometric.data import Batch, Data
from torch_geometric.utils import dropout_node, dropout_edge, k_hop_subgraph

from src.vocabulary import Vocabulary
from src.torch_data.graphs import SliceGraph
from src.torch_data.samples import SliceGraphBatch

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
dead_branch_conditions = ["if(0)", "if(1 == 0)", "if(false)"]
dead_loop_conditions = ["while(0)", "while(false)", "for(; 0; )"]
dead_body_statements = [";", "0;"]

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

def _fresh_aug_node_name(graph: nx.DiGraph, prefix: str) -> str:
    idx = 0
    node_name = f"{prefix}_{idx}"
    while node_name in graph:
        idx += 1
        node_name = f"{prefix}_{idx}"
    return node_name

def _choose_code_variant(candidates: List[str], vocab: Vocabulary):
    tokenized_candidates = [
        (candidate, tokenize_code_line(candidate, subtoken=False))
        for candidate in candidates
    ]
    in_vocab_candidates = [
        (candidate, tokens)
        for candidate, tokens in tokenized_candidates
        if all(token in vocab.token_to_id for token in tokens)
    ]
    return random.choice(in_vocab_candidates or tokenized_candidates)

def _add_symbolized_node(graph: nx.DiGraph, node_name: str, sym_code: str, tokens: List[str] = None):
    graph.add_node(node_name)
    graph.nodes[node_name]["sym_code"] = sym_code
    graph.nodes[node_name]["code_sym_token"] = tokens or tokenize_code_line(sym_code, subtoken=False)

def apply_printf_template(slice_graph: nx.DiGraph, vocab: Vocabulary, max_len: int) -> SliceGraph:
    augmented_graph = deepcopy(slice_graph)
    aug_node = _fresh_aug_node_name(augmented_graph, "printf_aug")
    _add_symbolized_node(augmented_graph, aug_node, global_augmentation1)
    return SliceGraph(slice_graph=augmented_graph)

def apply_unreachable_return_template(slice_graph: nx.DiGraph, vocab: Vocabulary, max_len: int) -> SliceGraph:
    augmented_graph = deepcopy(slice_graph)
    cond_node = _fresh_aug_node_name(augmented_graph, "unreachable_cond_aug")
    stmt_node = _fresh_aug_node_name(augmented_graph, "unreachable_stmt_aug")
    _add_symbolized_node(augmented_graph, cond_node, global_augmentation2[0])
    _add_symbolized_node(augmented_graph, stmt_node, global_augmentation2[1])
    all_nodes = list(slice_graph.nodes())
    controlled_nodes = {stmt_node}
    insert_point = None
    if random.choice([True, False]) and len(all_nodes) > 0:
        insert_point = random.choice(all_nodes)
    if insert_point is not None:
        forward_queue = [insert_point]
        visited = {insert_point}

        while len(forward_queue) > 0:
            current_line = forward_queue.pop(0)
            controlled_nodes.add(current_line)
            if current_line not in slice_graph._succ:
                continue
            for succ in slice_graph._succ[current_line]:
                if succ in visited:
                    continue
                visited.add(succ)
                forward_queue.append(succ)

    new_edges = [(cond_node, n, {"label": "CONTROLS", "direction": "forward"}) for n in controlled_nodes]
    augmented_graph.add_edges_from(new_edges)
    return SliceGraph(slice_graph=augmented_graph)

def apply_dead_branch_template(slice_graph: nx.DiGraph, vocab: Vocabulary, max_len: int) -> SliceGraph:
    augmented_graph = deepcopy(slice_graph)
    cond_code, cond_tokens = _choose_code_variant(dead_branch_conditions, vocab)
    body_code, body_tokens = _choose_code_variant(dead_body_statements, vocab)
    cond_node = _fresh_aug_node_name(augmented_graph, "dead_branch_cond_aug")
    body_node = _fresh_aug_node_name(augmented_graph, "dead_branch_body_aug")
    _add_symbolized_node(augmented_graph, cond_node, cond_code, cond_tokens)
    _add_symbolized_node(augmented_graph, body_node, body_code, body_tokens)
    augmented_graph.add_edge(cond_node, body_node, label="CONTROLS", direction="forward")
    return SliceGraph(slice_graph=augmented_graph)

def apply_dead_loop_template(slice_graph: nx.DiGraph, vocab: Vocabulary, max_len: int) -> SliceGraph:
    augmented_graph = deepcopy(slice_graph)
    loop_code, loop_tokens = _choose_code_variant(dead_loop_conditions, vocab)
    body_code, body_tokens = _choose_code_variant(dead_body_statements, vocab)
    loop_node = _fresh_aug_node_name(augmented_graph, "dead_loop_cond_aug")
    body_node = _fresh_aug_node_name(augmented_graph, "dead_loop_body_aug")
    _add_symbolized_node(augmented_graph, loop_node, loop_code, loop_tokens)
    _add_symbolized_node(augmented_graph, body_node, body_code, body_tokens)
    augmented_graph.add_edge(loop_node, body_node, label="CONTROLS", direction="forward")
    return SliceGraph(slice_graph=augmented_graph)

augmentation_templates: List[Callable[[nx.DiGraph, Vocabulary, int], SliceGraph]] = [
    apply_printf_template,
    apply_unreachable_return_template,
    apply_dead_branch_template,
    apply_dead_loop_template,
]

def generate_node_set_augmentation(slice_graph: nx.DiGraph, vocab: Vocabulary, max_len: int) -> SliceGraph:
    return apply_printf_template(slice_graph, vocab, max_len)

def generate_edge_set_augmentation(slice_graph: nx.DiGraph, vocab: Vocabulary, max_len: int) -> SliceGraph:
    return apply_unreachable_return_template(slice_graph, vocab, max_len)

def generate_template_augmentations(slice_graph: nx.DiGraph, vocab: Vocabulary, max_len: int, n_views: int = 2):
    if n_views <= len(augmentation_templates):
        selected_templates = random.sample(augmentation_templates, k=n_views)
    else:
        selected_templates = [
            random.choice(augmentation_templates)
            for _ in range(n_views)
        ]
    return [
        template(slice_graph, vocab, max_len).to_torch_graph(vocab, max_len)
        for template in selected_templates
    ]

def generate_SF_augmentations(batched_graph: SliceGraphBatch, vocab: Vocabulary, max_len: int):
    # Apply augmentations to the input graph
    view_lists = [[], []]

    slice_graph_list = [get_slice_graph(slice_path) for slice_path in batched_graph.slice_paths]

    for slice_graph in slice_graph_list:
        views = generate_template_augmentations(slice_graph, vocab, max_len, n_views=len(view_lists))
        for idx, view in enumerate(views):
            view_lists[idx].append(view)

    return [Batch.from_data_list(view_list) for view_list in view_lists]

if __name__ == "__main__":
    pass
