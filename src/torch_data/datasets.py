
import json

import networkx as nx

from os.path import exists
from omegaconf import DictConfig
from functools import lru_cache
from copy import deepcopy

from torch.utils.data import Dataset


from src.vocabulary import Vocabulary
from src.torch_data.graphs import SliceGraph
from src.torch_data.samples import SliceGraphSample
from src.swav.graph_augmentations import generate_node_set_augmentation, generate_edge_set_augmentation

def generate_SF_augmentations_per_sample(slice_graph: nx.DiGraph, vocab: Vocabulary, max_len: int):
    glob1_view = generate_node_set_augmentation(slice_graph, vocab, max_len).to_torch_graph(vocab, max_len)
    glob2_view = generate_edge_set_augmentation(slice_graph, vocab, max_len).to_torch_graph(vocab, max_len)

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
    control_edge_view = deepcopy(slice_graph)
    control_edge_view.remove_edges_from(dd_edges + post_dom_edges)
    isolated = [n for n in control_edge_view.nodes() if control_edge_view.degree(n) == 0]
    if control_edge_view.number_of_nodes() > len(isolated):
        control_edge_view.remove_nodes_from(isolated)
    control_edge_view = SliceGraph(slice_graph=control_edge_view).to_torch_graph(vocab, max_len)
    
    # if len(dd_edges) > 0:
    dd_edge_view = deepcopy(slice_graph)
    dd_edge_view.remove_edges_from(control_edges + post_dom_edges)
    isolated = [n for n in dd_edge_view.nodes() if dd_edge_view.degree(n) == 0]
    if dd_edge_view.number_of_nodes() > len(isolated):
        dd_edge_view.remove_nodes_from(isolated)
    dd_edge_view = SliceGraph(slice_graph=dd_edge_view).to_torch_graph(vocab, max_len)
    
    # if len(post_dom_edges) > 0:
    # post_dom_edge_view = deepcopy(slice_graph)
    # post_dom_edge_view.remove_edges_from(control_edges + dd_edges)
    # isolated = [n for n in post_dom_edge_view.nodes() if post_dom_edge_view.degree(n) == 0]
    # post_dom_edge_view.remove_nodes_from(isolated)
    # post_dom_edge_view = SliceGraph(slice_graph=post_dom_edge_view).to_torch_graph(vocab, max_len)

    return [glob1_view, glob2_view, control_edge_view, dd_edge_view]

class SliceDataset(Dataset):
    def __init__(self, slices_paths: str, config: DictConfig, vocab: Vocabulary, cache_size: int = 128) -> None:
        super().__init__()
        self.__config = config
        assert exists(slices_paths), f"{slices_paths} not exists!"
        with open(slices_paths, "r") as rfi:
            self.__slice_path_list = list(json.load(rfi))
        self.__vocab = vocab
        self.__max_len = config.dataset.token.max_parts
        # self.__slices = [SliceGraph(slice_path) for slice_path in self.__slice_path_list]
        self.__n_samples = len(self.__slice_path_list)
        self._load_slice = lru_cache(maxsize=cache_size)(self._load_slice_uncached)
    
    def _load_slice_uncached(self, slice_path: str):
        return SliceGraph(slice_path=slice_path)
    
    def clear_cache(self):
        self._load_slice.cache_clear()
    
    def get_cache_info(self):
        return self._load_slice.cache_info()
    
    def __len__(self) -> int:
        return self.__n_samples

    def __getitem__(self, index) -> SliceGraphSample:
        slice_path = self.__slice_path_list[index]
        slice_graph: SliceGraph = self._load_slice(slice_path)
        return SliceGraphSample(graph=slice_graph.to_torch_graph(self.__vocab, self.__config.dataset.token.max_parts),
                         label=slice_graph.label,
                         slice_path=self.__slice_path_list[index],
                         augmented_views=generate_SF_augmentations_per_sample(slice_graph.slice_graph, self.__vocab, self.__max_len)
                         )

    def get_n_samples(self):
        return self.__n_samples