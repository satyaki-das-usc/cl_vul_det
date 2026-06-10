
import json

import networkx as nx

from os.path import exists
from omegaconf import DictConfig
from functools import lru_cache

from torch.utils.data import Dataset


from src.vocabulary import Vocabulary
from src.torch_data.graphs import SliceGraph
from src.torch_data.samples import SliceGraphSample
from src.swav.graph_augmentations import generate_node_set_augmentation, generate_edge_set_augmentation

def generate_SF_augmentations_per_sample(slice_graph: nx.DiGraph, vocab: Vocabulary, max_len: int):
    glob1_view = generate_node_set_augmentation(slice_graph, vocab, max_len).to_torch_graph(vocab, max_len)
    glob2_view = generate_edge_set_augmentation(slice_graph, vocab, max_len).to_torch_graph(vocab, max_len)

    return [glob1_view, glob2_view]

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
