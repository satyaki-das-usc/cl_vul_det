
import json
from os.path import exists

from torch.utils.data import Dataset

from omegaconf import DictConfig

from src.vocabulary import Vocabulary
from src.torch_data.graphs import SliceGraph
from src.torch_data.samples import SliceGraphSample

class SliceDataset(Dataset):
    def __init__(self, slices_paths: str, config: DictConfig, vocab: Vocabulary) -> None:
        super().__init__()
        self.__config = config
        assert exists(slices_paths), f"{slices_paths} not exists!"
        with open(slices_paths, "r") as rfi:
            slice_path_list = list(json.load(rfi))
        self.__vocab = vocab
        self.__slices = [SliceGraph(slice_path) for slice_path in slice_path_list]
        self.__n_samples = len(self.__slices)
    
    def __len__(self) -> int:
        return self.__n_samples

    def __getitem__(self, index) -> SliceGraphSample:
        slice_graph: SliceGraph = self.__slices[index]
        return SliceGraphSample(graph=slice_graph.to_torch_graph(self.__vocab, self.__config.dataset.token.max_parts),
                         label=slice_graph.label)

    def get_n_samples(self):
        return self.__n_samples