from dataclasses import dataclass
from typing import List
from torch_geometric.data import Data, Batch
import torch

@dataclass
class SliceGraphSample:
    graph: Data
    label: int
    slice_path: str

class SliceGraphBatch:
    def __init__(self, slice_graphs: List[SliceGraphSample]):
        self.slice_paths = [slice_graph.slice_path for slice_graph in slice_graphs]
        self.labels = torch.tensor([slice_graph.label for slice_graph in slice_graphs],
                                   dtype=torch.long)
        self.graphs = Batch.from_data_list([slice_graph.graph for slice_graph in slice_graphs])
        self.sz = len(slice_graphs)

    def __len__(self):
        return self.sz

    def pin_memory(self) -> "SliceGraphBatch":
        self.labels = self.labels.pin_memory()
        self.graphs = self.graphs.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.labels = self.labels.to(device)
        self.graphs = self.graphs.to(device)