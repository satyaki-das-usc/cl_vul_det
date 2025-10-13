from dataclasses import dataclass
from typing import Optional, List
from torch_geometric.data import Data, Batch
import torch

@dataclass
class SliceGraphSample:
    graph: Data
    label: int
    slice_path: str
    augmented_views: Optional[List[Data]] = None  # Store pre-generated augmentations

class SliceGraphBatch:
    def __init__(self, slice_graphs: List[SliceGraphSample]):
        self.slice_paths = [slice_graph.slice_path for slice_graph in slice_graphs]
        self.labels = torch.tensor([slice_graph.label for slice_graph in slice_graphs],
                                   dtype=torch.long)
        self.graphs = Batch.from_data_list([slice_graph.graph for slice_graph in slice_graphs])
        augmented_views = []
        for slice_graph in slice_graphs:
            for idx, view in enumerate(slice_graph.augmented_views):
                if idx < len(augmented_views):
                    augmented_views[idx].append(view)
                    continue
                augmented_views.append([view])
        self.augmented_views = [Batch.from_data_list(view) for view in augmented_views]
        self.sz = len(slice_graphs)

    def __len__(self):
        return self.sz

    def pin_memory(self) -> "SliceGraphBatch":
        self.labels = self.labels.pin_memory()
        self.graphs = self.graphs.pin_memory()
        if self.augmented_views is not None:
            self.augmented_views = [view.pin_memory() if hasattr(view, 'pin_memory') else view 
                                   for view in self.augmented_views]
        return self

    def move_to_device(self, device: torch.device):
        self.labels = self.labels.to(device)
        self.graphs = self.graphs.to(device)
        if self.augmented_views is not None:
            self.augmented_views = [view.to(device) for view in self.augmented_views]