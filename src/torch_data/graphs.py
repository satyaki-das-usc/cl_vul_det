from dataclasses import dataclass
import networkx as nx

import pickle
import torch
from torch_geometric.data import Data
from src.vocabulary import Vocabulary

edge_type_map = {
    "REACHES": "COMESFROM",
    "CONTROLS": "FOLLOWS"
}

@dataclass
class SliceGraph:
    def __init__(self, slice_path: str = None, slice_graph: nx.DiGraph = None):
        if slice_graph is not None:
            self.__slice_graph = slice_graph
        elif slice_path is not None:
            with open(slice_path, "rb") as rbfi:
                self.__slice_graph: nx.DiGraph = pickle.load(rbfi)
        else:
            print(f"Neither slice graph path nor the slice graph object was provided. Terminating...")
            exit(-1)
        
        self.__init_graph()
    
    def __init_graph(self):
        self.tokens_list = []
        self.node_to_idx = {}
        for idx, n in enumerate(self.__slice_graph):
            tokens = self.__slice_graph.nodes[n]["code_sym_token"]
            self.tokens_list.append(tokens)
            self.node_to_idx[f"{n}"] = idx
        
        self.__label = self.__slice_graph.graph["label"]
        
    @property
    def label(self) -> int:
        return self.__label
    
    def to_torch_graph(self, vocab: Vocabulary, max_len: int):
        node_ids = torch.full((len(self.tokens_list), max_len),
                                    vocab.get_pad_id(),
                                    dtype=torch.long)
        
        for idx, tokens in enumerate(self.tokens_list):
            ids = vocab.convert_tokens_to_ids(tokens)
            less_len = min(max_len, len(ids))
            node_ids[idx, :less_len] = torch.tensor(ids[:less_len], dtype=torch.long)

        edge_index = []
        edge_attr = []

        for u, v, data in self.__slice_graph.edges(data=True):
            if "direction" not in data:
                raise ValueError(f"Edge {u} -> {v} does not have a direction attribute.")
            if data["direction"] == "forward":
                edge_index.append((self.node_to_idx[f"{u}"], self.node_to_idx[f"{v}"]))
                edge_attr.append((vocab.get_id(data["label"]), vocab.get_id(data["var"]) if "var" in data else vocab.get_pad_id()))
            elif data["direction"] == "backward":
                edge_index.append((self.node_to_idx[f"{v}"], self.node_to_idx[f"{u}"]))
                edge_attr.append((vocab.get_id(edge_type_map[data["label"]]), vocab.get_id(data["var"]) if "var" in data else vocab.get_pad_id()))
            
        return Data(x=node_ids, edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(), edge_attr=torch.tensor(edge_attr, dtype=torch.long))