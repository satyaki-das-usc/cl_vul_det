from omegaconf import DictConfig
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import TopKPooling, GCNConv, GINEConv, GATv2Conv, GatedGraphConv, GlobalAttention
import torch.nn.functional as F

from src.vocabulary import Vocabulary
from src.models.modules.common_layers import STEncoder

class GraphSwAVModel(torch.nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_dim=256, num_clusters=1000):
        super().__init__()
        self.gnn = GINEConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            ),
            edge_dim=edge_dim
        )
        gate_nn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        self.pool = GlobalAttention(gate_nn)
        self.head = torch.nn.Linear(hidden_dim, num_clusters)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = edge_attr.float()
        x = self.gnn(x, edge_index, edge_attr)  # uses edge_attr
        pooled = self.pool(x, batch)  # [num_graphs, hidden_dim]
        logits = self.head(pooled)
        return logits, pooled  # [num_graphs, num_clusters]

class GraphConvEncoder(torch.nn.Module):
    """

    Kipf and Welling: Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    (https://arxiv.org/pdf/1609.02907.pdf)

    """

    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        super(GraphConvEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)

        self.input_GCL = GCNConv(config.rnn.hidden_size, config.hidden_size)

        self.input_GPL = TopKPooling(config.hidden_size,
                                     ratio=config.pooling_ratio)

        for i in range(config.n_hidden_layers - 1):
            setattr(self, f"hidden_GCL{i}",
                    GCNConv(config.hidden_size, config.hidden_size))
            setattr(
                self, f"hidden_GPL{i}",
                TopKPooling(config.hidden_size,
                            ratio=config.pooling_ratio))

        self.attpool = GlobalAttention(torch.nn.Linear(config.hidden_size, 1))

    def forward(self, batched_graph: Batch):
        # [n nodes; rnn hidden]
        node_embedding = self.__st_embedding(batched_graph.x)
        edge_index = batched_graph.edge_index
        edge_attr = batched_graph.edge_attr.float().mean(dim=1, keepdim=False)
        batch = batched_graph.batch

        # Pass edge_attr to TopKPooling

        # node_embedding = F.relu(self.input_GCL(node_embedding, edge_index))
        node_embedding = F.relu(self.input_GCL(node_embedding, edge_index, edge_weight=edge_attr))
        # node_embedding, edge_index, _, batch, _, _ = self.input_GPL(node_embedding, edge_index, None, batch)
        node_embedding, edge_index, edge_attr, batch, _, _ = self.input_GPL(node_embedding, edge_index, edge_attr, batch)

        # [n_SliceGraph; SliceGraph hidden dim]
        out = self.attpool(node_embedding, batch)
        for i in range(self.__config.n_hidden_layers - 1):
            # node_embedding = F.relu(getattr(self, f"hidden_GCL{i}")(node_embedding, edge_index))
            node_embedding = F.relu(getattr(self, f"hidden_GCL{i}")(node_embedding, edge_index, edge_weight=edge_attr))
            # node_embedding, edge_index, _, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
            #     node_embedding, edge_index, None, batch)
            node_embedding, edge_index, edge_attr, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
                node_embedding, edge_index, edge_attr, batch)
            out += self.attpool(node_embedding, batch)

        # [n_SliceGraph; SliceGraph hidden dim]
        return out


class GatedGraphConvEncoder(torch.nn.Module):
    """

    from Li et al.: Gated Graph Sequence Neural Networks (ICLR 2016)
    (https://arxiv.org/pdf/1511.05493.pdf)

    """

    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        super(GatedGraphConvEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)

        self.input_GCL = GatedGraphConv(out_channels=config.hidden_size, num_layers=config.n_gru)

        self.input_GPL = TopKPooling(config.hidden_size,
                                     ratio=config.pooling_ratio)

        for i in range(config.n_hidden_layers - 1):
            setattr(self, f"hidden_GCL{i}",
                    GatedGraphConv(out_channels=config.hidden_size, num_layers=config.n_gru))
            setattr(
                self, f"hidden_GPL{i}",
                TopKPooling(config.hidden_size,
                            ratio=config.pooling_ratio))
        self.attpool = GlobalAttention(torch.nn.Linear(config.hidden_size, 1))

    def forward(self, batched_graph: Batch):
        # [n nodes; rnn hidden]
        node_embedding = self.__st_embedding(batched_graph.x)
        edge_index = batched_graph.edge_index
        edge_attr = batched_graph.edge_attr.float()
        batch = batched_graph.batch
        node_embedding = F.relu(self.input_GCL(node_embedding, edge_index, edge_weight=edge_attr))
        node_embedding, edge_index, edge_attr, batch, _, _ = self.input_GPL(node_embedding, edge_index, edge_attr, batch)
        # [n_SliceGraph; SliceGraph hidden dim]
        out = self.attpool(node_embedding, batch)
        for i in range(self.__config.n_hidden_layers - 1):
            node_embedding = F.relu(getattr(self, f"hidden_GCL{i}")(node_embedding, edge_index, edge_weight=edge_attr))
            node_embedding, edge_index, edge_attr, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
                node_embedding, edge_index, edge_attr, batch)
            out += self.attpool(node_embedding, batch)
        # [n_SliceGraph; SliceGraph hidden dim]
        return out
