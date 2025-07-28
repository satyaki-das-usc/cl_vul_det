from omegaconf import DictConfig
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import TopKPooling, GCNConv, GINEConv, GATv2Conv, GatedGraphConv, GlobalAttention
from torch_geometric.utils import subgraph
import torch.nn.functional as F

from src.vocabulary import Vocabulary
from src.models.modules.common_layers import STEncoder

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

class GINEConvEncoder(torch.nn.Module):

    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        super(GINEConvEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)

        self.input_GCL = GINEConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(config.rnn.hidden_size, config.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(config.hidden_size, config.hidden_size)
            ),
            edge_dim=config.edge_dim
        )

        self.input_GPL = TopKPooling(config.hidden_size,
                                     ratio=config.pooling_ratio)

        for i in range(config.n_hidden_layers - 1):
            setattr(self, f"hidden_GCL{i}",
                    GINEConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(config.rnn.hidden_size, config.hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(config.hidden_size, config.hidden_size)
            ),
            edge_dim=2
        ))
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

        # Pass edge_attr to TopKPooling

        # node_embedding = F.relu(self.input_GCL(node_embedding, edge_index))
        node_embedding = F.relu(self.input_GCL(node_embedding, edge_index, edge_attr))
        node_embedding_pooled, edge_index_pooled, _, batch_pooled, perm, score = self.input_GPL(node_embedding, edge_index, None, batch)
        edge_index_pooled, edge_attr_pooled = subgraph(
            subset=perm,
            edge_index=edge_index,
            edge_attr=edge_attr,
            relabel_nodes=True,
            num_nodes=node_embedding.size(0)
        )
        node_embedding, edge_index, edge_attr, batch = node_embedding_pooled, edge_index_pooled, edge_attr_pooled, batch_pooled
        # node_embedding, edge_index, edge_attr, batch, _, _ = self.input_GPL(node_embedding, edge_index, edge_attr, batch)

        # [n_SliceGraph; SliceGraph hidden dim]
        out = self.attpool(node_embedding, batch)
        for i in range(self.__config.n_hidden_layers - 1):
            # node_embedding = F.relu(getattr(self, f"hidden_GCL{i}")(node_embedding, edge_index))
            node_embedding = F.relu(self.input_GCL(node_embedding, edge_index, edge_attr))
            node_embedding_pooled, edge_index_pooled, _, batch_pooled, perm, score = getattr(self, f"hidden_GPL{i}")(
                node_embedding, edge_index, None, batch)
            edge_index_pooled, edge_attr_pooled = subgraph(
                subset=perm,
                edge_index=edge_index,
                edge_attr=edge_attr,
                relabel_nodes=True,
                num_nodes=node_embedding.size(0)
            )
            node_embedding, edge_index, edge_attr, batch = node_embedding_pooled, edge_index_pooled, edge_attr_pooled, batch_pooled
            # node_embedding, edge_index, edge_attr, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
            #     node_embedding, edge_index, edge_attr, batch)
            out += self.attpool(node_embedding, batch)

        # [n_SliceGraph; SliceGraph hidden dim]
        return out

class GraphSwAVModel(torch.nn.Module):

    _encoders = {
        "gcn": GraphConvEncoder,
        "ggnn": GatedGraphConvEncoder,
        "gine": GINEConvEncoder
    }

    def __init__(self, config: DictConfig, vocab: Vocabulary, vocabulary_size: int,
                 pad_idx: int):
        super().__init__()
        hidden_dim = config.gnn.hidden_size
        self.__graph_encoder = self._encoders[config.gnn.name](config.gnn, vocab, vocabulary_size,
                                                               pad_idx)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, config.gnn.projection_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(config.gnn.projection_dim, config.gnn.projection_dim)
        )

    def forward(self, batch: Batch):
        graph_activations = self.__graph_encoder(batch)  # [num_graphs, hidden_dim]
        logits = self.projection_head(graph_activations)
        logits = F.normalize(logits, dim=-1)
        return logits, graph_activations  # [num_graphs, num_clusters]