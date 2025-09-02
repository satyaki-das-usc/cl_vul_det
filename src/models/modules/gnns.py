from omegaconf import DictConfig
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import TopKPooling, GCNConv, GINEConv, GATv2Conv, GatedGraphConv, GlobalAttention, BatchNorm, AttentionalAggregation
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

        self.attpool = AttentionalAggregation(torch.nn.Linear(config.hidden_size, 1))

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
        self.attpool = AttentionalAggregation(torch.nn.Linear(config.hidden_size, 1))

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
        self.encoder = STEncoder(config, vocab, vocabulary_size, pad_idx)
        self.hidden = config.hidden_size
        self.edge_dim = config.edge_dim

        # Build sequence of conv/pool layers with skip and gating
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        in_dim = config.rnn.hidden_size  # from STEncoder
        num_layers = config.n_hidden_layers

        for _ in range(num_layers):
            self.convs.append(GINEConv(
                nn=torch.nn.Sequential(
                    torch.nn.Linear(in_dim, self.hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden, self.hidden)
                ),
                edge_dim=self.edge_dim  # ensure edge_feature→node_dim alignment
            ))
            self.bns.append(BatchNorm(self.hidden))
            self.pools.append(TopKPooling(self.hidden, ratio=config.pooling_ratio))
            in_dim = self.hidden

        self.global_att = AttentionalAggregation(torch.nn.Linear(self.hidden, 1))

    def forward(self, batched_graph: Batch):
        x = self.encoder(batched_graph.x)
        edge_index, edge_attr, batch = batched_graph.edge_index, batched_graph.edge_attr, batched_graph.batch

        out = 0
        for conv, bn, pool in zip(self.convs, self.bns, self.pools):
            x = conv(x, edge_index, edge_attr.float())
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)

            x, edge_index, edge_attr, batch, _, _ = pool(x, edge_index, edge_attr.float(), batch)
            # TopKPooling returns pooled edge_attr — no need for manual subgraph()

            out += self.global_att(x, batch)  # residual-summed graph vector

        return out  # graph-level embeddings

class GraphSwAVModel(torch.nn.Module):

    _encoders = {
        "gcn": GraphConvEncoder,
        "ggnn": GatedGraphConvEncoder,
        "gine": GINEConvEncoder
    }

    def __init__(self, config: DictConfig, vocab: Vocabulary, vocabulary_size: int,
                 pad_idx: int):
        super().__init__()
        self.l2norm = config.swav.l2norm
        self.__graph_encoder = self._encoders[config.gnn.name](config.gnn, vocab, vocabulary_size,
                                                               pad_idx)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(config.gnn.hidden_size, config.swav.hidden_mlp),
            torch.nn.BatchNorm1d(config.swav.hidden_mlp),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(config.swav.hidden_mlp, config.swav.projection_dim)
        )
        self.prototypes = torch.nn.Linear(config.swav.projection_dim, config.swav.nmb_prototypes, bias=False)

    def forward(self, batch: Batch):
        graph_activations = self.__graph_encoder(batch)
        embeddings = self.projection_head(graph_activations)
        if self.l2norm:
            embeddings = F.normalize(embeddings, dim=-1, p=2)
        return graph_activations, embeddings, self.prototypes(embeddings)