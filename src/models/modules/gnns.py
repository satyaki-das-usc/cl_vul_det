from omegaconf import DictConfig
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import TopKPooling, GCNConv, GINConv, GINEConv, GATv2Conv, GatedGraphConv, GlobalAttention, BatchNorm, AttentionalAggregation
from torch_geometric.utils import scatter, softmax, subgraph
import torch.nn.functional as F

from src.vocabulary import Vocabulary
from src.models.modules.common_layers import STEncoder


class AttentionReadout(torch.nn.Module):
    """Attention-weighted graph readout that also exposes node scores."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate_nn = torch.nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, batch: torch.Tensor):
        attention_logits = self.gate_nn(x).squeeze(-1)
        attention_weights = softmax(attention_logits, batch)
        graph_embeddings = scatter(
            attention_weights.unsqueeze(-1) * x,
            batch,
            dim=0,
            reduce="sum",
        )
        return graph_embeddings, attention_logits, attention_weights

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
        self.pad_idx = pad_idx
        self.attention_only = config.attention_only
        self.use_edge_attr = config.use_edge_attr
        self.edge_feature_dim = 2 * config.embed_size

        # Build sequence of conv/pool layers with skip and gating
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        in_dim = config.rnn.hidden_size  # from STEncoder
        num_layers = config.n_hidden_layers

        for _ in range(num_layers):
            if self.use_edge_attr:
                conv_layer = GINEConv(
                    nn=torch.nn.Sequential(
                        torch.nn.Linear(in_dim, self.hidden),
                        torch.nn.ReLU()
                    ),
                    train_eps=config.train_eps,
                    edge_dim=self.edge_feature_dim,
                )
            else:
                conv_layer = GINConv(
                    nn=torch.nn.Sequential(
                        torch.nn.Linear(in_dim, self.hidden),
                        torch.nn.ReLU()
                    ), train_eps=config.train_eps
                )
            self.convs.append(conv_layer)
            self.pools.append(TopKPooling(self.hidden, ratio=config.pooling_ratio))
            in_dim = self.hidden

        self.global_att = AttentionReadout(self.hidden)

    def _encode_edge_attributes(self, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_attr.dim() != 2 or edge_attr.size(1) != 2:
            raise ValueError(
                "Expected edge_attr with shape [num_edges, 2] containing "
                f"[edge_type_id, variable_id], got {tuple(edge_attr.shape)}."
            )

        edge_token_embeddings = self.encoder.embed_tokens(edge_attr)
        edge_type_embeddings = edge_token_embeddings[:, 0]
        variable_embeddings = edge_token_embeddings[:, 1]
        has_variable = edge_attr[:, 1] != self.pad_idx
        variable_embeddings = variable_embeddings * has_variable.unsqueeze(-1)
        return torch.cat([edge_type_embeddings, variable_embeddings], dim=-1)

    def _encode_and_readout(self, batched_graph: Batch):
        x = self.encoder(batched_graph.x)
        edge_index, edge_attr, batch = batched_graph.edge_index, batched_graph.edge_attr, batched_graph.batch
        if self.use_edge_attr:
            edge_attr = self._encode_edge_attributes(edge_attr)

        out = 0
        attention_logits = None
        attention_weights = None
        for conv, pool in zip(self.convs, self.pools):
            if self.use_edge_attr:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)
            if self.attention_only:
                continue

            x, edge_index, edge_attr, batch, _, _ = pool(x, edge_index, edge_attr, batch)
            # TopKPooling returns pooled edge_attr — no need for manual subgraph()

            layer_out, attention_logits, attention_weights = self.global_att(x, batch)
            out += layer_out  # residual-summed graph vector

        if self.attention_only:
            out, attention_logits, attention_weights = self.global_att(x, batch)

        return out, attention_logits, attention_weights

    def forward(self, batched_graph: Batch):
        out, _, _ = self._encode_and_readout(batched_graph)
        
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