from torch import nn
from omegaconf import DictConfig
import torch.nn.functional as F

from src.models.modules.gnns import GraphConvEncoder, GatedGraphConvEncoder, GINEConvEncoder
from torch_geometric.data import Batch
from src.vocabulary import Vocabulary

encoders = {
    "gcn": GraphConvEncoder,
    "ggnn": GatedGraphConvEncoder,
    "gine": GINEConvEncoder
}

class GraphSwAVVD(nn.Module):
    r"""vulnerability detection model to detect vulnerability using graph encoder from SwAV

    Args:
        config (DictConfig): configuration for the model
        vocabulary_size (int): the size of vacabulary
        pad_idx (int): the index of padding token
    """

    def __init__(self, config: DictConfig, vocab: Vocabulary, vocabulary_size: int,
                 pad_idx: int):
        super().__init__()
        self.__graph_encoder = encoders[config.gnn.name](config.gnn, vocab, vocabulary_size,
                                                               pad_idx)
        self.swav_l2norm = config.swav.l2norm
        self.__swav_projection_head = nn.Sequential(
            nn.Linear(config.gnn.hidden_size, config.swav.hidden_mlp),
            nn.BatchNorm1d(config.swav.hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(config.swav.hidden_mlp, config.swav.projection_dim)
        )
        self.swav_prototypes = nn.Linear(config.swav.projection_dim, config.swav.nmb_prototypes, bias=False)
        # hidden layers
        layers = [
            nn.Linear(config.gnn.hidden_size, config.classifier.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.classifier.drop_out)
        ]
        if config.classifier.n_hidden_layers < 1:
            raise ValueError(
                f"Invalid layers number ({config.classifier.n_hidden_layers})")
        for _ in range(config.classifier.n_hidden_layers - 1):
            layers += [
                nn.Linear(config.classifier.hidden_size, config.classifier.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.classifier.drop_out)
            ]
        self.__hidden_layers = nn.Sequential(*layers)
        self.__classifier = nn.Linear(config.classifier.hidden_size, config.classifier.n_classes)
    
    def forward(self, batch: Batch):
        """

        Args:
            batch (Batch): [n_SliceGraph (Data)]

        Returns: classifier results: [n_method; n_classes]
        """
        # [n_SliceGraph, hidden size]
        graph_encodings = self.__graph_encoder(batch)
        swav_embeddings = self.__swav_projection_head(graph_encodings)
        if self.swav_l2norm:
            swav_embeddings = F.normalize(swav_embeddings, dim=-1, p=2)

        activations = self.__hidden_layers(graph_encodings)
        # [n_SliceGraph; n_classes]
        return self.__classifier(activations), activations, graph_encodings, swav_embeddings, self.swav_prototypes(swav_embeddings)