from torch import nn
from omegaconf import DictConfig
import torch
from typing import Dict
from pytorch_lightning import LightningModule
from torch.nn import CosineEmbeddingLoss
from torch.optim import Adam, SGD, Adamax, RMSprop
import torch.nn.functional as F

from src.torch_data.samples import SliceGraphBatch
from src.models.modules.gnns import GraphConvEncoder, GatedGraphConvEncoder, GINEConvEncoder, GraphSwAVModel
from src.metrics import Statistic
from torch_geometric.data import Batch
from src.vocabulary import Vocabulary

# cosine_similarity = CosineSimilarity(dim=2)


optimizers = {
        "RMSprop": RMSprop,
        "Adam": Adam,
        "SGD": SGD,
        "Adamax": Adamax
    }

encoders = {
    "gcn": GraphConvEncoder,
    "ggnn": GatedGraphConvEncoder,
    "gine": GINEConvEncoder
}

def get_optimizer(name: str) -> torch.nn.Module:
        if name in optimizers:
            return optimizers[name]
        raise KeyError(f"Optimizer {name} is not supported")

class CLVulDet(LightningModule):
    r"""vulnerability detection model to detect vulnerability

    Args:
        config (DictConfig): configuration for the model
        vocabulary_size (int): the size of vacabulary
        pad_idx (int): the index of padding token
    """

    def __init__(self, config: DictConfig, vocab: Vocabulary, vocabulary_size: int,
                 pad_idx: int, graph_encoder: GraphSwAVModel = None):
        super().__init__()
        if graph_encoder is not None:
            self.save_hyperparameters(ignore=["graph_encoder"])
        else:
            self.save_hyperparameters()
        self.best_val_f1 = 0.0
        self.__config = config
        hidden_size = config.classifier.hidden_size
        self.trainable_parameters = []
        if graph_encoder is not None:
            self.__graph_encoder = nn.Sequential(*list(graph_encoder.children())[:-1])
            for param in self.__graph_encoder.parameters():
                param.requires_grad = False
        else:
            self.__graph_encoder = encoders[config.gnn.name](config.gnn, vocab, vocabulary_size,
                                                               pad_idx)
            self.trainable_parameters += [self.__graph_encoder.parameters()]
        # hidden layers
        layers = [
            nn.Linear(config.gnn.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.classifier.drop_out)
        ]
        if config.classifier.n_hidden_layers < 1:
            raise ValueError(
                f"Invalid layers number ({config.classifier.n_hidden_layers})")
        for _ in range(config.classifier.n_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(config.classifier.drop_out)
            ]
        self.__hidden_layers = nn.Sequential(*layers)
        self.trainable_parameters += [self.__hidden_layers.parameters()]
        self.__classifier = nn.Linear(hidden_size, config.classifier.n_classes)
        self.trainable_parameters += [self.__classifier.parameters()]
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []
    
    def set_checkpoint_path(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def forward(self, batch: Batch) -> torch.Tensor:
        """

        Args:
            batch (Batch): [n_SliceGraph (Data)]

        Returns: classifier results: [n_method; n_classes]
        """
        # [n_SliceGraph, hidden size]
        graph_activations = self.__graph_encoder(batch)
        activations = self.__hidden_layers(graph_activations)
        logits = self.__classifier(activations)
        # [n_SliceGraph; n_classes]
        return logits, activations, graph_activations

    def configure_optimizers(self) -> Dict:
        optimizer = get_optimizer(
            self.__config.hyper_parameters.optimizer)(
            [{
                "params": p
            } for p in self.trainable_parameters],
            self.__config.hyper_parameters.learning_rate)
        if self.__config.hyper_parameters.use_warmup_lr:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.__config.hyper_parameters.lr_warmup_epochs
            )

            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs - self.__config.hyper_parameters.lr_warmup_epochs,
                eta_min=1e-6
            )

            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.__config.hyper_parameters.lr_warmup_epochs]
            )
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: self.__config.hyper_parameters.decay_gamma
                                    ** epoch)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _log_training_step(self, results: Dict):
        self.log_dict(results, on_step=True, on_epoch=False)

    def get_contrastive_loss(self, activations, labels: torch.Tensor) -> torch.Tensor:
        if self.__config.distance_metric == "cosine":
            activations = F.normalize(activations, p=2, dim=1)
        n = activations.size(0)
        idx_i, idx_j = torch.triu_indices(n, n, offset=1, device=labels.device)
        if self.__config.exclude_NNs:
            valid_mask = (labels[idx_i] + labels[idx_j]) > 0
            idx_i_valid = idx_i[valid_mask]
            idx_j_valid = idx_j[valid_mask]
            act_i = activations[idx_i_valid]
            act_j = activations[idx_j_valid]
        else:
            act_i = activations[idx_i]
            act_j = activations[idx_j]
        
        if self.__config.distance_metric == "euclidean":
            # Prepare pairs for contrastive loss using Euclidean distance
            if self.__config.exclude_NNs:
                target = torch.where(
                    (labels[idx_i_valid] == 1) & (labels[idx_j_valid] == 1),
                    torch.tensor(1.0),
                    torch.tensor(-1.0)
                )
            else:
                target = torch.where(labels[idx_i] == labels[idx_j], 1.0, 0.0).to(activations.device)
            euclidean_distance = torch.norm(act_i - act_j, p=2, dim=1)
            margin = 1.0
            if act_i.size(0) > 0:
                contrastive_loss = (
                    target * euclidean_distance.pow(2) +
                    (1 - target) * F.relu(margin - euclidean_distance).pow(2)
                ).mean()
            else:
                contrastive_loss = torch.tensor(0.0, device=activations.device, requires_grad=True)
        elif self.__config.distance_metric == "cosine":
            # Prepare pairs for CosineEmbeddingLoss
            if self.__config.exclude_NNs:
                target = torch.where(
                    (labels[idx_i_valid] == 1) & (labels[idx_j_valid] == 1),
                    torch.tensor(1.0),
                    torch.tensor(-1.0)
                )
            else:
                target = torch.where(labels[idx_i] == labels[idx_j], 1.0, -1.0).to(activations.device)
            cosine_loss_fn = CosineEmbeddingLoss(margin=0.0)
            if act_i.size(0) > 0:
                contrastive_loss = cosine_loss_fn(act_i, act_j, target)
            else:
                contrastive_loss = torch.tensor(0.0, device=activations.device, requires_grad=True)
        else:
            raise ValueError(f"Unsupported distance metric: {self.__config.distance_metric}")
        return contrastive_loss
    
    def training_step(self, batch: SliceGraphBatch,
                      batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_SliceGraph; n_classes]
        logits, activations, graph_activations = self(batch.graphs)
        # loss = F.cross_entropy(logits, batch.labels)
        
        contrastive_loss = self.get_contrastive_loss(activations, batch.labels)
        ce_loss = F.cross_entropy(logits, batch.labels)

        if self.current_epoch < self.__config.hyper_parameters.contrastive_warmup_epochs:
            loss = ce_loss
        else:
            loss = ce_loss + contrastive_loss

        result: Dict = {"train_loss": loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="train")
            result.update(batch_metric)
            self._log_training_step(result)
            self.log("F1",
                     batch_metric["train_f1"],
                     prog_bar=True,
                     logger=False)
            self.logger.experiment.add_scalars(
                "train_batch/losses",
                {
                    "contrastive_loss": contrastive_loss.item(),
                    "cross_entropy_loss": ce_loss.item()
                },
                global_step=self.global_step
            )
            self.log('train_batch_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        step_output = {"loss": loss, "statistic": statistic}
        self.train_outputs.append(step_output)
        return step_output

    def validation_step(self, batch: SliceGraphBatch,
                        batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_SliceGraph; n_classes]
        logits, activations, graph_activations = self(batch.graphs)
        loss = F.cross_entropy(logits, batch.labels)

        result: Dict = {"val_loss": loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="val")
            result.update(batch_metric)
        step_output = {"loss": loss, "statistic": statistic}
        self.val_outputs.append(step_output)
        return step_output

    def test_step(self, batch: SliceGraphBatch,
                  batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_SliceGraph; n_classes]
        logits, activations, graph_activations = self(batch.graphs)
        loss = F.cross_entropy(logits, batch.labels)

        result: Dict = {"test_loss", loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="test")
            result.update(batch_metric)

        step_output = {"loss": loss, "statistic": statistic}
        self.test_outputs.append(step_output)
        return step_output

    # ========== EPOCH END ==========
    def _prepare_epoch_end_log(self, step_outputs: list,
                               step: str) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            losses = [
                so if isinstance(so, torch.Tensor) else so["loss"]
                for so in step_outputs
            ]
            mean_loss = torch.stack(losses).mean()
        return {f"{step}_loss": mean_loss}

    def _shared_epoch_end(self, step_outputs: list, group: str):
        log = self._prepare_epoch_end_log(step_outputs, group)
        statistic = Statistic.union_statistics(
            [out["statistic"] for out in step_outputs])
        batch_metric = statistic.calculate_metrics(group)
        if group == "val":
            current_f1 = batch_metric["val_f1"]
            if current_f1 > self.best_val_f1:
                self.best_val_f1 = current_f1
                self.trainer.save_checkpoint(self.checkpoint_path)
                self.log("best_val_f1", self.best_val_f1, prog_bar=True)
        log.update(batch_metric)
        self.log_dict(log, on_step=False, on_epoch=True)

    def on_train_epoch_end(self):
        self._shared_epoch_end(self.train_outputs, "train")

    def on_validation_epoch_end(self):
        self._shared_epoch_end(self.val_outputs, "val")

    def on_test_epoch_end(self):
        self._shared_epoch_end(self.test_outputs, "test")

class NoCLVulDet(LightningModule):
    r"""vulnerability detection model to detect vulnerability

    Args:
        config (DictConfig): configuration for the model
        vocabulary_size (int): the size of vacabulary
        pad_idx (int): the index of padding token
    """

    def __init__(self, config: DictConfig, vocab: Vocabulary, vocabulary_size: int,
                 pad_idx: int, graph_encoder: GraphSwAVModel = None):
        super().__init__()
        if graph_encoder is not None:
            self.save_hyperparameters(ignore=["graph_encoder"])
        else:
            self.save_hyperparameters()
        self.best_val_f1 = 0.0
        self.__config = config
        hidden_size = config.classifier.hidden_size
        self.trainable_parameters = []
        if graph_encoder is not None:
            self.__graph_encoder = nn.Sequential(*list(graph_encoder.children())[:-1])
            for param in self.__graph_encoder.parameters():
                param.requires_grad = False
        else:
            self.__graph_encoder = encoders[config.gnn.name](config.gnn, vocab, vocabulary_size,
                                                               pad_idx)
            self.trainable_parameters += [self.__graph_encoder.parameters()]
        # hidden layers
        layers = [
            nn.Linear(config.gnn.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.classifier.drop_out)
        ]
        if config.classifier.n_hidden_layers < 1:
            raise ValueError(
                f"Invalid layers number ({config.classifier.n_hidden_layers})")
        for _ in range(config.classifier.n_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(config.classifier.drop_out)
            ]
        self.__hidden_layers = nn.Sequential(*layers)
        self.trainable_parameters += [self.__hidden_layers.parameters()]
        self.__classifier = nn.Linear(hidden_size, config.classifier.n_classes)
        self.trainable_parameters += [self.__classifier.parameters()]
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

    def set_checkpoint_path(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
    
    def forward(self, batch: Batch) -> torch.Tensor:
        """

        Args:
            batch (Batch): [n_SliceGraph (Data)]

        Returns: classifier results: [n_method; n_classes]
        """
        # [n_SliceGraph, hidden size]
        self.graph_activations = self.__graph_encoder(batch)
        self.activations = self.__hidden_layers(self.graph_activations)
        # [n_SliceGraph; n_classes]
        return self.__classifier(self.activations)

    def configure_optimizers(self) -> Dict:
        optimizer = get_optimizer(
            self.__config.hyper_parameters.optimizer)(
            [{
                "params": p
            } for p in self.trainable_parameters],
            self.__config.hyper_parameters.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: self.__config.hyper_parameters.decay_gamma
                                    ** epoch)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _log_training_step(self, results: Dict):
        self.log_dict(results, on_step=True, on_epoch=False)

    def training_step(self, batch: SliceGraphBatch,
                      batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_SliceGraph; n_classes]
        logits = self(batch.graphs)
        loss = F.cross_entropy(logits, batch.labels)

        result: Dict = {"train_loss": loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="train")
            result.update(batch_metric)
            self._log_training_step(result)
            self.log("F1",
                     batch_metric["train_f1"],
                     prog_bar=True,
                     logger=False)
        step_output = {"loss": loss, "statistic": statistic}
        self.train_outputs.append(step_output)
        return step_output

    def validation_step(self, batch: SliceGraphBatch,
                        batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_SliceGraph; n_classes]
        logits = self(batch.graphs)
        loss = F.cross_entropy(logits, batch.labels)

        result: Dict = {"val_loss": loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="val")
            result.update(batch_metric)
        step_output = {"loss": loss, "statistic": statistic}
        self.val_outputs.append(step_output)
        return step_output

    def test_step(self, batch: SliceGraphBatch,
                  batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_SliceGraph; n_classes]
        logits = self(batch.graphs)
        loss = F.cross_entropy(logits, batch.labels)

        result: Dict = {"test_loss", loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="test")
            result.update(batch_metric)

        step_output = {"loss": loss, "statistic": statistic}
        self.test_outputs.append(step_output)
        return step_output

    # ========== EPOCH END ==========
    def _prepare_epoch_end_log(self, step_outputs: list,
                               step: str) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            losses = [
                so if isinstance(so, torch.Tensor) else so["loss"]
                for so in step_outputs
            ]
            mean_loss = torch.stack(losses).mean()
        return {f"{step}_loss": mean_loss}

    def _shared_epoch_end(self, step_outputs: list, group: str):
        log = self._prepare_epoch_end_log(step_outputs, group)
        statistic = Statistic.union_statistics(
            [out["statistic"] for out in step_outputs])
        batch_metric = statistic.calculate_metrics(group)
        if group == "val":
            current_f1 = batch_metric["val_f1"]
            if current_f1 > self.best_val_f1:
                self.best_val_f1 = current_f1
                self.trainer.save_checkpoint(self.checkpoint_path)
                self.log("best_val_f1", self.best_val_f1, prog_bar=True)
        log.update(batch_metric)
        self.log_dict(log, on_step=False, on_epoch=True)

    def on_train_epoch_end(self):
        self._shared_epoch_end(self.train_outputs, "train")

    def on_validation_epoch_end(self):
        self._shared_epoch_end(self.val_outputs, "val")

    def on_test_epoch_end(self):
        self._shared_epoch_end(self.test_outputs, "test")