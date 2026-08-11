"""Run a trained GraphSwAVVD classifier on unlabeled slice graphs."""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import cast

import networkx as nx
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common_utils import filter_warnings, init_log
from src.models.swav_vd import GraphSwAVVD
from src.torch_data.graphs import SliceGraph
from src.vocabulary import Vocabulary


class UnlabeledSliceDataset(Dataset):
    def __init__(
            self,
            test_file: Path,
            config: DictConfig,
            vocab: Vocabulary):
        with test_file.open("r") as rfi:
            slice_paths = json.load(rfi)
        if not isinstance(slice_paths, list):
            raise ValueError(
                f"Expected a JSON list of slice paths in {test_file}."
            )

        self.slice_paths = [
            self._resolve_slice_path(path, test_file.parent)
            for path in slice_paths
        ]
        self.config = config
        self.vocab = vocab

    @staticmethod
    def _resolve_slice_path(path, test_file_directory: Path) -> Path:
        if not isinstance(path, str):
            raise ValueError(
                "Every entry in the test JSON file must be a string path."
            )

        candidate = Path(path).expanduser()
        if not candidate.is_absolute() and not candidate.is_file():
            candidate = test_file_directory / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Slice graph not found: {candidate}")
        return candidate

    def __len__(self):
        return len(self.slice_paths)

    def __getitem__(self, index):
        slice_path = self.slice_paths[index]
        with slice_path.open("rb") as rbfi:
            graph: nx.DiGraph = pickle.load(rbfi)

        # SliceGraph's conversion code expects this metadata even though
        # inference itself never reads it. The loaded graph is local to this
        # process, so adding a placeholder does not modify the pickle file.
        graph.graph.setdefault("label", -1)
        slice_graph = SliceGraph(slice_graph=graph)
        torch_graph = slice_graph.to_torch_graph(
            self.vocab,
            self.config.dataset.token.max_parts,
        )
        return torch_graph, str(slice_path)


def collate_slices(samples):
    graphs, slice_paths = zip(*samples)
    return Batch.from_data_list(list(graphs)), list(slice_paths)


def get_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Predict vulnerability probabilities for unlabeled slice graphs."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a GraphSwAVVD model state-dict checkpoint.",
    )
    parser.add_argument(
        "--test-file",
        required=True,
        help="Path to a JSON list of slice graph pickle files.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/dwk.yaml",
        help="Model configuration used to create the checkpoint.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path. Defaults to <test-file-stem>_predictions.json "
            "beside the test file."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size; defaults to hyper_parameters.test_batch_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data-loading worker processes (default: 0).",
    )
    return parser


def resolve_arguments(args):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    test_file = Path(args.test_file).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    for description, path in (
            ("Checkpoint", checkpoint_path),
            ("Test JSON file", test_file),
            ("Config file", config_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")

    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    if args.output is None:
        output_path = test_file.with_name(
            f"{test_file.stem}_predictions.json"
        )
    else:
        output_path = Path(args.output).expanduser().resolve()
    return checkpoint_path, test_file, config_path, output_path


def load_model(
        config: DictConfig,
        checkpoint_path: Path,
        device: torch.device):
    dataset_root = Path(config.data_folder) / config.dataset.name
    vocab_path = dataset_root / "w2v.wv"
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")

    vocab = Vocabulary.from_w2v(str(vocab_path))
    model = GraphSwAVVD(
        config,
        vocab,
        vocab.get_vocab_size(),
        vocab.get_pad_id(),
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint)
    model.eval()
    return vocab, model


def predict(model, data_loader, device: torch.device):
    predictions = []
    with torch.no_grad():
        for graphs, slice_paths in tqdm(data_loader, desc="Predicting"):
            logits = model.forward_logits(
                graphs.to(device, non_blocking=True)
            )
            probabilities = torch.softmax(logits, dim=1).cpu().tolist()

            for slice_path, class_probabilities in zip(
                    slice_paths,
                    probabilities):
                if len(class_probabilities) != 2:
                    raise ValueError(
                        "Expected a binary classifier with exactly 2 classes, "
                        f"but the model returned {len(class_probabilities)}."
                    )
                non_vulnerable_probability = float(class_probabilities[0])
                vulnerable_probability = float(class_probabilities[1])
                predicted_label = int(
                    vulnerable_probability > non_vulnerable_probability
                )
                predictions.append({
                    "slice_path": slice_path,
                    "predicted_label": predicted_label,
                    "predicted_class": (
                        "vulnerable"
                        if predicted_label == 1
                        else "non_vulnerable"
                    ),
                    "confidence": max(
                        non_vulnerable_probability,
                        vulnerable_probability,
                    ),
                    "vulnerable_probability": vulnerable_probability,
                    "non_vulnerable_probability": (
                        non_vulnerable_probability
                    ),
                })

    predictions.sort(
        key=lambda item: item["vulnerable_probability"],
        reverse=True,
    )
    vulnerable_rank = 0
    for prediction in predictions:
        if prediction["predicted_label"] == 1:
            vulnerable_rank += 1
            prediction["vulnerable_rank"] = vulnerable_rank
        else:
            prediction["vulnerable_rank"] = None
    return predictions


def main():
    filter_warnings()
    args = get_argument_parser().parse_args()
    checkpoint_path, test_file, config_path, output_path = resolve_arguments(
        args
    )
    init_log(Path(__file__).stem)

    config = cast(DictConfig, OmegaConf.load(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    vocab, model = load_model(config, checkpoint_path, device)

    dataset = UnlabeledSliceDataset(test_file, config, vocab)
    batch_size = args.batch_size or int(
        config.hyper_parameters.test_batch_size
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_slices,
        pin_memory=device.type == "cuda",
    )

    predictions = predict(model, data_loader, device)
    vulnerable_predictions = [
        prediction
        for prediction in predictions
        if prediction["predicted_label"] == 1
    ]
    output = {
        "checkpoint": str(checkpoint_path),
        "test_file": str(test_file),
        "num_slices": len(predictions),
        "num_predicted_vulnerable": len(vulnerable_predictions),
        "vulnerable_predictions": vulnerable_predictions,
        "all_predictions": predictions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as wfi:
        json.dump(output, wfi, indent=2)
    logging.info(
        f"Predicted {len(vulnerable_predictions)} of {len(predictions)} "
        "slices as vulnerable."
    )
    logging.info(f"Ranked predictions written to: {output_path}")


if __name__ == "__main__":
    main()
