import json
import logging
import pickle

import networkx as nx

from os.path import basename, join, splitext
from omegaconf import DictConfig, OmegaConf
from typing import Iterable, List, cast

from tqdm import tqdm

from src.common_utils import get_arg_parser, init_log


def iter_slice_sentences(all_slices: Iterable[str]):
    for slice_path in tqdm(all_slices, desc="Slices"):
        with open(slice_path, "rb") as rbfi:
            slice_graph: nx.DiGraph = pickle.load(rbfi)
        for sentence in slice_graph.graph["slice_sym_token"]:
            yield sentence


def write_corpus(all_slices: List[str], corpus_filepath: str) -> int:
    num_sentences = 0
    num_tokens_with_whitespace = 0

    with open(corpus_filepath, "w", encoding="utf-8") as wfi:
        for sentence in iter_slice_sentences(all_slices):
            num_tokens_with_whitespace += sum(
                1 for token in sentence if any(char.isspace() for char in token)
            )
            wfi.write(" ".join(sentence))
            wfi.write("\n")
            num_sentences += 1

    if num_tokens_with_whitespace > 0:
        logging.warning(
            "Found %d tokens containing whitespace. Space-delimited corpus files "
            "cannot preserve those as single tokens.",
            num_tokens_with_whitespace,
        )

    return num_sentences


if __name__ == "__main__":
    arg_parser = get_arg_parser()
    arg_parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="Output corpus filepath. Defaults to <dataset_root>/w2v_corpus.txt.",
    )
    args = arg_parser.parse_args()
    init_log(splitext(basename(__file__))[0])

    config = cast(DictConfig, OmegaConf.load(args.config))

    dataset_root = join(config.data_folder, config.dataset.name)
    if args.use_temp_data:
        dataset_root = config.temp_root

    all_slices_filepath = join(dataset_root, config.all_slices_filename)
    corpus_filepath = args.output or join(dataset_root, "w2v_corpus.txt")

    logging.info(f"Loading all generated slices from {all_slices_filepath}...")
    with open(all_slices_filepath, "r") as rfi:
        all_slices = json.load(rfi)
    logging.info(f"Completed. Loaded {len(all_slices)} slices.")

    logging.info(f"Writing Word2Vec corpus to {corpus_filepath}...")
    num_sentences = write_corpus(all_slices, corpus_filepath)
    logging.info(f"Completed. Wrote {num_sentences} sentences.")
    logging.info("=========End session=========")
    logging.shutdown()
