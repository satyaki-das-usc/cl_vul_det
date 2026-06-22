import json
import logging
import pickle

import networkx as nx

from multiprocessing import Pool, cpu_count
from os.path import basename, join, splitext
from omegaconf import DictConfig, OmegaConf
from typing import Iterable, List, Tuple, cast

from tqdm import tqdm

from src.common_utils import get_arg_parser, init_log


def extract_slice_sentences(slice_path: str) -> List[List[str]]:
    with open(slice_path, "rb") as rbfi:
        slice_graph: nx.DiGraph = pickle.load(rbfi)
    return slice_graph.graph["slice_sym_token"]


def sentence_stats(sentence: Iterable[str]) -> Tuple[int, int]:
    return (
        1,
        sum(1 for token in sentence if any(char.isspace() for char in token)),
    )


def write_sentences(wfi, sentences: Iterable[List[str]]) -> Tuple[int, int]:
    num_sentences = 0
    num_tokens_with_whitespace = 0

    for sentence in sentences:
        sentence_count, whitespace_count = sentence_stats(sentence)
        wfi.write(" ".join(sentence))
        wfi.write("\n")
        num_sentences += sentence_count
        num_tokens_with_whitespace += whitespace_count

    return num_sentences, num_tokens_with_whitespace


def write_corpus(
    all_slices: List[str],
    corpus_filepath: str,
    num_workers: int,
    chunksize: int,
) -> int:
    total_sentences = 0
    total_tokens_with_whitespace = 0

    with open(corpus_filepath, "w", encoding="utf-8") as wfi:
        with Pool(processes=num_workers) as pool:
            slice_sentence_iter = pool.imap(
                extract_slice_sentences,
                all_slices,
                chunksize=chunksize,
            )
            for sentences in tqdm(
                slice_sentence_iter,
                desc="Slices",
                total=len(all_slices),
            ):
                num_sentences, num_tokens_with_whitespace = write_sentences(
                    wfi, sentences
                )
                total_sentences += num_sentences
                total_tokens_with_whitespace += num_tokens_with_whitespace

    if total_tokens_with_whitespace > 0:
        logging.warning(
            "Found %d tokens containing whitespace. Space-delimited corpus files "
            "cannot preserve those as single tokens.",
            total_tokens_with_whitespace,
        )

    return total_sentences


if __name__ == "__main__":
    arg_parser = get_arg_parser()
    arg_parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=str,
        help="Output corpus filepath. Defaults to <dataset_root>/w2v_corpus.txt.",
    )
    arg_parser.add_argument(
        "--chunksize",
        default=16,
        type=int,
        help="Number of slice paths sent to each worker task batch.",
    )
    args = arg_parser.parse_args()
    init_log(splitext(basename(__file__))[0])

    config = cast(DictConfig, OmegaConf.load(args.config))
    if config.num_workers != -1:
        use_cpu = min(config.num_workers, cpu_count())
    else:
        use_cpu = cpu_count()

    dataset_root = join(config.data_folder, config.dataset.name)
    if args.use_temp_data:
        dataset_root = config.temp_root

    all_slices_filepath = join(dataset_root, config.all_slices_filename)
    corpus_filepath = args.output or join(dataset_root, "w2v_corpus.txt")

    logging.info(f"Loading all generated slices from {all_slices_filepath}...")
    with open(all_slices_filepath, "r") as rfi:
        all_slices = json.load(rfi)
    logging.info(f"Completed. Loaded {len(all_slices)} slices.")

    logging.info(
        "Writing Word2Vec corpus to %s with %d workers and chunksize %d...",
        corpus_filepath,
        use_cpu,
        args.chunksize,
    )
    num_sentences = write_corpus(
        all_slices,
        corpus_filepath,
        use_cpu,
        args.chunksize,
    )
    logging.info(f"Completed. Wrote {num_sentences} sentences.")
    logging.info("=========End session=========")
    logging.shutdown()
