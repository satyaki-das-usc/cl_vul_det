import functools
import hashlib
import os
import json
import pickle

import networkx as nx
import logging

from multiprocessing import Pool, cpu_count
from os.path import join, splitext, basename
from omegaconf import DictConfig, OmegaConf
from typing import List, cast

from tqdm import tqdm

from src.common_utils import get_arg_parser, init_log
from src.slice_tokenizer import SliceTokenizer

config = None

SLICE_CODE_DIGEST_SIZE = 16


def slice_metadata(slice_path: str, slice_graph: nx.DiGraph):
    code_digest = hashlib.blake2b(
        slice_graph.graph["slice_sym_code"].encode("utf-8"),
        digest_size=SLICE_CODE_DIGEST_SIZE,
    ).digest()
    return slice_path, bool(slice_graph.graph["label"]), code_digest


@functools.lru_cache(maxsize=32)
def load_source_lines(src_cpp_path):
    with open(src_cpp_path, "r") as rfi:
        return rfi.readlines()

def slice_is_tokenized(slice_graph: nx.DiGraph) -> bool:
    return (
        "slice_sym_code" in slice_graph.graph
        and "slice_sym_token" in slice_graph.graph
        and all(
            "code_sym_token" in node_data
            for _, node_data in slice_graph.nodes(data=True)
        )
    )

def process_slice_parallel(slice_path):
    try:
        with open(slice_path, "rb") as rbfi:
            slice_graph: nx.DiGraph = pickle.load(rbfi)
    
        if slice_is_tokenized(slice_graph):
            return slice_metadata(slice_path, slice_graph)
        
        src_cpp_path = join(slice_path.partition(config.slice_folder)[0], config.source_root_folder, slice_graph.graph["file_paths"][0])
        src_lines = load_source_lines(src_cpp_path)

        tokenizer = SliceTokenizer(slice_graph, src_lines, config)
        tokenized_slice = tokenizer.tokenize_slice()

        if len(tokenized_slice.nodes) == 0:
            os.remove(slice_path)
            return None
        if len(tokenized_slice.edges) == 0:
            os.remove(slice_path)
            return None
        with open(slice_path, "wb") as wbfi:
            pickle.dump(tokenized_slice, wbfi, pickle.HIGHEST_PROTOCOL)
        
        return slice_metadata(slice_path, tokenized_slice)
        
    except Exception as e:
        logging.error(slice_path)
        raise e

if __name__ == "__main__":
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()
    init_log(splitext(basename(__file__))[0])
    
    config = cast(DictConfig, OmegaConf.load(args.config))
    if config.num_workers != -1:
        USE_CPU = min(config.num_workers, cpu_count())
    else:
        USE_CPU = cpu_count()

    dataset_root = join(config.data_folder, config.dataset.name)
    if args.use_temp_data:
        dataset_root = config.temp_root
    
    all_slices_filepath = join(dataset_root, config.all_slices_filename)
    logging.info(f"Loading all generated slices from {all_slices_filepath}...")
    with open(all_slices_filepath, "r") as rfi:
        all_slices = json.load(rfi)
    logging.info(f"Completed. Loaded {len(all_slices)} slices.")

    chunksize = max(
        1,
        min(256, len(all_slices) // max(1, USE_CPU * 8)),
    )
    logging.info(
        f"Using {USE_CPU} workers with multiprocessing chunksize {chunksize}."
    )
    logging.info(f"Going over {len(all_slices)} files...")
    non_empty_slice_paths: List = []
    slice_metadata_map = {}
    with Pool(USE_CPU) as pool:
        for metadata in tqdm(
            pool.imap_unordered(
                process_slice_parallel,
                all_slices,
                chunksize=chunksize,
            ),
            desc=f"Slices",
            total=len(all_slices),
        ):
            if metadata is None:
                continue
            slice_path, label, code_digest = metadata
            non_empty_slice_paths.append(slice_path)
            slice_metadata_map[slice_path] = (label, code_digest)
    
    logging.info(f"Tokenized {len(non_empty_slice_paths)} slices.")
    logging.info(f"Saving tokenized slices to {all_slices_filepath}...")
    with open(all_slices_filepath, "w") as wfi:
        json.dump(non_empty_slice_paths, wfi, indent=2)
    logging.info(f"Completed.")

    slice_metadata_filepath = join(dataset_root, config.slice_metadata_filename)
    logging.info(f"Saving slice metadata to {slice_metadata_filepath}...")
    with open(slice_metadata_filepath, "wb") as wfi:
        pickle.dump(slice_metadata_map, wfi, pickle.HIGHEST_PROTOCOL)
    logging.info("Completed.")
    logging.info("=========End session=========")
    logging.shutdown()
