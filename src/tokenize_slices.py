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

def code_sym_token_exists(slice: nx.DiGraph) -> bool:
    for n in slice:
        if "code_sym_token" in slice.nodes[n]:
            return True
    return False

def process_slice_parallel(slice_path):
    try:
        with open(slice_path, "rb") as rbfi:
            slice_graph: nx.DiGraph = pickle.load(rbfi)
    
        if code_sym_token_exists(slice_graph):
            return slice_path
        
        src_cpp_path = join(slice_path.partition(config.slice_folder)[0], config.source_root_folder, slice_graph.graph["file_paths"][0])
        with open(src_cpp_path, "r") as rfi:
            src_lines = rfi.readlines()

        tokenizer = SliceTokenizer(slice_graph, src_lines, config)
        tokenized_slice = tokenizer.tokenize_slice()

        if len(tokenized_slice.nodes) == 0:
            os.remove(slice_path)
            return ""
        if len(tokenized_slice.edges) == 0:
            os.remove(slice_path)
            return ""
        with open(slice_path, "wb") as wbfi:
            pickle.dump(tokenized_slice, wbfi, pickle.HIGHEST_PROTOCOL)
        
        return slice_path
        
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
    with Pool(USE_CPU) as pool:
        non_empty_slice_paths: List = [
            slice_path
            for slice_path in tqdm(
                pool.imap_unordered(
                    process_slice_parallel,
                    all_slices,
                    chunksize=chunksize,
                ),
                desc=f"Slices",
                total=len(all_slices),
            )
            if slice_path != ""
        ]
    
    logging.info(f"Tokenized {len(non_empty_slice_paths)} slices.")
    logging.info(f"Saving tokenized slices to {all_slices_filepath}...")
    with open(all_slices_filepath, "w") as wfi:
        json.dump(non_empty_slice_paths, wfi, indent=2)
    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()
