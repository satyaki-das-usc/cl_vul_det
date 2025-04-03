import functools
import os
import json
import pickle

import networkx as nx
import logging

from multiprocessing import Manager, Pool, Queue, cpu_count
from os.path import join, isdir
from omegaconf import DictConfig, OmegaConf
from typing import List, cast

from tqdm import tqdm

from src.common_utils import get_arg_parser
from src.slice_tokenizer import SliceTokenizer

config = None

def init_log():
    LOG_DIR = "logs"
    if not isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logging.basicConfig(
        handlers=[
            logging.FileHandler(join(LOG_DIR, "tokenize_slices.log")),
            logging.StreamHandler()
        ],
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("=========New session=========")
    logging.info(f"Logging dir: {LOG_DIR}")

def code_sym_token_exists(slice: nx.DiGraph) -> bool:
    for n in slice:
        if "code_sym_token" in slice.nodes[n]:
            return True
    return False

def process_slice_parallel(slice_path, queue: Queue):
    try:
        with open(slice_path, "rb") as rbfi:
            slice_graph: nx.DiGraph = pickle.load(rbfi)
    
        if code_sym_token_exists(slice_graph):
            return slice_graph
        
        src_cpp_path = join(slice_path.partition(config.slice_folder)[0], config.source_root_folder, slice_graph.graph["file_paths"][0])
        with open(src_cpp_path, "r") as rfi:
            src_lines = rfi.readlines()

        tokenizer = SliceTokenizer(slice_graph, src_lines, config)
        tokenized_slice = tokenizer.tokenize_slice()

        if len(tokenized_slice.nodes) == 0:
            os.system(f"rm {slice_path}")
            return ""
        if len(tokenized_slice.edges) == 0:
            os.system(f"rm {slice_path}")
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
    init_log()
    
    config = cast(DictConfig, OmegaConf.load(args.config))
    if config.num_workers != -1:
        USE_CPU = min(config.num_workers, cpu_count())
    else:
        USE_CPU = cpu_count()

    if args.use_temp_data:
        dataset_root = config.temp_root
    else:
        dataset_root = config.data_folder
    all_slices_filepath = join(dataset_root, config.all_slices_filename)
    logging.info(f"Loading all generated slices from {all_slices_filepath}...")
    with open(all_slices_filepath, "r") as rfi:
        all_slices = json.load(rfi)
    logging.info(f"Completed. Loaded {len(all_slices)} slices.")

    logging.info(f"Going over {len(all_slices)} files...")
    with Manager() as m:
        message_queue = m.Queue()  # type: ignore
        pool = Pool(USE_CPU)
        process_func = functools.partial(process_slice_parallel, queue=message_queue)
        non_empty_slice_paths: List = [
            slice_path
            for slice_path in tqdm(
                pool.imap_unordered(process_func, all_slices),
                desc=f"Slices",
                total=len(all_slices),
            )
            if slice_path != ""
        ]
        message_queue.put("finished")
        pool.close()
        pool.join()
    
    logging.info(f"Tokenized {len(non_empty_slice_paths)} slices.")
    logging.info(f"Saving tokenized slices to {all_slices_filepath}...")
    with open(all_slices_filepath, "w") as wfi:
        json.dump(non_empty_slice_paths, wfi, indent=2)
    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()