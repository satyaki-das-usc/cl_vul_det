from collections import defaultdict
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

config = None

def init_log():
    LOG_DIR = "logs"
    if not isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logging.basicConfig(
        handlers=[
            logging.FileHandler(join(LOG_DIR, "generate_file_slice_mapping.log")),
            logging.StreamHandler()
        ],
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("=========New session=========")
    logging.info(f"Logging dir: {LOG_DIR}")

def process_slice_parallel(slice_path, queue: Queue):
    try:
        with open(slice_path, "rb") as rbfi:
            slice_graph: nx.DiGraph = pickle.load(rbfi)
       
        src_cpp_path = join(slice_path.partition(config.slice_folder)[0], config.source_root_folder, slice_graph.graph["file_paths"][0])
        
        return src_cpp_path, slice_path
    
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

    dataset_root = join(config.data_folder, config.dataset.name)
    if args.use_temp_data:
        dataset_root = config.temp_root
    
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
        file_slices: List = [
            (src_cpp_path, slice_path)
            for src_cpp_path, slice_path in tqdm(
                pool.imap_unordered(process_func, all_slices),
                desc=f"Slices",
                total=len(all_slices),
            )
        ]
        message_queue.put("finished")
        pool.close()
        pool.join()
    
    logging.info(f"Completed. Converting to JSON format...")
    file_slices_map = defaultdict(list)
    for src_cpp_path, slice_path in tqdm(file_slices):
        file_slices_map[src_cpp_path].append(slice_path)
    logging.info(f"Completed. Found {len(file_slices_map)} unique source files.")
    
    file_slices_filepath = join(dataset_root, config.file_slices_filename)
    logging.info(f"Saving file slices map to {file_slices_filepath}...")
    with open(file_slices_filepath, "w") as wfi:
        json.dump(file_slices_map, wfi, indent=2)
    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()