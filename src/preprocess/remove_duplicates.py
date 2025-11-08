import functools
import os
import json
import pickle

import networkx as nx
import logging

from multiprocessing import Manager, Pool, Queue, cpu_count
from os.path import join, splitext, basename
from omegaconf import DictConfig, OmegaConf
from typing import List, cast

from tqdm import tqdm

from src.common_utils import get_arg_parser, init_log

file_slices = dict()

def process_file_parallel(cpp_path, queue: Queue):
    try:
        all_slices = file_slices[cpp_path]
        vul_slice_list = []
        nonvul_slice_list = []

        for slice_path in all_slices:
            with open(slice_path, "rb") as rbfi:
                slice_graph: nx.DiGraph = pickle.load(rbfi)
            if slice_graph.graph["label"]:
                vul_slice_list.append((slice_path, slice_graph))
            else:
                nonvul_slice_list.append((slice_path, slice_graph))
        
        unique_vul_slice_set = set()
        unique_vul_slice_list = []
        for entry in vul_slice_list:
            slice_path, slice_graph = entry
            if slice_graph.graph["slice_sym_code"] not in unique_vul_slice_set:
                unique_vul_slice_set.add(slice_graph.graph["slice_sym_code"])
                unique_vul_slice_list.append((slice_path, slice_graph))
            else:
                with open("duplicate_slices.txt", "a") as afi:
                    afi.write(f"{slice_path}\n")
        
        unique_nonvul_slice_set = set()
        unique_nonvul_slice_list = []
        for entry in nonvul_slice_list:
            slice_path, slice_graph = entry
            if slice_graph.graph["slice_sym_code"] not in unique_nonvul_slice_set:
                unique_nonvul_slice_set.add(slice_graph.graph["slice_sym_code"])
                unique_nonvul_slice_list.append((slice_path, slice_graph))
            else:
                with open("duplicate_slices.txt", "a") as afi:
                    afi.write(f"{slice_path}\n")
        
        final_unique_nonvul_slice_list = []
        for entry in unique_nonvul_slice_list:
            slice_path, slice_graph = entry
            if slice_graph.graph["slice_sym_code"] not in unique_vul_slice_set:
                final_unique_nonvul_slice_list.append((slice_path, slice_graph))
            else:
                with open("duplicate_slices.txt", "a") as afi:
                    afi.write(f"{slice_path}\n")
        
        return [entry[0] for entry in unique_vul_slice_list + final_unique_nonvul_slice_list]
    except Exception as e:
        logging.error(cpp_path)
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

    os.system(f"touch duplicate_slices.txt")

    file_slices_path = join(dataset_root, config.file_slices_filename)
    logging.info(f"Loading filewise generated slices from {file_slices_path}...")
    with open(file_slices_path, "r") as rfi:
        file_slices = json.load(rfi)
    logging.info(f"Completed. Loaded slices for {len(file_slices)} files.")

    logging.info(f"Going over {len(file_slices)} files...")
    cpp_paths = list(file_slices.keys())
    with Manager() as m:
        message_queue = m.Queue()  # type: ignore
        pool = Pool(USE_CPU)
        process_func = functools.partial(process_file_parallel, queue=message_queue)
        unique_slice_list: List = [
            file_slice
            for file_slices in tqdm(
                pool.imap_unordered(process_func, cpp_paths),
                desc=f"Cpp files",
                total=len(cpp_paths),
            )
            for file_slice in file_slices
        ]

        message_queue.put("finished")
        pool.close()
        pool.join()
    
    logging.info(f"Total unique slices: {len(unique_slice_list)}")
    all_slices_filepath = join(dataset_root, config.all_slices_filename)
    logging.info(f"Saving unique slices to {all_slices_filepath}...")
    with open(all_slices_filepath, "w") as wfi:
        json.dump(unique_slice_list, wfi, indent=2)
    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()