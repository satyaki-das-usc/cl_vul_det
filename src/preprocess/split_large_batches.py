import os
import json
import pickle

import logging

import networkx as nx

from multiprocessing import cpu_count
from os.path import join, isdir
from math import floor
from omegaconf import DictConfig, OmegaConf
from typing import cast

from tqdm import tqdm
from pytorch_lightning import seed_everything

from src.common_utils import get_arg_parser, filter_warnings

def init_log():
    LOG_DIR = "logs"
    if not isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logging.basicConfig(
        handlers=[
            logging.FileHandler(join(LOG_DIR, "split_large_batches.log")),
            logging.StreamHandler()
        ],
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("=========New session=========")
    logging.info(f"Logging dir: {LOG_DIR}")

def delete_indices(lst, indices):
    indices_set = set(indices)
    return [item for i, item in enumerate(lst) if i not in indices_set]

def split_into_n_lists(lst, n):
    k, r = divmod(len(lst), n)
    result = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < r else 0)
        result.append(lst[start:end])
        start = end
    return result

if __name__ == "__main__":
    filter_warnings()
    arg_parser = get_arg_parser()
    arg_parser.add_argument(
        "--merge_type",
        type=str,
        default="centroid",
        choices=["prototype", "centroid"],
        help="Type of cluster merging to use: 'prototype' or 'centroid'"
    )
    args = arg_parser.parse_args()

    config = cast(DictConfig, OmegaConf.load(args.config))
    seed_everything(config.seed, workers=True)

    init_log()

    if config.num_workers != -1:
        USE_CPU = min(config.num_workers, cpu_count())
    else:
        USE_CPU = cpu_count()

    if args.use_temp_data:
        dataset_root = config.temp_root
    else:
        dataset_root = config.data_folder
    
    train_dataset_path = join(dataset_root, config.train_slices_filename)
    logging.info(f"Loading training dataset from {train_dataset_path}")
    with open(train_dataset_path, "r") as rfi:
        train_slices = json.load(rfi)
    logging.info(f"Loaded {len(train_slices)} training slices")

    train_labels = []
    logging.info(f"Loading training labels...")
    for slice_path in tqdm(train_slices):
        with open(slice_path, "rb") as rbfi:
            slice_graph: nx.DiGraph = pickle.load(rbfi)
        train_labels.append(slice_graph.graph["label"])
    logging.info(f"Loaded {len(train_labels)} training labels")

    swav_batches_filepath = join(dataset_root, config.swav_batches_filename)
    logging.info(f"Loading minibatches from {swav_batches_filepath}...")
    with open(swav_batches_filepath, "r") as rfi:
        minibatches = json.load(rfi)
    logging.info(f"Loaded {len(minibatches)} minibatches")

    large_minibatches = [(idx, minibatch) for idx, minibatch in enumerate(minibatches) if len(minibatch) > (2 * config.hyper_parameters.batch_size)]

    new_minibatches = []
    for idx, minibatch in tqdm(large_minibatches):
        vul_indices = [i for i in minibatch if train_labels[i] == 1]
        nonvul_indices = [i for i in minibatch if train_labels[i] == 0]

        vul_cnt = len(vul_indices)
        nonvul_cnt = len(nonvul_indices)

        max_vul_split = floor(vul_cnt / config.min_vul_per_batch)
        max_nonvul_split = floor(nonvul_cnt / config.min_nonvul_per_batch)
        max_split = min(max_vul_split, max_nonvul_split)
        desired_split = len(minibatch) // config.hyper_parameters.batch_size

        final_split = min(max_split, desired_split)

        vul_splits = split_into_n_lists(vul_indices, final_split)
        nonvul_splits = split_into_n_lists(nonvul_indices, final_split)

        minibatch_splits = [vul_split + nonvul_split for vul_split, nonvul_split in zip(vul_splits, nonvul_splits)]

        new_minibatches += minibatch_splits

    minibatches = delete_indices(minibatches, [idx for idx, minibatch in large_minibatches]) + new_minibatches
    logging.info(f"Number of minibatches after splitting: {len(minibatches)}. Saving to {swav_batches_filepath}...")
    with open(swav_batches_filepath, "w") as wfi:
        json.dump(minibatches, wfi, indent=2)
    
    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()