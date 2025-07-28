import os
import json
import pickle
import numpy as np

import logging

import networkx as nx

from multiprocessing import cpu_count
from os.path import join, isdir, splitext
from omegaconf import DictConfig, OmegaConf
from typing import cast, List
from itertools import cycle
from random import sample

from tqdm import tqdm
from pytorch_lightning import seed_everything

from src.common_utils import get_arg_parser, filter_warnings

dataset_root = ""
train_slices = []

def init_log():
    LOG_DIR = "logs"
    if not isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logging.basicConfig(
        handlers=[
            logging.FileHandler(join(LOG_DIR, "generate_custom_balanced_batches.log")),
            logging.StreamHandler()
        ],
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("=========New session=========")
    logging.info(f"Logging dir: {LOG_DIR}")

def create_balanced_batches(tuple_list, max_per_label=64):
    """
    Create balanced batches from a list of (pos_indices, neg_indices) tuples, using all majority indices.
    
    Args:
        tuple_list: List of 2-tuples, each containing ([pos_indices], [neg_indices]).
        max_per_label: Maximum number of samples per label in a batch (default: 64).
    
    Returns:
        List of 2-tuples, each representing a batch ([pos_indices], [neg_indices]).
    """
    batches = []

    for pos_indices, neg_indices in tuple_list:
        pos_indices = np.array(pos_indices, dtype=int)
        neg_indices = np.array(neg_indices, dtype=int)
        num_pos = len(pos_indices)
        num_neg = len(neg_indices)

        # Skip if either list is empty
        if num_pos == 0 or num_neg == 0:
            continue

        # Determine minority and majority classes
        if num_pos <= num_neg:
            minority_indices = pos_indices
            majority_indices = neg_indices
            majority_size = num_neg
            is_pos_minority = True
        else:
            minority_indices = neg_indices
            majority_indices = pos_indices
            majority_size = num_pos
            is_pos_minority = False

        # Set batch size (min of minority size and max_per_label)
        batch_size = min(len(minority_indices), max_per_label)
        
        # Shuffle majority indices to distribute randomly
        np.random.shuffle(majority_indices)
        
        # Create cyclic iterator for minority indices
        minority_cycle = cycle(minority_indices)

        for i in range(0, majority_size, batch_size):
            # Calculate current batch size (last batch may be smaller)
            current_batch_size = min(batch_size, majority_size - i)
            if current_batch_size == 0:
                break
                
            # Select majority indices for this batch
            selected_majority = majority_indices[i:i + current_batch_size]
            
            # Select minority indices using cyclic iterator
            selected_minority = np.array([next(minority_cycle) for _ in range(current_batch_size)], dtype=int)
            
            # Assign selected indices to pos and neg based on minority class
            if is_pos_minority:
                selected_pos = selected_minority
                selected_neg = selected_majority
            else:
                selected_pos = selected_majority
                selected_neg = selected_minority
            
            batches.append((selected_pos.tolist(), selected_neg.tolist()))
    
    return batches

def get_feat_dict(instance_index_map):
    # Step 1: Group cpp_paths by feat_code with pos_indices length
    intermediate = []
    for cpp_path, data in instance_index_map.items():
        feat_code = str(data["feat_code"])  # Convert feat_code to string
        pos_indices_len = len(data["pos_indices"])  # Length for sorting
        intermediate.append((cpp_path, feat_code, pos_indices_len))
    
    # Step 2: Sort by feat_code and pos_indices length (descending)
    intermediate.sort(key=lambda x: (x[1], -x[2]))  # Sort by feat_code, then by pos_indices_len descending
    
    # Step 3: Build output dictionary
    output_dict = {}
    current_feat_code = None
    current_paths = []
    
    for cpp_path, feat_code, _ in intermediate:
        if feat_code != current_feat_code:
            if current_feat_code is not None:
                output_dict[f"feat_{current_feat_code}"] = current_paths
            current_feat_code = feat_code
            current_paths = [cpp_path]
        else:
            current_paths.append(cpp_path)
    
    # Add the last group
    if current_feat_code is not None:
        output_dict[f"feat_{current_feat_code}"] = current_paths

    return output_dict

def is_positive_inadequate_instance(instance_data) -> bool:
    """
    Check if the instance data has enough positive indices.
    """
    return len(instance_data["pos_indices"]) < 3

def is_negative_inadequate_instance(instance_data) -> bool:
    """
    Check if the instance data has enough negative indices.
    """
    return len(instance_data["neg_indices"]) == 0

def is_adequate_instance(instance_data) -> bool:
    """
    Check if the instance data is adequate.
    """
    if is_positive_inadequate_instance(instance_data):
        return False
    if is_negative_inadequate_instance(instance_data):
        return False
    return True

def get_instance_tuple_list(unperturbed_file_list: List[str]) -> List[tuple]:
    instance_index_map = {}

    other_train_data = {}

    logging.info("Creating instance index map...")
    for idx, slice_path in tqdm(enumerate(train_slices), total=len(train_slices)):
        with open(slice_path, "rb") as rbfi:
            slice_graph: nx.DiGraph = pickle.load(rbfi)
        src_cpp_path = join(slice_path.partition(config.slice_folder)[0], config.source_root_folder, slice_graph.graph["file_paths"][0])
        cpp_filepath = src_cpp_path.partition(config.source_root_folder)[-1][1:]
        if cpp_filepath.endswith("io.c"):
            continue
        if cpp_filepath in unperturbed_file_list:
            if cpp_filepath not in instance_index_map:
                instance_index_map[cpp_filepath] = {
                    "feat_code": slice_graph.graph["feat_code"],
                    "pos_indices": [],
                    "neg_indices": []
                }
            if slice_graph.graph["label"]:
                instance_index_map[cpp_filepath]["pos_indices"].append(idx)
            else:
                instance_index_map[cpp_filepath]["neg_indices"].append(idx)
        else:
            other_train_data[slice_path] = idx
    logging.info(f"Number of instances: {len(instance_index_map)}")

    logging.info("Assigning other instances to the same batch...")
    for cpp_filepath, slice_data in tqdm(instance_index_map.items(), total=len(instance_index_map)):
        cpp_filepath_no_ext = splitext(cpp_filepath)[0]
        keys_to_remove = []
        for slice_path, idx in other_train_data.items():
            if cpp_filepath_no_ext not in slice_path:
                continue
            with open(slice_path, "rb") as rbfi:
                slice_graph: nx.DiGraph = pickle.load(rbfi)
            if slice_graph.graph["label"]:
                slice_data["pos_indices"].append(idx)
            else:
                slice_data["neg_indices"].append(idx)
            keys_to_remove.append(slice_path)
        for key in keys_to_remove:
            del other_train_data[key]
    logging.info(f"Number of instances: {len(instance_index_map)}")

    keys_to_remove = []
    neutral_sensi_path = ""
    for slice_path, idx in other_train_data.items():
        if "neutral_sensi" not in slice_path:
            continue
        with open(slice_path, "rb") as rbfi:
            slice_graph: nx.DiGraph = pickle.load(rbfi)

        src_cpp_path = join(slice_path.partition(config.slice_folder)[0], config.source_root_folder, slice_graph.graph["file_paths"][0])
        cpp_filepath = src_cpp_path.partition(config.source_root_folder)[-1][1:]
        if neutral_sensi_path == "":
            neutral_sensi_path = cpp_filepath
        if neutral_sensi_path != cpp_filepath:
            keys_to_remove.append(slice_path)
            continue
        if cpp_filepath not in instance_index_map:
            instance_index_map[cpp_filepath] = {
                "feat_code": slice_graph.graph["feat_code"],
                "pos_indices": [],
                "neg_indices": []
            }
        if slice_graph.graph["label"]:
            instance_index_map[cpp_filepath]["pos_indices"].append(idx)
        else:
            instance_index_map[cpp_filepath]["neg_indices"].append(idx)
        keys_to_remove.append(slice_path)

    for key in keys_to_remove:
        del other_train_data[key]

    for slice_path, idx in other_train_data.items():
        with open(slice_path, "rb") as rbfi:
            slice_graph: nx.DiGraph = pickle.load(rbfi)

        src_cpp_path = join(slice_path.partition(config.slice_folder)[0], config.source_root_folder, slice_graph.graph["file_paths"][0])
        cpp_filepath = src_cpp_path.partition(config.source_root_folder)[-1][1:]

        if cpp_filepath not in instance_index_map:
            instance_index_map[cpp_filepath] = {
                "feat_code": slice_graph.graph["feat_code"],
                "pos_indices": [],
                "neg_indices": []
            }
        if slice_graph.graph["label"]:
            instance_index_map[cpp_filepath]["pos_indices"].append(idx)
        else:
            instance_index_map[cpp_filepath]["neg_indices"].append(idx)

    feat_dict = get_feat_dict(instance_index_map)

    keys_to_remove = []
    for cpp_filepath, instance_data in tqdm(instance_index_map.items(), total=len(instance_index_map)):
        if not is_positive_inadequate_instance(instance_data) and not is_negative_inadequate_instance(instance_data):
            continue
        if is_negative_inadequate_instance(instance_data):
            if is_positive_inadequate_instance(instance_data):
                merge_cpp_path = feat_dict[f"feat_{instance_data['feat_code']}"][0]
            else:
                merge_cpp_path = feat_dict[f"feat_{instance_data['feat_code']}"][-1]
        elif is_positive_inadequate_instance(instance_data):
            merge_cpp_path = feat_dict[f"feat_{instance_data['feat_code']}"][0]
        if cpp_filepath == merge_cpp_path:
            continue
        instance_index_map[merge_cpp_path]["pos_indices"].extend(instance_data["pos_indices"])
        instance_index_map[merge_cpp_path]["neg_indices"].extend(instance_data["neg_indices"])
        keys_to_remove.append(cpp_filepath)

    for key in keys_to_remove:
        del instance_index_map[key]

    return [(instance_data["pos_indices"], instance_data["neg_indices"]) for instance_data in instance_index_map.values()]

def get_vf_tuple_list() -> List[tuple]:
    curr_feat_indices = {}

    for idx, slice_path in tqdm(enumerate(train_slices), total=len(train_slices)):
        with open(slice_path, "rb") as rbfi:
            slice_graph: nx.DiGraph = pickle.load(rbfi)
        feat_code = slice_graph.graph["feat_code"]
        if f"{feat_code}" not in curr_feat_indices:
            curr_feat_indices[f"{feat_code}"] = {
                "pos_indices": [],
                "neg_indices": [],
            }
        if slice_graph.graph["label"]:
            curr_feat_indices[f"{feat_code}"]["pos_indices"].append(idx)
        else:
            curr_feat_indices[f"{feat_code}"]["neg_indices"].append(idx)
    return [(entry["pos_indices"], entry["neg_indices"]) for entry in curr_feat_indices.values()]

def get_swav_tuple_list(swav_batches: List[List[int]]) -> List[tuple]:
    tuple_list = []
    for batch in tqdm(swav_batches):
        pos_indices = []
        neg_indices = []
        for idx in batch:
            with open(train_slices[idx], "rb") as rbfi:
                slice_graph: nx.DiGraph = pickle.load(rbfi)
            if slice_graph.graph["label"]:
                pos_indices.append(idx)
            else:
                neg_indices.append(idx)
        tuple_list.append((pos_indices, neg_indices))
    return tuple_list

if __name__ == "__main__":
    filter_warnings()
    arg_parser = get_arg_parser()
    arg_parser.add_argument("-s", "--sampler", type=str, required=True)
    arg_parser.add_argument("--do_balancing", action='store_true', help="Generate balanced batches")
    args = arg_parser.parse_args()

    config = cast(DictConfig, OmegaConf.load(args.config))
    seed_everything(config.seed, workers=True)

    if args.sampler not in config.train_sampler_options:
        raise ValueError(f"Sampler {args.sampler} not in options: {config.train_sampler_options}")

    init_log()
    sampler = args.sampler

    if config.num_workers != -1:
        USE_CPU = min(config.num_workers, cpu_count())
    else:
        USE_CPU = cpu_count()

    if args.use_temp_data:
        dataset_root = config.temp_root
    else:
        dataset_root = config.data_folder
    
    if config.dataset.name == "Devign":
        dataset_root = join(config.data_folder, config.dataset.name)
    
    train_dataset_path = join(dataset_root, config.train_slices_filename)
    logging.info(f"Loading training dataset from {train_dataset_path}")
    with open(train_dataset_path, "r") as rfi:
        train_slices = json.load(rfi)
    logging.info(f"Loaded {len(train_slices)} training slices")

    tuple_list = []
    if sampler == "vf":
        logging.info("Creating VF tuple list...")
        tuple_list = get_vf_tuple_list()
        logging.info(f"VF tuple list created. Total tuples: {len(tuple_list)}")
    elif sampler == "instance":
        logging.info("Creating instance tuple list...")
        unperturbed_file_list = []
        logging.info("Loading unperturbed file list...")
        for feat_name in config.vul_feats:
            unperturbed_file_list_path = join(dataset_root, config.VF_perts_root, feat_name, config.unperturbed_files_filename)
            with open(unperturbed_file_list_path, "r") as rfi:
                unperturbed_file_list += json.load(rfi)
        unperturbed_file_list = list(set(unperturbed_file_list))
        logging.info(f"Number of unperturbed files: {len(unperturbed_file_list)}")
        tuple_list = get_instance_tuple_list(unperturbed_file_list)
        logging.info(f"Instance tuple list created. Total tuples: {len(tuple_list)}")
    elif sampler == "swav":
        logging.info("Creating SwAV tuple list...")
        swav_batches_filepath = join(dataset_root, config.swav_batches_filename)
        with open(swav_batches_filepath, "r") as rfi:
            swav_batches = json.load(rfi)
        tuple_list = get_swav_tuple_list(swav_batches)
        logging.info(f"SwAV tuple list created. Total tuples: {len(tuple_list)}")
    
    if args.do_balancing:    
        logging.info(f"Creating balanced batches from {len(tuple_list)} tuples...")
        batches = [sample(batch[0] + batch[1], len(batch[0]) + len(batch[1])) for batch in create_balanced_batches(tuple_list)] 
    else:
        batches = [batch[0] + batch[1] for batch in tuple_list]

    # Save the balanced batches to disk
    if args.do_balancing:
        output_filepath = join(dataset_root, f"{sampler}_balanced_batches.json")
    else:
        output_filepath = join(dataset_root, f"{sampler}_sub_datasets.json")
    logging.info(f"Created {len(batches)} batches. Saving to {output_filepath}...")
    with open(output_filepath, "w") as wfi:
        json.dump(batches, wfi)
    
    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()