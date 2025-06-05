import functools
import os
import json
import pickle
import logging
import networkx as nx

from os.path import isdir, exists, dirname, splitext, join
from collections import defaultdict

from tqdm import tqdm
from multiprocessing import Manager, Pool, cpu_count
from omegaconf import DictConfig, OmegaConf
from typing import cast

from src.common_utils import get_arg_parser

all_feats_name = ["incorr_calc_buff_size", "buff_access_src_size", "off_by_one", "buff_overread", "double_free", "use_after_free", "buff_underwrite", "buff_underread", "sensi_read", "sensi_write"]
spu_feats_name = ["edge_set", "node_set"]

dataset_root = ""

filewise_xfg_path_map = dict()
instance_perturbation_map = dict()
unperturbed_file_list = set()

def init_log():
    LOG_DIR = "logs"
    if not isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logging.basicConfig(
        handlers=[
            logging.FileHandler(join(LOG_DIR, "generate_instance_perturbation_mapping.log")),
            logging.StreamHandler()
        ],
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("=========New session=========")
    logging.info(f"Logging dir: {LOG_DIR}")

def process_slice_path_parallel(slice_path, queue):
    with open(slice_path, "rb") as rbfi:
        slice_graph: nx.DiGraph = pickle.load(rbfi)
    file_path = slice_graph.graph["file_paths"][0]
    if file_path in unperturbed_file_list:
        return None, file_path
    file_dir = dirname(file_path)
    relevant_unperturbed_files = [entry for entry in unperturbed_file_list if entry.startswith(dirname(file_dir))]
    for unperturbed_file_path in relevant_unperturbed_files:
        if not file_path.startswith(splitext(unperturbed_file_path)[0]):
            continue
        return unperturbed_file_path, file_path
    return None, file_path

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
    
    train_slices_filepath = join(dataset_root, config.train_slices_filename)
    logging.info(f"Loading train slices from {train_slices_filepath}...")
    with open(train_slices_filepath, "r") as rfi:
        train_slices = json.load(rfi)
    logging.info(f"Completed. Loaded {len(train_slices)} slices.")

    filename_map = defaultdict(set)

    for feat_name in all_feats_name:
        logging.info(f"Processing {feat_name}...")
        feat_dir = join(dataset_root, config.VF_perts_root, feat_name)

        feat_slices = [slice_path for slice_path in train_slices if slice_path.startswith(feat_dir)]
        logging.info(f"Found {len(feat_slices)} slices for {feat_name}.")
        unperturbed_file_list_path = join(feat_dir, config.unperturbed_files_filename)
        assert exists(unperturbed_file_list_path), f"File {unperturbed_file_list_path} does not exist"
        logging.info(f"Loading unperturbed file list from {unperturbed_file_list_path}...")
        with open(unperturbed_file_list_path, "r") as rfi:
            unperturbed_file_list = set(json.load(rfi))
        logging.info(f"Loaded {len(unperturbed_file_list)} unperturbed files.")

        logging.info(f"Going over {len(feat_slices)} files...")
        with Manager() as m:
            message_queue = m.Queue()  # type: ignore
            pool = Pool(USE_CPU)
            process_func = functools.partial(process_slice_path_parallel, queue=message_queue)
            [filename_map[unperturbed_file_path].add(file_path)
                for unperturbed_file_path, file_path in tqdm(
                    pool.imap_unordered(process_func, feat_slices),
                    desc=f"Slice Graph paths",
                    total=len(feat_slices),) if unperturbed_file_path is not None]

            message_queue.put("finished")
            pool.close()
            pool.join()
        logging.info(f"Finished processing {feat_name}")
    
    for key, value in filename_map.items():
        instance_perturbation_map[key] = list(value)
    
    instance_perturbation_map_filepath = join(dataset_root, config.instance_perturbation_map_filename)
    logging.info(f"Writing instance perturbation map to {instance_perturbation_map_filepath}...")
    with open(join(dataset_root, config.instance_perturbation_map_filename), "w") as wfi:
        json.dump(instance_perturbation_map, wfi, indent=2)
    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()