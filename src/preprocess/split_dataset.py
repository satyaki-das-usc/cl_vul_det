import os
import json

import logging

from multiprocessing import cpu_count
from os.path import join, isdir
from omegaconf import DictConfig, OmegaConf
from typing import cast

from sklearn.model_selection import train_test_split

from src.common_utils import get_arg_parser

def init_log():
    LOG_DIR = "logs"
    if not isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logging.basicConfig(
        handlers=[
            logging.FileHandler(join(LOG_DIR, "split_dataset.log")),
            logging.StreamHandler()
        ],
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("=========New session=========")
    logging.info(f"Logging dir: {LOG_DIR}")

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

    train_slices, test_slices = train_test_split(all_slices, test_size=0.2)
    test_slices, val_slices = train_test_split(test_slices, test_size=0.5)

    train_slices_filepath = join(dataset_root, config.train_slices_filename)
    logging.info(f"Saving {len(train_slices)} slices to {train_slices_filepath}...")
    with open(train_slices_filepath, "w") as wfi:
        json.dump(train_slices, wfi)
    logging.info("Completed.")

    val_slices_filepath = join(dataset_root, config.val_slices_filename)
    logging.info(f"Saving {len(val_slices)} slices to {val_slices_filepath}...")
    with open(val_slices_filepath, "w") as wfi:
        json.dump(val_slices, wfi)
    logging.info("Completed.")

    test_slices_filepath = join(dataset_root, config.test_slices_filename)
    logging.info(f"Saving {len(test_slices)} slices to {test_slices_filepath}...")
    with open(test_slices_filepath, "w") as wfi:
        json.dump(test_slices, wfi)
    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()