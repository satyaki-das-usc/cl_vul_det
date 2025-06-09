import os
import json
import pickle

import networkx as nx
import logging

from gensim.models import Word2Vec
from multiprocessing import cpu_count
from os.path import join, isdir
from omegaconf import DictConfig, OmegaConf
from typing import cast

from tqdm import tqdm

from src.common_utils import get_arg_parser
from src.vocabulary import Vocabulary

def init_log():
    LOG_DIR = "logs"
    if not isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logging.basicConfig(
        handlers=[
            logging.FileHandler(join(LOG_DIR, "run.log")),
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
    
    vocab = Vocabulary.from_w2v(join(dataset_root, "w2v.wv"))
    vocab_size = vocab.get_vocab_size()
    pad_idx = vocab.get_pad_id()