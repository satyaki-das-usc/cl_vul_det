import json
import pickle

import networkx as nx
import logging

from gensim.models import Word2Vec
from multiprocessing import cpu_count
from os.path import join, splitext, basename
from omegaconf import DictConfig, OmegaConf
from typing import cast

from tqdm import tqdm

from src.common_utils import get_arg_parser, init_log

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

    all_tokens_list = set()
    logging.info(f"Going over {len(all_slices)} files...")
    for slice_path in tqdm(all_slices):
        with open(slice_path, "rb") as rbfi:
            slice_graph: nx.DiGraph = pickle.load(rbfi)
        all_tokens_list.update(slice_graph.graph['slice_sym_token'])
    logging.info(f"Completed. Total tokens collected: {len(all_tokens_list)}")

    model = Word2Vec(sentences=[list(all_tokens_list)], min_count=3, vector_size=config.gnn.embed_size,
                    max_vocab_size=config.dataset.token.vocabulary_size, workers=USE_CPU, sg=1, epochs=10)
    logging.info(f"Word2Vec model created with {len(model.wv.index_to_key)} unique tokens...")
    w2v_save_path = join(dataset_root, "w2v.wv")
    logging.info(f"Saving Word2Vec model to {w2v_save_path}...")
    model.wv.save(w2v_save_path)
    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()