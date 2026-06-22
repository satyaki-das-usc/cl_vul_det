import logging

from gensim.models import Word2Vec
from multiprocessing import cpu_count
from os.path import exists, join, splitext, basename
from omegaconf import DictConfig, OmegaConf
from typing import cast

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
    
    corpus_filepath = join(dataset_root, "w2v_corpus.txt")
    if not exists(corpus_filepath):
        raise FileNotFoundError(
            f"Word2Vec corpus file not found: {corpus_filepath}. "
            "Generate it first with src/preprocess/generate_word2vec_corpus.py."
        )

    logging.info(f"Training Word2Vec model from corpus file {corpus_filepath}...")
    model = Word2Vec(corpus_file=corpus_filepath, min_count=3, vector_size=config.gnn.embed_size,
                    max_vocab_size=config.dataset.token.vocabulary_size, workers=USE_CPU, sg=1,
                    epochs=10, batch_words=50000)
    logging.info(f"Word2Vec model created with {len(model.wv.index_to_key)} unique tokens...")
    w2v_save_path = join(dataset_root, "w2v.wv")
    logging.info(f"Saving Word2Vec model to {w2v_save_path}...")
    model.wv.save(w2v_save_path)
    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()
