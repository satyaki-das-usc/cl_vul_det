import os
import json
import torch

import logging

from multiprocessing import cpu_count
from os.path import join, isdir, basename, exists
from omegaconf import DictConfig, OmegaConf
from typing import cast

from commode_utils.callbacks import ModelCheckpointWithUploadCallback
from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from src.common_utils import get_arg_parser, filter_warnings
from src.vocabulary import Vocabulary
from src.torch_data.custom_samplers import InstanceSampler, VFSampler, SwAVSampler
from src.torch_data.datamodules import SliceDataModule
from src.models.vul_det import CLVulDet, NoCLVulDet

sampler = ""

sample_name_map = {
    "instance": "Instance",
    "vf": "VF",
    "swav": "SwAV"
}

gnn_name_map = {
    "gcn": "GCN",
    "gin": "GIN",
    "gine": "GINE",
    "ggnn": "GGNN",
    "gatv2": "GATv2",
    "gated": "Gated",
    "st": "ST"
}

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

def train(model: LightningModule, data_module: LightningDataModule,
          config: DictConfig, no_cl: bool = False):
    # Define logger
    model_name = model.__class__.__name__
    gnn_name = gnn_name_map[config.gnn.name]
    sampler_name = sample_name_map[sampler]
    nn_text = "ExcludeNN" if config.exclude_NNs else "IncludeNN"
    lr_warmup_text = "NoLRWarmup" if config.hyper_parameters.use_warmup_lr else "LRWarmup"
    cl_warmup_text = "NoCLWarmup" if config.hyper_parameters.contrastive_warmup_epochs > 0 else "CLWarmup"
    dataset_name = basename(config.dataset.name)
    # tensorboard logger
    tensorlogger = TensorBoardLogger(join("ts_logger", model_name, gnn_name, sampler_name, nn_text, lr_warmup_text, cl_warmup_text),
                                     dataset_name)
    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=join(tensorlogger.log_dir, "checkpoints"),
        monitor="val_loss",
        filename="{epoch:02d}-{step:02d}-{val_loss:.4f}",
        every_n_epochs=1,
        save_top_k=5,
    )
    upload_weights = ModelCheckpointWithUploadCallback(
        join(tensorlogger.log_dir, "checkpoints"))

    early_stopping_callback = EarlyStopping(patience=config.hyper_parameters.patience,
                                            monitor="val_loss",
                                            verbose=True,
                                            mode="min")

    lr_logger = LearningRateMonitor("step")
    # print_epoch_results = PrintEpochResultCallback(split_symbol="_",
    #                                                after_test=False)

    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(
        max_epochs=config.hyper_parameters.n_epochs,
        gradient_clip_val=config.hyper_parameters.clip_norm,
        deterministic=True,
        val_check_interval=config.hyper_parameters.val_every_step,
        log_every_n_steps=config.hyper_parameters.log_every_n_steps,
        logger=[tensorlogger],
        devices=gpu,
        accelerator="gpu" if gpu else "cpu",
        callbacks=[
            lr_logger, early_stopping_callback, checkpoint_callback, upload_weights, TQDMProgressBar(refresh_rate=config.hyper_parameters.progress_bar_refresh_rate)
        ],
    )
    
    checkpoint_dir = join(config.model_save_dir, gnn_name, sampler_name, nn_text, lr_warmup_text, cl_warmup_text)
    if not exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = join(checkpoint_dir, f"{model_name}.ckpt")
    model.set_checkpoint_path(checkpoint_path)
    if not exists(checkpoint_path):
        logging.info(f"Checkpoint at {checkpoint_path} not found. Starting training from scratch.")
        trainer.fit(model=model, datamodule=data_module)
    else:
        logging.info(f"Checkpoint found at {checkpoint_path}. Resuming training.")
        trainer.fit(model=model, datamodule=data_module, ckpt_path=checkpoint_path)
    # trainer.save_checkpoint(checkpoint_path)
    trainer.test(model=model, datamodule=data_module)

if __name__ == "__main__":
    filter_warnings()
    arg_parser = get_arg_parser()
    arg_parser.add_argument("-s", "--sampler", type=str, required=True)
    arg_parser.add_argument("--no_cl", action='store_true', help="Use contrastive learning")
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
    
    vocab = Vocabulary.from_w2v(join(dataset_root, "w2v.wv"))
    vocab_size = vocab.get_vocab_size()
    pad_idx = vocab.get_pad_id()

    train_slices_filepath = join(dataset_root, config.train_slices_filename)
    logging.info(f"Loading training slice paths list from {train_slices_filepath}...")
    with open(train_slices_filepath, "r") as rfi:
        train_slices = json.load(rfi)
    logging.info(f"Completed. Loaded {len(train_slices)} slices.")

    if sampler == "vf":
        logging.info("Creating VF sampler...")
        train_sampler = VFSampler(train_slices, config.hyper_parameters.batch_size)
        logging.info("VF sampler created.")
    elif sampler == "instance":
        all_feats_name = ["incorr_calc_buff_size", "buff_access_src_size", "off_by_one", "buff_overread", "double_free", "use_after_free", "buff_underwrite", "buff_underread", "sensi_read", "sensi_write"]
        unperturbed_file_list = []
        logging.info("Loading unperturbed file list...")
        for feat_name in all_feats_name:
            unperturbed_file_list_path = join(dataset_root, config.VF_perts_root, feat_name,  config.unperturbed_files_filename)
            with open(unperturbed_file_list_path, "r") as rfi:
                unperturbed_file_list += json.load(rfi)
        unperturbed_file_list = list(set(unperturbed_file_list))
        logging.info(f"Number of unperturbed files: {len(unperturbed_file_list)}")
        logging.info("Creating Instance sampler...")
        train_sampler = InstanceSampler(train_slices, config.hyper_parameters.batch_size, config, unperturbed_file_list)
        logging.info("Instance sampler created.")
    elif sampler == "swav":
        logging.info("Creating SwAV sampler...")
        swav_batches_filepath = join(dataset_root, config.swav_batches_filename)
        train_sampler = SwAVSampler(swav_batches_filepath)
        logging.info("SwAV sampler created.")
    else:
        train_sampler = None

    logging.info("Loading data module...")
    data_module = SliceDataModule(config, vocab, train_sampler=train_sampler, use_temp_data=args.use_temp_data)
    logging.info("Data module loading completed.")

    if args.no_cl:
        model = NoCLVulDet(config, vocab, vocab_size, pad_idx)
    else:
        model = CLVulDet(config, vocab, vocab_size, pad_idx)

    train(model, data_module, config, args.no_cl)