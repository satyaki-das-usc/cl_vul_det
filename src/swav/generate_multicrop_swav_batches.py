import random
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

import logging

from multiprocessing import cpu_count
from omegaconf import DictConfig, OmegaConf
from typing import List, cast
from tqdm import tqdm
from math import ceil
from os.path import join, isdir, exists
from collections import defaultdict

from pytorch_lightning import seed_everything

import torch.nn.functional as F

from src.common_utils import get_arg_parser
from src.vocabulary import Vocabulary
from src.torch_data.graphs import SliceGraph
from src.torch_data.samples import SliceGraphSample, SliceGraphBatch

from src.models.modules.gnns import GraphSwAVModel
from src.models.modules.losses import InfoNCEContrastiveLoss

from src.swav.assignment_protocols import sinkhorn, uot_sinkhorn_gpu
from src.swav.graph_augmentations import augment, augment_multicrop

def init_log():
    LOG_DIR = "logs"
    if not isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logging.basicConfig(
        handlers=[
            logging.FileHandler(join(LOG_DIR, "generate_multicrop_swav_batches.log")),
            logging.StreamHandler()
        ],
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("=========New session=========")
    logging.info(f"Logging dir: {LOG_DIR}")

def get_merged_clusters(cluster_ids: List[int], prototypes, min_size: int = 2) -> List[int]:
    """
    Merge small clusters into larger ones.
    """
    N = prototypes.size(1)
    with torch.no_grad():
        proto_sim = prototypes.T @ prototypes

    cluster_ids_tensor = torch.tensor(cluster_ids, dtype=torch.int64)
    all_ids = torch.arange(N, device=cluster_ids_tensor.device)

    unique, counts = torch.unique(cluster_ids_tensor, return_counts=True)
    small_clusters = unique[counts < min_size].tolist()

    while len(small_clusters) > 0:
        for cid in small_clusters:
            c_proto_sym = proto_sim[cid]
            c_proto_sym[cid] = float('-inf')  # Exclude self-similarity
            non_empty_proto_mask = torch.isin(all_ids, unique)
            c_proto_sym[~non_empty_proto_mask] = float('-inf')  # Exclude empty clusters
            best_match = c_proto_sym.argmax().item()
            cluster_ids_tensor[cluster_ids_tensor == cid] = best_match
            unique, counts = torch.unique(cluster_ids_tensor, return_counts=True)
            small_clusters = unique[counts < min_size].tolist()
            if len(small_clusters) == 0:
                break
    
    return cluster_ids_tensor.tolist()

assignment_functions = {
    "sinkhorn": sinkhorn,
    "uot_sinkhorn": uot_sinkhorn_gpu
}

if __name__ == "__main__":
    arg_parser = get_arg_parser()
    arg_parser.add_argument("--do_train", action="store_true", help="Enable training; if not set, use pretrained model.")
    arg_parser.add_argument("--nmb_views", type=int, default=[2, 6], nargs="+",
                    help="list of number of views (example: [2, 6])")
    arg_parser.add_argument("--views_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of global view indices used for computing assignments")
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
    
    seed_everything(config.seed, workers=True)

    max_len = config.dataset.token.max_parts
    vocab = Vocabulary.from_w2v(config.gnn.w2v_path)
    vocab_size = vocab.get_vocab_size()
    pad_idx = vocab.get_pad_id()
    
    train_slices_filepath = join(dataset_root, config.train_slices_filename)
    logging.info(f"Loading training slice paths list from {train_slices_filepath}...")
    with open(train_slices_filepath, "r") as rfi:
        train_slices = json.load(rfi)
    logging.info(f"Completed. Loaded {len(train_slices)} slices.")

    logging.info(f"Going over {len(train_slices)} files...")

    sample_list = []
    for slice_path in tqdm(train_slices, desc=f"Slice files"):
        slice_graph = SliceGraph(slice_path)
        sample_list.append(SliceGraphSample(graph=slice_graph.to_torch_graph(vocab, max_len),
                            label=slice_graph.label))
    logging.info(f"Completed. Loaded {len(sample_list)} samples.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSwAVModel(config, vocab, vocab_size, pad_idx).to(device)
    optimizer = torch.optim.AdamW([{
                "params": p
            } for p in model.parameters()], config.swav.learning_rate)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #             optimizer,
    #             lr_lambda=lambda epoch: config.hyper_parameters.decay_gamma
    #                                 ** epoch)

    K = max(1024, ceil(len(train_slices) / config.hyper_parameters.batch_size))
    D = config.gnn.projection_dim
    prototypes = torch.nn.Parameter(torch.randn(D, K, device=device))
    logging.info(f"Initialized prototypes with shape {prototypes.shape}.")
    with torch.no_grad():
        prototypes.data = F.normalize(prototypes.data, dim=0)

    BATCH_SIZE = 256
    
    contrastive_criterion = InfoNCEContrastiveLoss(temperature=config.swav.contrastive.temperature)

    swav_losses = []
    contrast_losses = []

    if args.do_train:
        for epoch in range(config.swav.n_epochs):
            logging.info(f"Epoch {epoch + 1}")
            freeze_proto = (epoch == 0)
            if freeze_proto:
                optimizer = torch.optim.AdamW([{
                    "params": p
                } for p in model.parameters()], config.hyper_parameters.learning_rate)
            else:
                optimizer = torch.optim.AdamW([{
                    "params": p
                } for p in list(model.parameters()) + [prototypes]], config.hyper_parameters.learning_rate)
            model.train()
            random.shuffle(sample_list)
            progress_bar = tqdm(range(0, len(sample_list), BATCH_SIZE))

            epoch_swav_losses = []
            epoch_contrast_losses = []

            for i in progress_bar:
                with torch.no_grad():
                    prototypes.data = F.normalize(prototypes.data, dim=0)
                
                batched_graph = SliceGraphBatch(sample_list[i:i + BATCH_SIZE])
                batched_graph = batched_graph.graphs.to(device)
                inputs = augment_multicrop(batched_graph, mask_id=vocab.get_unk_id(), nmb_views=args.nmb_views)
                logits, graph_activations = zip(*(model(inp) for inp in inputs))
                output = [lg @ prototypes for lg in logits]

                swav_loss = 0
                for view_id in args.views_for_assign:
                    with torch.no_grad():
                        out = output[view_id].detach()
                        q = assignment_functions[config.swav.assignment_protocol](out)
                    subloss = 0
                    for v in np.delete(np.arange(np.sum(args.nmb_views)), view_id):
                        x = output[v] / config.swav.temperature
                        subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=-1), dim=-1))
                    swav_loss += subloss / (np.sum(args.nmb_views) - 1)
                swav_loss /= len(args.views_for_assign)

                h1, h2 = F.normalize(graph_activations[0], dim=-1), F.normalize(graph_activations[1], dim=-1)
                contrastive_loss = contrastive_criterion(h1, h2)
                
                loss = swav_loss + config.swav.contrastive.lambda_h * contrastive_loss

                epoch_swav_losses.append(swav_loss.item())
                epoch_contrast_losses.append(contrastive_loss.item())

                swav_losses.append(swav_loss.item())
                contrast_losses.append(contrastive_loss.item())
                progress_bar.set_postfix({
                    'swav': swav_loss.item(),
                    'contrast': contrastive_loss.item()
                })

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # scheduler.step()
            logging.info(f"Epoch {epoch + 1} - SwAV Loss: {np.mean(epoch_swav_losses):.4f}, Contrastive Loss: {np.mean(epoch_contrast_losses):.4f}")

        plt.plot(swav_losses, label='SwAV Loss')
        plt.plot(contrast_losses, label='Contrastive Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.title('Training Loss Convergence')
        plt.savefig(join(dataset_root, 'training_losses.png'), bbox_inches='tight')
        plt.close()
    else:
        logging.info("Skipping training. Using pretrained model and prototypes.")
        model_save_path = join(dataset_root, config.swav.model_save_path)
        if not exists(model_save_path):
            raise FileNotFoundError(f"Model save path {model_save_path} does not exist.")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        logging.info(f"Model loaded from {model_save_path}")

        prototypes_save_path = join(dataset_root, config.swav.prototypes_save_path)
        if not exists(prototypes_save_path):
            raise FileNotFoundError(f"Prototypes save path {prototypes_save_path} does not exist.")
        prototypes.data = torch.load(prototypes_save_path, map_location=device)
        logging.info(f"Prototypes loaded from {prototypes_save_path}")

    logging.info("Generating cluster IDs...")
    progress_bar = tqdm(range(0, len(sample_list), BATCH_SIZE))
    model.eval()
    all_cluster_ids = []
    for i in progress_bar:
        batched_graph = SliceGraphBatch(sample_list[i:i + BATCH_SIZE])
        batched_graph = batched_graph.graphs.to(device)
        global_view = augment(batched_graph, mask_id=vocab.get_unk_id())
        with torch.no_grad():
            logits, _ = model(global_view)
            output = logits @ prototypes
            q = assignment_functions[config.swav.assignment_protocol](out)
            cluster_ids = q.argmax(dim=1)
            all_cluster_ids.extend(cluster_ids.cpu().numpy().tolist())
    logging.info("Obtained cluster IDs. Merging small clusters...")
    all_cluster_ids = get_merged_clusters(all_cluster_ids, prototypes, min_size=config.hyper_parameters.batch_size)
    
    logging.info("Finalized cluster IDs. Generating groups...")
    cluster_id_maps = defaultdict(list)
    for idx, cluster_id in enumerate(tqdm(all_cluster_ids)):
        cluster_id_maps[f"{cluster_id}"].append(idx)
    
    swav_multicrop_batches = [indices for indices in cluster_id_maps.values()]
    swav_batches_filepath = join(dataset_root, config.swav_batches_filename)
    logging.info(f"Generated {len(swav_multicrop_batches)} groups. Saving to {swav_batches_filepath}...")
    with open(swav_batches_filepath, "w") as wfo:
        json.dump(swav_multicrop_batches, wfo)

    logging.info(f"Saving model...")
    model_save_path = join(dataset_root, config.swav.model_save_path)
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")
    logging.info(f"Saving prototypes...")
    prototypes_save_path = join(dataset_root, config.swav.prototypes_save_path)
    torch.save(prototypes.data, prototypes_save_path)
    logging.info(f"Prototypes saved to {prototypes_save_path}")

    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()