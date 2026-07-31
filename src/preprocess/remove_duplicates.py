import json
import pickle

import logging

from multiprocessing import Pool, cpu_count
from os.path import join, splitext, basename
from omegaconf import DictConfig, OmegaConf
from typing import cast, Dict

from tqdm import tqdm

from src.common_utils import get_arg_parser, init_log

file_slices = dict()
slice_metadata = dict()


def process_file_parallel(cpp_path):
    try:
        all_slices = file_slices[cpp_path]
        unique_vul_slice_set = set()
        unique_vul_slice_list = []
        unique_nonvul_slice_by_code: Dict[bytes, str] = {}
        duplicate_slice_list = []

        for slice_path in all_slices:
            label, code_digest = slice_metadata[slice_path]
            if label:
                if code_digest not in unique_vul_slice_set:
                    unique_vul_slice_set.add(code_digest)
                    unique_vul_slice_list.append(slice_path)
                else:
                    duplicate_slice_list.append(slice_path)
            else:
                if code_digest not in unique_nonvul_slice_by_code:
                    unique_nonvul_slice_by_code[code_digest] = slice_path
                else:
                    duplicate_slice_list.append(slice_path)

        final_unique_nonvul_slice_list = []
        for code_digest, slice_path in unique_nonvul_slice_by_code.items():
            if code_digest not in unique_vul_slice_set:
                final_unique_nonvul_slice_list.append(slice_path)
            else:
                duplicate_slice_list.append(slice_path)

        return (
            unique_vul_slice_list + final_unique_nonvul_slice_list,
            duplicate_slice_list,
        )

    except Exception as e:
        logging.error(cpp_path)
        raise e

def build_cpp_batches(cpp_paths, num_workers):
    total_work = sum(max(1, len(file_slices[cpp_path])) for cpp_path in cpp_paths)
    target_batch_work = max(1, total_work // (num_workers * 8))

    batches = []
    current_batch = []
    current_batch_work = 0
    for cpp_path in cpp_paths:
        cpp_work = max(1, len(file_slices[cpp_path]))
        if current_batch and current_batch_work + cpp_work > target_batch_work:
            batches.append(current_batch)
            current_batch = []
            current_batch_work = 0

        current_batch.append(cpp_path)
        current_batch_work += cpp_work

    if current_batch:
        batches.append(current_batch)

    return batches

def process_file_batch(cpp_paths):
    unique_slices = []
    duplicate_slices = []
    for cpp_path in cpp_paths:
        file_unique_slices, file_duplicate_slices = process_file_parallel(cpp_path)
        unique_slices.extend(file_unique_slices)
        duplicate_slices.extend(file_duplicate_slices)
    return unique_slices, duplicate_slices

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

    file_slices_path = join(dataset_root, config.file_slices_filename)
    logging.info(f"Loading filewise generated slices from {file_slices_path}...")
    with open(file_slices_path, "r") as rfi:
        file_slices = json.load(rfi)
    logging.info(f"Completed. Loaded slices for {len(file_slices)} files.")

    slice_metadata_path = join(dataset_root, config.slice_metadata_filename)
    logging.info(f"Loading slice metadata from {slice_metadata_path}...")
    with open(slice_metadata_path, "rb") as rfi:
        slice_metadata = pickle.load(rfi)
    logging.info(f"Completed. Loaded metadata for {len(slice_metadata)} slices.")

    logging.info(f"Going over {len(file_slices)} files...")
    cpp_paths = list(file_slices.keys())
    unique_slice_list = []
    duplicate_slices = set()
    if USE_CPU > 1:
        cpp_batches = build_cpp_batches(cpp_paths, USE_CPU)
        with Pool(USE_CPU) as pool:
            for file_unique_slices, file_duplicate_slices in tqdm(
                pool.imap_unordered(process_file_batch, cpp_batches),
                desc="Cpp file batches",
                total=len(cpp_batches),
            ):
                unique_slice_list.extend(file_unique_slices)
                duplicate_slices.update(file_duplicate_slices)
    else:
        for cpp_path in tqdm(cpp_paths, desc="Cpp files", total=len(cpp_paths)):
            file_unique_slices, file_duplicate_slices = process_file_parallel(cpp_path)
            unique_slice_list.extend(file_unique_slices)
            duplicate_slices.update(file_duplicate_slices)

    with open("duplicate_slices.txt", "w") as wfi:
        wfi.writelines(f"{slice_path}\n" for slice_path in duplicate_slices)

    unique_slice_list = list(set(unique_slice_list) - duplicate_slices)
    
    logging.info(f"Total unique slices: {len(unique_slice_list)}")
    all_slices_filepath = join(dataset_root, config.all_slices_filename)
    logging.info(f"Saving unique slices to {all_slices_filepath}...")
    with open(all_slices_filepath, "w") as wfi:
        json.dump(unique_slice_list, wfi, indent=2)
    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()