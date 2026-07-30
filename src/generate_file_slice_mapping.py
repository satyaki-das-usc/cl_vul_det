from collections import defaultdict
from functools import lru_cache
import json
import logging

from os.path import join, splitext, basename
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import cast

from tqdm import tqdm

from src.common_utils import get_arg_parser, init_log

SOURCE_EXTENSIONS = (".c", ".cpp", ".h")


@lru_cache(maxsize=None)
def resolve_source_path(source_base: str) -> str:
    matches = [
        f"{source_base}{extension}"
        for extension in SOURCE_EXTENSIONS
        if Path(f"{source_base}{extension}").is_file()
    ]

    if not matches:
        raise FileNotFoundError(f"Could not find source file for {source_base}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous source files for {source_base}: {matches}")

    return matches[0]


def source_base_from_slice(
    slice_path: str,
    slice_folder: str,
    source_root_folder: str,
) -> str:
    path = Path(slice_path)
    parts = list(path.parts)

    try:
        slice_index = parts.index(slice_folder)
    except ValueError as error:
        raise ValueError(
            f"Path does not contain the '{slice_folder}' directory: {slice_path}"
        ) from error

    source_stem, separator, _ = path.name.partition("___")
    if not separator:
        raise ValueError(f"Unexpected slice filename: {path.name}")

    return str(
        Path(
            *parts[:slice_index],
            source_root_folder,
            *parts[slice_index + 1:-1],
            source_stem,
        )
    )

if __name__ == "__main__":
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()
    init_log(splitext(basename(__file__))[0])
    
    config = cast(DictConfig, OmegaConf.load(args.config))

    dataset_root = join(config.data_folder, config.dataset.name)
    if args.use_temp_data:
        dataset_root = config.temp_root
    
    all_slices_filepath = join(dataset_root, config.all_slices_filename)
    logging.info(f"Loading all generated slices from {all_slices_filepath}...")
    with open(all_slices_filepath, "r") as rfi:
        all_slices = json.load(rfi)
    logging.info(f"Completed. Loaded {len(all_slices)} slices.")

    logging.info(f"Going over {len(all_slices)} files...")
    file_slices_map = defaultdict(list)
    for slice_path in tqdm(all_slices, desc="Slices"):
        source_base = source_base_from_slice(
            slice_path,
            config.slice_folder,
            config.source_root_folder,
        )
        try:
            src_cpp_path = resolve_source_path(source_base)
        except (FileNotFoundError, RuntimeError):
            logging.error("Failed to resolve source for slice %s", slice_path)
            raise
        file_slices_map[src_cpp_path].append(slice_path)
    logging.info(f"Completed. Found {len(file_slices_map)} unique source files.")
    
    file_slices_filepath = join(dataset_root, config.file_slices_filename)
    logging.info(f"Saving file slices map to {file_slices_filepath}...")
    with open(file_slices_filepath, "w") as wfi:
        json.dump(file_slices_map, wfi, indent=2)
    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()