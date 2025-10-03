import functools
import os
import json
import pickle

import networkx as nx
import logging

import argparse
from multiprocessing import Manager, Pool, Queue, cpu_count
from os.path import join, exists, isdir, dirname, basename, splitext, relpath
from omegaconf import DictConfig, OmegaConf
from typing import List, Set, Tuple, Dict, cast

from tqdm import tqdm

from src.common_utils import read_csv, extract_nodes_with_location_info, extract_line_number, get_arg_parser, init_log
from src.cpg_query import get_line_nodes

csv_path = ""
slices_root = ""
ground_truth = dict()
unperturbed_files = list()
sensi_api_path = ""
USE_CPU = cpu_count()

feat_name_code_map = dict()

def get_forward_slice_graph(CPG: nx.DiGraph, line_no: int):
    slice_lines = set()

    forward_queue = []
    visited = set()
    forward_queue.append(line_no)
    visited.add(line_no)

    while len(forward_queue) > 0:
        current_line = forward_queue.pop(0)
        slice_lines.add(current_line)
        if current_line not in CPG._succ:
            continue
        for succ in CPG._succ[current_line]:
            if succ in visited:
                continue
            visited.add(succ)
            forward_queue.append(succ)
    if len(slice_lines) == 0:
        return None
    
    slice_graph = CPG.subgraph(list(slice_lines)).copy()
    for u, v, edge_data in slice_graph.edges(data=True):
        edge_data["direction"] = "forward"

    return slice_graph, slice_lines

def get_backward_slice_graph(CPG: nx.DiGraph, line_no: int):
    slice_lines = set()

    backward_queue = []
    visited = set()
    backward_queue.append(line_no)
    visited.add(line_no)

    while len(backward_queue) > 0:
        current_line = backward_queue.pop(0)
        slice_lines.add(current_line)
        if current_line not in CPG._pred:
            continue
        for pred in CPG._pred[current_line]:
            if pred in visited:
                continue
            visited.add(pred)
            backward_queue.append(pred)
    if len(slice_lines) == 0:
        return None
    
    slice_graph = CPG.subgraph(list(slice_lines)).copy()
    for u, v, edge_data in slice_graph.edges(data=True):
        edge_data["direction"] = "backward"

    return slice_graph, slice_lines

def get_slices_with_direction(CPG: nx.DiGraph, line_no: int, vul_lines: Set[int], direction: str) -> List[nx.DiGraph]:
    if direction == "forward":
        slice_graph, slice_lines = get_forward_slice_graph(CPG, line_no)
    elif direction == "backward":
        slice_graph, slice_lines = get_backward_slice_graph(CPG, line_no)
    elif direction == "both":
        forward_slice_graph, forward_slice_lines = get_forward_slice_graph(CPG, line_no)
        backward_slice_graph, backward_slice_lines = get_backward_slice_graph(CPG, line_no)
        slice_lines = forward_slice_lines | backward_slice_lines
        slice_graph = CPG.subgraph(list(slice_lines)).copy()
        for u, v, edge_data in slice_graph.edges(data=True):
            edge_data["direction"] = "forward"
    else:
        raise ValueError(f"Invalid slice direction: {direction}")

    label = len(slice_lines.intersection(vul_lines)) > 0
    slice_graph.graph["label"] = label
    slice_graph.graph["key_line"] = line_no
    slice_graph.graph["type"] = direction

    return [slice_graph]

def get_slices(CPG: nx.DiGraph, key_line_map: Dict[str, Set[int]], vul_lines: Set[int]) -> Dict[str, List[nx.DiGraph]]:
    if CPG is None:
        return None
    if key_line_map is None:
        return None
    
    slices = {
        "checking_call": [],
        "alloc_call": [],
        "dealloc_call": [],
        "read_call": [],
        "write_call": [],
        "other_call": [],
        "array": [],
        "ptr": [],
        "arith": []
    }

    slice_directions = {
        "checking_call": "forward",
        "alloc_call": "forward",
        "dealloc_call": "backward",
        "read_call": "forward",
        "write_call": "backward",
        "other_call": "both",
        "array": "backward",
        "ptr": "both",
        "arith": "both"
    }

    for key, lines in key_line_map.items():
        if key not in slices.keys():
            continue
        if key not in slice_directions.keys():
            continue
        slice_direction = slice_directions[key]
        for line_no in lines:
            slices[key].extend(get_slices_with_direction(CPG, line_no, vul_lines, slice_direction))

    return slices

def has_data_dependency(CPG: nx.DiGraph, line_num: int) -> bool:
    return sum([1 for start, end, edge_data in CPG.in_edges(line_num, data=True) if edge_data["label"] == "REACHES"] + [1 for start, end, edge_data in CPG.out_edges(line_num, data=True) if edge_data["label"] == "REACHES"]) > 0

def remove_dependency_free_node(CPG: nx.DiGraph) -> nx.DiGraph:
    filtered_CPG = CPG.copy()

    edges_to_remove = set()
    nodes_to_remove = set()

    for start, end, edge_data in CPG.edges(data=True):
        if "label" not in edge_data:
            continue
        if edge_data["label"] != "CONTROLS":
            continue
        if has_data_dependency(CPG, end):
            continue
        edges_to_remove.add((start, end))
        nodes_to_remove.add(end)
    
    filtered_CPG.remove_edges_from(edges_to_remove)
    filtered_CPG.remove_nodes_from(nodes_to_remove)

    return filtered_CPG

def is_static_condition(joern_nodes: List[Dict], line_num: int) -> bool:
    line_nodes = get_line_nodes(joern_nodes, line_num)
    for node in line_nodes:
        if node["type"] != "Condition":
            continue
        code = node["code"].strip()
        code = code.replace("&&", "and").replace("||", "or")
        try:
            value = eval(code)
            if value is not None:
                return True
        except:
            continue
    
    return False

def remove_static_control_dependency(CPG: nx.DiGraph, joern_nodes: List[Dict]) -> nx.DiGraph:
    filtered_CPG = CPG.copy()
    
    edges_to_remove = set()
    nodes_to_remove = set()
    
    for start, end, edge_data in CPG.edges(data=True):
        if edge_data["label"] != "CONTROLS":
            continue
        if not is_static_condition(joern_nodes, start):
            continue
        edges_to_remove.add((start, end))
        nodes_to_remove.add(start)
    
    filtered_CPG.remove_edges_from(edges_to_remove)
    filtered_CPG.remove_nodes_from(nodes_to_remove)

    return filtered_CPG


def build_CPG(code_path: str,
              source_path: str) -> Tuple[nx.DiGraph, Dict[str, Set[int]]]:
    
    assert exists(sensi_api_path), f"{sensi_api_path} not exists!"
    with open(sensi_api_path, "r") as rfi:
        sensi_api_map = json.load(rfi)
    
    nodes_path = join(code_path, "nodes.csv")
    edges_path = join(code_path, "edges.csv")
    if not exists(nodes_path) or not exists(edges_path):
        print(f"Missing nodes or edges for {source_path}")
        return None, None
    nodes = read_csv(nodes_path)
    edges = read_csv(edges_path)
    checking_call_lines = set()
    alloc_call_lines = set()
    dealloc_call_lines = set()
    read_call_lines = set()
    write_call_lines = set()
    other_call_lines = set()
    array_lines = set()
    ptr_lines = set()
    arithmatic_lines = set()
    if len(nodes) == 0:
        print(f"No nodes generated for {source_path}")
        return None, None
    
    for node_idx, node in enumerate(nodes):
        node_type = node['type'].strip()
        if node_type == "CallExpression":
            function_name = nodes[node_idx + 1]['code']
            if function_name is None:
                continue
            function_name = function_name.strip()
            if function_name == "":
                continue
            line_no = extract_line_number(node_idx, nodes)
            if line_no < 1:
                continue
            if function_name in sensi_api_map["checking"]:
                checking_call_lines.add(line_no)
            if function_name in sensi_api_map["alloc"]:
                alloc_call_lines.add(line_no)
            if function_name in sensi_api_map["dealloc"]:
                dealloc_call_lines.add(line_no)
            if function_name in sensi_api_map["read"]:
                read_call_lines.add(line_no)
            if function_name in sensi_api_map["write"]:
                write_call_lines.add(line_no)
            if function_name in sensi_api_map["other"]:
                other_call_lines.add(line_no)
        if node_type == "ArrayIndexing":
            line_no = extract_line_number(node_idx, nodes)
            if line_no < 1:
                continue
            array_lines.add(line_no)
        if node_type == "PtrMemberAccess":
            line_no = extract_line_number(node_idx, nodes)
            if line_no < 1:
                continue
            ptr_lines.add(line_no)
        elif node['operator'].strip() in ['+', '-', '*', '/']:
            line_no = extract_line_number(node_idx, nodes)
            if line_no < 1:
                continue
            arithmatic_lines.add(line_no)

    CPG = nx.DiGraph(file_paths=[source_path])
    control_edges, data_edges, post_dom_edges, def_use_edges = list(), list(), list(), list()
    node_id_to_ln = extract_nodes_with_location_info(nodes)
    for edge in edges:
        edge_type = edge['type'].strip()
        start_node_id = edge['start'].strip()
        end_node_id = edge['end'].strip()
        if edge_type in ["DEF", "USE"]:
            if start_node_id not in node_id_to_ln.keys():
                continue
            start_ln = node_id_to_ln[start_node_id]
            end_ln = (-1) * int(end_node_id)
            if edge_type == "USE":
                end_ln *= 2
            symbol_used = [node for node in nodes if node["key"].strip() == end_node_id][0]["code"].strip()
            def_use_edges.append((start_ln, end_ln, {"label": edge_type, "symbol": symbol_used}))
            continue
        if start_node_id not in node_id_to_ln.keys(
        ) or end_node_id not in node_id_to_ln.keys():
            continue
        start_ln = node_id_to_ln[start_node_id]
        end_ln = node_id_to_ln[end_node_id]
        if edge_type == 'CONTROLS':  # Control
            control_edges.append((start_ln, end_ln, {"label": edge_type}))
        if edge_type == 'REACHES':  # Data
            data_edges.append((start_ln, end_ln, {"label": edge_type, "var": edge["var"].strip()}))
        if edge_type == 'POST_DOM': # Post dominance
            post_dom_edges.append((start_ln, end_ln, {"label": edge_type}))
    
    CPG.add_edges_from(control_edges)
    CPG.add_edges_from(data_edges)
    # CPG.add_edges_from(post_dom_edges)
    # CPG.add_edges_from(def_use_edges)

    # CPG = remove_static_control_dependency(remove_dependency_free_node(CPG), nodes)

    # CPG.remove_edges_from(def_use_edges)
    
    return CPG, {
        "checking_call": checking_call_lines,
        "alloc_call": alloc_call_lines,
        "dealloc_call": dealloc_call_lines,
        "read_call": read_call_lines,
        "write_call": write_call_lines,
        "other_call": other_call_lines,
        "array": array_lines,
        "ptr": ptr_lines,
        "arith": arithmatic_lines
    }

def write_slices(slices: Dict[str, List[nx.DiGraph]], cpp_path: str) -> Dict[str, bool]:
    slice_dir = join(slices_root, dirname(cpp_path))

    if not isdir(slice_dir):
        os.makedirs(slice_dir, exist_ok=True)
    
    done_list = []

    for key, slice_graphs in slices.items():
        for slice_graph in slice_graphs:
            slice_graph_path = join(slice_dir, f"{splitext(basename(cpp_path))[0]}___{key}__{slice_graph.graph['key_line']}__{slice_graph.graph['type']}.pkl")
            with open(slice_graph_path, "wb") as wfi:
                pickle.dump(slice_graph, wfi, pickle.HIGHEST_PROTOCOL)
            done_list.append(slice_graph_path)
    
    return done_list

def process_file_parallel(cpp_path, queue: Queue):
    try:
        file_cpg_root = join(csv_path, cpp_path)
        # if cpp_path in unperturbed_files:
        #     return []
        file_vul_lines = ground_truth[cpp_path] if cpp_path in ground_truth else []
        
        CPG, key_line_map = build_CPG(file_cpg_root, cpp_path)
        CPG.graph["feat_code"] = feat_name_code_map.get(csv_path.split("/")[-2], -1)
        slices = get_slices(CPG, key_line_map, set(file_vul_lines))
        return write_slices(slices, cpp_path)
    except Exception as e:
        logging.error(cpp_path)
        raise e

def process_dataset(dataset_root: str, config: DictConfig, only_clear_slices: bool):
    global csv_path, slices_root, ground_truth, unperturbed_files, USE_CPU
    if not exists(dataset_root):
        logging.info(f"{dataset_root} not exists!")
        return []
    source_root_path = join(dataset_root, config.source_root_folder)
    csv_path = join(dataset_root, config.csv_folder)
    slices_root = join(dataset_root, config.slice_folder)

    if only_clear_slices:
        logging.info(f"Clearing existing slices in {slices_root}...")
        os.system(f"rm -rf {slices_root}")
        return []
    
    if not isdir(slices_root):
        os.makedirs(slices_root, exist_ok=True)
    
    cpp_paths_filepath = join(dataset_root, config.cpp_paths_filename)
    cpp_paths = []
    if not exists(cpp_paths_filepath):
        logging.info(f"{cpp_paths_filepath} not found. Retriving all source code files from {source_root_path}...")
        for root, dirs, files in os.walk(source_root_path, topdown=True):
            for file_name in files:
                rel_dir = relpath(root, source_root_path)
                rel_file = join(rel_dir, file_name)
                if not rel_file.endswith(".c") and not rel_file.endswith(".cpp") and not rel_file.endswith(".h"):
                    continue
                cpp_paths.append(rel_file)
        logging.info(f"Successfully retrieved {len(cpp_paths)} files. Writing all cpp_paths to {cpp_paths_filepath}...")
        with open(cpp_paths_filepath, "w") as wfi:
            json.dump(cpp_paths, wfi)
    else:
        logging.info(f"Reading cpp filepaths from {cpp_paths_filepath}...")
        with open(cpp_paths_filepath, "r") as rfi:
            cpp_paths = json.load(rfi)
        logging.info(f"Completed. Retrieved {len(cpp_paths)} cpp filepaths.")

    ignore_list = []
    ignore_list_filepath = join(dataset_root, config.ignore_list_filename)
    if exists(ignore_list_filepath):
        logging.info(f"Reading ignore list from {ignore_list_filepath}...")
        with open(ignore_list_filepath, "r") as rfi:
            ignore_list = json.load(rfi)
        logging.info(f"Completed. Retrieved {len(ignore_list)} entries from ignore list.")
    else:
        logging.info(f"{ignore_list_filepath} not found. No files will be ignored.")
    cpp_paths = set(cpp_paths) - set(ignore_list)
    
    unperturbed_files_filepath = join(dataset_root, config.unperturbed_files_filename)
    if not exists(unperturbed_files_filepath):
        logging.info(f"{unperturbed_files_filepath} not found. Setting all files as perturbed...")
        unperturbed_files = []
    else:
        logging.info(f"Reading filepaths of unperturbed files from {unperturbed_files_filepath}...")
        with open(unperturbed_files_filepath, "r") as rfi:
            unperturbed_files = json.load(rfi)
        logging.info(f"Completed. Retrieved {len(unperturbed_files)} unperturbed filepaths.")
    
    ground_truth_filepath = join(dataset_root, config.ground_truth_filename)
    if not exists(ground_truth_filepath):
        logging.info(f"{ground_truth_filepath} not found. Setting ground truth as empty...")
        ground_truth = dict()
    else:
        logging.info(f"Reading ground truth from {ground_truth_filepath}...")
        with open(ground_truth_filepath, "r") as rfi:
            ground_truth = json.load(rfi)
        logging.info(f"Completed.")
    
    logging.info(f"Going over {len(cpp_paths)} files...")
    with Manager() as m:
        message_queue = m.Queue()  # type: ignore
        pool = Pool(USE_CPU)
        process_func = functools.partial(process_file_parallel, queue=message_queue)
        feat_slice_list: List = [
            file_slice
            for file_slices in tqdm(
                pool.imap_unordered(process_func, cpp_paths),
                desc=f"Cpp files",
                total=len(cpp_paths),
            )
            for file_slice in file_slices
        ]

        message_queue.put("finished")
        pool.close()
        pool.join()
    
    return feat_slice_list

def main(args: argparse.Namespace):
    global sensi_api_path, USE_CPU
    config = cast(DictConfig, OmegaConf.load(args.config))
    sensi_api_path = join(config.data_folder, config.sensi_api_map_filename)
    
    if config.num_workers != -1:
        USE_CPU = min(config.num_workers, cpu_count())
    else:
        USE_CPU = cpu_count()

    dataset_root = join(config.data_folder, config.dataset.name)
    if args.use_temp_data:
        dataset_root = config.temp_root
    
    if args.use_temp_data:
        logging.info(f"Processing temp data from {dataset_root}...")
    else:
        logging.info(f"Processing {config.dataset.name} data from {dataset_root}...")

    all_slice_list = []
    if config.dataset.name != config.VF_perts_root or args.use_temp_data:
        all_slice_list += process_dataset(dataset_root, config, args.only_clear_slices)
    else:
        global feat_name_code_map
        feat_name_code_map = {feat_name: idx for idx, feat_name in enumerate(config.vul_feats)}
        for feat_name in config.vul_feats:
            logging.info(f"Processing {feat_name}...")
            feat_dir = join(dataset_root, feat_name)
            all_slice_list += process_dataset(feat_dir, config, args.only_clear_slices)

    all_slices_filepath = join(dataset_root, config.all_slices_filename)
    logging.info(f"Writing {len(all_slice_list)} slices to {all_slices_filepath}...")
    with open(all_slices_filepath, "w") as wfi:
        json.dump(all_slice_list, wfi, indent=2)
    logging.info(f"Completed.")
    logging.info("=========End session=========")
    logging.shutdown()

if __name__ == "__main__":
    arg_parser = get_arg_parser()
    arg_parser.add_argument("--only_clear_slices", action='store_true', 
                            help="Whether to only clear existing slices")
    args = arg_parser.parse_args()
    init_log(splitext(basename(__file__))[0])
    main(args)