import re
import json

from os.path import join

from src.common_utils import read_csv

def get_incoming_dd_edges_for_var(CPG, v, var):
    in_data_var = set()
    for edge in CPG.in_edges(v):
        u = edge[0]
        edge_data = CPG.get_edge_data(u, v)
        if edge_data["label"] != "REACHES":
            continue
        edge_var = edge_data["var"]
        if edge_var != var:
            continue
        in_data_var.add(u)
    
    return in_data_var

def get_incoming_cd_edges(CPG, v):
    in_ctrl_v = set()
    for edge in CPG.in_edges(v):
        u = edge[0]
        edge_data = CPG.get_edge_data(u, v)
        if edge_data["label"] != "CONTROLS":
            continue
        in_ctrl_v.add(u)
    
    return in_ctrl_v

def get_start_node_idx(joern_nodes, line_num):
    for idx, node in enumerate(joern_nodes):
        if len(node["location"]) == 0:
            continue
        loc = int(node["location"].split(":")[0])
        if line_num != loc:
            continue
        return idx
    return -1

def get_end_node_idx(joern_nodes, start_idx):
    for idx, node in enumerate(joern_nodes[start_idx + 1:]):
        if len(node["location"]) == 0:
            continue
        if node["isCFGNode"] != "True":
            continue
        return start_idx + 1 + idx
    return -1

def get_line_nodes(joern_nodes, line_num):
    start_idx = get_start_node_idx(joern_nodes, line_num)
    end_idx = get_end_node_idx(joern_nodes, start_idx)

    if end_idx < 0:
        return joern_nodes[start_idx:]
    
    return joern_nodes[start_idx: end_idx]

def is_buffer_write_function_call(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)
    if len([node for node in line_nodes if node["type"] == "CallExpression"]) == 0:
        return False
    return len([node for node in line_nodes if node["type"] == "Callee" and node["code"].strip() == "memcpy"]) > 0

def is_buffer_allocation_function_call(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)
    if len([node for node in line_nodes if node["type"] == "CallExpression"]) == 0:
        return False
    return len([node for node in line_nodes if node["type"] == "Callee" and node["code"].strip() == "malloc"]) > 0

def is_arr_decl(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)
    id_decl_nodes = [node for node in line_nodes if node["type"] == "IdentifierDeclType"]
    if len(id_decl_nodes) == 0:
        return False
    
    for node in id_decl_nodes:
        if "[" not in node["code"]:
            continue
        if "]" not in node["code"]:
            continue
        return True
    
    return False

def is_arr_index_write_operation(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)
    arr_indexing_nodes = [node for node in line_nodes if node["type"] == "ArrayIndexing"]
    if len(arr_indexing_nodes) == 0:
        return False
    arr_indexing_str = arr_indexing_nodes[0]["code"].strip().replace(" ", "")
    asgnmt_expr_nodes = [node for node in line_nodes if node["type"] == "AssignmentExpression"]

    for node in asgnmt_expr_nodes:
        asgnmt_operator = node["operator"].strip()
        asgnmt_code_str = node["code"].strip()
        asgnmt_operand1, asgnmt_operand2 = [operand.replace(" ", "").strip() for operand in asgnmt_code_str.split(asgnmt_operator, maxsplit=1)]
        if asgnmt_operand1 != arr_indexing_str:
            continue
        return True
    
    return False

def is_arr_index_read_operation(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)
    arr_indexing_nodes = [node for node in line_nodes if node["type"] == "ArrayIndexing"]
    if len(arr_indexing_nodes) == 0:
        return False
    arr_indexing_str = arr_indexing_nodes[0]["code"].strip().replace(" ", "")
    asgnmt_expr_nodes = [node for node in line_nodes if node["type"] == "AssignmentExpression"]
    
    for node in asgnmt_expr_nodes:
        asgnmt_operator = node["operator"].strip()
        asgnmt_code_str = node["code"].strip()
        asgnmt_operand1, asgnmt_operand2 = [operand.replace(" ", "").strip() for operand in asgnmt_code_str.split(asgnmt_operator, maxsplit=1)]
        if asgnmt_operand1 != arr_indexing_str:
            continue
        return False
    
    return True

def is_buffer_copy_function_call(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)
    if len([node for node in line_nodes if node["type"] == "CallExpression"]) == 0:
        return False
    return len([node for node in line_nodes if node["type"] == "Callee" and node["code"].strip() == "strncpy"]) > 0

def is_buffer_deallocation_function_call(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)
    if len([node for node in line_nodes if node["type"] == "CallExpression"]) == 0:
        return False
    return len([node for node in line_nodes if node["type"] == "Callee" and node["code"].strip() == "free"]) > 0

def is_relational_expression(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)

    return len([node for node in line_nodes if node["type"] == "RelationalExpression"]) > 0

def is_arithmetic_expression(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)

    return len([node for node in line_nodes if node["operator"].strip() in ['+', '-', '*', '/']]) > 0

def can_follow(CPG, u, v):
    if CPG.has_edge(v, u) and CPG.get_edge_data(v, u)["label"] == "POST_DOM":
        return True
    u_ctrl_edges = [edge for edge in CPG.in_edges(u) if CPG.get_edge_data(edge[0], edge[1])["label"] == "CONTROLS"]
    v_ctrl_edges = [edge for edge in CPG.in_edges(v) if CPG.get_edge_data(edge[0], edge[1])["label"] == "CONTROLS"]
    for x in u_ctrl_edges:
        for y in v_ctrl_edges:
            if CPG.has_edge(y[0], x[0]) and CPG.get_edge_data(y[0], x[0])["label"] == "POST_DOM":
                return True
    return False

def is_sensi_read_function_call(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)
    callee_nodes = [node for node in line_nodes if node["type"] == "Callee"]

    sensi_read_apis = ["getch", "fgets", "gets", "getchar", "fgetc"]
    for node in callee_nodes:
        callee_func = node["code"].strip()
        if callee_func not in sensi_read_apis:
            continue
        return True
    
    return False

def is_sensi_write_function_call(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)
    callee_nodes = [node for node in line_nodes if node["type"] == "Callee"]

    sensi_write_apis = ["memcpy", "strncpy", "strcpy", "wcsncpy", "memset", "wmemset"]
    for node in callee_nodes:
        callee_func = node["code"].strip()
        if callee_func not in sensi_write_apis:
            continue
        return True
    
    return False

def get_called_sensi_funcs(joern_nodes, v, key):
    line_nodes = get_line_nodes(joern_nodes, v)
    callee_nodes = [node for node in line_nodes if node["type"] == "Callee"]

    sensi_apis = []

    if key == "sensi_read_callee":
        sensi_apis = ["getch", "fgets", "gets", "getchar", "fgetc"]
    if key == "sensi_write_callee":
        sensi_apis = ["memcpy", "strncpy", "strcpy", "wcsncpy", "memset", "wmemset"]
    
    called_sensi_funcs = set()
    for node in callee_nodes:
        callee_func = node["code"].strip()
        if callee_func not in sensi_apis:
            continue
        called_sensi_funcs.add(callee_func)
    
    return list(called_sensi_funcs)

def get_node_type(joern_nodes, v):
    node_types = []
    if is_arithmetic_expression(joern_nodes, v):
        node_types.append("ArithmeticExpression")
    if is_buffer_write_function_call(joern_nodes, v):
        return node_types + ["WF"]
    elif is_buffer_allocation_function_call(joern_nodes, v):
        return node_types + ["AF"]
    elif is_arr_decl(joern_nodes, v):
        return node_types + ["AD"]
    elif is_buffer_copy_function_call(joern_nodes, v):
        return node_types + ["CF"]
    elif is_buffer_deallocation_function_call(joern_nodes, v):
        return node_types + ["DF"]
    elif is_arr_index_write_operation(joern_nodes, v):
        return node_types + ["AIW"]
    elif is_arr_index_read_operation(joern_nodes, v):
        return node_types + ["AIR"]
    elif is_relational_expression(joern_nodes, v):
        return node_types + ["RE"]
    elif is_sensi_read_function_call(joern_nodes, v):
        return node_types + ["SRF"]
    elif is_sensi_write_function_call(joern_nodes, v):
        return node_types + ["SWF"]
    return node_types + ["UNK"]

def get_start_wf_idx(line_nodes):
    for idx, node in enumerate(line_nodes):
        if node["type"].strip() != "Callee":
            continue
        if node["code"].strip() not in ["memcpy", "strncpy"]:
            continue

        return idx

def binary_search_insert(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left

def get_wf_arg_nodes(line_nodes, start_idx):
    if start_idx is None:
        with open("error.log", "a") as afi:
            for node in line_nodes:
                afi.write(f"\nType: {node['type']}; Code: {node['code']}")
    
    arg_indices = [(start_idx + idx) for idx, node in enumerate(line_nodes[start_idx:]) if node["type"].strip() == "Argument"]
    rm_arg_indices = [binary_search_insert(arg_indices, start_idx + idx) for idx, node in enumerate(line_nodes[start_idx:]) if node["type"].strip() == "CallExpression"]

    return [line_nodes[arg_node_idx] for idx, arg_node_idx in enumerate(arg_indices) if idx not in rm_arg_indices]

def get_buffer_write_dest(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)
    start_idx = get_start_wf_idx(line_nodes)
    arg_nodes = get_wf_arg_nodes(line_nodes, start_idx)

    if len(arg_nodes) != 3:
        cpp_path = joern_nodes[0]["code"].strip().split("source-code")[-1][1:]
        with open("error.log", "a") as afi:
            afi.write(f"{cpp_path}::{v}\n")
    
    assert len(arg_nodes) == 3, f"ERROR: Buffer write with {len(arg_nodes)} arguments"

    return arg_nodes[0]["code"].strip()

def get_buffer_write_src(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)
    start_idx = get_start_wf_idx(line_nodes)
    arg_nodes = get_wf_arg_nodes(line_nodes, start_idx)

    if len(arg_nodes) != 3:
        cpp_path = joern_nodes[0]["code"].strip().split("source-code")[-1][1:]
        with open("error.log", "a") as afi:
            afi.write(f"{cpp_path}::{v}\n")
    
    assert len(arg_nodes) == 3, f"ERROR: Buffer write with {len(arg_nodes)} arguments"

    return arg_nodes[1]["code"].strip()

def can_be_evaluated(math_expression):
    valid_pattern = re.compile(r'^[\d\+\-\*/\(\)\.\s]+$')
    
    if not valid_pattern.match(math_expression):
        return False
    
    try:
        result = eval(math_expression, {"__builtins__": None}, {})
    except:
        return False
    
    return True

def get_numeric_part(size_str):
    expression = f"{size_str}".replace(" ", "").replace("]", "")
    replacements = [
        "sizeof(char)",
        "sizeof(unsignedchar)",
        "sizeof(signedchar)",
        "sizeof(int)",
        "sizeof(unsignedint)",
        "sizeof(short)",
        "sizeof(unsignedshort)",
        "sizeof(long)",
        "sizeof(unsignedlong)",
        "sizeof(int64_t)",
        "sizeof(twoIntsStruct)",
        "char[",
        "unsignedchar[",
        "signedchar[",
        "int[",
        "unsignedint[",
        "short[",
        "unsignedshort[",
        "long[",
        "unsignedlong[",
        "int64_t[",
        "twoIntsStruct[",
    ]
    math_ops = ["+", "-", "*", "/"]
    for search_key in replacements:
        expression = expression.replace(search_key, "")
    for op in math_ops:
        expression = expression.lstrip(op).rstrip(op)
    
    return expression

def evaluate_size(size_str):
    expression = f"{size_str}".replace(" ", "").replace("]", "")
    with open("size_replacements.json", "r") as rfi:
        replacements = json.load(rfi)
        for key, value in replacements.items():
            expression = expression.replace(key, value)

    if not can_be_evaluated(expression):
        return -2147483648
    return eval(expression)

def get_len_func_start_idx(line_nodes):
    for idx, node in enumerate(line_nodes):
        if node["type"].strip() != "Callee":
            continue
        if "len" not in node["code"]:
            continue

        return idx
    
    return None

def is_wcslen_call(line_nodes, len_func_start_idx):
    for node in line_nodes[len_func_start_idx:]:
        if node["type"].strip() != "Callee":
            continue
        if node["code"].strip() != "wcslen":
            continue
        return True
    return False

def get_concrete_buffer_write_byte_count_str(CPG, nodes_dir, joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)
    arg_nodes = [node for node in line_nodes if node["type"] == "Argument"]

    buffer_write_byte_count_str = arg_nodes[2]["code"].strip()
    if len(arg_nodes) == 3:
        return buffer_write_byte_count_str
    
    len_func_start_idx = get_len_func_start_idx(line_nodes)
    if len_func_start_idx is None:
        return buffer_write_byte_count_str

    len_func_arg_nodes = [node for node in line_nodes[len_func_start_idx:] if node["type"].strip() == "Argument"]
    if len(len_func_arg_nodes) == 0:
        return buffer_write_byte_count_str

    len_var = len_func_arg_nodes[0]["code"].strip().replace(" ", "")
    dd_len_var = list(get_incoming_dd_edges_for_var(CPG, v, len_var))

    if len(dd_len_var) == 0:
        return buffer_write_byte_count_str
    
    retrived_len = get_buffer_length(joern_nodes, dd_len_var[0], mu(nodes_dir, "type", dd_len_var[0]))
    if is_wcslen_call(line_nodes, len_func_start_idx):
        retrived_len = int(retrived_len / 4)
    replace_from = line_nodes[len_func_start_idx - 1]["code"].strip()

    if replace_from in buffer_write_byte_count_str:
        return buffer_write_byte_count_str.replace(replace_from, f"{retrived_len}")
    
    replace_from = replace_from.replace(" ", "")
    return buffer_write_byte_count_str.replace(replace_from, f"{retrived_len}")

def get_buffer_write_byte_count_str(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)
    start_idx = get_start_wf_idx(line_nodes)
    arg_nodes = get_wf_arg_nodes(line_nodes, start_idx)
    
    if len(arg_nodes) != 3:
        cpp_path = joern_nodes[0]["code"].strip().split("source-code")[-1][1:]
        with open("error.log", "a") as afi:
            afi.write(f"{cpp_path}::{v}\n")
    
    assert len(arg_nodes) == 3, f"ERROR: Buffer write with {len(arg_nodes)} arguments"

    return arg_nodes[2]["code"].strip()

def get_buffer_write_byte_count(CPG, nodes_dir, joern_nodes, v):
    return evaluate_size(get_concrete_buffer_write_byte_count_str(CPG, nodes_dir, joern_nodes, v))

def get_buffer_length_str(joern_nodes, v, node_type):
    line_nodes = get_line_nodes(joern_nodes, v)
    if node_type == "AF":
        arg_nodes = [node for node in line_nodes if node["type"] == "Argument"]

        if len(arg_nodes) != 1:
            cpp_path = joern_nodes[0]["code"].strip().split("source-code")[-1][1:]
            with open("error.log", "a") as afi:
                afi.write(f"{cpp_path}::{v}\n")
        
        assert len(arg_nodes) == 1, f"ERROR: Buffer alloc with {len(arg_nodes)} arguments"

        return arg_nodes[0]["code"].strip()
    elif node_type == "AD":
        size_str = [node for node in line_nodes if node["type"] == "IdentifierDeclType"][0]["code"]

        return size_str
    
    return "-1"

def get_buffer_length(joern_nodes, v, node_type):
    return evaluate_size(get_buffer_length_str(joern_nodes, v, node_type))

def get_buffer_alloc_dest(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)
    asgnmnt_nodes = [node for node in line_nodes if node["type"] == "AssignmentExpression"]

    if len(asgnmnt_nodes) != 1:
        cpp_path = joern_nodes[0]["code"].strip().split("source-code")[-1][1:]
        with open("error.log", "a") as afi:
            afi.write(f"{cpp_path}::{v}\n")
        
        return None
    
    assert len(asgnmnt_nodes) == 1, f"ERROR: Buffer alloc with {len(asgnmnt_nodes)} assignment expressions"

    return asgnmnt_nodes[0]["code"].split("=", maxsplit=1)[0].strip()

def get_array_indexing_info(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)
    arr_indexing_node = [node for node in line_nodes if node["type"] == "ArrayIndexing"][0]
    dest_arr, dest_index = [part.strip() for part in arr_indexing_node["code"].split("[", maxsplit=1)]
    dest_index = dest_index.replace("]", "").strip()

    return dest_arr, dest_index

def get_deallocated_buffer(joern_nodes, v):
    line_nodes = get_line_nodes(joern_nodes, v)
    arg_nodes = [node for node in line_nodes if node["type"] == "Argument"]

    if len(arg_nodes) != 1:
        cpp_path = joern_nodes[0]["code"].strip().split("source-code")[-1][1:]
        with open("error.log", "a") as afi:
            afi.write(f"{cpp_path}::{v}\n")
    
    assert len(arg_nodes) == 1, f"ERROR: Buffer dealloc with {len(arg_nodes)} arguments"

    return arg_nodes[0]["code"].strip()

def mu(nodes_dir, key, v, CPG=None):
    nodes_path = join(nodes_dir, "nodes.csv")
    joern_nodes = read_csv(nodes_path)

    if key == "type":
        return get_node_type(joern_nodes, v)
    if key == "arg_dest":
        return get_buffer_write_dest(joern_nodes, v)
    if key == "arg_src":
        return get_buffer_write_src(joern_nodes, v)
    if key == "arg_count":
        return get_buffer_write_byte_count(CPG, nodes_dir, joern_nodes, v)
    if key == "arg_count_str":
        return get_concrete_buffer_write_byte_count_str(CPG, nodes_dir, joern_nodes, v)
    if key == "len":
        return get_buffer_length(joern_nodes, v, mu(nodes_dir, "type", v))
    if key == "dest":
        return get_buffer_alloc_dest(joern_nodes, v)
    if key == "arr_idx":
        return get_array_indexing_info(joern_nodes, v)
    if key == "dealloc_buff":
        return get_deallocated_buffer(joern_nodes, v)
    if key in ["sensi_read_callee", "sensi_write_callee"]:
        return get_called_sensi_funcs(joern_nodes, v, key)

    return None

if __name__ == "__main__":
    pass