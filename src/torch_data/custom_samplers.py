import pickle
import networkx as nx

from os.path import join, splitext

from omegaconf import DictConfig
from tqdm import tqdm

from torch.utils.data import Sampler

from src.models.modules.gnns import GraphConvEncoder, GatedGraphConvEncoder
from src.vocabulary import Vocabulary

encoders = {
    "gcn": GraphConvEncoder,
    "ggnn": GatedGraphConvEncoder
}

class InstanceSampler(Sampler):
    def __init__(self, dataset, batch_size: int, config: DictConfig, unperturbed_file_list):
        self.dataset = dataset
        self.batch_size = batch_size
        # self.custom_batches = custom_batches  # Predefined list of index batches
        self.custom_batches = self.get_custom_batches(unperturbed_file_list, config.source_root_folder, config.slice_folder)  # Predefined list of index batches

    def split_sublists(self, list_of_lists):
        result = []
        for sublist in list_of_lists:
            if len(sublist) > self.batch_size:
                result.extend([sublist[i:i + self.batch_size] for i in range(0, len(sublist), self.batch_size)])
            else:
                result.append(sublist)
        return result
    
    def get_custom_batches(self, unperturbed_file_list, source_root_foldername, slice_foldername):
        custom_batch_indices = []

        instance_index_map = {}

        other_train_data = {}
        # sample_io_c_filepath = "000/119/825/io.c"
        # instance_index_map[sample_io_c_filepath] = []
        # logging.info("Creating map for io.c...")
        # for idx, slice_graph_sample in tqdm(enumerate(dataset), total=len(dataset)):
        #     slice_graph_path = slice_graph_sample.path
        #     cpp_filepath = "/".join(slice_graph_path.split("slice/")[-1].split("/")[:-2])
        #     if cpp_filepath != sample_io_c_filepath:
        #         continue
        #     instance_index_map[sample_io_c_filepath].append(idx)
        # logging.info(f"Mapping for io.c completed. Number of instances: {len(instance_index_map[sample_io_c_filepath])}")

        print("Creating instance index map...")
        for idx, slice_path in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            with open(slice_path, "rb") as rbfi:
                slice_graph: nx.DiGraph = pickle.load(rbfi)
            src_cpp_path = join(slice_path.partition(slice_foldername)[0], source_root_foldername, slice_graph.graph["file_paths"][0])
            cpp_filepath = src_cpp_path.partition(source_root_foldername)[-1][1:]
            if cpp_filepath.endswith("io.c"):
                continue
            if cpp_filepath in unperturbed_file_list:
                if cpp_filepath not in instance_index_map:
                    instance_index_map[cpp_filepath] = []
                instance_index_map[cpp_filepath].append(idx)
            else:
                other_train_data[slice_path] = idx
        print(f"Number of instances: {len(instance_index_map)}")
        
        print("Assigning other instances to the same batch...")
        for cpp_filepath, indices in tqdm(instance_index_map.items(), total=len(instance_index_map)):
            cpp_filepath_no_ext = splitext(cpp_filepath)[0]
            keys_to_remove = []
            for slice_path, idx in other_train_data.items():
                if cpp_filepath_no_ext not in slice_path:
                    continue
                indices.append(idx)
                keys_to_remove.append(slice_path)
            for key in keys_to_remove:
                del other_train_data[key]
        print(f"Number of instances: {len(instance_index_map)}")
        
        instance_index_map["other"] = list(other_train_data.values())

        custom_batch_indices = list(instance_index_map.values())
        
        return self.split_sublists(custom_batch_indices)

    def __iter__(self):
        # Yield predefined batches (list of lists of indices)
        for batch in self.custom_batches:
            yield batch

    def __len__(self):
        return len(self.custom_batches)

class VFSampler(Sampler):
    def __init__(self, dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        # self.custom_batches = custom_batches  # Predefined list of index batches
        self.custom_batches = self.get_custom_batches()  # Predefined list of index batches
    
    def get_custom_batches(self):
        custom_batch_indices = []
        curr_feat_indices = [-1 for i in range(10)]

        for idx, slice_path in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            with open(slice_path, "rb") as rbfi:
                slice_graph: nx.DiGraph = pickle.load(rbfi)
            feat_index = curr_feat_indices[slice_graph.graph["feat_code"]]
            if feat_index == -1 or len(custom_batch_indices[feat_index]) >= self.batch_size:
                custom_batch_indices.append([])
                feat_index = len(custom_batch_indices) - 1
                curr_feat_indices[slice_graph.graph["feat_code"]] = feat_index
            custom_batch_indices[feat_index].append(idx)
        
        return custom_batch_indices

    def __iter__(self):
        # Yield predefined batches (list of lists of indices)
        for batch in self.custom_batches:
            yield batch

    def __len__(self):
        return len(self.custom_batches)

class SwAVSampler(Sampler):
    def __init__(self, dataset, batch_size: int, config: DictConfig, vocab: Vocabulary, vocabulary_size: int,
                 pad_idx: int):
        self.dataset = dataset
        self.__graph_encoder = encoders[config.gnn.name](config.gnn, vocab, vocabulary_size,
                                                               pad_idx)
        self.batch_size = batch_size
        # self.custom_batches = custom_batches  # Predefined list of index batches
        self.custom_batches = self.get_custom_batches()  # Predefined list of index batches
    
    def get_custom_batches(self):
        custom_batch_indices = []
        
        
        return custom_batch_indices

    def __iter__(self):
        # Yield predefined batches (list of lists of indices)
        for batch in self.custom_batches:
            yield batch

    def __len__(self):
        return len(self.custom_batches)