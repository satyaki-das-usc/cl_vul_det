import torch

from os.path import join

from multiprocessing import cpu_count

from omegaconf import DictConfig
from typing import List, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.vocabulary import Vocabulary
from src.torch_data.datasets import SliceDataset
from src.torch_data.samples import SliceGraphSample, SliceGraphBatch

class SliceDataModule(LightningDataModule):
    def __init__(self, config: DictConfig, vocab: Vocabulary, train_batch_size, train_sampler=None, use_temp_data: bool = False):
        super().__init__()
        self.__vocab = vocab
        self.__config = config
        self.__train_sampler = train_sampler
        self.__train_batch_size = train_batch_size
        self.__train_dataset = None
        self.__val_dataset = None
        self.__test_dataset = None

        if self.__config.num_workers != -1:
            self.__n_workers = min(self.__config.num_workers, cpu_count())
        else:
            self.__n_workers = cpu_count()
        
        self.__dataset_root = join(self.__config.data_folder, self.__config.dataset.name)
        if use_temp_data:
            self.__dataset_root = self.__config.temp_root

    @staticmethod
    def collate_wrapper(batch: List[SliceGraphSample]) -> SliceGraphBatch:
        return SliceGraphBatch(batch)
    
    def __create_dataset(self, data_path: str) -> Dataset:
        return SliceDataset(data_path, self.__config, self.__vocab, cache_size=self.__train_batch_size)
    
    def set_train_batch_size(self, batch_size: int):
        self.__train_batch_size = batch_size

    def clear_cache(self):
        if self.__train_dataset is not None:
            self.__train_dataset.clear_cache()
        if self.__val_dataset is not None:
            self.__val_dataset.clear_cache()
        if self.__test_dataset is not None:
            self.__test_dataset.clear_cache()
    
    def get_cache_info(self):
        """Get cache statistics from all datasets"""
        info = {}
        if self.__train_dataset is not None:
            info['train'] = self.__train_dataset.get_cache_info()
        if self.__val_dataset is not None:
            info['val'] = self.__val_dataset.get_cache_info()
        if self.__test_dataset is not None:
            info['test'] = self.__test_dataset.get_cache_info()
        return info
    
    def train_dataloader(self) -> DataLoader:
        if self.__config.dataset.name == "BigVul":
            train_dataset_path = join(self.__dataset_root, f"{self.__config.dataset.version}_{self.__config.train_slices_filename}")
        else:
            train_dataset_path = join(self.__dataset_root, self.__config.train_slices_filename)
        self.__train_dataset = self.__create_dataset(train_dataset_path)

        if self.__train_sampler:
            return DataLoader(
                self.__train_dataset,
                batch_size=self.__train_batch_size,
                sampler=self.__train_sampler,
                num_workers=self.__n_workers,
                collate_fn=self.collate_wrapper,
                pin_memory=False,
                persistent_workers=False
            )
        return DataLoader(
            self.__train_dataset,
            batch_size=self.__train_batch_size,
            shuffle=True,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=False,
            persistent_workers=False
        )

    def val_dataloader(self) -> DataLoader:
        if self.__config.dataset.name == "BigVul":
            val_dataset_path = join(self.__dataset_root, f"{self.__config.dataset.version}_{self.__config.val_slices_filename}")
        else:
            val_dataset_path = join(self.__dataset_root, self.__config.val_slices_filename)
        self.__val_dataset = self.__create_dataset(val_dataset_path)
        return DataLoader(
            self.__val_dataset,
            batch_size=self.__config.hyper_parameters.test_batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=False,
            persistent_workers=False
        )

    def test_dataloader(self) -> DataLoader:
        if self.__config.dataset.name == "BigVul":
            test_dataset_path = join(self.__dataset_root, f"{self.__config.dataset.version}_{self.__config.test_slices_filename}")
        else:
            test_dataset_path = join(self.__dataset_root, self.__config.test_slices_filename)
        self.__test_dataset = self.__create_dataset(test_dataset_path)
        return DataLoader(
            self.__test_dataset,
            batch_size=self.__config.hyper_parameters.test_batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=False,
            persistent_workers=False
        )
    
    def transfer_batch_to_device(
            self,
            batch: SliceGraphBatch,
            device: Optional[torch.device] = None,
            dataloader_idx=None) -> SliceGraphBatch:
        if device is not None:
            batch.move_to_device(device)
        return batch
