import torch

from os.path import join

from multiprocessing import cpu_count

from omegaconf import DictConfig
from typing import List, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler

from src.vocabulary import Vocabulary
from src.torch_data.datasets import SliceDataset
from src.torch_data.samples import SliceGraphSample, SliceGraphBatch

class CustomSliceGraphBatchSampler(Sampler):
    def __init__(self, dataset, custom_batches):
        self.dataset = dataset
        self.custom_batches = custom_batches  # Predefined list of index batches

    def __iter__(self):
        # Yield predefined batches (list of lists of indices)
        for batch in self.custom_batches:
            yield batch

    def __len__(self):
        return len(self.custom_batches)

class SliceDataModule(LightningDataModule):
    def __init__(self, config: DictConfig, vocab: Vocabulary, use_temp_data: bool = False):
        super().__init__()
        self.__vocab = vocab
        self.__config = config

        if self.__config.num_workers != -1:
            self.__n_workers = min(self.__config.num_workers, cpu_count())
        else:
            self.__n_workers = cpu_count()
        
        if use_temp_data:
            self.__dataset_root = self.__config.temp_root
        else:
            self.__dataset_root = self.__config.data_folder

    @staticmethod
    def collate_wrapper(batch: List[SliceGraphSample]) -> SliceGraphBatch:
        return SliceGraphBatch(batch)
    
    def __create_dataset(self, data_path: str) -> Dataset:
        return SliceDataset(data_path, self.__config, self.__vocab)
    
    def train_dataloader(self) -> DataLoader:
        train_dataset_path = join(self.__dataset_root, self.__config.train_slices_filename)
        train_dataset = self.__create_dataset(train_dataset_path)
        return DataLoader(
            train_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=self.__config.hyper_parameters.shuffle_data,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        val_dataset_path = join(self.__dataset_root, self.__config.val_slices_filename)
        val_dataset = self.__create_dataset(val_dataset_path)
        return DataLoader(
            val_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        test_dataset_path = join(self.__dataset_root, self.__config.test_slices_filename)
        test_dataset = self.__create_dataset(test_dataset_path)
        return DataLoader(
            test_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )
    
    def transfer_batch_to_device(
            self,
            batch: SliceGraphBatch,
            device: Optional[torch.device] = None) -> SliceGraphBatch:
        if device is not None:
            batch.move_to_device(device)
        return batch
