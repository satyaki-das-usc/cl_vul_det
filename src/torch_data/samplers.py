import math
import random
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List

from torch.utils.data import Sampler


class BalancedBinaryBatchSampler(Sampler):
    def __init__(self, labels: Iterable[int], batch_size: int, seed: int = 0):
        self.labels = [int(label) for label in labels]
        self.batch_size = int(batch_size)
        if self.batch_size < 2 or self.batch_size % 2 != 0:
            raise ValueError("BalancedBinaryBatchSampler requires an even batch_size >= 2.")

        self.samples_per_class = self.batch_size // 2
        self.seed = int(seed)
        self._epoch = 0

        class_indices: Dict[int, List[int]] = defaultdict(list)
        for index, label in enumerate(self.labels):
            class_indices[label].append(index)
        if len(class_indices) != 2:
            raise ValueError(
                "BalancedBinaryBatchSampler requires exactly two classes, "
                f"got {sorted(class_indices)}."
            )
        if any(len(indices) == 0 for indices in class_indices.values()):
            raise ValueError("BalancedBinaryBatchSampler cannot sample from an empty class.")

        self.class_indices = {
            label: indices
            for label, indices in sorted(class_indices.items())
        }
        majority_count = max(len(indices) for indices in self.class_indices.values())
        self.num_batches = math.ceil(majority_count / self.samples_per_class)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        total_per_class = self.num_batches * self.samples_per_class
        draws_by_class = {
            label: self._draw_class_indices(indices, total_per_class, rng)
            for label, indices in self.class_indices.items()
        }

        for batch_index in range(self.num_batches):
            start = batch_index * self.samples_per_class
            end = start + self.samples_per_class
            batch = []
            for label in self.class_indices:
                batch.extend(draws_by_class[label][start:end])
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.num_batches

    @staticmethod
    def _draw_class_indices(indices: List[int], n_draws: int, rng: random.Random) -> List[int]:
        draws = []
        while len(draws) < n_draws:
            shuffled = list(indices)
            rng.shuffle(shuffled)
            draws.extend(shuffled)
        return draws[:n_draws]
