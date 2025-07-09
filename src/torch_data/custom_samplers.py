import json

from torch.utils.data import Sampler

class BalancedSampler(Sampler):
    def __init__(self, balanced_batches_filepath):
        with open(balanced_batches_filepath, "r") as rfi:
            self.custom_batches = json.load(rfi)

    def __iter__(self):
        # Yield predefined batches (list of lists of indices)
        for batch in self.custom_batches:
            yield batch

    def __len__(self):
        return len(self.custom_batches)