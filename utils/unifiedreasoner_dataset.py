import copy
import os
from typing import List

import torch
from filelock import FileLock
from torch.utils.data import Dataset

from utils.data_utils import InputFeatures


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.features: List[InputFeatures] = []

    def cache(self, data_dir, mode, tokenizer_name, max_seq_length):
        cached_features_file = os.path.join(data_dir,
                                            f"cached_{mode.value}_{tokenizer_name}_{max_seq_length}_{os.environ['RUN_NAME'].split('@')[0]}")
        with FileLock(f'{cached_features_file}.lock'):
            if os.path.exists(cached_features_file):
                print(f'loading cache from {cached_features_file}')
                self.features = torch.load(cached_features_file)
                return 'load_from_cache'
            if len(self.features) != 0:
                print(f'saving cache to {cached_features_file}')
                torch.save(self.features, cached_features_file)
        # else:
        #    raise NotImplementedError
