import sys

import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import cpu_count, Pool

class DerivativeModel:
    def __init__(self, topk, batch_size=500):
        self.topk = topk
        self.batch_size = batch_size

    def predict(self, tops, bottoms):
        derivates_list = []

        bottoms = bottoms.unsqueeze(0)

        bar_format = '{desc}|{bar:10}|[{elapsed}<{remaining},{rate_fmt}]'
        with tqdm(range(len(tops) // self.batch_size), file=sys.stdout, bar_format=bar_format, position=0, leave=True) as pbar:
            for i in range(0, len(tops), self.batch_size):
                current_tops = tops[i:i + self.batch_size]
                current_tops = current_tops.unsqueeze(1)

                derivates = torch.mean(torch.abs((current_tops[:, :, -1] - bottoms[:, :, 0])), -1)
                _, top_elements_indices = torch.topk(-derivates, self.topk, dim=-1)
                derivates_list.extend(top_elements_indices)

                pbar.update(1)

        return derivates_list

