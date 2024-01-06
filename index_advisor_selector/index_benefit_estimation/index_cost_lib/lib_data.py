# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: lib_data
# @Author: Wei Zhou
# @Time: 2023/6/7 21:52

import torch
from torch.utils.data import Dataset


def collate_fn4lib(samples):
    data, label = map(list, zip(*samples))

    # Find the maximum number of index optimizable operations
    max_l, min_l = 0, 999
    max_l = max(max_l, max([len(da) for da in data]))
    min_l = min(min_l, min([len(da) for da in data]))

    # Pad data to facilitate batch training/testing
    pad_data, mask, pad_element = list(), list(), [0 for _ in range(0, 12)]
    for data_point in data:
        new_data = []
        point_mask = [0 for _ in range(0, max_l)]

        for j in range(0, len(data_point)):
            # (1006): newly modified.
            new_data.append(data_point[j])
            # new_data.append(data_point[j][:-1])  # ?[:-1]
            point_mask[j] = 1

        if max_l - len(data_point) > 0:
            for k in range(0, max_l - len(data_point)):
                new_data.append(pad_element)

        pad_data.append(new_data)
        mask.append(point_mask)

    return torch.tensor(pad_data), torch.tensor(mask), torch.tensor(label)


class LIBDataset(Dataset):
    def __init__(self, data_label):
        self.data = list()
        self.label = list()
        self.max_l = 0
        self.min_l = 999

        self.format(data_label)

    def format(self, data_label):
        # Remove empty points
        data, label = list(), list()
        for i in range(0, len(data_label)):
            if len(data_label[i][0]) == 0:
                continue
            else:
                data.append(data_label[i][0])
                label.append(data_label[i][1])

        self.data, self.label = data, label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        assert len(self.data) == len(self.label)
        return len(self.data)
