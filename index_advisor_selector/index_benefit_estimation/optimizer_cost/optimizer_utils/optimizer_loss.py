# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: tree_cost_loss
# @Author: Wei Zhou
# @Time: 2022/9/4 21:33

import torch
import torch.nn as nn

import numpy as np


def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


def cal_mape(preds, targets):
    mape = ((np.abs(preds - targets) / targets)).mean()
    return mape


class QError(nn.Module):
    def __init__(self):
        super(QError, self).__init__()

    def forward(self, preds, targets, out="mean",
                # min_val=-16.997074, max_val=2.9952054,
                min_val=None, max_val=None):
        if min_val and max_val:
            preds = unnormalize_torch(preds, min_val, max_val)
            targets = unnormalize_torch(targets, min_val, max_val)
        else:
            preds = preds + 1e-8
            targets = targets + 1e-8

        x = preds / targets
        y = targets / preds
        qerror = torch.where(preds > targets, x, y)

        if out == "mean":
            return torch.mean(qerror)
        elif out == "raw":
            return qerror
