# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: tree_cost_loss
# @Author: Wei Zhou
# @Time: 2022/9/4 21:33

import torch
import torch.nn as nn

import numpy as np

from index_advisor_selector.index_benefit_estimation.tree_model.tree_cost_dataset import unnormalize


def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


def cal_mape(preds, targets):
    mape = ((np.abs(preds - targets) / targets)).mean()
    return mape


class RatioLoss(nn.Module):
    def __init__(self):
        super(RatioLoss, self).__init__()

    def forward(self, preds, targets):
        loss = torch.abs(preds - targets) / targets
        return torch.mean(loss)


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
        # qerror = torch.where(x > y, x, y)
        # qerror = []

        # for i in range(len(targets)):
        #     # if (preds[i] > targets[i]).cpu().data.numpy()[0]:
        #     if preds[i] > targets[i]:
        #         qerror.append(preds[i] / targets[i])
        #     else:
        #         qerror.append(targets[i] / preds[i])
        # return torch.mean(torch.cat(qerror))
        if out == "mean":
            return torch.mean(qerror)
        elif out == "raw":
            return qerror


def xgb_loss(preds, targets):
    name = "ratio_loss"
    targets = targets.get_label()
    val = np.mean(np.abs(preds - targets) / targets)

    return name, val


def xgb_MSE(preds_unnorm, targets_unnorm):
    """
    :param preds_unnorm:
    :param targets_unnorm:
    :return:
    """
    name = "mse"
    # val = np.sqrt(((preds_unnorm - targets_unnorm.get_label()) ** 2).mean())
    val = ((np.exp(preds_unnorm) - np.exp(targets_unnorm.get_label())) ** 2).mean()
    return name, val


def xgb_QError(preds, targets, out="mean",
               min_val=-16.997074, max_val=2.9952054):
    """
    qerror(cost_i, cost_i') = max(cost_i, cost_i) / min(costi, cost_i)

    :param preds:
    :param targets:
    :param out:
    :param min_val:
    :param max_val:
    :return:
    """
    name = "qerror"
    targets = targets.get_label()
    if min_val and max_val:
        preds = unnormalize(preds, min_val, max_val)
        targets = unnormalize(targets, min_val, max_val)
    else:
        preds = preds + 1e-8
        targets = targets + 1e-8

    qerror = list()
    for i in range(len(targets)):
        if preds[i] > targets[i]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    if out == "mean":
        val = np.mean(qerror)
    else:
        val = qerror
    return name, val


def lgb_loss(preds, targets):
    metric_name = "ratio_loss"
    targets = targets.label
    # preds = preds.label
    value = np.mean(np.abs(preds - targets) / targets)
    is_higher_better = False

    return metric_name, value, is_higher_better


def lgb_MSE(preds, targets):
    metric_name = "mse"
    # value = ((preds - targets.label) ** 2).mean()
    value = ((np.exp(preds) - np.exp(targets.label)) ** 2).mean()
    is_higher_better = False

    return metric_name, value, is_higher_better


def lgb_QError(preds, targets, out="mean",
               min_val=-16.997074, max_val=2.9952054):
    """
    qerror(cost_i, cost_i') = max(cost_i, cost_i) / min(costi, cost_i)
    :param preds:
    :param targets:
    :param out:
    :param min_val:
    :param max_val:
    :return:
    """
    metric_name = "qerror"
    targets = targets.label
    if min_val and max_val:
        preds = unnormalize(preds, min_val, max_val)
        targets = unnormalize(targets, min_val, max_val)
    else:
        preds = preds + 1e-8
        targets = targets + 1e-8

    qerror = list()
    for i in range(len(targets)):
        if preds[i] > targets[i]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    if out == "mean":
        value = np.mean(qerror)
    else:
        value = qerror
    is_higher_better = False

    return metric_name, value, is_higher_better
