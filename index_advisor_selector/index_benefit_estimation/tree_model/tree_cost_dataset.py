# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: tree_cost_dataset
# @Author: Wei Zhou
# @Time: 2022/6/26 13:16

import numpy as np

from torch.utils.data import Dataset, random_split, DataLoader


def normalize(x, min_card_log, max_card_log):
    return np.maximum(np.minimum((np.log(x) - min_card_log) / (max_card_log - min_card_log), 1.0), 0.0)


def unnormalize(x, min_card_log, max_card_log):
    return np.exp(x * (max_card_log - min_card_log) + min_card_log)


class PlanPairDataset(Dataset):
    def __init__(self, plan_data, plan_num=4, feat_chan="cost_row",
                 feat_conn="concat", label_type="cla", cla_min_ratio=0.2):
        self.plan_data = plan_data
        self.plan_num = plan_num
        self.feat_chan = feat_chan
        self.feat_conn = feat_conn
        self.label_type = label_type
        self.cla_min_ratio = cla_min_ratio

    def __getitem__(self, index):
        item = self.plan_data[index]

        channels = list()
        if "cost_row" in self.feat_chan:
            channels.extend(["p_cost_feat", "p_row_feat", "p_wcost_feat", "p_wrow_feat"])
        elif "cost" in self.feat_chan:
            channels.extend(["p_cost_feat", "p_wcost_feat"])
        elif "row" in self.feat_chan:
            channels.extend(["p_row_feat", "p_wrow_feat"])

        # 1. feature selection -> 2. feature connection
        feat = list()
        for channel in channels:
            src_wo, src_hypo, tgt_wo, tgt_hypo = list(), list(), list(), list()
            if self.feat_chan in ["cost", "row", "cost_row"]:
                if self.plan_num == 1:
                    src_wo = item["feat"][channel]
                elif self.plan_num == 2:
                    src_wo = item[0]["feat"][channel]
                    src_hypo = item[1]["feat"][channel]
                elif self.plan_num == 4:
                    src_wo = item[0]["feat"][channel]
                    src_hypo = item[1]["feat"][channel]
                    tgt_wo = item[2]["feat"][channel]
                    tgt_hypo = item[3]["feat"][channel]

            # 2. feature connection: mathematical transformation
            if self.feat_conn == "norm_concat":
                assert "cost" in self.feat_chan or "row" in self.feat_chan, \
                    "the value of `feat_chan` should contain `row` or `cost` when `norm_concat`."
                f = list(src_wo) + list(src_hypo) + list(tgt_wo) + list(tgt_hypo)
                f = (f - np.mean(f)) / np.std(f)
                feat.extend(f)
            if self.feat_conn == "pair_diff":
                assert self.plan_num == 2 or self.plan_num == 4, \
                    "the value of `plan_num` should be 2 or 4 when `pair_diff`."
                feat.extend(src_wo - src_hypo)
                if self.plan_num == 4:
                    feat.extend(tgt_wo - tgt_hypo)
            elif self.feat_conn == "pair_diff_ratio":
                assert self.plan_num == 2 or self.plan_num == 4, \
                    "the value of `plan_num` should be 2 or 4 when `pair_diff_ratio`."
                # : divided by zero, value clipping.
                feat.extend((src_wo - src_hypo) / src_wo)
                if self.plan_num == 4:
                    feat.extend((tgt_wo - tgt_hypo) / tgt_wo)
            elif self.feat_conn == "pair_diff_norm":
                assert self.plan_num == 2 or self.plan_num == 4, \
                    "the value of `plan_num` should be 2 or 4 when `pair_diff_norm`."
                feat.extend((src_wo - src_hypo) / (src_wo + src_hypo))
                if self.plan_num == 4:
                    feat.extend((tgt_wo - tgt_hypo) / (tgt_wo + tgt_hypo))
            else:  # plain concat
                feat.extend(list(src_wo) + list(src_hypo) + list(tgt_wo) + list(tgt_hypo))

        # 3. label generation
        if self.label_type == "ratio" or self.label_type == "log_ratio":
            assert self.plan_num == 2 or self.plan_num == 4, \
                "the value of `plan_num` should be 2 or 4 when `ratio`."
            label = item[1]["label"] / item[0]["label"]
            if self.plan_num == 4:
                label = (item[3]["label"] / item[2]["label"]) / label
        elif self.label_type == "diff_ratio" or self.label_type == "log_diff_ratio":
            assert self.plan_num == 2 or self.plan_num == 4, \
                "the value of `plan_num` should be 2 or 4 when `diff_ratio`."
            label = 1 - item[1]["label"] / item[0]["label"]
            if self.plan_num == 4:
                label = (1 - item[3]["label"] / item[2]["label"]) / label
        elif self.label_type == "cla":
            assert self.plan_num == 2 or self.plan_num == 4, \
                "the value of `plan_num` should be 2 or 4 when `cla`."
            label = 1 - item[1]["label"] / item[0]["label"]
            if self.plan_num == 4:
                if label == 0:
                    label = 0
                else:
                    after = 1 - item[3]["label"] / item[2]["label"]
                    label = 1 - after / label
            # : min_ratio?
            label = 1 if label > self.cla_min_ratio else 0
        elif self.label_type == "raw" or self.label_type == "log_raw":
            assert self.plan_num == 1, "the value of `plan_num` should be 1 when `raw`."
            label = item["label"]

        if self.feat_chan in ["seq_ind"]:
            data_type = np.int32
        else:
            data_type = np.float32

        return np.array(feat, dtype=data_type), np.array(label, dtype=np.float32)

    def __len__(self):
        return len(self.plan_data)
