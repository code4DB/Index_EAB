# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: distill_dataset
# @Author: Wei Zhou
# @Time: 2022/6/26 13:16

import numpy as np

from torch.utils.data import Dataset, random_split, DataLoader

from index_advisor_selector.index_candidate_generation.distill_model.distill_utils.distill_const import ops_dict, rule_dict, ind_ops_dict


def normalize(x, min_card_log, max_card_log):
    return np.maximum(np.minimum((np.log(x) - min_card_log) / (max_card_log - min_card_log), 1.0), 0.0)


def unnormalize(x, min_card_log, max_card_log):
    return np.exp(x * (max_card_log - min_card_log) + min_card_log)


class PlanPairDataset(Dataset):
    def __init__(self, plan_data):
        self.plan_data = plan_data

    def __getitem__(self, index):
        item = self.plan_data[index]

        # ["utility", "query_shape", "index_shape", "physical_operator"]
        feat = list()

        # 1. utility (1)
        feat.append(item["feat"]["utility"])

        # 2. query_shape (6 * 7)
        max_tbl, max_depth = 6, 7

        shape_vec = list()
        for shape in item["feat"]["query_shape"]:
            vec = list()
            for op in shape:
                vec.append(ops_dict.get(op, -1) + 1)
            vec = vec + [0] * max(0, (max_depth - len(shape)))
            vec = vec[:max_depth]

            shape_vec.extend(vec)

        shape_vec = shape_vec + [0 for _ in range(max_depth)] * max(0, (max_tbl * max_depth - len(shape_vec)))
        shape_vec = shape_vec[:max_tbl * max_depth]

        feat.extend(shape_vec)

        # 3. index_shape (1 * 2)
        max_col = 2

        shape_vec = list()
        for op in item["feat"]["index_shape"]:
            shape_vec.append(rule_dict.get(op, 0) + 1)
        shape_vec = shape_vec + [0] * max(0, (max_col - len(item["feat"]["index_shape"])))
        shape_vec = shape_vec[:max_col]

        feat.extend(shape_vec)

        # 4. physical_operator (10)
        ops_vec = [0 for _ in ind_ops_dict.keys()]
        for ops in item["feat"]["physical_operator"].keys():
            if ops in ind_ops_dict.keys():
                ops_vec[ind_ops_dict[ops]] = np.max(item["feat"]["physical_operator"][ops])

        feat.extend(ops_vec)

        label = item["label act"]

        # data_type = np.float32  # np.int32, np.float32

        # return np.array(feat, dtype=data_type), np.array(label, dtype=np.float32)
        return feat, label

    def __len__(self):
        return len(self.plan_data)
