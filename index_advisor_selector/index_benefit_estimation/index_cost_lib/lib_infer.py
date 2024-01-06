# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: lib_infer
# @Author: Wei Zhou
# @Time: 2023/10/7 15:51

import os
import json
import logging
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import random_split, DataLoader

from sklearn.linear_model import LinearRegression

from index_advisor_selector.index_benefit_estimation.index_cost_lib.lib_train import get_parser
from index_advisor_selector.index_benefit_estimation.index_cost_lib.lib_data import collate_fn4lib
from index_advisor_selector.index_benefit_estimation.index_cost_lib.lib_model import make_model, self_attn_model, q_error, QError

from index_advisor_selector.index_benefit_estimation.benefit_utils.get_plan_info import tra_plan_ite
from index_advisor_selector.index_benefit_estimation.benefit_utils.benefit_const import ops_join_dict, ops_sort_dict, \
    ops_group_dict, ops_scan_dict

bench = "tpch"

# parser = get_parser()
# args = parser.parse_args()
#
# args.model_load = "/data/wz/index/index_eab/eab_benefit/index_cost_lib/exp_res/exp_lib_tpch_tgt_ep500_bat2048/model/lib_LIB_200.pt"

stats_load = f"/data/wz/index/index_eab/eab_benefit/index_cost_lib/data/db_stats_{bench}.json"


def load_model_lib():
    model_load = "/data/wz/index/index_eab/eab_benefit/index_cost_lib/exp_res/exp_lib_tpch_tgt_ep500_bat2048/model/lib_LIB_200.pt"

    encoder_model, pooling_model = make_model(32, 6, 128, 8, dropout=0.2)
    # encoder_model, pooling_model = make_model(args.dim1, args.n_encoder_layers,
    #                                           args.dim3, args.n_heads, dropout=args.dropout_r)

    model = self_attn_model(encoder_model, pooling_model, 12, 32, 64)
    # model = self_attn_model(encoder_model, pooling_model, args.input_dim, args.dim1, args.dim2)

    # if os.path.exists(args.model_load):
    #     checkpoint = torch.load(args.model_load, map_location="cpu")
    #     model.load_state_dict(checkpoint["model"])

    if os.path.exists(model_load):
        checkpoint = torch.load(model_load, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    return model


def get_lib_est_res(model, indexes, plan):
    # cuda environment is recommended
    # device = torch.device(f"cuda:{args.gpu_no}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    with open(stats_load, "r") as rf:
        stats = json.load(rf)

    nodes = tra_plan_ite(plan)

    index_ops = list()
    for ind in indexes:
        tbl, cols = ind.split("#")[0], ind.split("#")[1].split(",")

        for no, col in enumerate(cols):
            for node in nodes:
                if col in str(node["detail"]):
                    # 1. operation information (5)
                    # join, sort, group, scan_range, scan_equal
                    vec = [0 for _ in range(5)]
                    typ = node["type"]
                    if typ in ops_join_dict:
                        vec[0] = 1
                    elif typ in ops_sort_dict:
                        vec[1] = 1
                    elif typ in ops_group_dict:
                        vec[2] = 1
                    elif typ in ops_scan_dict:
                        # (1005): to be improved. columns with the same name.
                        if f"{col} =" in str(node["detail"]):
                            vec[3] = 1
                        else:
                            vec[4] = 1

                    # 2. database statistics (4)
                    card = np.log(node["detail"]["Plan Rows"])
                    row = np.log(stats[f"{tbl}.{col}"]["rows"])
                    null = stats[f"{tbl}.{col}"]["null"]
                    dist = stats[f"{tbl}.{col}"]["dist"]

                    vec.extend([card, row, null, dist])

                    # 3. index information (3)
                    if len(col) == 1:
                        vec.extend([1, 0, 0])
                    else:
                        vec.extend([0, 1, no + 1])

                    index_ops.append(vec)

    if len(index_ops) == 0:
        return [1.]

    test_data_pre = [[index_ops, 1.]]
    # test_loader = DataLoader(dataset=test_data_pre, batch_size=args.batch_size,
    #                          shuffle=True, collate_fn=collate_fn4lib, drop_last=False)

    test_loader = DataLoader(dataset=test_data_pre, batch_size=1024,
                             shuffle=True, collate_fn=collate_fn4lib, drop_last=False)

    model.to(device)

    model.eval()
    total_pred_rr = list()
    pro_bar = enumerate(test_loader)
    for bi, batch in pro_bar:
        pad_data, mask, label = batch
        pad_data, mask, label = pad_data.to(device), mask.to(device), label.to(device)

        pred_rr = model(pad_data.to(torch.float32), mask).flatten().detach().cpu().numpy()
        total_pred_rr.extend(pred_rr)

    y_pred = total_pred_rr

    return y_pred


def infer(args):
    # cuda environment is recommended
    device = torch.device(f"cuda:{args.gpu_no}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    with open(args.test_data_file, "r") as rf:
        test_data = json.load(rf)

    # min(1, item["w/ actual cost"] / item["w/o actual cost"])
    # test_data_pre = [[item["feat"], item["w/ actual cost"] / item["w/o actual cost"]] for item in test_data
    #                  if len(item["feat"]) > 0 and item["w/ actual cost"] / item["w/o actual cost"] <= 1]

    test_data_pre = [[item["feat"], item["w/ actual cost"] / item["w/o actual cost"]] for item in test_data]

    logging.info(f"Load the test data from `{args.test_data_file}` ({len(test_data)}).")

    test_loader = DataLoader(dataset=test_data_pre, batch_size=1024,  # args.batch_size
                             shuffle=True, collate_fn=collate_fn4lib, drop_last=False)

    encoder_model, pooling_model = make_model(args.dim1, args.n_encoder_layers,
                                              args.dim3, args.n_heads, dropout=args.dropout_r)
    model = self_attn_model(encoder_model, pooling_model, args.input_dim, args.dim1, args.dim2)

    # args.model_load = "/data/wz/index/index_eab/eab_benefit/index_cost_lib/exp_res/exp_lib_tpch_tgt_ep500_bat2048/model/lib_LIB_200.pt"
    if os.path.exists(args.model_load):
        checkpoint = torch.load(args.model_load, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    model.to(device)

    model.eval()
    total_pred_rr = list()
    pro_bar = tqdm(enumerate(test_loader))

    time_start = time.time()

    for bi, batch in pro_bar:
        pad_data, mask, label = batch
        pad_data, mask, label = pad_data.to(device), mask.to(device), label.to(device)

        pred_rr = model(pad_data, mask).flatten().detach().cpu().numpy()
        total_pred_rr.extend(pred_rr)

    time_end = time.time()

    print(f"The time overhead ({len(total_pred_rr)}) is {time_end - time_start}.")

    # y_pred = np.array(total_pred_rr) * np.array([item["w/o actual cost"] for item in test_data])
    y_pred = np.array(total_pred_rr) * np.array([item["w/o estimated cost"] for item in test_data])
    y_test = np.array([item["w/ actual cost"] for item in test_data])

    # linear regression
    model = LinearRegression()
    model.fit(np.log(y_pred).reshape(-1, 1), np.log(y_test).reshape(-1, 1))
    y_pred = model.predict(np.log(y_pred).reshape(-1, 1))

    qerror = QError()(torch.tensor(np.exp(y_pred)), torch.tensor(y_test.reshape(-1, 1)), out="raw")
    # qerror = QError()(torch.tensor(y_pred), torch.tensor(y_test), out="raw")

    qerror = qerror.numpy()

    # (1114): newly added.
    # for dat, pred, err in zip(test_data, y_pred, qerror):
    #     dat["y_pred"] = float(pred)
    #     dat["qerror"] = float(err)
    #
    # data_save = args.test_data_file.replace(".json", "_res.json")
    # with open(data_save, "w") as wf:
    #     json.dump(test_data, wf, indent=2)

    print(f"The experimental result:\n"
          f"mean qerror: {round(np.mean(qerror), 4)}\n"
          f"median qerror: {round(np.median(qerror), 4)}\n"
          f"90th qerror: {round(np.quantile(qerror, 0.9), 4)}\n"
          f"95th qerror: {round(np.quantile(qerror, 0.95), 4)}\n"
          f"max qerror: {round(np.max(qerror), 4)}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Setting random seed to facilitate reproduction
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    infer(args)
