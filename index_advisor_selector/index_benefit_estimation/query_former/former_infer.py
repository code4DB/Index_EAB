# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: former_infer
# @Author: Wei Zhou
# @Time: 2023/10/7 11:06

import os
import json
import copy
from tqdm import tqdm

import torch
import numpy as np

from index_advisor_selector.index_benefit_estimation.query_former.former_train import get_parser
from index_advisor_selector.index_benefit_estimation.query_former.model.model import QueryFormer

from index_advisor_selector.index_benefit_estimation.query_former.model.dataset import PlanTreeDataset
from index_advisor_selector.index_benefit_estimation.query_former.model.trainer import train, evaluate
from index_advisor_selector.index_benefit_estimation.query_former.model.database_util import get_hist_file, get_job_table_sample, collator
from index_advisor_selector.index_benefit_estimation.benefit_utils.benefit_const import alias2table_tpch, alias2table_job, alias2table_tpcds


# (1007): to be removed.
# parser = get_parser()
# args = parser.parse_args()
#
# args.test_data_file = "/data/wz/index/index_eab/eab_benefit/cost_data/tpch/tpch_cost_data_tgt_test.json"
# args.model_load = "/data/wz/index/index_eab/eab_benefit/query_former/exp_res/exp_former_tpch_tgt_ep500_bat1024/model/former_FORMER_200.pt"
#
# args.encoding_load = "/data/wz/index/index_eab/eab_benefit/query_former/data/tpch/encoding_tpch_v2.pt"
# args.cost_norm_load = "/data/wz/index/index_eab/eab_benefit/query_former/data/tpch/cost_norm_tpch_v2.pt"


class Args:
    pass


def load_model_former(model_load, encoding_load, cost_norm_load):
    # device = torch.device(f"cuda:{args.gpu_no}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # model_load = "/data/wz/index/index_eab/eab_benefit/query_former/exp_res/exp_former_tpch_tgt_ep500_bat1024/model/former_FORMER_200.pt"
    # encoding_load = "/data/wz/index/index_eab/eab_benefit/query_former/data/tpch/encoding_tpch_v2.pt"
    # cost_norm_load = "/data/wz/index/index_eab/eab_benefit/query_former/data/tpch/cost_norm_tpch_v2.pt"

    encoding = torch.load(encoding_load)
    cost_norm = torch.load(cost_norm_load)

    model = QueryFormer(emb_size=64, ffn_dim=128, head_size=12,
                        dropout=0.1, n_layers=8, encoding=encoding,
                        use_sample=False, use_hist=False, pred_hid=128)
    if os.path.exists(model_load):
        # "checkpoints/cost_model.pt"
        checkpoint = torch.load(model_load, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    # encoding = torch.load(args.encoding_load)
    # cost_norm = torch.load(args.cost_norm_load)
    #
    # model = QueryFormer(emb_size=args.embed_size, ffn_dim=args.ffn_dim, head_size=args.head_size,
    #                     dropout=args.dropout, n_layers=args.n_layers, encoding=encoding,
    #                     use_sample=args.use_sample, use_hist=args.use_hist, pred_hid=args.pred_hid)
    # if os.path.exists(args.model_load):
    #     # "checkpoints/cost_model.pt"
    #     checkpoint = torch.load(args.model_load, map_location="cpu")
    #     model.load_state_dict(checkpoint["model"])

    model.to(device)

    # methods = {
    #     "get_sample": get_job_table_sample,
    #     "encoding": encoding,
    #     "cost_norm": cost_norm,
    #     "hist_file": None,
    #     "model": model,
    #     "device": device,
    #     "bs": args.batch_size,
    # }

    methods = {
        "get_sample": get_job_table_sample,
        "encoding": encoding,
        "cost_norm": cost_norm,
        "hist_file": None,
        "model": model,
        "device": device,
        "bs": 1024,
    }

    return methods


def get_former_est_res(methods, data):
    # device = torch.device(f"cuda:{args.gpu_no}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    test_data_file = "/data/wz/index/index_eab/eab_benefit/cost_data/tpch/tpch_cost_data_tgt_test.json"

    model = methods["model"]

    # if "tpch" in args.test_data_file:
    #     alias2tbl = alias2table_tpch
    # elif "tpcds" in args.test_data_file:
    #     alias2tbl = alias2table_tpcds
    # elif "job" in args.test_data_file:
    #     alias2tbl = alias2table_job

    if "tpch" in test_data_file:
        alias2tbl = alias2table_tpch
    elif "tpcds" in test_data_file:
        alias2tbl = alias2table_tpcds
    elif "job" in test_data_file:
        alias2tbl = alias2table_job

    ds = PlanTreeDataset(data, None,
                         methods["encoding"], methods["hist_file"], methods["cost_norm"],
                         methods["cost_norm"], "cost", None, alias2tbl=alias2tbl)

    model.eval()
    cost_predss = np.empty(0)
    with torch.no_grad():
        for i in range(0, len(ds), methods["bs"]):
            batch, batch_labels = collator(list(zip(*[ds[j] for j in range(i, min(i + methods["bs"], len(ds)))])))

            batch = batch.to(device)

            cost_preds, _ = model(batch)
            cost_preds = cost_preds.squeeze()

            cost_predss = np.append(cost_predss, cost_preds.cpu().detach().numpy())

        cost_predss = methods["cost_norm"].unnormalize_labels(cost_predss)

    return cost_predss


def eval_work(methods, device):
    # args.test_data_file = "/data/wz/index/index_eab/eab_benefit/cost_data/tpch/tpch_cost_data_tgt_test.json"
    with open(args.test_data_file, "r") as rf:
        test_data = json.load(rf)

    test_data_copy = copy.deepcopy(test_data)

    if "tpch" in args.test_data_file:
        alias2tbl = alias2table_tpch
    elif "tpcds" in args.test_data_file:
        alias2tbl = alias2table_tpcds
    elif "job" in args.test_data_file:
        alias2tbl = alias2table_job

    ds = PlanTreeDataset(test_data, None,
                         methods["encoding"], methods["hist_file"], methods["cost_norm"],
                         methods["cost_norm"], "cost", None, alias2tbl=alias2tbl)

    # args.model_load = "/data/wz/index/index_eab/eab_benefit/query_former/exp_res/" \
    #                   "exp_former_tpch_tgt_ep500_bat1024/model/former_FORMER_200.pt"
    # if os.path.exists(args.model_load):
    #     # "checkpoints/cost_model.pt"
    #     checkpoint = torch.load(args.model_load, map_location="cpu")
    #     methods["model"].load_state_dict(checkpoint["model"])
    methods["model"].to(device)

    y_pred, qerror, eval_score, corr = evaluate(methods["model"], ds, methods["bs"], methods["cost_norm"],
                                                methods["device"], True)

    # (1114): newly added.
    # for dat, pred, err in zip(test_data_copy, y_pred, qerror):
    #     dat["y_pred"] = float(pred)
    #     dat["qerror"] = float(err)
    #
    # data_save = args.test_data_file.replace(".json", "_res.json")
    # with open(data_save, "w") as wf:
    #     json.dump(test_data_copy, wf, indent=2)

    return eval_score


def get_eval_res(args):
    device = torch.device(f"cuda:{args.gpu_no}" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")

    methods = load_model_former(args.model_load, args.encoding_load, args.cost_norm_load)
    # methods = pre_est_obj(args)

    eval_score = eval_work(methods, device)

    print(eval_score)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    get_eval_res(args)
