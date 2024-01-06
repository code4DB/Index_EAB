# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: tree_cost_infer
# @Author: Wei Zhou
# @Time: 2022/8/18 9:58

import json
import random
import time

import torch
import numpy as np

from index_advisor_selector.index_benefit_estimation.benefit_utils.benefit_const import ops_dict
from index_advisor_selector.index_benefit_estimation.tree_model.pre_tree_data import get_plan_info
from index_advisor_selector.index_benefit_estimation.tree_model.tree_cost_utils import tree_cost_com
from index_advisor_selector.index_benefit_estimation.tree_model.tree_cost_model import XGBoost, LightGBM, RandomForest
from index_advisor_selector.index_benefit_estimation.tree_model.tree_cost_utils.tree_cost_loss import QError, xgb_QError, lgb_QError, cal_mape
from index_advisor_selector.index_benefit_estimation.tree_model.tree_cost_dataset import PlanPairDataset, unnormalize, normalize

# tree_parser = tree_cost_com.get_parser()
# tree_args = tree_parser.parse_args()

# tree_args.model_type = "XGBoost"
# tree_args.feat_chan = "cost_row"
#
# tree_args.model_load = "/data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_tpch_tgt_round5k/model/reg_xgb_cost.xgb.model"
# tree_args.scale_load = "/data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_tpch_tgt_round5k/data/train_scale_data.pt"


def load_model_tree():
    model_type = "XGBoost"
    model_load = "/data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_tpch_tgt_round5k/model/reg_xgb_cost.xgb.model"

    if model_type == "XGBoost":
        model = XGBoost(path=model_load)
    elif model_type == "LightGBM":
        model = LightGBM(model_load)
    elif model_type == "RandomForest":
        model = RandomForest(model_load)

    # if tree_args.model_type == "XGBoost":
    #     model = XGBoost(path=tree_args.model_load)
    # elif tree_args.model_type == "LightGBM":
    #     model = LightGBM(tree_args.model_load)
    # elif tree_args.model_type == "RandomForest":
    #     model = RandomForest(tree_args.model_load)

    return model


def get_tree_est_res(model, plan):
    feat_chan = "cost_row"
    scale_load = "/data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_tpch_tgt_round5k/data/train_scale_data.pt"

    p_info = get_plan_info(plan)

    p_cost_feat = [p_info["node_cost_sum"][key] if key in p_info["node_cost_sum"] else 0 for key in ops_dict.keys()]
    p_row_feat = [p_info["node_row_sum"][key] if key in p_info["node_row_sum"] else 0 for key in ops_dict.keys()]
    p_wcost_feat = [p_info["node_cost_wsum"][key] if key in p_info["node_cost_wsum"] else 0 for key in ops_dict.keys()]
    p_wrow_feat = [p_info["node_row_wsum"][key] if key in p_info["node_row_wsum"] else 0 for key in ops_dict.keys()]

    data = [{"feat": {"p_cost_feat": p_cost_feat, "p_row_feat": p_row_feat,
                      "p_wcost_feat": p_wcost_feat, "p_wrow_feat": p_wrow_feat},
             "label": 666, "info": p_info}]

    # test_set = PlanPairDataset(data, plan_num=tree_args.plan_num, feat_chan=tree_args.feat_chan, feat_conn=tree_args.feat_conn,
    #                            label_type=tree_args.label_type, cla_min_ratio=tree_args.cla_min_ratio)

    test_set = PlanPairDataset(data, plan_num=1, feat_chan=feat_chan,
                               feat_conn="concat", label_type="raw", cla_min_ratio=0.2)

    X_test = [sample[0] for sample in test_set]

    # Scale features of X according to feature_range
    # if tree_args.feat_chan in ["cost", "row", "cost_row"]:
    #     scaler = torch.load(tree_args.scale_load)
    #     X_test = np.array(scaler.transform(X_test), dtype=np.float32)
    if feat_chan in ["cost", "row", "cost_row"]:
        scaler = torch.load(scale_load)
        X_test = np.array(scaler.transform(X_test), dtype=np.float32)

    y_pred = model.estimate(X_test)

    return y_pred


def eval_model():
    # : 1. get the params.
    parser = tree_cost_com.get_parser()
    args = parser.parse_args()

    # : 4. set the torch random_seed (consistent with args!!!).
    seed = 666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # : 1. Data preparation (unnormalized).
    # args.test_data_load = "/data/wz/index/index_eab/eab_benefit/tree_model/data/tpch/tree_tpch_cost_data_tgt_test.json"
    with open(args.test_data_load, "r") as rf:
        data = json.load(rf)

    test_set = PlanPairDataset(data, plan_num=args.plan_num, feat_chan=args.feat_chan, feat_conn=args.feat_conn,
                               label_type=args.label_type, cla_min_ratio=args.cla_min_ratio)

    X_test = [sample[0] for sample in test_set]
    y_test = [sample[1] for sample in test_set]

    # Scale features of X according to feature_range
    if args.feat_chan in ["cost", "row", "cost_row"]:
        # args.scale_load = "/data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_tpch_tgt_round5k/data/train_scale_data.pt"
        scaler = torch.load(args.scale_load)
        X_test = np.array(scaler.transform(X_test), dtype=np.float32)

    # args.model_type = "XGBoost"
    # args.model_load = "/data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_tpch_tgt_round5k/model/reg_xgb_cost.xgb.model"

    if args.model_type == "XGBoost":
        model = XGBoost(path=args.model_load)
    elif args.model_type == "LightGBM":
        model = LightGBM(args.model_load)
    elif args.model_type == "RandomForest":
        model = RandomForest(args.model_load)

    time_start = time.time()

    y_pred = model.estimate(X_test)

    time_end = time.time()

    print(f"The time overhead ({len(X_test)}) is {time_end - time_start}.")

    qerror = QError()(torch.from_numpy(np.exp(np.array(y_pred))),
                      torch.from_numpy(np.array(y_test)),
                      out="raw", min_val=None, max_val=None)
    qerror = qerror.numpy()

    # (1114): newly added.
    # for dat, pred, err in zip(data, y_pred, qerror):
    #     dat["y_pred"] = float(pred)
    #     dat["qerror"] = float(err)

    # data_save = args.test_data_load.replace(".json", "_res.json")
    # with open(data_save, "w") as wf:
    #     json.dump(data, wf, indent=2)

    # metric = [torch.mean(qerror).item(), torch.median(qerror).item(), torch.quantile(qerror, 0.9).item(),
    #           torch.quantile(qerror, 0.95).item(), torch.quantile(qerror, 0.99).item(), torch.max(qerror).item()]
    # print("\t".join(map(str, metric)))

    print(f"The experimental result:\n"
          f"mean qerror: {round(np.mean(qerror), 4)}\n"
          f"median qerror: {round(np.median(qerror), 4)}\n"
          f"90th qerror: {round(np.quantile(qerror, 0.9), 4)}\n"
          f"95th qerror: {round(np.quantile(qerror, 0.95), 4)}\n"
          f"max qerror: {round(np.max(qerror), 4)}")


if __name__ == "__main__":
    eval_model()
