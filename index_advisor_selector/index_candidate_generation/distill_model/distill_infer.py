# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: distill_infer
# @Author: Wei Zhou
# @Time: 2022/8/18 9:58

import json
import random
import time

import torch

import numpy as np
from sklearn.metrics import f1_score

from index_advisor_selector.index_candidate_generation.distill_model.distill_utils.distill_const import ops_dict
from index_advisor_selector.index_candidate_generation.distill_model.distill_utils import distill_com
from index_advisor_selector.index_candidate_generation.distill_model.distill_model import XGBoost, LightGBM, RandomForest
from index_advisor_selector.index_candidate_generation.distill_model.distill_utils.distill_loss import QError, xgb_QError, lgb_QError, cal_mape
from index_advisor_selector.index_candidate_generation.distill_model.distill_dataset import PlanPairDataset, unnormalize, normalize


def eval_model():
    # : 1. get the params.
    parser = distill_com.get_parser()
    args = parser.parse_args()

    # : 4. set the torch random_seed (consistent with args!!!).
    seed = 666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # : 1. Data preparation (unnormalized).
    # args.test_data_load = "/data1/wz/index/index_eab/eab_other/distill_model/data/tpch/tree_tpch_cost_data_tgt_test.json"

    # args.test_data_load = "/data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpch_test.json"
    # args.test_data_load = "/data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpcds_test.json"
    args.test_data_load = "/data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_job_test.json"

    with open(args.test_data_load, "r") as rf:
        data = json.load(rf)

    data = [dat for dat in data if dat["label act"] != 0.]

    test_set = PlanPairDataset(data)

    X_test = [sample[0] for sample in test_set]
    y_test = [sample[1] for sample in test_set]

    args.model_type = "PG"
    args.model_type = "XGBoost"
    if args.model_type == "PG":
        y_pred = [dat["label est"] for dat in data if dat["label act"] != 0.]
    else:
        # Scale features of X according to feature_range
        # if args.feat_chan in ["cost", "row", "cost_row"]:

        # args.model_type = "XGBoost"
        # args.scale_load = "/data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_xgb_job_round5k/data/train_scale_data.pt"

        # args.model_type = "LightGBM"
        # args.scale_load = "/data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_lgb_job_round5k/data/train_scale_data.pt"

        args.model_type = "RandomForest"
        args.scale_load = "/data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_rf_job_round5k/data/train_scale_data.pt"

        scaler = torch.load(args.scale_load)
        X_test = np.array(scaler.transform(X_test), dtype=np.float32)

        # args.model_type = "XGBoost"
        # args.model_load = "/data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_xgb_job_round5k/model/reg_xgb_cost.xgb.model"

        # args.model_type = "LightGBM"
        # args.model_load = "/data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_lgb_job_round5k/model/reg_job_cost.lgb.model"
        # args.model_load = "/data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_lgb_job_round5k/model/reg_lgb_cost.lgb.model"

        args.model_type = "RandomForest"
        args.model_load = "/data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_rf_job_round5k/model/reg_rf_cost.rf.joblib"

        if args.model_type == "XGBoost":
            model = XGBoost(path=args.model_load)
        elif args.model_type == "LightGBM":
            model = LightGBM(args.model_load)
        elif args.model_type == "RandomForest":
            model = RandomForest(path=args.model_load)

        time_start = time.time()

        y_pred = model.estimate(X_test)

        time_end = time.time()

        print(f"The time overhead ({len(X_test)}) is {time_end - time_start}.")

    qerror = QError()(torch.from_numpy(np.exp(np.array(y_pred))),
                      torch.from_numpy(np.array(y_test)),
                      out="raw", min_val=None, max_val=None)
    qerror = qerror.numpy()

    for threshold in [0.05, 0.2, 0.5]:
        if args.model_type == "PG":
            y_pred_f1 = list(map(int, np.array(y_pred) > threshold))
        else:
            y_pred_f1 = list(map(int, np.exp(np.array(y_pred)) > threshold))
        y_test_f1 = list(map(int, np.array(y_test) > threshold))

        f1 = f1_score(y_pred_f1, y_test_f1)
        print(np.round(f1, 2))

    # (1114): newly added.
    # for dat, pred, err in zip(data, y_pred, qerror):
    #     dat["y_pred"] = float(pred)
    #     dat["qerror"] = float(err)

    # data_save = args.test_data_load.replace(".json", "_res.json")
    # with open(data_save, "w") as wf:
    #     json.dump(data, wf, indent=2)

    metric = [round(np.mean(qerror), 2),
              round(np.median(qerror), 2),
              round(np.quantile(qerror, 0.9), 2),
              round(np.quantile(qerror, 0.95), 2),
              round(np.quantile(qerror, 0.99), 2),
              round(np.max(qerror), 2)]
    print("\t".join(map(str, metric)))
    #
    # print(f"The experimental result:\n"
    #       f"mean qerror: {round(np.mean(qerror), 4)}\n"
    #       f"median qerror: {round(np.median(qerror), 4)}\n"
    #       f"90th qerror: {round(np.quantile(qerror, 0.9), 4)}\n"
    #       f"95th qerror: {round(np.quantile(qerror, 0.95), 4)}\n"
    #       f"max qerror: {round(np.max(qerror), 4)}")


if __name__ == "__main__":
    eval_model()
