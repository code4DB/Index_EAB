# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: get_opt_res
# @Author: Wei Zhou
# @Time: 2023/11/27 13:17

import torch
import numpy as np

import json
import configparser

from sklearn.metrics import f1_score

from index_advisor_selector.index_benefit_estimation.benefit_utils.openGauss_dbms import openGaussDatabaseConnector
from index_advisor_selector.index_candidate_generation.distill_model.distill_utils.distill_loss import QError


def get_gauss_res():
    for bench in ["tpch", "tpcds", "job"]:
        db_file = "/data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf"

        db_conf = configparser.ConfigParser()
        db_conf.read(db_file)

        db_conf["openGauss"]["host"] = "10.26.42.166"
        db_conf["openGauss"]["port"] = "5433"
        db_conf["openGauss"]["user"] = "postgres"
        db_conf["openGauss"]["password"] = "dmai4db2021."
        if bench == "job":
            db_conf["openGauss"]["database"] = "imdb_load103"
        else:
            db_conf["openGauss"]["database"] = f"{bench}_1gb103"

        database_connector = openGaussDatabaseConnector(db_conf, autocommit=True)

        data_load = f"/data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_{bench}_test.json"
        with open(data_load, "r") as rf:
            data = json.load(rf)

        plan_data = list()
        for dat in data:
            wo_plan = database_connector.get_ind_plan(dat["query"], "")
            w_plan = database_connector.get_ind_plan(dat["query"], dat["indexes"])

            dat["w/o plan openGauss"] = wo_plan
            dat["w/ plan openGauss"] = w_plan

            plan_data.append(dat)

        data_save = f"/data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_openGauss_{bench}_test.json"
        with open(data_save, "w") as wf:
            json.dump(plan_data, wf, indent=2)


def ana_gauss_res():
    for bench in ["tpch", "tpcds", "job"]:
        data_load = f"/data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_openGauss_{bench}_test.json"
        with open(data_load, "r") as rf:
            data = json.load(rf)

        y_pred = [1 - dat["w/ plan openGauss"]["Total Cost"] / dat["w/o plan openGauss"]["Total Cost"]
                  for dat in data if dat["label act"] != 0.
                  and (1 - dat["w/ plan openGauss"]["Total Cost"] / dat["w/o plan openGauss"]["Total Cost"]) > 0.]
        y_test = [dat["label act"] for dat in data if dat["label act"] != 0.
                  and (1 - dat["w/ plan openGauss"]["Total Cost"] / dat["w/o plan openGauss"]["Total Cost"]) > 0.]

        y_pred_f1 = list(map(int, np.array(y_pred) > 0.2))
        y_test_f1 = list(map(int, np.array(y_test) > 0.2))

        f1 = f1_score(y_pred_f1, y_test_f1)
        print(np.round(f1, 2))

        qerror = QError()(torch.from_numpy(np.array(y_pred)),
                          torch.from_numpy(np.array(y_test)),
                          out="raw", min_val=None, max_val=None)
        qerror = qerror.numpy()

        metric = [round(np.mean(qerror), 2),
                  round(np.median(qerror), 2),
                  round(np.quantile(qerror, 0.9), 2),
                  round(np.quantile(qerror, 0.95), 2),
                  round(np.quantile(qerror, 0.99), 2),
                  round(np.max(qerror), 2)]
        print("\t".join(map(str, metric)))

        print(1)


if __name__ == "__main__":
    # get_gauss_res()
    ana_gauss_res()
