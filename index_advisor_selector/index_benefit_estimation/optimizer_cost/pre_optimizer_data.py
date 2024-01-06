# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: pre_optimizer_data
# @Author: Wei Zhou
# @Time: 2023/10/6 20:41

import os
import json
import configparser
from tqdm import tqdm

from index_advisor_selector.index_benefit_estimation.benefit_utils.postgres_dbms import PostgresDatabaseConnector
from index_advisor_selector.index_benefit_estimation.benefit_utils.openGauss_dbms import openGaussDatabaseConnector


def get_optimizer_cost(dbms="postgresql"):
    benchmarks = ["tpch", "tpcds", "job"]
    for bench in tqdm(benchmarks):
        if bench == "job":
            db_conf_file = f"/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_{bench}.conf"
        else:
            db_conf_file = f"/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_{bench}_1gb.conf"
        db_conf = configparser.ConfigParser()
        db_conf.read(db_conf_file)

        if dbms == "openGauss":
            db_conf["openGauss"]["host"] = "xx.xx.xx.xx"
            if bench == "job":
                db_conf["openGauss"]["database"] = f"imdb_load103"
            else:
                db_conf["openGauss"]["database"] = f"{bench}_1gb103"
            db_conf["openGauss"]["port"] = "xxxx"
            db_conf["openGauss"]["user"] = "xxxx"
            db_conf["openGauss"]["password"] = "xxxx"

        for fid in ["src", "tgt"]:
            for did in ["train", "valid", "test"]:
                if dbms == "postgresql":
                    connector = PostgresDatabaseConnector(db_conf, autocommit=True)
                elif dbms == "openGauss":
                    connector = openGaussDatabaseConnector(db_conf, autocommit=True)

                data_load = f"/data/wz/index/index_eab/eab_benefit/cost_data/{bench}/{bench}_cost_data_{fid}_{did}.json"
                with open(data_load, "r") as rf:
                    data = json.load(rf)

                total_data = list()
                for item in data:
                    wo_plan = connector.get_ind_plan(item["sql"], "")
                    w_plan = connector.get_ind_plan(item["sql"], item["indexes"])

                    total_data.append({"sql": item["sql"], "indexes": item["indexes"],
                                       "w/o estimated cost": wo_plan["Total Cost"], "w/ estimated cost": w_plan["Total Cost"],
                                       "w/o actual cost": item["w/o actual cost"], "w/ actual cost": item["w/ actual cost"],
                                       "w/o plan": wo_plan, "w/ plan": w_plan})

                data_save = f"/data/wz/index/index_eab/eab_benefit/optimizer_cost/data/" \
                            f"{bench}/{dbms}_{bench}_cost_data_{fid}_{did}.json"
                if not os.path.exists(os.path.dirname(data_save)):
                    os.makedirs(os.path.dirname(data_save))
                with open(data_save, "w") as wf:
                    json.dump(total_data, wf, indent=2)


if __name__ == "__main__":
    # dbms = "PG"
    # get_optimizer_cost(dbms=dbms)

    dbms = "openGauss"
    get_optimizer_cost(dbms=dbms)
