# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: get_db_stats
# @Author: Wei Zhou
# @Time: 2023/10/5 16:15

import os
import json
from decimal import Decimal

from tqdm import tqdm
import configparser

from index_advisor_selector.index_benefit_estimation.benefit_utils.postgres_dbms import PostgresDatabaseConnector


class IndexEncoder(json.JSONEncoder):
    def default(self, obj):
        # 👇️ if passed in object is instance of Decimal
        # convert it to a float
        if isinstance(obj, Decimal):
            return float(obj)

        # 👇️ otherwise use the default behavior
        return json.JSONEncoder.default(self, obj)


def get_row_null_distinct(db_conf):
    connector = PostgresDatabaseConnector(db_conf, autocommit=True)

    tables = connector.get_tables()
    rows = connector.get_row_count(tables)

    tbl_info, col_info = dict(), dict()
    for tbl, row in tqdm(zip(tables, rows)):
        columns = connector.get_cols(tbl)
        tbl_info[tbl] = {"columns": columns, "rows": row[-1]}

        null = connector.get_null_frac(tbl, columns)
        dist = connector.get_dist_frac(tbl, columns)

        for col, n, d in zip(columns, null, dist):
            col_info[f"{tbl}.{col}"] = {"rows": row[-1], "null": n, "dist": d}

    return col_info


if __name__ == "__main__":
    benchmarks = ["tpch"]  # "tpch", "tpcds", "job"
    for bench in benchmarks:
        if bench == "job":
            db_conf_file = f"/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_{bench}.conf"
        else:
            db_conf_file = f"/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_{bench}_1gb.conf"
        db_conf = configparser.ConfigParser()
        db_conf.read(db_conf_file)

        # db_conf["postgresql"]["host"] = "10.26.42.166"
        # db_conf["postgresql"]["database"] = "tpch_1gb103_skew"
        # db_conf["postgresql"]["port"] = "5432"
        # db_conf["postgresql"]["user"] = "wz"
        # db_conf["postgresql"]["password"] = "ai4db2021"

        col_info = get_row_null_distinct(db_conf)

        data_save = f"/data/wz/index/index_eab/eab_benefit/index_cost_lib/data/db_stats_{bench}.json"
        if not os.path.exists(os.path.dirname(data_save)):
            os.makedirs(os.path.dirname(data_save))
        with open(data_save, "w") as wf:
            json.dump(col_info, wf, indent=2, cls=IndexEncoder)
