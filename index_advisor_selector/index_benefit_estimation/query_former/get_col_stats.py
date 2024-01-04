# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: get_col_stats
# @Author: Wei Zhou
# @Time: 2023/10/5 20:29

import os
import json

import torch
import pandas as pd
from tqdm import tqdm

import configparser

from index_advisor_selector.index_benefit_estimation.benefit_utils.benefit_const import ops_dict
from index_advisor_selector.index_benefit_estimation.query_former.model.database_util import Encoding
from index_advisor_selector.index_selection.heu_selection.heu_utils.postgres_dbms import PostgresDatabaseConnector


def pre_min_max(db_conf):
    connector = PostgresDatabaseConnector(db_conf, autocommit=True)

    tables = connector.get_tables()
    rows = connector.get_row_count(tables)

    tbl_info = dict()
    col_info = {"name": list(), "min": list(), "max": list(),
                "cardinality": list(), "num_unique_values": list()}
    for tbl, row in tqdm(zip(tables, rows)):
        columns = connector.get_col_name_types(tbl)
        tbl_info[tbl] = {"columns": columns, "rows": row[-1]}

        col_names = [col[0] for col in columns]

        mins = connector.get_min_vals(tbl, col_names)
        maxs = connector.get_max_vals(tbl, col_names)
        card = row[-1]
        num_unique = connector.get_num_uniques(tbl, col_names)

        for col, mi, ma, uni in zip(columns, mins, maxs, num_unique):
            col_info["name"].append(f"{tbl}.{col[0]}")
            # todo(1005): to be improved.
            if col[1] not in ["integer", "int", "bigint", "smallint", "numeric", "decimal", "real", "double precision"]:
                col_info["min"].append(0)
                col_info["max"].append(0)
            else:
                col_info["min"].append(mi)
                col_info["max"].append(ma)
            col_info["cardinality"].append(card)
            col_info["num_unique_values"].append(uni)

    data_df = pd.DataFrame(col_info)

    return data_df


def pre_encoding_obj(db_conf):
    connector = PostgresDatabaseConnector(db_conf, autocommit=True)

    tables = connector.get_tables()
    columns = list()
    for tbl in tables:
        columns.extend([f"{tbl}.{col}" for col in connector.get_cols(tbl)])
    col2idx = {"NA": 0}
    col2idx.update({col: no + 1 for no, col in enumerate(columns)})

    min_max_info = pre_min_max(db_conf)
    column_min_max_vals = dict()
    for col, mi, ma in zip(min_max_info["name"], min_max_info["min"], min_max_info["max"]):
        column_min_max_vals[col] = [mi, ma]

    op2idx = {">": 0, "=": 1, "<": 2, ">=": 3, "<=": 4, "NA": 5}
    encoding = Encoding(column_min_max_vals, col2idx, op2idx=op2idx)

    for tbl in tables:
        encoding.encode_table(tbl)

    joins = connector.get_join_relations()
    encoding.encode_join(None)
    for join in joins:
        join = f"{join[1]}.{join[2]} = {join[3]}.{join[4]}"
        encoding.encode_join(join)

    for typ in ops_dict.keys():
        encoding.encode_type(typ)

    return encoding


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

        # data_df = pre_min_max_file(db_conf)
        # data_save = f"/data/wz/index/index_eab/eab_benefit/query_former/data/{bench}/column_min_max_vals_{bench}.csv"
        # if not os.path.exists(os.path.dirname(data_save)):
        #     os.makedirs(os.path.dirname(data_save))
        # data_df.to_csv(data_save)

        encoding = pre_encoding_obj(db_conf)
        data_save = f"/data/wz/index/index_eab/eab_benefit/query_former/data/{bench}/encoding_{bench}.pt"
        if not os.path.exists(os.path.dirname(data_save)):
            os.makedirs(os.path.dirname(data_save))
        torch.save(encoding, data_save)
