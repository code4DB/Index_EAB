# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: gen_workload
# @Author: Wei Zhou
# @Time: 2023/10/3 15:32

import os
import json
import random

from workload_generator.perturbation_based.perturb_utils.mod_sql import random_gen, vec2sql
from workload_generator.perturbation_based.perturb_utils.sql_token import tokenize_sql


def random_perturb(bench, sql_list, seed=666, mode="all", max_diff=10, perturb_prop=0.5, is_check=False):
    word2idx_json = f"/data/wz/index/index_eab/eab_data/db_info_vocab/word2idx_{bench}.json"
    idx2word_json = f"/data/wz/index/index_eab/eab_data/db_info_vocab/idx2word_{bench}.json"
    wordinfo_json = f"/data/wz/index/index_eab/eab_data/db_info_vocab/wordinfo_{bench}.json"
    colinfo_json = f"/data/wz/index/index_eab/eab_data/db_info_conf/colinfo_{bench}.json"

    with open(word2idx_json, "r") as rf:
        word2idx = json.load(rf)
    with open(idx2word_json, "r") as rf:
        idx2word = json.load(rf)
    with open(wordinfo_json, "r") as rf:
        word_info = json.load(rf)
    with open(colinfo_json, "r") as rf:
        col_info = json.load(rf)

    tbls = list(set([col_info[col]["table"] for col in col_info.keys()]))
    cols = col_info.keys()

    # 1. tokenize the input sql
    sql_tokens, except_tokens = tokenize_sql(sql_list, tbls, cols, word2idx)
    assert len(except_tokens) == 0, "Error occurs when performing perturbations!"

    # 2. conduct perturbation over the input sql
    valid_tokens, except_tokens, sql_vecs = random_gen(sql_tokens, word2idx, idx2word, word_info,
                                                       col_info, mode=mode, max_diff=max_diff,
                                                       perturb_prop=perturb_prop,
                                                       seed=seed, is_check=is_check)

    assert len(except_tokens) == 0, "Error occurs when performing perturbations!"

    sql_res_perted = vec2sql(valid_tokens, sql_vecs, idx2word, col_info, mode="without_table")

    return sql_res_perted


def pre_query_drift_group(bench, workload, temp_num=10, seed=666):
    random.seed(seed)

    query_temp = dict()
    for no, query in enumerate(workload):
        if isinstance(query, list):
            query_temp[no + 1] = [query[1]]
        else:
            query_temp[no + 1] = [query]

        for _ in range(temp_num):
            mode = random.choice(["all", "column", "value"])
            max_diff = random.randint(3, 8)
            perturb_prop = random.uniform(0.3, 0.8)

            if isinstance(query, list):
                pert_res = random_perturb(bench, [query[1]], mode=mode, seed=random.randint(0, seed),
                                          max_diff=max_diff, perturb_prop=perturb_prop)
            else:
                pert_res = random_perturb(bench, [query], mode=mode, seed=random.randint(0, seed),
                                          max_diff=max_diff, perturb_prop=perturb_prop)

            query_temp[no + 1].append(pert_res[0]["sql_text"])

    return query_temp


def pre_drift_data(bench, workload, temp_num, work_size, work_num, eval_num, test_num, drift_typ, seed=666):
    random.seed(seed)

    query_temp = pre_query_drift_group(bench, workload, temp_num)

    work_list = list()
    while len(work_list) < work_num + eval_num + test_num:
        sql_list = list()
        if drift_typ == "drift_unique" or len(work_list) >= work_num:
            for cls in query_temp.keys():
                # 1. determine the instance
                sql = random.choice(query_temp[cls])

                # 2. determine the frequency
                freq = random.randint(1, 1000)

                tup = [cls, sql, freq]
                sql_list.append(tup)
        else:
            while len(sql_list) < work_size:
                # 1. determine the class
                cls = random.sample(query_temp.keys(), 1)[0]

                # 2. determine the instance
                sql = random.choice(query_temp[cls])

                # 3. determine the frequency
                freq = random.randint(1, 1000)

                tup = [int(cls), sql, freq]
                sql_list.append(tup)

            sql_list = sorted(sql_list, key=lambda x: int(x[0]))

        work_list.append(sql_list)

    assert len(set([str(work) for work in work_list])) == work_num + eval_num + test_num, "Duplicate workload exists!"

    data_save = f"/data/wz/index/index_eab/eab_olap/bench_drift/{bench}_v2/" \
                f"{bench}_work_{drift_typ}_multi_w{work_size}_n{work_num}.json"
    if not os.path.exists(os.path.dirname(data_save)):
        os.makedirs(os.path.dirname(data_save))
    with open(data_save, "w") as wf:
        json.dump(work_list[:work_num], wf, indent=2)

    data_save = f"/data/wz/index/index_eab/eab_olap/bench_drift/{bench}_v2/" \
                f"{bench}_work_drift_multi_w{work_size}_n{eval_num}_eval.json"
    if not os.path.exists(os.path.dirname(data_save)):
        os.makedirs(os.path.dirname(data_save))
    with open(data_save, "w") as wf:
        json.dump(work_list[work_num:work_num + eval_num], wf, indent=2)

    data_save = f"/data/wz/index/index_eab/eab_olap/bench_drift/{bench}_v2/" \
                f"{bench}_work_drift_multi_w{work_size}_n{test_num}_test.json"
    if not os.path.exists(os.path.dirname(data_save)):
        os.makedirs(os.path.dirname(data_save))
    with open(data_save, "w") as wf:
        json.dump(work_list[work_num + eval_num:work_num + eval_num + test_num], wf, indent=2)


if __name__ == "__main__":
    work_num = {"tpch": 1000, "tpch_skew": 1000,
                "tpcds": 5000, "dsb": 3000,
                "job": 3000}

    work_size = {"tpch": 18, "tpch_skew": 18,
                 "tpcds": 79, "dsb": 53,
                 "job": 33}

    seed = 666

    temp_num = 100
    eval_num, test_num = 10, 100
    drift_typ = "drift_duplicate"  # drift_unique, drift_duplicate

    benchmarks = ["tpch", "tpcds", "job", "tpch_skew", "dsb"]
    for bench in benchmarks:
        work_load = "path to the workload file"
        with open(work_load, "r") as rf:
            workload = json.load(rf)

        pre_drift_data(bench, workload, temp_num, work_size[bench], work_num[bench],
                       eval_num, test_num, drift_typ, seed=seed)
