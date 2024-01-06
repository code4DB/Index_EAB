# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: pre_filter_data
# @Author: Wei Zhou
# @Time: 2023/11/26 19:49

import os
import re

import math
import json

import random
from tqdm import tqdm

from index_advisor_selector.index_candidate_generation.distill_model.distill_utils.distill_workload import Index
from index_advisor_selector.index_candidate_generation.distill_model.distill_utils.distill_const import ops_join_dict, ops_scan_dict, ops_sort_dict, ops_group_dict


def traverse_plan(seq, terminal, node, parent):
    node["parent"] = parent
    node["ID"] = len(seq)

    seq.append(node)

    if "Plans" in node.keys():
        for i, n in enumerate(node["Plans"]):
            traverse_plan(seq, terminal, n, node["ID"])
            if i == 0:
                node["left"] = n["ID"]
            elif i == 1:
                node["right"] = n["ID"]
    else:
        terminal.append(node["ID"])


def extract_feat(query, root, index, row, stats):
    # 1. tag the node in the plan
    seq = list()
    terminal = list()
    traverse_plan(seq, terminal, root, None)

    # utility
    red = 0
    scan = [s for s in seq if s["Node Type"] in ops_scan_dict.keys()]
    for sc in scan:
        pred_num = 1 + len(re.findall(r" and ", str(sc))) + len(re.findall(r" or ", str(sc)))
        for col in index.columns:
            if col in str(sc) and index.table in str(sc):
                if "Relation Name" not in sc.keys() and "Alias" not in sc.keys():
                    continue

                if sc["Relation Name"] in row.keys():
                    select = sc["Plan Rows"] / row[sc["Relation Name"].lower()]
                elif sc["Alias"] in row.keys():
                    select = sc["Plan Rows"] / row[sc["Alias"].lower()]

                select = 1 - math.pow((1 - select), 1 / pred_num)

                red += (1 - select) * sc["Total Cost"]

    join = [s for s in seq if s["Node Type"] in ops_join_dict.keys()]
    for jo in join:
        jo_str = [jo[key] for key in jo.keys() if key != "Plans"]
        for col in index.columns:
            if col in str(jo_str) and index.table in str(jo_str):
                select = jo["Plan Rows"] / (seq[jo["left"]]["Plan Rows"] * seq[jo["right"]]["Plan Rows"])
                select = math.sqrt(select)

                red += (1 - select) * sc["Total Cost"]

    utility = red / root["Total Cost"]

    # query shape
    query_shape_total = list()
    for n in terminal:
        n_temp = seq[n]

        query_shape = list()
        while n_temp["parent"] is not None:
            if n_temp["Node Type"] not in query_shape:
                query_shape.append(n_temp["Node Type"])
            n_temp = seq[n_temp["parent"]]

        query_shape_total.append(query_shape)

    # index shape
    sort_str = [[s[key] for key in s.keys() if key != "Plans"] for s in seq if s["Node Type"] in ops_sort_dict.keys()]
    group_str = [[s[key] for key in s.keys() if key != "Plans"] for s in seq if s["Node Type"] in ops_group_dict.keys()]
    join_str = [[s[key] for key in s.keys() if key != "Plans"] for s in seq if s["Node Type"] in ops_join_dict.keys()]
    scan_str = [[s[key] for key in s.keys() if key != "Plans"] for s in seq if s["Node Type"] in ops_scan_dict.keys()]

    index_shape = list()
    for col in index.columns:
        if col in str(sort_str) and index.table in str(sort_str):
            index_shape.append("order-by")
        elif col in str(group_str) and index.table in str(group_str):
            index_shape.append("group-by")
        elif col in str(join_str) and index.table in str(join_str):
            index_shape.append("join")
        elif col in str(scan_str) and index.table in str(scan_str):
            index_shape.append("selection")

    if len(index_shape) == 1 and index_shape[0] in ["order-by", "group-by"]:
        for col in index.columns:
            if col in str(join_str) and index.table in str(join_str):
                index_shape = ["join"]
                break
            elif col in str(scan_str) and index.table in str(scan_str):
                index_shape = ["selection"]
                break

    # physical operator
    op_feat = dict()
    scan = [s for s in seq if s["Node Type"] in ops_scan_dict.keys()]
    for sc in scan:
        pred_num = 1 + len(re.findall(r" and ", str(sc))) + len(re.findall(r" and ", str(sc)))
        for col in index.columns:
            if col in str(sc) and index.table in str(sc):
                if "Relation Name" not in sc.keys() and "Alias" not in sc.keys():
                    continue

                if sc["Relation Name"] in row.keys():
                    select = sc["Plan Rows"] / row[sc["Relation Name"].lower()]
                elif sc["Alias"] in row.keys():
                    select = sc["Plan Rows"] / row[sc["Alias"].lower()]

                select = 1 - math.pow((1 - select), 1 / pred_num)

                if sc["Node Type"] not in op_feat.keys():
                    op_feat[sc["Node Type"]] = list()

                op_feat[sc["Node Type"]].append(select)

    join = [s for s in seq if s["Node Type"] in ops_join_dict.keys()]
    for jo in join:
        jo_str = [jo[key] for key in jo.keys() if key != "Plans"]
        for col in index.columns:
            if col in str(jo_str) and index.table in str(jo_str):
                select = jo["Plan Rows"] / (seq[jo["left"]]["Plan Rows"] * seq[jo["right"]]["Plan Rows"])
                select = math.sqrt(select)

                if jo["Node Type"] not in op_feat.keys():
                    op_feat[jo["Node Type"]] = list()

                op_feat[jo["Node Type"]].append(select)

    sort = [s for s in seq if s["Node Type"] in ops_sort_dict.keys()]
    for so in sort:
        so_str = [so[key] for key in so.keys() if key != "Plans"]
        for col in index.columns:
            if col in str(so_str) and index.table in str(so_str):
                density = stats[f"{index.table}.{col}"]["dist"]

                if so["Node Type"] not in op_feat.keys():
                    op_feat[so["Node Type"]] = list()

                op_feat[so["Node Type"]].append(density)

    group = [s for s in seq if s["Node Type"] in ops_group_dict.keys()]
    for gr in group:
        gr_str = [gr[key] for key in gr.keys() if key != "Plans"]
        for col in index.columns:
            if col in str(gr_str) and index.table in str(gr_str):
                density = stats[f"{index.table}.{col}"]["dist"]

                if gr["Node Type"] not in op_feat.keys():
                    op_feat[gr["Node Type"]] = list()

                op_feat[gr["Node Type"]].append(density)

    return {"utility": utility, "query_shape": query_shape_total,
            "index_shape": index_shape, "physical_operator": op_feat}


def gen_data_batch():
    benchmarks = ["tpch", "tpcds", "job"]
    for bench in tqdm(benchmarks):
        data_dir = f"/data1/wz/index/index_eab/eab_other/cost_data/{bench}"

        plan_data, selected = list(), list()
        for file in os.listdir(data_dir):
            with open(f"{data_dir}/{file}", "r") as rf:
                data = json.load(rf)

            for dat in data:
                for typ in ["src", "tgt"]:
                    if f"{typ}_act_wo_plan" not in dat.keys():
                        continue

                    if f"{typ}_act_not_hypo_plan" not in dat.keys():
                        continue

                    if "Actual Total Time" not in dat[f"{typ}_act_wo_plan"].keys():
                        continue

                    if "Actual Total Time" not in dat[f"{typ}_act_not_hypo_plan"].keys():
                        continue

                    if dat[f"{typ}_est_wo_plan"]["Total Cost"] == 0:
                        continue

                    if 1 - dat[f"{typ}_est_hypo_plan"]["Total Cost"] / dat[f"{typ}_est_wo_plan"]["Total Cost"] == 0:
                        continue

                    if 1 - dat[f"{typ}_act_not_hypo_plan"]["Actual Total Time"] / dat[f"{typ}_act_wo_plan"][
                        "Actual Total Time"] < 0:
                        continue

                    if len(dat[f"{typ}_inds"]) == 1 and (dat[f"{typ}_sql"], str(dat[f"{typ}_inds"])) not in selected:
                        plan_data.append({"query": dat[f"{typ}_sql"], "indexes": dat[f"{typ}_inds"],

                                          "label est": 1 - dat[f"{typ}_est_hypo_plan"]["Total Cost"] / dat[f"{typ}_est_wo_plan"][
                                              "Total Cost"],
                                          "label act": 1 - dat[f"{typ}_act_not_hypo_plan"]["Actual Total Time"] /
                                                       dat[f"{typ}_act_wo_plan"]["Actual Total Time"],

                                          "wo plan est": dat[f"{typ}_est_wo_plan"],
                                          "wo plan act": dat[f"{typ}_act_wo_plan"],
                                          "w plan est": dat[f"{typ}_est_hypo_plan"],
                                          "w plan act": dat[f"{typ}_act_not_hypo_plan"]})

                        selected.append((dat[f"{typ}_sql"], str(dat[f"{typ}_inds"])))

        data_save = f"/data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_{bench}.json"
        with open(data_save, "w") as wf:
            json.dump(plan_data, wf, indent=2)


def gen_feat_batch():
    """

    1. utility: 1,
    2. query_shape: 6 (table_num) * 7 (tree_depth),
    3. index_shape: 1 (index_num) * 2 (column_num),
    4. physical_operator: 10

    :return:
    """

    benchmarks = ["tpch", "tpcds", "job"]
    for bench in tqdm(benchmarks):
        feat_data = list()

        data_load = f"/data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_{bench}.json"
        with open(data_load, "r") as rf:
            data = json.load(rf)

        stats_load = f"/data1/wz/index/index_eab/eab_benefit/index_cost_lib/data/db_stats_{bench}.json"
        with open(stats_load, "r") as rf:
            stats = json.load(rf)

        if bench in ["tpch", "tpcds"]:
            schema_load = f"/data1/wz/index/index_eab/eab_data/db_info_conf/schema_{bench}_1gb.json"
        elif bench in ["job"]:
            schema_load = f"/data1/wz/index/index_eab/eab_data/db_info_conf/schema_{bench}.json"

        with open(schema_load, "r") as rf:
            schema = json.load(rf)
        row = dict()
        for item in schema:
            row[item["table"]] = item["rows"]

        for no, dat in enumerate(data):
            query = dat["query"]
            root = dat["wo plan est"]

            index = dat["indexes"][0]
            table = index.split("#")[0]
            columns = index.split("#")[1].split(",")
            index = Index(columns, table)

            feat = extract_feat(query, root, index, row, stats)

            dat["feat"] = feat
            feat_data.append(dat)

        data_save = f"/data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_{bench}.json"
        with open(data_save, "w") as wf:
            json.dump(feat_data, wf, indent=2)


if __name__ == "__main__":
    # gen_data_batch()
    gen_feat_batch()
