# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: pre_lib_data
# @Author: Wei Zhou
# @Time: 2023/10/5 16:11

import os
import json
import numpy as np
from tqdm import tqdm

from index_advisor_selector.index_benefit_estimation.benefit_utils.get_plan_info import tra_plan_ite
from index_advisor_selector.index_benefit_estimation.benefit_utils.benefit_const import ops_join_dict, ops_sort_dict, \
    ops_group_dict, ops_scan_dict


def pre_lib_plan():
    benchmarks = ["tpch", "tpcds", "job"]
    for bench in tqdm(benchmarks):
        for fid in ["src", "tgt"]:
            for did in ["train", "valid", "test"]:
                data_load = f"/data/wz/index/index_eab/eab_benefit/cost_data/{bench}/{bench}_cost_data_{fid}_{did}.json"
                with open(data_load, "r") as rf:
                    data = json.load(rf)

                stats_load = f"/data/wz/index/index_eab/eab_benefit/index_cost_lib/data/db_stats_{bench}.json"
                with open(stats_load, "r") as rf:
                    stats = json.load(rf)

                total_data = list()
                for item in data:
                    indexes = item["indexes"]
                    wo_plan = item["w/o plan"]
                    nodes = tra_plan_ite(wo_plan)

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
                    total_data.append({"feat": index_ops,
                                       "w/o estimated cost": item["w/o estimated cost"],
                                       "w/ estimated cost": item["w/ estimated cost"],
                                       "w/o actual cost": item["w/o actual cost"],
                                       "w/ actual cost": item["w/ actual cost"]})

                data_save = f"/data/wz/index/index_eab/eab_benefit/index_cost_lib/data/" \
                            f"{bench}/lib_{bench}_cost_data_{fid}_{did}.json"
                if not os.path.exists(os.path.dirname(data_save)):
                    os.makedirs(os.path.dirname(data_save))
                with open(data_save, "w") as wf:
                    json.dump(total_data, wf, indent=2)


if __name__ == "__main__":
    pre_lib_plan()
