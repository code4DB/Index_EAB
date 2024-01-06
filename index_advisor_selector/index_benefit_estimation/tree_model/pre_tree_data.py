# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: pre_tree_data
# @Author: Wei Zhou
# @Time: 2022/8/9 22:48

import os
import json

from tqdm import tqdm

from index_advisor_selector.index_benefit_estimation.benefit_utils.benefit_const import ops_dict


example_plan1 = {"Node Type": "Hash Join", "Total Cost": 35, "Plan Rows": 200, "Plans": [
    {"Node Type": "Hash Join", "Total Cost": 20, "Plan Rows": 200, "Plans": [
        {"Node Type": "Index Seek", "Total Cost": 10, "Plan Rows": 200},
        {"Node Type": "Index Scan", "Total Cost": 30, "Plan Rows": 1000}
    ]},
    {"Node Type": "Index Scan", "Total Cost": 50, "Plan Rows": 1000}
]}

example_plan2 = {"Node Type": "Hash Join", "Total Cost": 35, "Plan Rows": 200, "Plans": [
    {"Node Type": "Hash Join", "Total Cost": 20, "Plan Rows": 1000, "Plans": [
        {"Node Type": "Index Seek", "Total Cost": 20, "Plan Rows": 1000},
        {"Node Type": "Index Scan", "Total Cost": 30, "Plan Rows": 1000}
    ]},
    {"Node Type": "Index Scan", "Total Cost": 10, "Plan Rows": 200}
]}


# traverse the plan.
def tra_plan_ite(plan):
    """
    1. est_plan: 9
    ['Node Type', 'Parallel Aware', 'Startup Cost', 'Total Cost', 'Plan Rows',
    'Plan Width', 'Workers Planned', 'Single Copy', 'Plans']

    2. act_plan: 24
    ['Node Type', 'Parallel Aware', 'Startup Cost', 'Total Cost', 'Plan Rows',
    'Plan Width', 'Actual Startup Time', 'Actual Total Time', 'Actual Rows',
    'Actual Loops', 'Workers Planned', 'Workers Launched', 'Single Copy',
    'Shared Hit Blocks', 'Shared Read Blocks', 'Shared Dirtied Blocks',
    'Shared Written Blocks', 'Local Hit Blocks', 'Local Read Blocks',
    'Local Dirtied Blocks', 'Local Written Blocks', 'Temp Read Blocks',
    'Temp Written Blocks', 'Plans']
    :param plan:
    :return:
    """
    height = 1
    stack, node = [{"height": height, "node": plan}], list()
    while stack:
        cnode = stack.pop()  # index: default=last
        node.append({"height": cnode["height"], "type": cnode["node"]["Node Type"]})
        if "Plans" in cnode["node"].keys():
            height += 1
            for nod in reversed(cnode["node"]["Plans"]):
                stack.append({"height": height, "node": nod})
    return node


def tra_plan_rec(node, plan, height, parent):
    # node.append(plan["Node Type"])
    # node.append({"height": height, "type": plan["Node Type"]})
    # node.append({"height": height, "plan": plan})
    node.append({"id": len(node), "height": height,
                 "parent": parent, "children": list(),
                 "detail": {f"{item[0]}": item[1] for item
                            in plan.items() if item[0] not in ["Plans"]}})
    if parent != -1:  # not root node.
        node[parent]["children"].append(node[-1]["id"])
    parent = node[-1]["id"]
    if "Plans" in plan.keys():
        for splan in plan["Plans"]:
            node = tra_plan_rec(node, splan, height + 1, parent)

    return node


def get_tra_order(node, plan, mode="pre"):
    """
    Get the traverse order sequence: pre/in/post order.
    :param node:
    :param plan:
    :param mode:
    :return:
    """
    if mode == "pre":
        node.append({"type": plan["Node Type"],
                     "cost": plan["Total Cost"],
                     "row": plan["Plan Rows"]})
        if "Plans" in plan.keys():
            for splan in plan["Plans"]:
                node = get_tra_order(node, splan, mode)
    elif mode == "in":
        if "Plans" in plan.keys():
            splans = plan["Plans"]
            node = get_tra_order(node, splans[0], mode)
            node.append({"type": plan["Node Type"],
                         "cost": plan["Total Cost"],
                         "row": plan["Plan Rows"]})
            if len(splans) == 2:
                node = get_tra_order(node, splans[1], mode)
        else:
            node.append({"type": plan["Node Type"],
                         "cost": plan["Total Cost"],
                         "row": plan["Plan Rows"]})
    elif mode == "post":
        if "Plans" in plan.keys():
            for splan in plan["Plans"]:
                node = get_tra_order(node, splan, mode)
        node.append({"type": plan["Node Type"],
                     "cost": plan["Total Cost"],
                     "row": plan["Plan Rows"]})

    return node


def tag_post_order(count, plan):
    if "Plans" in plan.keys():
        for i, splan in enumerate(plan["Plans"]):
            count, plan["Plans"][i] = tag_post_order(count, splan)

    plan["id"] = count
    count += 1

    return count, plan


def tra_plan_post_rec(node, stack, plan):
    """
    Postorder traversal for the value of the height.

    :param node:
    :param stack:
    :param plan:
    :return:
    """
    if "Plans" in plan.keys():
        for splan in plan["Plans"]:
            node, stack = tra_plan_post_rec(node, stack, splan)

    # node.append(plan["Node Type"])
    # node.append({"height": height, "type": plan["Node Type"]})
    # node.append({"height": height, "plan": plan})
    node.append({"id": len(node), "height": -1, "children": list(),
                 "detail": {f"{item[0]}": item[1] for item
                            in plan.items() if item[0] not in ["Plans"]}})
    if "Plans" in plan.keys():
        cid = list()
        for _ in range(len(plan["Plans"])):
            cnode = stack.pop()
            cid.append(cnode["id"])
        node[-1]["children"] = cid  # sorted(cid)
        # : height -> the min/max-level of its children?
        node[-1]["height"] = max([node[cid]["height"] for cid in node[-1]["children"]]) + 1
    else:
        node[-1]["children"] = list()
        node[-1]["height"] = 1
    stack.append(node[-1])

    # if parent != -1:  # not root node.
    #     node[parent]["children"] = cid

    return node, stack


# count the operators(node types) in the plan.
def count_ops():
    """
    1. Jan Kossmann, Alexander Kastius, Rainer Schlosser:
    SWIRL: Selection of Workload-aware Indexes using Reinforcement Learning. EDBT 2022: 2:155-2:168.

    ["Seq Scan", "Hash Join", "Nested Loop", "CTE Scan",
    "Index Only Scan", "Index Scan", "Merge Join", "Sort"]

    2. all node types in the plan.
    ['Aggregate', 'BitmapAnd', 'Hash Join', 'Gather',
    'Nested Loop', 'Gather Merge', 'Materialize', 'Bitmap Heap Scan',
    'Index Scan', 'Merge Join', 'Seq Scan', 'Sort', 'Hash',
    'Index Only Scan', 'Bitmap Index Scan']
    :return:
    """
    runtime_load = "/data/wz/index/index_eab/eab_benefit/cost_data/job_cost_data_src.json"
    with open(runtime_load, "r") as rf:
        runtime_info = json.load(rf)
    ops = set()
    for info in runtime_info:
        ops = ops.union(set([node["type"] for node in tra_plan_ite(info["w/o plan"])]))
        ops = ops.union(set([node["type"] for node in tra_plan_ite(info["w/ plan"])]))

    print(len(ops), sorted(ops))
    return ops


# : 1. Est Cost / Row (Weighted Sum).
# Bailu Ding, Sudipto Das, Ryan Marcus, Wentao Wu, Surajit Chaudhuri, Vivek R. Narasayya:
# AI Meets AI: Leveraging Query Executions to Improve Index Recommendations.
# SIGMOD Conference 2019: 1241-1258.


def cal_weighted_sum(node):
    """
    WeightedSum(structural information): sum(height * weight).
    :param node:
    :return:
    """
    for nod in node:
        if not nod["children"]:
            nod["EstCostWeightedSum"] = nod["detail"]["Total Cost"]
            nod["EstRowWeightedSum"] = nod["detail"]["Plan Rows"]
        else:
            nod["EstCostWeightedSum"] = sum([node[cid]["height"] * node[cid]["EstCostWeightedSum"]
                                             for cid in nod["children"]])
            nod["EstRowWeightedSum"] = sum([node[cid]["height"] * node[cid]["EstRowWeightedSum"]
                                            for cid in nod["children"]])

    return node


def sum_node_info(node_list):
    cost_sum, row_sum = dict(), dict()
    cost_wsum, row_wsum = dict(), dict()
    for node in node_list:
        node_type = node["detail"]["Node Type"]
        # 1) Total Cost
        if node_type not in cost_sum.keys():
            cost_sum[node_type] = node["detail"]["Total Cost"]
        else:
            cost_sum[node_type] += node["detail"]["Total Cost"]

        # 2) Plan Rows
        if node_type not in row_sum.keys():
            row_sum[node_type] = node["detail"]["Plan Rows"]
        else:
            row_sum[node_type] += node["detail"]["Plan Rows"]

        # 3) EstCostWeightedSum
        if node_type not in cost_wsum.keys():
            cost_wsum[node_type] = node["EstCostWeightedSum"]
        else:
            cost_wsum[node_type] += node["EstCostWeightedSum"]

        # 4) EstRowWeightedSum
        if node_type not in row_wsum.keys():
            row_wsum[node_type] = node["EstRowWeightedSum"]
        else:
            row_wsum[node_type] += node["EstRowWeightedSum"]

    return cost_sum, row_sum, cost_wsum, row_wsum


def get_plan_info(plan):
    """

    :param plan:
    :return:
    """
    node, stack = list(), list()
    node, stack = tra_plan_post_rec(node, stack, plan)

    info = dict()
    info["plan_node"] = cal_weighted_sum(node)
    info["node_cost_sum"], info["node_row_sum"], \
    info["node_cost_wsum"], info["node_row_wsum"] = sum_node_info(node)

    return info


def pre_tree_plan():
    bench = ["tpch", "tpcds", "job"]
    for ben in tqdm(bench):
        for fid in ["src", "tgt"]:
            for did in ["train", "valid", "test"]:
                data_load = f"/data/wz/index/index_eab/eab_benefit/cost_data/" \
                            f"{ben}/{ben}_cost_data_{fid}_{did}.json"
                with open(data_load, "r") as rf:
                    data = json.load(rf)

                data_pre = list()
                for item in data:
                    p_info = get_plan_info(item["w/ plan"])

                    p_cost_feat = [p_info["node_cost_sum"][key] if key in p_info["node_cost_sum"] else 0 for key in ops_dict.keys()]
                    p_row_feat = [p_info["node_row_sum"][key] if key in p_info["node_row_sum"] else 0 for key in ops_dict.keys()]
                    p_wcost_feat = [p_info["node_cost_wsum"][key] if key in p_info["node_cost_wsum"] else 0 for key in ops_dict.keys()]
                    p_wrow_feat = [p_info["node_row_wsum"][key] if key in p_info["node_row_wsum"] else 0 for key in ops_dict.keys()]

                    data_pre.append({"feat": {"p_cost_feat": p_cost_feat, "p_row_feat": p_row_feat,
                                              "p_wcost_feat": p_wcost_feat, "p_wrow_feat": p_wrow_feat},
                                     "label": item["w/ actual cost"],
                                     "info": p_info, "raw": item})

                data_save = f"/data/wz/index/index_eab/eab_benefit/tree_model/data/" \
                            f"{ben}/tree_{ben}_cost_data_{fid}_{did}.json"
                if not os.path.exists(os.path.dirname(data_save)):
                    os.makedirs(os.path.dirname(data_save))
                with open(data_save, "w") as wf:
                    json.dump(data_pre, wf, indent=2)


if __name__ == "__main__":
    # count_ops()

    # query_plan = example_plan1
    # info = get_plan_info(query_plan)

    pre_tree_plan()
