# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: get_plan_info
# @Author: Wei Zhou
# @Time: 2023/10/5 17:29

example_plan = {"Node Type": "Hash Join", "Total Cost": 35, "Plan Rows": 200, "Plans": [
    {"Node Type": "Hash Join", "Total Cost": 20, "Plan Rows": 200, "Plans": [
        {"Node Type": "Index Seek", "Total Cost": 10, "Plan Rows": 200},
        {"Node Type": "Index Scan", "Total Cost": 30, "Plan Rows": 1000}
    ]},
    {"Node Type": "Index Scan", "Total Cost": 50, "Plan Rows": 1000}
]}


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
        node.append({"height": cnode["height"], "type": cnode["node"]["Node Type"],
                     "detail": {key: val for key, val in cnode["node"].items() if key != "Plans"}})
        if "Plans" in cnode["node"].keys():
            height += 1
            for nod in reversed(cnode["node"]["Plans"]):
                stack.append({"height": height, "node": nod,
                              "detail": {key: val for key, val in nod.items() if key != "Plans"}})
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


if __name__ == "__main__":
    plan = example_plan
    node = tra_plan_ite(plan)
