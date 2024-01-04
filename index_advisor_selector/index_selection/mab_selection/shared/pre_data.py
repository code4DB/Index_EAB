# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: pre_data
# @Author: Wei Zhou
# @Time: 2023/7/31 22:01

import re
import json
import psqlparse


def parse(parsed_res, item_res, tbl_col):
    if "fromClause" not in parsed_res.keys():
        for key in parsed_res.keys():
            if isinstance(parsed_res[key], dict):
                item_res = parse(parsed_res[key], item_res, tbl_col)
            else:
                return item_res
    else:
        table_alias = dict()
        for tbl_info in parsed_res["fromClause"]:
            if "RangeSubselect" in tbl_info.keys():
                item_res = parse(tbl_info["RangeSubselect"]["subquery"]["SelectStmt"], item_res, tbl_col)
            elif "JoinExpr" in tbl_info.keys():
                for key in tbl_info["JoinExpr"].keys():
                    if isinstance(tbl_info["JoinExpr"][key], dict) and \
                            "RangeVar" in tbl_info["JoinExpr"][key].keys():
                        tbl_name = tbl_info["JoinExpr"][key]["RangeVar"]["relname"]
                        if tbl_name not in table_alias.keys():
                            table_alias[tbl_name] = list()

                        if "alias" in tbl_info["JoinExpr"][key]["RangeVar"].keys():
                            alias = tbl_info["JoinExpr"][key]["RangeVar"]["alias"]["Alias"]["aliasname"]
                            if alias not in table_alias[tbl_name]:
                                table_alias[tbl_name].append(alias)

                for tbl in table_alias.keys():
                    if tbl not in tbl_col.keys():
                        continue
                    for col in tbl_col[tbl]:
                        if len(table_alias[tbl]) != 0:
                            for alias in table_alias[tbl]:
                                col_ref = "{'ColumnRef': {'fields': [{'String': {'str': '" + alias + "'}}, {'String': {'str': '" + col + "'}}"
                                if col_ref in str(tbl_info["JoinExpr"][key]):
                                    if tbl.upper() not in item_res["predicates"].keys():
                                        item_res["predicates"][tbl.upper()] = list()
                                    item_res["predicates"][tbl.upper()].append(col.upper())
                                    break
                        else:
                            col_ref = "{'ColumnRef': {'fields': [{'String': {'str': '" + col + "'}}"
                            if col_ref in str(tbl_info["JoinExpr"][key]):
                                if tbl.upper() not in item_res["predicates"].keys():
                                    item_res["predicates"][tbl.upper()] = list()
                                item_res["predicates"][tbl.upper()].append(col.upper())
            else:
                tbl_name = tbl_info["RangeVar"]["relname"]
                if tbl_name not in table_alias.keys():
                    table_alias[tbl_name] = list()

                # alias = tbl_name
                if "alias" in tbl_info["RangeVar"].keys():
                    alias = tbl_info["RangeVar"]["alias"]["Alias"]["aliasname"]
                    if alias not in table_alias[tbl_name]:
                        table_alias[tbl_name].append(alias)

        if "withClause" in parsed_res.keys():
            for cte in parsed_res["withClause"]["WithClause"]["ctes"]:
                if cte["CommonTableExpr"]["ctename"] in table_alias.keys():
                    table_alias.pop(cte["CommonTableExpr"]["ctename"])
                item_res = parse(cte["CommonTableExpr"]["ctequery"]["SelectStmt"], item_res, tbl_col)

        clause_map = {"targetList": "payload",
                      "whereClause": "predicates",
                      "groupClause": "group_by",
                      "sortClause": "order_by"}

        for clause in clause_map.keys():
            if clause not in parsed_res.keys():
                continue

            for tbl in table_alias.keys():
                if tbl not in tbl_col.keys():
                    continue
                for col in tbl_col[tbl]:
                    if len(table_alias[tbl]) != 0:
                        for alias in table_alias[tbl]:
                            col_ref = "{'ColumnRef': {'fields': [{'String': {'str': '" + alias + "'}}, {'String': {'str': '" + col + "'}}"
                            if col_ref in str(parsed_res[clause]):
                                if tbl.upper() not in item_res[clause_map[clause]].keys():
                                    item_res[clause_map[clause]][tbl.upper()] = list()
                                item_res[clause_map[clause]][tbl.upper()].append(col.upper())
                                break
                    else:
                        col_ref = "{'ColumnRef': {'fields': [{'String': {'str': '" + col + "'}}"
                        if col_ref in str(parsed_res[clause]):
                            if tbl.upper() not in item_res[clause_map[clause]].keys():
                                item_res[clause_map[clause]][tbl.upper()] = list()
                            item_res[clause_map[clause]][tbl.upper()].append(col.upper())

        return item_res


def parse_column():
    parsed_load = "/data/wz/index/index_eab/eab_algo/mab_selection/resources/workloads/tpc_h_skew_static_100_pre.json"
    with open(parsed_load, "r") as rf:
        parsed_data = json.load(rf)

    work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql"
    # work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpcds_template_99.sql"
    # work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/job_template_33.sql"
    #
    # work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_template_multi_query.json"
    if work_load.endswith(".sql"):
        with open(work_load, "r") as rf:
            work_data = rf.readlines()
    elif work_load.endswith(".json"):
        with open(work_load, "r") as rf:
            work_data = json.load(rf)
        work_list = list()
        for typ in work_data.keys():
            for queries in work_data[typ].values():
                work_list.append(queries[0])
        work_data = work_list

    schema_load = "/data/wz/index/attack/data_resource/db_info/schema_tpch_1gb.json"
    # schema_load = "/data/wz/index/attack/data_resource/db_info/schema_tpcds_1gb.json"
    # schema_load = "/data/wz/index/attack/data_resource/db_info/schema_job.json"
    #
    # schema_load = "/data/wz/index/attack/data_resource/db_info/schema_tpcds_1gb.json"
    with open(schema_load, "r") as rf:
        schema_info = json.load(rf)

    tbl_col = dict()
    for info in schema_info:
        tbl_col[info["table"]] = [col["name"] for col in info["columns"]]

    all_res = list()
    for no, sql in enumerate(work_data):
        item_res = {"id": no, "query_string": sql, "predicates": {}, "payload": {}, "group_by": {}, "order_by": {}}
        parsed_res = psqlparse.parse_dict(sql)[0]["SelectStmt"]

        item_res = parse(parsed_res, item_res, tbl_col)

        all_res.append(item_res)

    parse_save = "/data/wz/index/index_eab/eab_algo/mab_selection/resources/workloads/tpch_template_18_parsed.json"
    with open(parse_save, "w") as wf:
        json.dump(all_res, wf, indent=2)
    pass


# Updates query syntax to work in PostgreSQL
def update_query_text(text):
    text = text.replace(";\nlimit ", " limit ").replace("limit -1", "")
    text = re.sub(r" ([0-9]+) days\)", r" interval '\1 days')", text)

    # TPC-H: Q1
    # DATEADD(dd, -112, CAST('1998-12-01' AS date))
    # DATE '1998-12-01' - INTERVAL '112' DAY
    text = re.sub(r"DATEADD\(dd, -([0-9]+), CAST\('1998-12-01' AS date\)\)",
                  r"DATE '1998-12-01' - INTERVAL '\1' DAY", text)

    # TPC-H: Q4
    # DATEADD(mm, 3, CAST('1994-11-01' AS date))
    # DATE '1994-11-01' + INTERVAL '3' MONTH
    text = re.sub(r"DATEADD\(mm, ([0-9]+), CAST\('([\-0-9]+)' AS date\)\)",
                  r"DATE '\2' + INTERVAL '\1' MONTH", text)

    # TPC-H: Q5
    # DATEADD(yy, 1, CAST('1997-01-01' AS date))
    # DATE '1997-01-01' + INTERVAL '1' YEAR
    text = re.sub(r"DATEADD\(yy, 1, CAST\('([\-0-9]+)' AS date\)\)",
                  r"DATE '\1' + INTERVAL '1' YEAR", text)

    # TPC-H: Q6
    # DATEADD(yy, 1, CAST('1997-01-01' AS date))
    # DATE '1997-01-01' + INTERVAL '1' YEAR
    text = re.sub(r"DATEADD\(yy, 1, CAST\('([\-0-9]+)' AS date\)\)",
                  r"DATE '\1' + INTERVAL '1' YEAR", text)

    # TPC-H: Q7
    # YEAR(l_shipdate) as l_year
    # EXTRACT(YEAR FROM l_shipdate) AS l_year
    text = text.replace("YEAR(l_shipdate) as l_year",
                        "EXTRACT(YEAR FROM l_shipdate) AS l_year")

    # TPC-H: Q8
    # YEAR(o_orderdate) as o_year
    # EXTRACT(YEAR FROM o_orderdate) AS o_year
    text = text.replace("YEAR(o_orderdate) as o_year",
                        "EXTRACT(YEAR FROM o_orderdate) AS o_year")

    # TPC-H: Q9
    # YEAR(o_orderdate) as o_year
    # EXTRACT(YEAR FROM o_orderdate) AS o_year
    text = text.replace("YEAR(o_orderdate) as o_year",
                        "EXTRACT(YEAR FROM o_orderdate) AS o_year")

    # TPC-H: Q10
    # DATEADD(mm, 3, CAST('1994-06-01' AS date))
    # DATE '1994-06-01' + INTERVAL '3' MONTH
    text = re.sub(r"DATEADD\(mm, 3, CAST\('([\-0-9]+)' AS date\)\)",
                  r"DATE '\1' + INTERVAL '3' MONTH", text)

    # TPC-H: Q12
    # DATEADD(yy, 1, CAST('1997-01-01' AS date))
    # DATE '1997-01-01' + INTERVAL '1' YEAR
    text = re.sub(r"DATEADD\(yy, 1, CAST\('([\-0-9]+)' AS date\)\)",
                  r"DATE '\1' + INTERVAL '1' YEAR", text)

    # TPC-H: 14
    # DATEADD(mm, 1, CAST('1997-12-01' AS date))
    # DATE '1997-12-01' + INTERVAL '1' MONTH
    text = re.sub(r"DATEADD\(mm, 1, CAST\('([\-0-9]+)' AS date\)\)",
                  r"DATE '\1' + INTERVAL '1' MONTH", text)

    # TPC-H: 20
    # DATEADD(yy, 1, CAST('1996-01-01' AS date))
    # DATE '1996-01-01' + INTERVAL '1' YEAR
    text = re.sub(r"DATEADD\(yy, 1, CAST\('([\-0-9]+)' AS date\)\)",
                  r"DATE '\1' + INTERVAL '1' YEAR", text)

    text = add_alias_subquery(text)

    return text


def add_alias_subquery(query_text):
    # PostgreSQL requires an alias for subqueries.
    text = query_text.lower()
    positions = list()
    for match in re.finditer(r"((from)|,)[  \n]*\(", text):
        counter = 1
        pos = match.span()[1]
        while counter > 0:
            char = text[pos]
            if char == "(":
                counter += 1
            elif char == ")":
                counter -= 1
            pos += 1
        next_word = query_text[pos:].lstrip().split(" ")[0].split("\n")[0]
        if next_word[0] in [")", ","] or next_word in [
            "limit",
            "group",
            "order",
            "where",
        ]:
            positions.append(pos)
    for pos in sorted(positions, reverse=True):
        query_text = query_text[:pos] + " as alias123 " + query_text[pos:]

    return query_text


def pre_sql():
    data_load = "/data/wz/index/code_aidb/DBABandits/resources/workloads/tpc_h_static_100.json"
    # data_load = "/data/wz/index/code_aidb/DBABandits/resources/workloads/tpc_h_skew_static_100.json"
    with open(data_load, "r") as rf:
        sql_data = rf.readlines()

    data_pre = list()
    for item in sql_data:
        item = eval(item)
        item["query_string"] = update_query_text(item["query_string"])
        data_pre.append(item)
        # data_pre.append(item.replace("'", '"'))

    data_save = "/data/wz/index/code_aidb/DBABandits/resources/workloads/tpc_h_static_100_pre.json"
    # data_save = "/data/wz/index/code_aidb/DBABandits/resources/workloads/tpc_h_skew_static_100_pre.json"
    # with open(data_save, "w") as wf:
    #     for item in data_pre:
    #         wf.writelines(f"{item}\n")

    with open(data_save, "w") as wf:
        json.dump(data_pre, wf, indent=2)


if __name__ == "__main__":
    # pre_sql()
    parse_column()
