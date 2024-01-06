# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: sql_tokenize
# @Author: Wei Zhou
# @Time: 2022/6/23 17:23

import re
import json
from tqdm import tqdm

import traceback
import configparser

import sqlparse
import mo_sql_parsing as mosqlparse

from . import perturb_const, mod_sql


def do_tokenization(token_list):
    tokens, types = list(), list()
    for token in token_list:
        if (type(token) is sqlparse.sql.Token or
            type(token) is sqlparse.sql.Identifier) and \
                " ASC" not in token.normalized and \
                " DESC" not in token.normalized:
            tokens.append(token.normalized)
        else:
            tokens.extend(do_tokenization(token.tokens))

    return tokens


def do_assign_type(clk, tbls, cols, tokens):
    types = list()

    # (0406): newly added. for `IN`.
    # pass

    for i in range(len(tokens)):
        pre2, pre4, pre5, pre7, nxt2, nxt4 = "", "", "", "", "", ""
        if i - 2 >= 0:
            pre2 = tokens[i - 2]
        if i - 4 >= 0:
            pre4 = tokens[i - 4]
        if i - 5 >= 0:
            pre5 = tokens[i - 5]
        if i - 7 >= 0:
            pre7 = tokens[i - 7]
        if i + 2 < len(tokens):
            nxt2 = tokens[i + 2]
        if i + 4 < len(tokens):
            nxt4 = tokens[i + 4]

        types.append(assign_type(clk, tbls, cols, pre2, pre4, pre5,
                                 pre7, tokens[i], nxt2, nxt4))

    return types


def assign_type(clk, tbls, cols, pre2, pre4, pre5,
                pre7, token, nxt2, nxt4):
    # *(0411): IN VALUE.
    # if token == " ":
    #     return "whitespace"
    if token in perturb_const.punctuation:
        return token
    elif token in perturb_const.parenthesis:
        return token
    elif token.lower() in perturb_const.keyword:
        return token.lower()
    # (0411): newly modified. `INNER JOIN` etc.
    elif token.upper() in perturb_const.join:
        return token.lower()
    # : join column: col1 `op` col2
    elif (pre2.lower().split(".")[-1] in cols
          or pre2.lower() in cols) and \
            token in perturb_const.operator and \
            (nxt2.lower().split(".")[-1] in cols
             or nxt2.lower() in cols):
        return f"{clk}#join_operator"
    # *: join column? subquery?
    elif token in perturb_const.operator:
        return "operator"
    # (0411): newly added. for `IS`, tokenization.
    elif token.lower() in perturb_const.null_operator:
        return "null_operator"
    elif token.lower() in perturb_const.aggregator:
        return f"{clk}#aggregator"
    elif token in perturb_const.order_by_key:
        return "order_by_key"
    elif token.upper() in perturb_const.conjunction:
        return "conjunction"
    elif token.lower() in perturb_const.null_predicate:
        return "null_predicate"
    elif token.lower() in perturb_const.in_predicate:
        return "in_predicate"
    elif token.lower() in perturb_const.exists_predicate:
        return "exists_predicate"
    elif token.lower() in perturb_const.like_predicate:
        return "like_predicate"
    # : aggregate column: agg(`col1`)
    elif pre2.lower() in perturb_const.aggregator \
            and (token.lower().split(".")[-1] in cols
                 or token.lower() in cols):
        return f"{clk}#aggregate_column"
    # : join column: `col1` op col2
    elif (nxt4.lower().split(".")[-1] in cols
          or nxt4.lower() in cols) \
            and nxt2.lower() in perturb_const.operator \
            and (token.lower().split(".")[-1] in cols
                 or token.lower() in cols):
        return f"{clk}#join_column"
    # : join column: col1 op `col2`
    elif (pre4.lower().split(".")[-1] in cols
          or pre4.lower() in cols) \
            and pre2.lower() in perturb_const.operator \
            and (token.lower().split(".")[-1] in cols
                 or token.lower() in cols):
        return f"{clk}#join_column"  # where?
    # (0413): newly modified. why after `value`?
    elif (token.lower().split(".")[-1] in cols
          or token.lower() in cols):
        return f"{clk}#column"
    # (0411): newly added.
    #  predicate value(like): col op `val`, op: like, not like.
    elif (pre4.lower().split(".")[-1] in cols
          or pre4.lower() in cols) \
            and pre2.lower() in perturb_const.like_predicate \
            and (token.lower().split(".")[-1] not in cols
                 or token.lower() not in cols) \
            and token.lower().split(".")[-1] not in tbls:
        if len(list(cols)[0].split(".")) == 2:
            # *(0411): column name, with table, '.'
            return f"{pre4.lower()}#like_value"
        else:
            return f"{pre4.lower().split('.')[-1]}#like_value"
    # : predicate value: col op `val`
    # : token.lower().split(".")[-1] not in tbls? (removed? join column?)
    elif (pre4.lower().split(".")[-1] in cols
          or pre4.lower() in cols) \
            and (token.lower().split(".")[-1] not in cols
                 or token.lower() not in cols) \
            and token.lower().split(".")[-1] not in tbls:
        if len(list(cols)[0].split(".")) == 2:
            # *(0411): column name, with table, '.'
            return f"{pre4.lower()}#value"
        else:
            return f"{pre4.lower().split('.')[-1]}#value"
    # : aggregate value: agg(col) op `val`, agg: "max", "min", "count"
    # : count(), `numeric` or not? (multiple choice)
    # : token.lower().split(".")[-1] not in tbls? removed?,
    #  pre7.lower().split(".")[-1] in perturb_const.aggregator[:2]? removed?
    elif (pre7.lower().split(".")[-1] in perturb_const.aggregator[:2]
          or pre7.lower() in perturb_const.aggregator[:2]) \
            and (pre5.lower().split(".")[-1] in cols
                 or pre5.lower() in cols) \
            and (token.lower().split(".")[-1] not in cols
                 or token.lower() not in cols) \
            and token.lower().split(".")[-1] not in tbls:
        if len(list(cols)[0].split(".")) == 2:
            # *(0411): column name, with table, '.'
            return f"{pre5.lower()}#aggregate_value"
        else:
            return f"{pre5.lower().split('.')[-1]}#aggregate_value"
    # : aggregate value: agg(col) op `val`, agg: "count", "avg", "sum"
    # : count(), `numeric` or not? (multiple choice)
    # : token.lower().split(".")[-1] not in tbls? removed?
    elif (pre7.lower().split(".")[-1] in perturb_const.aggregator[-3:]
          or pre7.lower() in perturb_const.aggregator[-3:]) \
            and (pre5.lower().split(".")[-1] in cols
                 or pre5.lower() in cols) \
            and (token.lower().split(".")[-1] not in cols
                 or token.lower() not in cols) \
            and token.lower().split(".")[-1] not in tbls:
        if len(list(cols)[0].split(".")) == 2:
            # *(0411): column name, with table, '.'
            return f"{pre5.lower()}#numeric_aggregate_value"
        else:
            return f"{pre5.lower().split('.')[-1]}#numeric_aggregate_value"
    elif (pre2.upper() in perturb_const.join
          or nxt2.upper() in perturb_const.join) and \
            token.lower() in tbls:
        return f"{clk}#join_table"
    elif token.lower() in tbls:
        return f"{clk}#table"
    elif perturb_const.wildcard in token:
        return "wildcard"
    # # (0411): newly added. for unknown.
    #   removed to check type assignment error.
    # else:
    #     return "unknown"


def check_type(token_load):
    with open(token_load, "r") as rf:
        json_data = json.load(rf)

    for data in json_data:
        for typ in data["types"]:
            if typ is None:
                print("Type Error!")


def split_sql(sql):
    sql = sql.strip("\n").strip(";") + "##"
    for key in perturb_const.keyword:
        res = re.findall(f"{key} ", sql, re.IGNORECASE)
        for r in res:
            sql = sql.replace(r, f"{key} ")

    pat_wgho = r"select (.*?) from (.*?) where (.*?) group by (.*?) having (.*?) order by (.*?)##"
    subclauses = re.findall(pat_wgho, sql)
    if subclauses:
        subclauses = subclauses[0]
        clause_dic = {"select": subclauses[0], "from": subclauses[1], "where": subclauses[2],
                      "group by": subclauses[3], "having": subclauses[4], "order by": subclauses[5]}
        return "wgho", clause_dic

    pat_wgo = r"select (.*?) from (.*?) where (.*?) group by (.*?) order by (.*?)##"
    subclauses = re.findall(pat_wgo, sql)
    if subclauses:
        subclauses = subclauses[0]
        clause_dic = {"select": subclauses[0], "from": subclauses[1], "where": subclauses[2],
                      "group by": subclauses[3], "order by": subclauses[4]}
        return "wgo", clause_dic

    pat_wgh = r"select (.*?) from (.*?) where (.*?) group by (.*?) having (.*?)##"
    subclauses = re.findall(pat_wgh, sql)
    if subclauses:
        subclauses = subclauses[0]
        clause_dic = {"select": subclauses[0], "from": subclauses[1], "where": subclauses[2],
                      "group by": subclauses[3], "having": subclauses[4]}
        return "wgh", clause_dic

    pat_wo = r"select (.*?) from (.*?) where (.*?) order by (.*?)##"
    subclauses = re.findall(pat_wo, sql)
    if subclauses:
        subclauses = subclauses[0]
        clause_dic = {"select": subclauses[0], "from": subclauses[1],
                      "where": subclauses[2], "order by": subclauses[3]}
        return "wo", clause_dic

    pat_wg = r"select (.*?) from (.*?) where (.*?) group by (.*?)##"
    subclauses = re.findall(pat_wg, sql)
    if subclauses:
        subclauses = subclauses[0]
        clause_dic = {"select": subclauses[0], "from": subclauses[1],
                      "where": subclauses[2], "group by": subclauses[3]}
        return "wg", clause_dic

    pat_gho = r"select (.*?) from (.*?) group by (.*?) having (.*?) order by (.*?)##"
    subclauses = re.findall(pat_gho, sql)
    if subclauses:
        subclauses = subclauses[0]
        clause_dic = {"select": subclauses[0], "from": subclauses[1],
                      "group by": subclauses[2], "having": subclauses[3], "order by": subclauses[4]}
        return "gho", clause_dic

    pat_go = r"select (.*?) from (.*?) group by (.*?) order by (.*?)##"
    subclauses = re.findall(pat_go, sql)
    if subclauses:
        subclauses = subclauses[0]
        clause_dic = {"select": subclauses[0], "from": subclauses[1],
                      "group by": subclauses[2], "order by": subclauses[3]}
        return "go", clause_dic

    pat_gh = r"select (.*?) from (.*?) group by (.*?) having (.*?)##"
    subclauses = re.findall(pat_gh, sql)
    if subclauses:
        subclauses = subclauses[0]
        clause_dic = {"select": subclauses[0], "from": subclauses[1],
                      "group by": subclauses[2], "having": subclauses[3]}
        return "gh", clause_dic

    pat_o = r"select (.*?) from (.*?) order by (.*?)##"
    subclauses = re.findall(pat_o, sql)
    if subclauses:
        subclauses = subclauses[0]
        clause_dic = {"select": subclauses[0], "from": subclauses[1],
                      "order by": subclauses[2]}
        return "o", clause_dic

    pat_g = r"select (.*?) from (.*?) group by (.*?)##"
    subclauses = re.findall(pat_g, sql)
    if subclauses:
        subclauses = subclauses[0]
        clause_dic = {"select": subclauses[0], "from": subclauses[1],
                      "group by": subclauses[2]}
        return "g", clause_dic

    pat_n = r"select (.*?) from (.*?) where (.*?)##"
    subclauses = re.findall(pat_n, sql)
    if subclauses:
        subclauses = subclauses[0]
        clause_dic = {"select": subclauses[0], "from": subclauses[1], "where": subclauses[2]}
        return "n", clause_dic

    pat_s = r"select (.*?) from (.*?)##"
    subclauses = re.findall(pat_s, sql)
    if subclauses:
        subclauses = subclauses[0]
        clause_dic = {"select": subclauses[0], "from": subclauses[1]}
        return "s", clause_dic


def tokenize_sql(sql_list, tbls, cols, word2idx):
    """
    1. clause identification; 2. SQL tokenization; 3. type assignment.

    :param sql_list:
    :param tbls:
    :param cols:
    :param word2idx: sql2vec(), for SQL vectors.
    :return:
    """
    sql_res, except_item = list(), list()
    for i, sql in tqdm(enumerate(sql_list)):
        try:
            # : verify the equality.
            # format: 1) keyword -> `upper case`;
            #         2) join column -> `add space`;
            #         3) `!=` -> `<>`.
            sql = mosqlparse.format(mosqlparse.parse(sql))
            # *(0411): nested query, function (recursive)? problematic
            # mosqlparse.parse(sql).keys()  # clauses
            # mosqlparse.format({"select": mosqlparse.parse(sql)["select"]})
            _, clauses = split_sql(sql)  # : nested query?
            # try:
            #     _, clauses = split_sql(sql)
            # except:
            #     split_sql(sql)
            #     print("split_sql")

            cla_res = {"sql": sql, "tokens": list(), "types": list(),
                       "pre_tokens": list(), "pre_types": list()}
            for clk in clauses.keys():
                cla_res[clk] = dict()
                token_list = sqlparse.parse(clauses[clk])[0].tokens
                if clk == "select":
                    cla_res[clk]["token"] = [clk, " "] + do_tokenization(token_list)
                    # cla_res[clk]["token"] = do_tokenization(token_list)
                    cla_res["tokens"].extend(cla_res[clk]["token"])
                    cla_res[clk]["type"] = do_assign_type(clk, tbls, cols, cla_res[clk]["token"])
                    cla_res["types"].extend(cla_res[clk]["type"])
                else:
                    cla_res[clk]["token"] = [" ", clk, " "] + do_tokenization(token_list)
                    # cla_res[clk]["token"] = [" "] + do_tokenization(token_list)
                    cla_res["tokens"].extend(cla_res[clk]["token"])
                    cla_res[clk]["type"] = do_assign_type(clk, tbls, cols, cla_res[clk]["token"])
                    cla_res["types"].extend(cla_res[clk]["type"])

                # remove the punctuation
                cla_res[clk]["pre_token"], cla_res[clk]["pre_type"] = list(), list()
                for tok in cla_res[clk]["token"]:
                    # (0412): newly added.
                    punctuation = perturb_const.punctuation
                    if word2idx.get("(", -1) == -1 and word2idx.get(")", -1) == -1:
                        punctuation.extend(["(", ")"])
                    if tok not in punctuation:
                        cla_res[clk]["pre_token"].append(tok)
                cla_res["pre_tokens"].extend(cla_res[clk]["pre_token"])
                for tok in cla_res[clk]["type"]:
                    # (0412): newly added.
                    punctuation = perturb_const.punctuation
                    if word2idx.get("(", -1) == -1 and word2idx.get(")", -1) == -1:
                        punctuation.extend(["(", ")"])
                    if tok not in punctuation:
                        cla_res[clk]["pre_type"].append(tok)
                cla_res["pre_types"].extend(cla_res[clk]["pre_type"])

            cla_res["pno_tokens"] = mod_sql.sql2vec([cla_res], word2idx)[0]

            sql_res.append(cla_res)
        except Exception as e:
            print(e, sql)
            print(traceback.format_exc())
            except_item.append((i, sql, str(e)))

    return sql_res, except_item
