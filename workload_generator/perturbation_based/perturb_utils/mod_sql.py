# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: mod_sql
# @Author: Wei Zhou
# @Time: 2022/6/28 16:16

# https://www.postgresql.org/docs/15/functions-aggregate.html
# Computes the average (arithmetic mean) of all the non-null input values.
# avg (smallint/integer/bigint/numeric) → numeric
# avg (real) → double precision
# avg (double precision) → double precision
# avg (interval) → interval

# Computes the sum of the non-null input values.
# sum(smallint/integer) → bigint
# sum(bigint/numeric) → numeric
# sum(real) → real
# sum(double precision) → double precision
# sum(interval) → interval
# sum(money) → money

import json
import random
import numpy as np
from tqdm import tqdm

import copy
import logging
import traceback

from . import perturb_const


# : GROUP BY / ORDER BY / WHERE
# WHERE: repetitive columns
# GROUP BY + HAVING -> SELECT ()
# HAVING -> GROUP BY ()
# ORDER BY -> SELECT ()


def valid_cand(token, table, step, ptok_nos, word2idx,
               idx2word, word_info, col_info, max_diff=5):
    """
    :param token: dict
    :param table: list(str), table names
    todo: only increase when modified inplace for `add`/`del` (x).
          change the `token['pre_tokens']` dynamically for `add`/`del`.
          append new payload, predicates to the end of the corresponding clause.
    :param step: int, current decoded time-step
    :param ptok_nos: list(int), tokens decoded already
    :param word2idx:
    :param idx2word:
    :param word_info:
    :param col_info:
    :param max_diff:
    :return:
    """
    # : time-step exceed the max_len.
    if step >= len(token["pre_types"]):
        return [perturb_const.PAD]

    # : newly added
    if token["pno_tokens"][step] == perturb_const.SEP:
        return [perturb_const.SEP]

    # (might not executable): perturbation step constraint, forcibly truncated.
    # if np.sum(np.array(src_vec[:step]) != np.array(ptok_nos)) >= max_diff \
    #         and src_vec[step] in cand:
    #     return [src_vec[step]]

    cand = list()
    # 1) reserved: grammar keyword
    if token["pre_types"][step] in perturb_const.keyword:
        cand = [word2idx[token["pre_tokens"][step].lower()]]
    elif token["pre_types"][step].upper() in perturb_const.join:
        cand = [word2idx[token["pre_tokens"][step].upper()]]
    # is, *
    elif "null_operator" in token["pre_types"][step]:
        cand = [word2idx[token["pre_tokens"][step].lower()]]
    elif "wildcard" in token["pre_types"][step]:
        # cand = [word2idx[token["pre_tokens"][step]]]
        # (0413): newly added.
        cand = [perturb_const.UNK]
    elif token["pre_types"][step] in perturb_const.parenthesis:
        cand = [word2idx[token["pre_tokens"][step]]]

    # 2) tables
    elif "#join_table" in token["pre_types"][step]:
        cand = [word2idx[token["pre_tokens"][step].lower()]]
    # : join order
    elif "#table" in token["pre_types"][step]:
        cand = [word2idx[token["pre_tokens"][step].lower()]]
        # cand = [word2idx[tbl] for tbl in table if word2idx[tbl] not in ptok_nos]

    # 3) columns
    # elif "from#column" in token["pre_types"][i]:
    elif "#join_column" in token["pre_types"][step]:  # from/where
        cand = [word2idx.get(token["pre_tokens"][step].split(".")[-1], 1)]
        if cand[0] == 1:
            cand = [word2idx[token["pre_tokens"][step].lower()]]
    # : special column (aggregate)
    elif token["pre_types"][step] == "select#aggregate_column":
        # (0412): newly added. for `(`, `)`
        if "(" in idx2word.values() and ")" in idx2word.values():
            aggregator = idx2word[str(ptok_nos[-2])]
        else:
            aggregator = idx2word[str(ptok_nos[-1])]

        for tbl in table:
            tbl_col = list(range(word_info[f"{tbl}#column name"]["start_id"],
                                 word_info[f"{tbl}#column name"]["end_id"] + 1))
            # 3.1) max()/min(): column of all types.
            if aggregator in perturb_const.aggregator[:2]:
                # if aggregator in perturb_const.aggregator[:3]:
                cand.extend(tbl_col)
            # : count()
            # elif ptoken[-1] == perturb_const.aggregator[2]:
            #     cand.extend([col for col in tbl_col
            #                  if col_info[idx2word[str(col)]]["type"] != "date"])
            # 3.2) count()/avg()/sum(): column of numeric types.
            elif aggregator in perturb_const.aggregator[-3:]:
                # elif aggregator in perturb_const.aggregator[-2:]:
                cand.extend([col for col in tbl_col
                             if col_info[idx2word[str(col)]]["type"]
                             in ["smallint", "integer", "bigint", "numeric"]])
        # : numeric aggregate column can be the same, filter columns applied under the same aggregator selected already.
        cand_bak = copy.deepcopy(cand)
        cand = list(set(cand) - set([no for i, no in enumerate(ptok_nos)
                                     if token["pre_types"][i] == token["pre_types"][step]]))
        if len(cand) == 0:
            # not enough numeric column, numeric aggregator static(unchanged)
            if aggregator in perturb_const.aggregator[-3:]:
                cand = list(set(cand_bak) - set([no for i, no in enumerate(ptok_nos)
                                                 if token["pno_tokens"][i - 1] == token["pno_tokens"][step - 1]]))
            else:  # : newly added, to be removed.?
                cand = list(set(cand_bak) - set([no for i, no in enumerate(ptok_nos)
                                                 if token["pre_types"][i] == "select#aggregate_column"
                                                 and ptok_nos[i - 1] == ptok_nos[step - 1]]))
    # : special column (group by)
    elif token["pre_types"][step] == "group by#column":
        cand = list(set([no for i, no in enumerate(ptok_nos) if token["pre_types"][i] == "select#column"])
                    - set([no for i, no in enumerate(ptok_nos) if token["pre_types"][i] == "group by#column"]))
    # : special column (having)
    elif token["pre_types"][step] == "having#aggregate_column":
        cand = list(set([tok_no for i, tok_no in enumerate(ptok_nos) if
                         i >= 1 and token["pre_types"][i - 1] == "select#aggregator" and
                         ptok_nos[i - 1] == ptok_nos[step - 1]])
                    # - set([no for i, no in enumerate(ptok_nos) if
                    #        token["pre_types"][i] == "having#aggregate_column"]))
                    - set([no for i, no in enumerate(ptok_nos) if
                           token["pre_types"][i] == "having#aggregate_column" and
                           ptok_nos[i - 1] == ptok_nos[step - 1]]))
    # : special column (order by)
    elif token["pre_types"][step] == "order by#column":
        cand = list(set([no for i, no in enumerate(ptok_nos) if token["pre_types"][i] == "select#column"])
                    - set([no for i, no in enumerate(ptok_nos) if token["pre_types"][i] == "order by#column"]))
    elif token["pre_types"][step] == "order by#aggregate_column":
        cand = list(set([no for i, no in enumerate(ptok_nos) if
                         i >= 1 and token["pre_types"][i - 1] == "select#aggregator" and
                         ptok_nos[i - 1] == ptok_nos[step - 1]])
                    # - set([no for i, no in enumerate(ptok_nos) if
                    #        token["pre_types"][i] == "order by#aggregate_column"]))
                    - set([no for i, no in enumerate(ptok_nos) if
                           token["pre_types"][i] == "order by#aggregate_column" and
                           ptok_nos[i - 1] == ptok_nos[step - 1]]))
    # : repeated column
    elif "#column" in token["pre_types"][step]:  # select/where
        for tbl in table:
            cand.extend(list(range(word_info[f"{tbl}#column name"]["start_id"],
                                   word_info[f"{tbl}#column name"]["end_id"] + 1)))
        # cand = list(set(cand) - set([no for i, no in enumerate(ptok_nos)
        #                              if token["pre_types"][i] == token["pre_types"][step]]))
        # (0413): newly modified. repeated column allowed in `WHERE` clause.
        if token["pre_types"][step] != "where#column":
            cand = list(set(cand) - set([no for i, no in enumerate(ptok_nos)
                                         if token["pre_types"][i] == token["pre_types"][step]]))
        else:
            cand = list(set(cand))

        # : to be removed.
        # try:
        #     cand = list(set(cand) - set([no for i, no in enumerate(ptok_nos)
        #                                  if token["pre_types"][i] == token["pre_types"][step]]))
        # except:
        #     print("#column")

    # 4) values
    # 4.1) common values: column values and min()/max() aggregate values.
    elif "#value" in token["pre_types"][step]:
        cand = list(range(word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["start_id"],
                          word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["end_id"] + 1))
    elif "#aggregate_value" in token["pre_types"][step]:
        # (0412): newly added. for `(`, `)`
        if "(" in idx2word.values() and ")" in idx2word.values():
            aggregator = idx2word[str(ptok_nos[-3])]
        else:
            aggregator = idx2word[str(ptok_nos[-2])]

        # cand = list(range(word_info[f"{aggregator}#column values"]["start_id"],
        #                   word_info[f"{aggregator}#column values"]["end_id"] + 1))
        try:
            cand = list(range(word_info[f"{aggregator}#column values"]["start_id"],
                              word_info[f"{aggregator}#column values"]["end_id"] + 1))
            sql_res = vec2sql([token], [ptok_nos], idx2word, col_info)[0]
            # sql_res["pre_tokens"] = token
            # sql_res["ptok_nos"] = ptok_nos.tolist()
            # with open("./test.json", "w") as wf:
            #     json.dump(sql_res, wf, indent=2)
        except:
            traceback.print_exc()
            print("#aggregate_value")
            logging.error("#aggregate_value")
            logging.error(traceback.print_exc())
            sql_res = vec2sql([token], [ptok_nos], idx2word, col_info)[0]
            sql_res["pre_tokens"] = token
            # sql_res["ptok_nos"] = ptok_nos.tolist()
            with open("./except.json", "w") as wf:
                json.dump(sql_res, wf, indent=2)

    # elif "#aggregate_value" in token["pre_types"][tno]:
    #     cand = [-1]  # list(range(74, 3008))  # [-1]
    # 4.2) special values: count()/avg()/sum() numeric aggregate values.
    # only one.
    elif len(ptok_nos) >= 3 and "(" not in idx2word.values() and ")" not in idx2word.values() and \
            idx2word[str(ptok_nos[-3])] in perturb_const.aggregator[-3:] and \
            "#numeric_aggregate_value" in token["pre_types"][step]:
        cand = [perturb_const.UNK]  # UNK
    elif len(ptok_nos) >= 5 and "(" in idx2word.values() and ")" in idx2word.values() and \
            idx2word[str(ptok_nos[-5])] in perturb_const.aggregator[-3:] and \
            "#numeric_aggregate_value" in token["pre_types"][step]:
        cand = [perturb_const.UNK]  # UNK
    # more than one. ?
    elif "#numeric_aggregate_value" in token["pre_types"][step]:
        # (0412): newly added. for `(`, `)`
        if "(" in idx2word.values() and ")" in idx2word.values():
            aggregator = idx2word[str(ptok_nos[-3])]
        else:
            aggregator = idx2word[str(ptok_nos[-2])]

        # cand = list(range(word_info[f"{aggregator}#column values"]["start_id"],
        #                   word_info[f"{aggregator}#column values"]["end_id"] + 1))
        try:
            cand = list(range(word_info[f"{aggregator}#column values"]["start_id"],
                              word_info[f"{aggregator}#column values"]["end_id"] + 1))
        except:
            print(traceback.format_exc())
            print("#numeric_aggregate_value")
            logging.error("#numeric_aggregate_value")
            logging.error(traceback.print_exc())
            sql_res = vec2sql([token], [ptok_nos], idx2word, col_info)[0]
            sql_res["pre_tokens"] = token
            with open("./except.json", "w") as wf:
                json.dump(sql_res, wf, indent=2)

    # elif "#numeric_aggregate_value" in token["pre_types"][step]:
    #     cand = [perturb_const.UNK]  # UNK
    # (0412): newly added.
    elif "#like_value" in token["pre_types"][step]:
        # (0414): newly added. for like value.
        if idx2word[str(ptok_nos[-1])] in perturb_const.like_predicate:
            cand = [perturb_const.UNK]
        else:
            cand = list(range(word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["start_id"],
                              word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["end_id"] + 1))

    # 5) operations
    elif "#join_operator" in token["pre_types"][step]:
        cand = [word2idx[token["pre_tokens"][step]]]
    elif "operator" in token["pre_types"][step]:
        cand = list(range(word_info["operator"]["start_id"],
                          word_info["operator"]["end_id"] + 1))

    # : numeric_aggregate_value? min/max
    # elif "select#aggregator" in token["pre_types"][step]:
    #     cand = list(range(word_info["aggregator"]["start_id"],
    #                       word_info["aggregator"]["end_id"] + 1))
    # min/max
    elif token["pre_types"][step] == "select#aggregator" and \
            token["pre_tokens"][step].lower() in perturb_const.aggregator[:2]:
        cand = list(range(word_info["aggregator"]["start_id"],
                          word_info["aggregator"]["start_id"] + 2))
    # : count/avg/sum, static?
    elif token["pre_types"][step] == "select#aggregator" and \
            token["pre_tokens"][step].lower() in perturb_const.aggregator[-3:]:
        cand = [word2idx[token["pre_tokens"][step].lower()]]
    # elif token["pre_types"][step] == "select#aggregator":
    #     cand = [word2idx[token["pre_tokens"][step].lower()]]

    elif token["pre_types"][step] == "having#aggregator":
        cand = [tok_no for i, tok_no in enumerate(ptok_nos)
                if token["pre_types"][i] == "select#aggregator"]
        existed = [tok_no for i, tok_no in enumerate(ptok_nos)
                   if token["pre_types"][i] == "having#aggregator"]
        # not a set, multiple aggregators in the select clause.
        for agg in existed:
            cand.remove(agg)
    elif "order by#aggregator" in token["pre_types"][step]:
        cand = [tok_no for i, tok_no in enumerate(ptok_nos)
                if token["pre_types"][i] == "select#aggregator"]
        existed = [tok_no for i, tok_no in enumerate(ptok_nos)
                   if token["pre_types"][i] == "order by#aggregator"]
        # not a set, multiple aggregators in the select clause.
        for agg in existed:
            cand.remove(agg)
    elif "conjunction" in token["pre_types"][step]:
        cand = list(range(word_info["conjunction"]["start_id"],
                          word_info["conjunction"]["end_id"] + 1))
    elif "order_by_key" in token["pre_types"][step]:
        cand = list(range(word_info["order by key"]["start_id"],
                          word_info["order by key"]["end_id"] + 1))

    # 6) predicate
    elif "null_predicate" in token["pre_types"][step]:
        cand = list(range(word_info["null"]["start_id"],
                          word_info["null"]["end_id"] + 1))
    elif "in_predicate" in token["pre_types"][step]:
        cand = list(range(word_info["in"]["start_id"],
                          word_info["in"]["end_id"] + 1))
    elif "exists_predicate" in token["pre_types"][step]:
        cand = list(range(word_info["exists"]["start_id"],
                          word_info["exists"]["end_id"] + 1))
    elif "like_predicate" in token["pre_types"][step]:
        cand = list(range(word_info["operator"]["start_id"],
                          word_info["operator"]["end_id"] + 1))
        if col_info[idx2word[str(ptok_nos[-1])]]["type"] in ["character", "character varying"]:
            cand.extend(list(range(word_info["like"]["start_id"],
                                   word_info["like"]["end_id"] + 1)))

    # : perturbation step constraint, already decoded words difference (not forcibly).
    #  group by / order by / having clause.
    if np.sum(np.array(token["pno_tokens"][:step]) != np.array(ptok_nos)) >= max_diff \
            and token["pno_tokens"][step] in cand:
        return [token["pno_tokens"][step]]
    # elif is_check and token["pre_types"][step] == "select#column" and \
    #         token["pno_tokens"][step] in cand:
    #     sql_tok = copy.deepcopy(ptok_nos)
    #     temp_cand = copy.deepcopy(cand)
    #     temp_cand.remove(token["pno_tokens"][step])
    #     sql_tok.append(random.choice(temp_cand))
    #
    #     for step in range(len(sql_tok), len(token["pre_types"])):
    #         vec = token["pno_tokens"]
    #         cand = valid_cand(token, table, step, sql_tok, word2idx,
    #                           idx2word, word_info, col_info, max_diff=max_diff)
    #
    #         if vec[step] in cand:
    #             selected = vec[step]
    #             sql_tok.append(selected)
    #         else:
    #             selected = random.choice(cand)
    #             sql_tok.append(selected)
    #
    #     if np.sum(np.array(token["pno_tokens"]) != np.array(sql_tok)) > max_diff:
    #         return [token["pno_tokens"][step]]
    #     else:
    #         return cand
    # elif is_check and token["pre_types"][step] == "where#column" and \
    #         token["pno_tokens"][step] in cand:
    #     sql_tok = copy.deepcopy(ptok_nos)
    #     temp_cand = copy.deepcopy(cand)
    #     temp_cand.remove(token["pno_tokens"][step])
    #     sql_tok.append(random.choice(temp_cand))
    #
    #     for step in range(len(sql_tok), len(token["pre_types"])):
    #         vec = token["pno_tokens"]
    #         cand = valid_cand(token, table, step, sql_tok, word2idx,
    #                           idx2word, word_info, col_info, max_diff=max_diff)
    #
    #         if vec[step] in cand:
    #             selected = vec[step]
    #             sql_tok.append(selected)
    #         else:
    #             selected = random.choice(cand)
    #             sql_tok.append(selected)
    #
    #     if np.sum(np.array(token["pno_tokens"]) != np.array(sql_tok)) > max_diff:
    #         return [token["pno_tokens"][step]]
    #     else:
    #         return cand
    else:
        return cand


def valid_cand_col(token, table, step, ptok_nos, column_left, word2idx,
                   idx2word, word_info, col_info, max_diff=5):
    """

    :param token: dict
    :param table: list(str), table names
    :param step: int, current decoded time-step
    :param ptok_nos: list(int), tokens decoded already
    :param column_left : list(int), the left column candidates
    :param word2idx:
    :param idx2word:
    :param word_info:
    :param col_info:
    :param max_diff:
    :return:
    """
    # : newly added.
    # if step == len(token["pno_tokens"]):
    #     return [perturb_const.SEP]

    # : time-step exceed the max_len.
    # : newly added. "pre_types" -> "pno_tokens"
    if step >= len(token["pno_tokens"]):
        return [perturb_const.PAD]

    # (might not executable): perturbation step constraint, forcibly truncated.
    # if np.sum(np.array(src_vec[:step]) != np.array(ptok_nos)) >= max_diff \
    #         and src_vec[step] in cand:
    #     return [src_vec[step]]

    # only the value is associated with the column selected.
    if "#column" not in token["pre_types"][step] and \
            "#aggregate_column" not in token["pre_types"][step] and \
            "#value" not in token["pre_types"][step] and \
            "#like_value" not in token["pre_types"][step] and \
            "#aggregate_value" not in token["pre_types"][step] and \
            "#numeric_aggregate_value" not in token["pre_types"][step]:
        return [token["pno_tokens"][step]]

    cand = list()
    # 3) columns
    # : special column (aggregate)
    #  type for min()/avg()/count()
    if token["pre_types"][step] == "select#aggregate_column":
        # (0412): newly added. for `(`, `)`
        if "(" in idx2word.values() and ")" in idx2word.values():
            aggregator = idx2word[str(ptok_nos[-2])]
        else:
            aggregator = idx2word[str(ptok_nos[-1])]

        # 3.1) max()/min(): column of all types.
        if aggregator in perturb_const.aggregator[:2]:
            # if aggregator in perturb_const.aggregator[:3]:
            cand.extend(column_left)
        # : count()
        # elif ptoken[-1] == perturb_const.aggregator[2]:
        #     cand.extend([col for col in tbl_col
        #                  if col_info[idx2word[str(col)]]["type"] != "date"])
        # 3.2) count()/avg()/sum(): column of numeric types.
        # elif aggregator in perturb_const.aggregator[-3:]:
        #     cand.extend([col for col in column_left
        #                  if col_info[idx2word[str(col)]]["type"]
        #                  in ["smallint", "integer", "bigint", "numeric"]])
        # *todo(0412): ?
        # filter columns selected in the same clause already.
        cand = list(set(cand) - set([no for i, no in enumerate(ptok_nos)
                                     if token["pre_types"][i] == token["pre_types"][step]]))
        # the only `["smallint", "integer", "bigint", "numeric"] #aggregate_column` has been chosen.
        if len(cand) == 0:
            cand = list(set([no for i, no in enumerate(token["pno_tokens"])
                             if token["pno_tokens"][i - 1] == token["pno_tokens"][step - 1]])
                        - set([no for i, no in enumerate(ptok_nos)
                               if ptok_nos[i - 1] == ptok_nos[step - 1]]))
            # cand.append(token["pno_tokens"][step])
    # : special column (group by)
    elif token["pre_types"][step] == "group by#column":
        cand = list(set([no for i, no in enumerate(ptok_nos) if token["pre_types"][i] == "select#column"])
                    - set([no for i, no in enumerate(ptok_nos) if token["pre_types"][i] == "group by#column"]))
    # : special column (having)
    elif token["pre_types"][step] == "having#aggregate_column":
        cand = list(set([tok_no for i, tok_no in enumerate(ptok_nos) if
                         i >= 1 and token["pre_types"][i - 1] == "select#aggregator" and
                         ptok_nos[i - 1] == ptok_nos[step - 1]])
                    - set([no for i, no in enumerate(ptok_nos) if
                           token["pre_types"][i] == "having#aggregate_column" and
                           ptok_nos[i - 1] == ptok_nos[step - 1]]))
    # : special column (order by)
    elif token["pre_types"][step] == "order by#column":
        cand = list(set([no for i, no in enumerate(ptok_nos) if token["pre_types"][i] == "select#column"])
                    - set([no for i, no in enumerate(ptok_nos) if token["pre_types"][i] == "order by#column"]))
    elif token["pre_types"][step] == "order by#aggregate_column":
        cand = list(set([no for i, no in enumerate(ptok_nos) if
                         i >= 1 and token["pre_types"][i - 1] == "select#aggregator" and
                         ptok_nos[i - 1] == ptok_nos[step - 1]])
                    - set([no for i, no in enumerate(ptok_nos) if
                           token["pre_types"][i] == "order by#aggregate_column" and
                           ptok_nos[i - 1] == ptok_nos[step - 1]]))
    # : repeated column
    elif "#column" in token["pre_types"][step]:  # select/from/where
        # cand = list(set(column_left) - set([no for i, no in enumerate(ptok_nos)
        #                                     if token["pre_types"][i] == token["pre_types"][step]]))
        # (0413): newly modified. repeated column allowed in `WHERE` clause.
        if token["pre_types"][step] != "where#column":  # "from#column"
            # (0417): newly added. for `like_predicate`.
            if len(token["pre_types"]) > step + 1 and token["pre_types"][step + 1] == "like_predicate":
                # cand = list(set(column_left) - set([no for i, no in enumerate(ptok_nos)
                #                                     if token["pre_types"][i] == token["pre_types"][step] and
                #                                     col_info[idx2word[str(no)]]["type"] in
                #                                     ["character", "character varying"]]))
                cand = list(set([no for no in column_left if col_info[idx2word[str(no)]]["type"] in
                                 ["character", "character varying"]])
                            - set([no for i, no in enumerate(ptok_nos) if
                                   token["pre_types"][i] == token["pre_types"][step]]))
            else:
                cand = list(set(column_left) - set([no for i, no in enumerate(ptok_nos)
                                                    if token["pre_types"][i] == token["pre_types"][step]]))
        else:
            # (0417): newly added. for `like_predicate`.
            if len(token["pre_types"]) > step + 1 and token["pre_types"][step + 1] == "like_predicate":
                cand = [no for no in list(set(column_left)) if
                        col_info[idx2word[str(no)]]["type"] in ["character", "character varying"]]
            else:
                cand = list(set(column_left))
        if len(cand) == 0 and (token["pre_types"][step] == "from#column"
                               or token["pre_types"][step] == "where#column"):  # "from#column"
            # (0417): newly added. for `like_predicate`.
            if len(token["pre_types"]) > step + 1 and token["pre_types"][step + 1] == "like_predicate":
                cand = list(set([no for i, no in enumerate(ptok_nos)
                                 if (token["pre_types"][i] == "select#column"
                                     or token["pre_types"][i] == "select#aggregate_column") and
                                 col_info[idx2word[str(no)]]["type"] in ["character", "character varying"]])
                            - set([no for i, no in enumerate(ptok_nos)
                                   if token["pre_types"][i] == token["pre_types"][step]]))
            else:
                cand = list(set([no for i, no in enumerate(ptok_nos)
                                 if token["pre_types"][i] == "select#column"
                                 or token["pre_types"][i] == "select#aggregate_column"])
                            - set([no for i, no in enumerate(ptok_nos)
                                   if token["pre_types"][i] == token["pre_types"][step]]))
        # cand = column_left  # : to be removed, for repeated columns.

    # 4) values
    # 4.1) common values: column values and min()/max() aggregate values.
    elif "#value" in token["pre_types"][step]:
        cand = list(range(word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["start_id"],
                          word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["end_id"] + 1))
    elif "#aggregate_value" in token["pre_types"][step]:
        # (0412): newly added. for `(`, `)`
        if "(" in idx2word.values() and ")" in idx2word.values():
            aggregator = idx2word[str(ptok_nos[-3])]
        else:
            aggregator = idx2word[str(ptok_nos[-2])]

        # cand = list(range(word_info[f"{aggregator}#column values"]["start_id"],
        #                   word_info[f"{aggregator}#column values"]["end_id"] + 1))
        try:
            cand = list(range(word_info[f"{aggregator}#column values"]["start_id"],
                              word_info[f"{aggregator}#column values"]["end_id"] + 1))
        except:
            print(traceback.format_exc())
            print("#aggregate_value")
            logging.error("#aggregate_value")
            logging.error(traceback.print_exc())
            sql_res = vec2sql([token], [ptok_nos], idx2word, col_info)[0]
            sql_res["pre_tokens"] = token
            with open("./except.json", "w") as wf:
                json.dump(sql_res, wf, indent=2)
    # elif "#aggregate_value" in token["pre_types"][tno]:
    #     cand = [-1]  # list(range(74, 3008))  # [-1]
    # 4.2) special values: count()/avg()/sum() numeric aggregate values.
    # only one.
    elif len(ptok_nos) >= 3 and "(" not in idx2word.values() and ")" not in idx2word.values() and \
            idx2word[str(ptok_nos[-3])] in perturb_const.aggregator[-3:] and \
            "#numeric_aggregate_value" in token["pre_types"][step]:
        cand = [perturb_const.UNK]  # UNK
    elif len(ptok_nos) >= 5 and "(" in idx2word.values() and ")" in idx2word.values() and \
            idx2word[str(ptok_nos[-5])] in perturb_const.aggregator[-3:] and \
            "#numeric_aggregate_value" in token["pre_types"][step]:
        cand = [perturb_const.UNK]  # UNK
    # more than one.
    elif "#numeric_aggregate_value" in token["pre_types"][step]:
        # (0412): newly added. for `(`, `)`
        if "(" in idx2word.values() and ")" in idx2word.values():
            aggregator = idx2word[str(ptok_nos[-3])]
        else:
            aggregator = idx2word[str(ptok_nos[-2])]

        # cand = list(range(word_info[f"{aggregator}#column values"]["start_id"],
        #                   word_info[f"{aggregator}#column values"]["end_id"] + 1))
        try:
            cand = list(range(word_info[f"{aggregator}#column values"]["start_id"],
                              word_info[f"{aggregator}#column values"]["end_id"] + 1))
        except:
            print(traceback.format_exc())
            print("#numeric_aggregate_value")
            logging.error("#numeric_aggregate_value")
            logging.error(traceback.print_exc())
            sql_res = vec2sql([token], [ptok_nos], idx2word, col_info)[0]
            sql_res["pre_tokens"] = token
            with open("./except.json", "w") as wf:
                json.dump(sql_res, wf, indent=2)
    # (0412): newly added.
    elif "#like_value" in token["pre_types"][step]:
        # (0414): newly added. for like value.
        if idx2word[str(ptok_nos[-1])] in perturb_const.like_predicate:
            cand = [perturb_const.UNK]
        else:
            cand = list(range(word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["start_id"],
                              word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["end_id"] + 1))

    try:
        assert len(cand) != 0, "The list of `cand` is empty."
    except:
        print(traceback.format_exc())
        print("cand empty error!")
        logging.error("cand empty error!")
        logging.error(traceback.print_exc())
        # valid_cand_col(token, table, step, ptok_nos, column_left, word2idx,
        #                idx2word, word_info, col_info, max_diff)
        sql_res = vec2sql([token], [ptok_nos], idx2word, col_info)[0]
        sql_res["pre_tokens"] = token
        sql_res["ptok_nos"] = ptok_nos.tolist()
        sql_res["column_left"] = column_left
        with open("./col_cand_except.json", "w") as wf:
            json.dump(sql_res, wf, indent=2)

    # : perturbation step constraint, already decoded words difference (not forcibly).
    #  group by / order by / having clause.
    if np.sum(np.array(token["pno_tokens"][:step]) != np.array(ptok_nos)) >= max_diff \
            and token["pno_tokens"][step] in cand:
        return [token["pno_tokens"][step]]
    else:
        return cand


def valid_cand_val(token, table, step, ptok_nos, word2idx,
                   idx2word, word_info, col_info, max_diff=5):
    """

    :param token: dict
    :param table: list(str), table names
    :param step: int, current decoded time-step
    :param ptok_nos: list(int), tokens decoded already
    :param word2idx:
    :param idx2word:
    :param word_info:
    :param col_info:
    :param max_diff:
    :return:
    """
    # : time-step exceed the max_len.
    if step >= len(token["pre_types"]):
        return [perturb_const.PAD]

    # : newly added
    if token["pno_tokens"][step] == perturb_const.SEP:
        return [perturb_const.SEP]

    # (might not executable): perturbation step constraint, forcibly.
    # if np.sum(np.array(src_vec[:step]) != np.array(ptok_nos)) >= max_diff \
    #         and src_vec[step] in cand:
    #     return [src_vec[step]]

    if "#value" not in token["pre_types"][step] and \
            "#aggregate_value" not in token["pre_types"][step] and \
            "#numeric_aggregate_value" not in token["pre_types"][step]:
        return [token["pno_tokens"][step]]

    cand = list()
    # 1) values
    # 1.1) common values: column values and min()/max() aggregate values.
    if "#value" in token["pre_types"][step]:
        cand = list(range(word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["start_id"],
                          word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["end_id"] + 1))
    elif "#aggregate_value" in token["pre_types"][step]:
        # (0412): newly added. for `(`, `)`
        if "(" in idx2word.values() and ")" in idx2word.values():
            aggregator = idx2word[str(ptok_nos[-3])]
        else:
            aggregator = idx2word[str(ptok_nos[-2])]
        try:
            cand = list(range(word_info[f"{aggregator}#column values"]["start_id"],
                              word_info[f"{aggregator}#column values"]["end_id"] + 1))
        except:
            print("#aggregate_value")
            print(traceback.format_exc())
            logging.error("#aggregate_value")
            logging.error(traceback.print_exc())
            # valid_cand_val(token, table, step, ptok_nos, column_left, word2idx,
            #                idx2word, word_info, col_info, max_diff)
            sql_res = vec2sql([token], [ptok_nos], idx2word, col_info)[0]
            sql_res["pre_tokens"] = token
            sql_res["ptok_nos"] = ptok_nos.tolist()
            with open("./except.json", "w") as wf:
                json.dump(sql_res, wf, indent=2)
    # 1.2) special values: count()/avg()/sum() numeric aggregate values.
    # only one.
    elif len(ptok_nos) >= 3 and "(" not in idx2word.values() and ")" not in idx2word.values() and \
            idx2word[str(ptok_nos[-3])] in perturb_const.aggregator[-3:] and \
            "#numeric_aggregate_value" in token["pre_types"][step]:
        cand = [perturb_const.UNK]  # UNK
    elif len(ptok_nos) >= 5 and "(" in idx2word.values() and ")" in idx2word.values() and \
            idx2word[str(ptok_nos[-5])] in perturb_const.aggregator[-3:] and \
            "#numeric_aggregate_value" in token["pre_types"][step]:
        cand = [perturb_const.UNK]  # UNK
    # more than one.
    elif "#numeric_aggregate_value" in token["pre_types"][step]:
        # (0412): newly added. for `(`, `)`
        if "(" in idx2word.values() and ")" in idx2word.values():
            aggregator = idx2word[str(ptok_nos[-3])]
        else:
            aggregator = idx2word[str(ptok_nos[-2])]
        try:
            cand = list(range(word_info[f"{aggregator}#column values"]["start_id"],
                              word_info[f"{aggregator}#column values"]["end_id"] + 1))
        except:
            print("#numeric_aggregate_value")
            print(traceback.format_exc())
            logging.error("#numeric_aggregate_value")
            logging.error(traceback.print_exc())
            # valid_cand_val(token, table, step, ptok_nos, column_left, word2idx,
            #                idx2word, word_info, col_info, max_diff)
            sql_res = vec2sql([token], [ptok_nos], idx2word, col_info)[0]
            sql_res["pre_tokens"] = token
            sql_res["ptok_nos"] = ptok_nos.tolist()
            with open("./except.json", "w") as wf:
                json.dump(sql_res, wf, indent=2)
    # (0412): newly added.
    elif "#like_value" in token["pre_types"][step]:
        # (0414): newly added. for like value.
        if idx2word[str(ptok_nos[-1])] in perturb_const.like_predicate:
            cand = [perturb_const.UNK]
        else:
            cand = list(range(word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["start_id"],
                              word_info[f"{idx2word[str(ptok_nos[-2])]}#column values"]["end_id"] + 1))

    # : perturbation step constraint, already decoded words difference (not forcibly).
    #  group by / order by / having clause.
    if np.sum(np.array(token["pno_tokens"][:step]) != np.array(ptok_nos)) >= max_diff \
            and token["pno_tokens"][step] in cand:
        return [token["pno_tokens"][step]]
    else:
        return cand


def sql2vec(sql_tokens, word2idx):
    """
    format the sql in token_res already.
    sql = mosqlparse.format(mosqlparse.parse(sql))
    :param sql_tokens:
    :param word2idx:
    :return:
    """
    vectors = list()
    for item in sql_tokens:
        # vec = list(map(lambda x: word2idx.get(
        #     str(x).strip("'").split(".")[-1], 3),
        #                item["pre_tokens"]))
        vec = list()
        for i in range(len(item["pre_tokens"])):
            key = str(item["pre_tokens"][i]).strip("'").strip(" ")
            if "operator" in item["pre_types"][i] and \
                    item["pre_tokens"][i] == "<>":
                key = "!="
            elif "aggregator" in item["pre_types"][i]:
                key = key.lower()
            elif "column" in item["pre_types"][i]:
                # : `table`.`column`. add for `JOB`, lower()
                key = key.lower()
                if word2idx.get(key, 1) == 1:
                    key = key.split(".")[-1]
            # : newly added for `<unk>`.
            elif "#numeric_aggregate_value" in item["pre_types"][i]:
                key = "<unk>"
            elif "value" in item["pre_types"][i]:
                key = f"{item['pre_types'][i].split('#')[0]}#_#{key}"
            # (0417): newly added.
            elif "null_predicate" in item["pre_types"][i]:
                key = key.lower()
            elif "in_predicate" in item["pre_types"][i]:
                key = key.lower()
            elif "exists_predicate" in item["pre_types"][i]:
                key = key.lower()
            elif "like_predicate" in item["pre_types"][i]:
                key = key.lower()
            elif "null_operator" in item["pre_types"][i]:
                key = key.lower()

            # only "numeric_aggregate_value" is allowed to be <unk>.
            # if word2idx.get(key, 1) == 1 and \
            #         "numeric_aggregate_value" not in item['pre_types'][i]:
            #     print(item['pre_types'][i])
            vec.append(word2idx.get(key, 1))  # <unk>: 1
        # if np.sum(np.array(vec) == 1) != 0:
        #     print(1)

        vectors.append(vec)

    return vectors


def vec2sql(sql_tokens, sql_vectors, idx2word, col_info, mode="without_table"):
    """

    :param sql_tokens: "<unk>"
    :param sql_vectors:
    :param idx2word:
    :param col_info:
    :param mode:
    :return:
    """

    columns = list(col_info.keys())
    tables = list(set([col_info[key]["table"] for key in col_info]))

    sql_res = list()
    for tok, vec in zip(sql_tokens, sql_vectors):
        res = {"sql_text": "", "sql_token": list(), "pno_tokens": list(map(int, vec))}
        for to, no in zip(tok["pre_tokens"], vec):
            # Filter the special token.
            if no in [perturb_const.BOS, perturb_const.EOS, perturb_const.PAD]:
                continue

            pre1, pre2, pre3, pre5 = "", "", "", ""
            if len(res["sql_token"]) - 1 > 0:
                pre1 = res["sql_token"][len(res["sql_token"]) - 1]
            if len(res["sql_token"]) - 2 > 0:
                pre2 = res["sql_token"][len(res["sql_token"]) - 2]
            if len(res["sql_token"]) - 3 > 0:
                pre3 = res["sql_token"][len(res["sql_token"]) - 3]
            if len(res["sql_token"]) - 5 > 0:
                pre5 = res["sql_token"][len(res["sql_token"]) - 5]

            cur = idx2word[str(no)]

            # (1003): newly added.
            if cur == "OR":
                cur = "AND"
            if cur == "!=":
                cur = "="

            # : newly added. for "\n"
            if isinstance(cur, str):
                cur = cur.replace("\n", " ")
            # for 1) numeric aggregate value, 2) like predicate value, 3) wildcard
            if cur == "<unk>":
                if pre1 in perturb_const.like_predicate \
                        and not str(to).startswith("'"):
                    cur = f"'{to}'"
                else:
                    cur = to
            res["sql_token"].append(cur)

            # (0413): newly modified for wildcard('*').
            # if pre1 in perturb_const.aggregator and cur in columns:
            if pre1 in perturb_const.aggregator and (cur in columns or "*" in str(cur)):
                if mode == "with_table":
                    # (0412): newly added. for parenthesis: `(`, `)`
                    if "(" in idx2word.values() and ")" in idx2word.values():
                        cur = f"{col_info[cur]['table']}.{cur}"
                    else:
                        cur = f"({col_info[cur]['table']}.{cur})"
                else:
                    # (0412): newly added. for parenthesis: `(`, `)`
                    if "(" in idx2word.values() and ")" in idx2word.values():
                        cur = f"{cur}"
                    else:
                        cur = f"({cur})"
            # (0417): newly modified. MIN()/MAX() aggregator,
            #  perturb_const.operator[:2] -> perturb_const.aggregator[:2], agg col op val
            # cur in col_info[pre2]["value"] and \
            elif ((pre3 in perturb_const.aggregator[:2] and
                   "(" not in idx2word.values() and ")" not in idx2word.values()) or
                  (pre5 in perturb_const.aggregator[:2] and
                   "(" in idx2word.values() and ")" in idx2word.values())) and \
                    pre2 in columns and \
                    (pre1 in perturb_const.operator or pre1.lower() in perturb_const.like_predicate) and \
                    cur not in columns and \
                    col_info[pre2]["type"] not in ["smallint", "integer", "bigint", "numeric"]:
                if not cur.startswith("'"):
                    cur = f"'{cur}'"
            # (0417): newly modified. predicate: string value, (agg) col op val
            elif ((pre3 not in perturb_const.aggregator and
                   "(" not in idx2word.values() and ")" not in idx2word.values()) or
                  (pre5 not in perturb_const.aggregator and
                   "(" in idx2word.values() and ")" in idx2word.values())) and \
                    pre2 in columns and \
                    (pre1 in perturb_const.operator or pre1.lower() in perturb_const.like_predicate) and \
                    cur not in columns and \
                    col_info[pre2]["type"] not in ["smallint", "integer", "bigint", "numeric"]:
                if not cur.startswith("'"):
                    cur = f"'{cur}'"

            # (0413): newly modified for wildcard('*').
            if (pre1 in columns or pre1 in tables or "*" in str(pre1)) and \
                    (cur in columns or cur in tables or "*" in str(cur)):
                if cur in columns and mode == "with_table":
                    res["sql_text"] += f", {col_info[cur]['table']}.{cur}"
                else:
                    res["sql_text"] += f", {cur}"
            elif pre1 in columns and cur in perturb_const.aggregator:
                res["sql_text"] += f", {cur}"
            elif pre1 in perturb_const.order_by_key and \
                    (cur in columns or cur in perturb_const.aggregator):
                if cur in columns and mode == "with_table":
                    res["sql_text"] += f", {col_info[cur]['table']}.{cur}"
                else:
                    res["sql_text"] += f", {cur}"
            elif pre1 in perturb_const.aggregator:
                res["sql_text"] += cur
            else:
                if cur in columns and mode == "with_table":
                    res["sql_text"] += f" {col_info[cur]['table']}.{cur}"
                else:
                    res["sql_text"] += f" {cur}"

        res["sql_text"] = res["sql_text"].strip(" ")
        sql_res.append(res)

    return sql_res


def verify_maxdiff(sql_tok, token, vec, table, column_left,
                   word2idx, idx2word, word_info, col_info,
                   mode="value", max_diff=5):
    for step in range(len(sql_tok), len(token["pre_types"])):
        if mode == "all":
            cand = valid_cand(token, table, step, sql_tok, word2idx,
                              idx2word, word_info, col_info, max_diff=max_diff)
        elif mode == "column":
            cand = valid_cand_col(token, table, step, sql_tok, column_left, word2idx,
                                  idx2word, word_info, col_info, max_diff=max_diff)
        elif mode == "value":
            cand = valid_cand_val(token, table, step, sql_tok, word2idx,
                                  idx2word, word_info, col_info, max_diff=max_diff)

        if vec[step] in cand:
            selected = vec[step]
        else:
            selected = random.choice(cand)

        sql_tok.append(selected)
        # only for mode='column'
        if mode == "column" and selected in column_left and \
                (token["pre_types"][step] == "select#column" or
                 (token["pre_types"][step] == "select#aggregate_column" and
                  idx2word[str(token["pno_tokens"][step - 1])] in perturb_const.aggregator[:2]) or
                 token["pre_types"][step] == "from#column" or token["pre_types"][step] == "where#column"):
            column_left.remove(selected)

    return sql_tok


def random_gen(sql_token, word2idx, idx2word, word_info, col_info,
               mode="value", max_diff=5, perturb_prop=0.5,
               seed=666, is_check=False):
    """

    :param sql_token:
    :param word2idx:
    :param idx2word:
    :param word_info:
    :param col_info:
    :param mode:
    :param max_diff:
    :param seed:
    :return:
    """

    random.seed(seed)
    valid_tokens, except_tokens, sql_vectors = list(), list(), list()
    for ino, token in tqdm(enumerate(sql_token)):
        # if ino == 616:
        #     print(1)
        try:
            vec = token["pno_tokens"]
            table = [token["pre_tokens"][i] for i, typ in
                     enumerate(token["pre_types"]) if "table" in typ]
            # only for mode='column'
            column_left = list()
            if mode == "column":
                column_left = [token["pno_tokens"][i] for i, typ in
                               enumerate(token["pre_types"]) if
                               typ == "select#column" or
                               (typ == "select#aggregate_column" and
                                idx2word[str(token["pno_tokens"][i - 1])] in perturb_const.aggregator[:2]) or
                               typ == "from#column" or typ == "where#column"]

            sql_tok = list()
            for step in range(len(token["pre_types"])):
                if mode == "all":
                    cand = valid_cand(token, table, step, sql_tok, word2idx,
                                      idx2word, word_info, col_info, max_diff=max_diff)
                elif mode == "column":
                    cand = valid_cand_col(token, table, step, sql_tok, column_left, word2idx,
                                          idx2word, word_info, col_info, max_diff=max_diff)
                elif mode == "value":
                    cand = valid_cand_val(token, table, step, sql_tok, word2idx,
                                          idx2word, word_info, col_info, max_diff=max_diff)

                if random.uniform(0, 1) > perturb_prop and vec[step] in cand:
                    selected = vec[step]
                else:
                    selected = random.choice(cand)
                    # : further ensure current perturbation won't
                    #       exceed the maximum edit distance allowed.
                    if is_check and selected != vec[step] and vec[step] in cand:
                        temp_sql_tok = copy.deepcopy(sql_tok)
                        temp_sql_tok.append(selected)

                        temp_token = copy.deepcopy(token)
                        temp_token["pno_tokens"][:len(temp_sql_tok)] = temp_sql_tok
                        temp_column_left = copy.deepcopy(column_left)

                        temp_sql_tok = verify_maxdiff(temp_sql_tok, temp_token, vec, table,
                                                      temp_column_left, word2idx, idx2word,
                                                      word_info, col_info, mode=mode, max_diff=max_diff)

                        if np.sum(np.array(temp_sql_tok) != np.array(token["pno_tokens"])) > max_diff:
                            selected = vec[step]

                sql_tok.append(selected)
                # only for mode='column'
                if mode == "column" and selected in column_left and \
                        (token["pre_types"][step] == "select#column" or
                         (token["pre_types"][step] == "select#aggregate_column" and
                          idx2word[str(token["pno_tokens"][step - 1])] in perturb_const.aggregator[:2]) or
                         token["pre_types"][step] == "from#column" or token["pre_types"][step] == "where#column"):
                    column_left.remove(selected)
                    # try:
                    #     sql_tok.append(random.choice(cand))
                    # except:
                    #     print("sql_tok.append(random.choice(cand))")
                # sql_tok.append(random.choice(cand))
            valid_tokens.append(token)
            sql_vectors.append(sql_tok)
        except Exception as e:
            traceback.print_exc()
            vec2sql(sql_token, [sql_tok], idx2word, col_info)
            # cand = valid_cand(token, table, step, sql_tok, word2idx,
            #                   idx2word, word_info, col_info, max_diff=max_diff)
            # cand = valid_cand_col(token, table, step, sql_tok, column_left, word2idx,
            #                       idx2word, word_info, col_info, max_diff=max_diff)
            except_tokens.append((ino, token, str(e)))

    return valid_tokens, except_tokens, sql_vectors
