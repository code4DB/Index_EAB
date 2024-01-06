import json
import pickle
import psqlparse

import itertools
import logging

from .index import Index
from .postgres_dbms import PostgresDatabaseConnector

import index_advisor_selector.index_selection.dqn_selection.dqn_utils.Encoding as en
import index_advisor_selector.index_selection.dqn_selection.dqn_utils.ParserForIndex as pi


def candidates_per_query(workload, max_index_width, candidate_generator):
    candidates = list()
    for query in workload.queries:
        candidates.append(candidate_generator(query, max_index_width))

    return candidates


def syntactically_relevant_indexes(query, max_index_width):
    # "Smart column Enumeration for Index Scans (SAEFIS)" or "Brute Force and Ignorance (BFI)"
    # See paper linked in DB2Advis algorithm
    # This implementation is "BFI" and uses all syntactically relevant indexes.
    columns = query.columns
    logging.debug(f"{query}")
    logging.debug(f"Indexable columns: {len(columns)}")

    indexable_columns_per_table = {}
    for column in columns:
        if column.table not in indexable_columns_per_table:
            indexable_columns_per_table[column.table] = set()
        indexable_columns_per_table[column.table].add(column)

    possible_column_combinations = set()
    for table in indexable_columns_per_table:
        columns = indexable_columns_per_table[table]
        for index_length in range(1, max_index_width + 1):
            possible_column_combinations |= set(
                itertools.permutations(columns, index_length)
            )

    logging.debug(f"Potential indexes: {len(possible_column_combinations)}")

    # (0804): newly added. for reproduction.
    return sorted([Index(p) for p in possible_column_combinations])


# (0917): newly added. index candidates generation rules by `DQN`.
def syntactically_relevant_indexes_dqn_rule(db_conf, query_texts, columns, max_index_width):
    result_column_combinations = list()

    enc = en.encoding_schema(db_conf)
    parser = pi.Parser(enc["attr"])

    # (0822): newly added.
    workload_ori = list()
    for texts in query_texts:
        # (0917): newly modified.
        if isinstance(texts, list):
            workload_ori.extend(texts)
        if isinstance(texts, str):
            workload_ori.append(texts)

    workload_ori = list(set(workload_ori))

    for no, query in enumerate(workload_ori):
        b = psqlparse.parse_dict(query)
        parser.parse_stmt(b[0])
        parser.gain_candidates()
    index_candidates = parser.index_candidates
    index_candidates = list(index_candidates)
    index_candidates.sort()

    column_dict = dict()
    for column in columns:
        # (0822): newly modified.
        column_dict[f"{column.table}.{column.name}"] = column
        # column_dict[column.name] = column

    prom_ind = dict()
    for cand in index_candidates:
        # (0822): newly modified.
        tbl = cand.split("#")[0]
        ind_cols = cand.split("#")[-1].split(",")

        # (0917): newly added.
        if len(ind_cols) > max_index_width:
            continue

        if len(ind_cols) not in prom_ind.keys():
            prom_ind[len(ind_cols)] = list()
        try:
            # (0822): newly modified.
            prom_ind[len(ind_cols)].append([column_dict[f"{tbl}.{col}"] for col in ind_cols])
        except Exception as e:
            logging.info("Some columns in the indexes are not involved in the global column set.")

    for key in sorted(prom_ind.keys()):
        result_column_combinations.extend(prom_ind[key])

    # (0917): newly modified.
    return sorted([Index(col) for col in result_column_combinations])


# (0917): newly added. index candidates generation by openGauss.
def syntactically_relevant_indexes_openGauss(db_conf, query_texts, columns, max_index_width):
    result_column_combinations = list()

    col_dict = dict()
    for col in columns:
        col_dict[str(col)] = col

    db_conf["postgresql"]["host"] = "xx.xx.xx.xx"
    db_conf["postgresql"]["user"] = "xxxx"
    db_conf["postgresql"]["password"] = "xxxx"
    db_conf["postgresql"]["port"] = "xxxx"
    connector = PostgresDatabaseConnector(db_conf, autocommit=True)

    api_query = "SELECT \"table\", \"column\" FROM gs_index_advise('{}')"
    for query in query_texts:
        query = query.replace("2) ratio,", "2) AS ratio,")
        tbl_col = connector.exec_fetch(api_query.format(query.replace("'", "''")), one=False)

        for tbl, col in tbl_col:
            cs = col.split(",")
            if cs == [""] or len(cs) > max_index_width:
                continue

            result_column_combinations.append([col_dict[f"{tbl}.{c}"] for c in cs])

    return sorted(list(set([Index(cols) for cols in result_column_combinations])))
