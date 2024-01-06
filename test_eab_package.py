# -*- coding: utf-8 -*-
# @Project: index_test
# @Module: test_eab_package
# @Author: Wei Zhou
# @Time: 2024/1/1 0:55

import json

from index_eab.eab_utils.workload import Workload
from index_eab.eab_utils.common_utils import get_columns_from_schema, read_row_query
from index_eab.eab_utils.postgres_dbms import PostgresDatabaseConnector

from index_eab.index_advisor.extend_algorithm import ExtendAlgorithm


def test_case():
    # 1. Configuration Setup
    host = "-- your host --"
    port = "-- your port --"
    db_name = "-- your database --"

    user = "-- your user --"
    password = "-- your password --"

    connector = PostgresDatabaseConnector(autocommit=True, host=host, port=port,
                                          db_name=db_name, user=user, password=password)

    # 2. Data Preparation
    schema_load = "/path/your database schema.json"
    with open(schema_load, "r") as rf:
        schema_list = json.load(rf)
    _, columns = get_columns_from_schema(schema_list)

    work_load = "/path/testing workload.json"
    with open(work_load, "r") as rf:
        work_list = json.load(rf)[:1]

    for work in work_list:
        workload = Workload(read_row_query(work, columns,
                                           varying_frequencies=True, seed=666))

        # 3. Index Advisor Evaluation
        config = {"budget_MB": 500, "max_index_width": 2, "max_indexes": 5, "constraint": "storage"}
        index_advisor = ExtendAlgorithm(connector, config)

        indexes = index_advisor.calculate_best_indexes(workload, columns=columns)

        break

    return indexes


if __name__ == "__main__":
    indexes = test_case()
    print(indexes)
