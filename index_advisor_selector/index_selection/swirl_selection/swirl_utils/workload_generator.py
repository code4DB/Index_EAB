import json
import numpy as np

import copy
import logging
import random

from tqdm import tqdm

import index_advisor_selector.index_selection.swirl_selection.swirl_utils.embedding_utils as embedding_utils
from index_advisor_selector.index_selection.swirl_selection.swirl_utils.cost_evaluation import CostEvaluation
from index_advisor_selector.index_selection.swirl_selection.swirl_utils.swirl_com import get_utilized_indexes

from index_advisor_selector.index_selection.swirl_selection.swirl_utils.candidate_generation import (
    candidates_per_query,
    syntactically_relevant_indexes,
)
from index_advisor_selector.index_selection.swirl_selection.swirl_utils.postgres_dbms import PostgresDatabaseConnector
from index_advisor_selector.index_selection.swirl_selection.swirl_utils.workload import Query, Workload
from index_advisor_selector.index_selection.swirl_selection.swirl_utils.constants import tpch_tables, tpcds_tables, job_table_alias

from .workload_embedder import WorkloadEmbedder


class WorkloadGenerator(object):
    def __init__(self, work_config, work_type, work_file, db_config,
                 schema_columns, random_seed, work_num=-1, temp_num=None,
                 work_gen="random", experiment_id=None, is_query_cache=False,
                 is_filter_workload_cols=True, is_filter_utilized_cols=False):
        # assert work_config["benchmark"] in [
        #     "TPCH",
        #     "TPCDS",
        #     "JOB",
        # ], f"Benchmark '{work_config['benchmark']}' is currently not supported."

        self.rnd = random.Random()
        self.rnd.seed(random_seed)
        self.np_rnd = np.random.default_rng(seed=random_seed)

        # For create view statement differentiation
        self.experiment_id = experiment_id

        self.db_config = db_config
        # all the columns in the schema after the filter of `TableNumRowsFilter`.
        self.schema_columns = schema_columns

        self.benchmark = work_config["benchmark"]  # default: TPC-H

        self.work_num = work_num
        self.temp_num = temp_num
        self.work_gen = work_gen
        self.is_query_cache = is_query_cache

        self.is_varying_frequencies = work_config["varying_frequencies"]  # default: false
        if work_type == "template":
            # : 22/99/33
            self.number_of_query_classes = self._set_number_of_query_classes()
            # default: 2, 17, 20 for TPC-H
            self.excluded_query_classes = set(work_config["excluded_query_classes"])
            # self.query_texts is `list of lists`.
            # Outer list for query classes, inner list for instances of this class
            # load queries that are not template. list(list()): (22, 1),
            # the first row/example in each query_class file.
            self.query_texts = self._retrieve_query_texts(work_file)
        else:
            self.query_texts = self._load_query_classes(work_file)
            # (0822): newly modified.
            if self.temp_num is not None:
                self.number_of_query_classes = self.temp_num
            else:
                self.number_of_query_classes = len(self.query_texts)
            # self.excluded_query_classes = set(work_config["excluded_query_classes"])
            self.excluded_query_classes = set()

        self.query_classes = set(range(1, self.number_of_query_classes + 1))
        self.available_query_classes = self.query_classes - self.excluded_query_classes

        # self.is_filter_utilized_cols = is_filter_utilized_cols

        # schema_columns: 47 -> workload_columns: 40.
        # select the indexable columns according to the workload of `available_query_classes`.
        self.globally_indexable_columns = self._select_indexable_columns(is_filter_workload_cols,
                                                                         is_filter_utilized_cols)
        assert work_config["size"] > 1 or not (
                self.work_gen == "random" and work_config["size"] == 1 and work_config["training_instances"] +
                work_config["validation_testing"]["number_of_workloads"]  # (0820): newly modified. add "random"
                > self.number_of_query_classes), "Can not generate the workload satisfied!"

        num_validation_instances = work_config["validation_testing"]["number_of_workloads"]
        num_test_instances = work_config["validation_testing"]["number_of_workloads"]
        self.wl_validation = []
        self.wl_testing = []

        # : workloads generation, work_config["similar_workloads"] -> `training`.
        if self.work_gen == "random":
            if work_config["similar_workloads"] and work_config["unknown_queries"] == 0:
                # : this branch can probably be removed.
                assert self.is_varying_frequencies, "Similar workloads can only be created with varying frequencies."
                self.wl_validation = [None]
                self.wl_testing = [None]
                _, self.wl_validation[0], self.wl_testing[0] = self._generate_workloads(
                    0, num_validation_instances, num_test_instances, work_config["size"])
                # : `query_class_change_frequency`?
                if "query_class_change_frequency" not in work_config \
                        or work_config["query_class_change_frequency"] is None:
                    self.wl_training = self._generate_similar_workloads(work_config["training_instances"],
                                                                        work_config["size"])
                else:
                    self.wl_training = self._generate_similar_workloads_qccf(
                        work_config["training_instances"], work_config["size"],
                        work_config["query_class_change_frequency"])
            elif work_config["unknown_queries"] > 0:
                assert (
                        work_config["validation_testing"]["unknown_query_probabilities"][-1] > 0
                ), "Query unknown_query_probabilities should be larger 0."

                embedder_connector = PostgresDatabaseConnector(self.db_config, autocommit=True)
                embedder = WorkloadEmbedder(
                    query_texts=self.query_texts,
                    representation_size=0,
                    database_connector=embedder_connector,
                    # : single-column only? Transform `globally_indexable_columns` to list of lists.
                    globally_index_candidates=[list(map(lambda x: [x], self.globally_indexable_columns))],
                    retrieve_plans=True)
                self.unknown_query_classes = embedding_utils.which_queries_to_remove(
                    embedder.plans, work_config["unknown_queries"], random_seed)  # self.excluded_query_classes
                self.unknown_query_classes = frozenset(self.unknown_query_classes) - self.excluded_query_classes

                # `missing_classes`: caused by the operation of `excluded`.
                missing_classes = work_config["unknown_queries"] - len(self.unknown_query_classes)
                # complement if missing, randomly sampled from the set of available query class.
                self.unknown_query_classes = self.unknown_query_classes | frozenset(
                    self.rnd.sample(self.available_query_classes - frozenset(self.unknown_query_classes),
                                    missing_classes)
                )
                assert len(self.unknown_query_classes) == work_config["unknown_queries"]
                # : newly added.
                embedder_connector.close()

                self.known_query_classes = self.available_query_classes - frozenset(self.unknown_query_classes)

                for query_class in self.excluded_query_classes:
                    assert query_class not in self.unknown_query_classes

                logging.critical(f"Global unknown query classes: {sorted(self.unknown_query_classes)}")
                logging.critical(f"Global known query classes: {sorted(self.known_query_classes)}")

                for unknown_query_probability in work_config["validation_testing"]["unknown_query_probabilities"]:
                    _, wl_validation, wl_testing = self._generate_workloads(
                        0,
                        num_validation_instances,
                        num_test_instances,
                        work_config["size"],
                        unknown_query_probability=unknown_query_probability,
                    )
                    self.wl_validation.append(wl_validation)
                    self.wl_testing.append(wl_testing)

                assert (
                        len(self.wl_validation)
                        == len(work_config["validation_testing"]["unknown_query_probabilities"])
                        == len(self.wl_testing)
                ), "Validation/Testing workloads length fail"

                # We are temporarily restricting the available query classes now to exclude certain classes for training
                original_available_query_classes = self.available_query_classes
                self.available_query_classes = self.known_query_classes

                if work_config["similar_workloads"]:
                    if work_config["query_class_change_frequency"] is not None:
                        logging.critical(
                            f"Similar workloads with query_class_change_frequency: {work_config['query_class_change_frequency']}"
                        )
                        self.wl_training = self._generate_similar_workloads_qccf(
                            work_config["training_instances"], work_config["size"],
                            work_config["query_class_change_frequency"]
                        )
                    else:
                        self.wl_training = self._generate_similar_workloads(work_config["training_instances"],
                                                                            work_config["size"])
                else:
                    self.wl_training, _, _ = self._generate_workloads(work_config["training_instances"], 0, 0,
                                                                      work_config["size"])
                # We are removing the restriction now.
                self.available_query_classes = original_available_query_classes
            else:
                self.wl_validation = [None]
                self.wl_testing = [None]
                self.wl_training, self.wl_validation[0], self.wl_testing[0] = self._generate_workloads(
                    work_config["training_instances"], num_validation_instances,
                    num_test_instances, work_config["size"])
        elif self.work_gen == "load":
            self.wl_validation = [None]
            self.wl_testing = [None]

            workloads = self._load_no_temp_workload(work_file)

            num_training_instances = len(workloads) - num_validation_instances - num_test_instances

            # (0818): newly added.
            if num_training_instances <= 0:
                num_training_instances = 1

            self.wl_training = self.rnd.sample(workloads, num_training_instances)
            self.wl_validation[0] = self.rnd.sample(workloads, num_validation_instances)
            self.wl_testing[0] = self.rnd.sample(workloads, num_test_instances)

        # logging.critical(f"Sample instances from training workloads: {self.rnd.sample(self.wl_training, 10)}")
        logging.critical(f"Sample instances from training workloads ({len(self.wl_training)}): {self.rnd.sample(self.wl_training, 1)}")
        logging.info("Finished generating workloads.")

    def _set_number_of_query_classes(self):
        """
        the number of the template queries.
        TPC-H: 22, TPC-DS: 99, JOB: 113.
        :return:
        """
        # (0822): newly modified.
        if self.benchmark == "TPCH":
            return 22
        elif self.benchmark == "TPCH-SKEW":
            return 22
        elif self.benchmark == "TPCDS":
            return 99
        elif self.benchmark == "DSB":
            return 53
        elif self.benchmark == "JOB":
            # return 113
            return 33
        else:
            raise ValueError("Unsupported Benchmark type provided, only TPCH, TPCDS, and JOB supported.")

    def _retrieve_query_texts(self, work_file):
        """
        todo: load all the template queries of each query_class/template?
        :return:
        """
        query_files = [open(f"{work_file}/{self.benchmark}/{self.benchmark}_{file_number}.txt", "r")
                       for file_number in range(1, self.number_of_query_classes + 1)]

        finished_queries = []
        for query_file in query_files:
            queries = query_file.readlines()
            # queries = query_file.readlines()[:1]  # : only the first row/example
            # remove `limit x`, replace the name of the view with the experiment id
            queries = self._preprocess_queries(queries)

            finished_queries.append(queries)
            query_file.close()

        assert len(finished_queries) == self.number_of_query_classes

        return finished_queries

    def _load_query_classes(self, work_file):
        sql_list = list()
        if work_file.endswith(".sql"):
            with open(work_file, "r") as rf:
                sql_list = rf.readlines()

            # (0820): newly modified.
            if self.work_num != -1:
                sql_list = sql_list[:self.work_num]
        elif work_file.endswith(".json"):
            with open(work_file, "r") as rf:
                data = json.load(rf)

            # (0820): newly modified.
            if self.work_num != -1:
                data = data[:self.work_num]

            # (0822). newly added
            query_classes = dict()

            # (0804): newly modified / added.
            for item in data:
                if isinstance(item, dict) and "workload" in item.keys():
                    sql_list.extend([it["sql"] for it in item["workload"]])
                elif isinstance(item, dict) and "sql" in item.keys():
                    sql_list.append(item["sql"])
                elif isinstance(item, list):
                    # (0822): newly modified. data:[item:[info:[]]]
                    if isinstance(item[0], list):
                        for info in item:
                            if info[0] not in query_classes.keys():
                                query_classes[info[0]] = list()
                            query_classes[info[0]].append(info[1])
                    elif isinstance(item[0], str):
                        sql_list.extend(item)
                    elif isinstance(item[0], int):
                        if item[0] not in query_classes.keys():
                            query_classes[item[0]] = list()
                        query_classes[item[0]].append(item[1])
                elif isinstance(item, str):
                    sql_list.append(item)

        # (0822): newly added.
        if len(query_classes) != 0:
            sql_list = [list(set(value)) for key, value in sorted(query_classes.items())]
            finished_queries = list()
            for sql in sql_list:
                # remove `limit x`, replace the name of the view with the experiment id
                queries = self._preprocess_queries(sql)
                finished_queries.append(queries)
        else:
            # remove the duplicate queries
            sql_list = list(set(sql_list))
            finished_queries = list()
            for sql in sql_list:
                # remove `limit x`, replace the name of the view with the experiment id
                queries = self._preprocess_queries([sql])
                finished_queries.append(queries)

        logging.info(f"Load the query classes ({len(finished_queries)}) from `{work_file}`.")

        return finished_queries

    def _load_no_temp_workload(self, work_file):
        # (0825): newly added.
        query_cache = dict()

        workload = list()
        if work_file.endswith(".sql"):
            with open(work_file, "r") as rf:
                data = rf.readlines()
            if self.work_num != -1:
                data = data[:self.work_num]

            # (0818): newly modified.
            sql_list = list()
            for no, sql in enumerate(data):
                if self.is_varying_frequencies:
                    # : to be modified. load directly.
                    frequency = self.np_rnd.integers(1, 10000, 1)[0]
                else:
                    frequency = 1
                query = Query(no, self._preprocess_queries([sql])[0], frequency=frequency)

                # (0825): newly modified.
                if self.is_query_cache and query.nr in query_cache.keys():
                    query.columns = query_cache[query.nr].columns
                else:
                    self._store_indexable_columns(query)
                    query_cache[query.nr] = query

                assert len(query.columns) > 0, f"Query columns should have length > 0: {query.text}"

                sql_list.append(query)

            workload.append(Workload(sql_list))
        elif work_file.endswith(".json"):
            with open(work_file, "r") as rf:
                data = json.load(rf)

            if self.work_num != -1:
                data = data[:self.work_num]

            qno = 0

            # (0820): newly added.
            # [Q1, Q2, ...]
            if isinstance(data[0], str):
                queries = list()
                for sql in data:
                    if self.is_varying_frequencies:
                        frequency = self.np_rnd.integers(1, 10000, 1)[0]
                    else:
                        frequency = 1
                    query = Query(qno, self._preprocess_queries([sql])[0], frequency=frequency)
                    qno += 1

                    # (0825): newly modified.
                    if self.is_query_cache and query.nr in query_cache.keys():
                        query.columns = query_cache[query.nr].columns
                    else:
                        self._store_indexable_columns(query)
                        query_cache[query.nr] = query

                    assert len(query.columns) > 0, f"Query columns should have length > 0: {query.text}"

                    queries.append(query)

                workload.append(Workload(queries))
            else:
                for item in tqdm(data):
                    # [{"workload":[]}, {"workload":[]}, ...]
                    # [{"sql":[]}, {"sql":[]}, ...]
                    if isinstance(item, dict):
                        if "workload" in item.keys():
                            queries = list()
                            for it in item["workload"]:
                                if self.is_varying_frequencies:
                                    frequency = self.np_rnd.integers(1, 10000, 1)[0]
                                else:
                                    frequency = 1
                                query = Query(qno, self._preprocess_queries([it["sql"]])[0], frequency=frequency)
                                qno += 1

                                # (0825): newly modified.
                                if self.is_query_cache and query.nr in query_cache.keys():
                                    query.columns = query_cache[query.nr].columns
                                else:
                                    self._store_indexable_columns(query)
                                    query_cache[query.nr] = query

                                assert len(query.columns) > 0, f"Query columns should have length > 0: {query.text}"

                                queries.append(query)

                            workload.append(Workload(queries))
                        else:
                            if self.is_varying_frequencies:
                                frequency = self.np_rnd.integers(1, 10000, 1)[0]
                            else:
                                frequency = 1
                            query = Query(qno, self._preprocess_queries([item["sql"]])[0], frequency=frequency)
                            qno += 1

                            # (0825): newly modified.
                            if self.is_query_cache and query.nr in query_cache.keys():
                                query.columns = query_cache[query.nr].columns
                            else:
                                self._store_indexable_columns(query)
                                query_cache[query.nr] = query

                            assert len(query.columns) > 0, f"Query columns should have length > 0: {query.text}"

                            workload.append(Workload([query]))

                    # (0822): newly modified.
                    # [[Q1, Q2, ...], [Q1, Q2, ...], ...]
                    # [[qid1, Q1, freq1], [qid2, Q2, freq2], ...]
                    # [[[qid1, Q1, freq1], [qid2, Q2, freq2], ...], [[qid1, Q1, freq1], [qid2, Q2, freq2], ...], ...]
                    elif isinstance(item, list):
                        if isinstance(item[0], str):
                            queries = list()
                            for it in item:
                                if self.is_varying_frequencies:
                                    frequency = self.np_rnd.integers(1, 10000, 1)[0]
                                else:
                                    frequency = 1
                                query = Query(qno, self._preprocess_queries([it])[0], frequency=frequency)
                                qno += 1

                                # (0825): newly modified.
                                if self.is_query_cache and query.nr in query_cache.keys():
                                    query.columns = query_cache[query.nr].columns
                                else:
                                    self._store_indexable_columns(query)
                                    query_cache[query.nr] = query

                                assert len(query.columns) > 0, f"Query columns should have length > 0: {query.text}"

                                queries.append(query)
                            workload.append(Workload(queries))

                        elif isinstance(item[0], int):
                            queries = list()
                            if self.is_varying_frequencies:
                                frequency = item[-1]
                            else:
                                frequency = 1
                            query = Query(item[0], self._preprocess_queries([item[1]])[0], frequency=frequency)

                            # (0825): newly modified.
                            if self.is_query_cache and query.nr in query_cache.keys():
                                query.columns = query_cache[query.nr].columns
                            else:
                                self._store_indexable_columns(query)
                                query_cache[query.nr] = query

                            assert len(query.columns) > 0, f"Query columns should have length > 0: {query.text}"

                            queries.append(query)
                            workload.append(Workload(queries))

                        elif isinstance(item[0], list):
                            queries = list()
                            for it in item:
                                if self.is_varying_frequencies:
                                    frequency = it[-1]
                                else:
                                    frequency = 1
                                query = Query(it[0], self._preprocess_queries([it[1]])[0], frequency=frequency)

                                # (0825): newly modified.
                                if self.is_query_cache and query.nr in query_cache.keys():
                                    query.columns = query_cache[query.nr].columns
                                else:
                                    self._store_indexable_columns(query)
                                    query_cache[query.nr] = query

                                assert len(query.columns) > 0, f"Query columns should have length > 0: {query.text}"

                                queries.append(query)
                            workload.append(Workload(queries))

        logging.info(f"Load the workload from `{work_file}`.")

        return workload

    def _preprocess_queries(self, queries):
        """
        Remove `limit x`, Replace the name of the view with the experiment_id.
        called by `_retrieve_query_texts()`.
        :param queries:
        :return:
        """
        processed_queries = []
        for query in queries:
            # (0818): newly removed.
            # query = query.replace("limit 100", "")
            # query = query.replace("limit 20", "")
            # query = query.replace("limit 10", "")
            query = query.strip()

            if "create view revenue0" in query:
                query = query.replace("revenue0", f"revenue0_{self.experiment_id}")

            processed_queries.append(query)

        return processed_queries

    def _select_indexable_columns(self, is_filter_workload_cols, is_filter_utilized_cols):
        """
        todo: generate a workload randomly based on
        all the available query classes -> get the `indexable_columns`?
        :param only_utilized_indexes:
        :return:
        """
        if is_filter_workload_cols:
            available_query_classes = tuple(self.available_query_classes)
            query_class_frequencies = tuple([1 for _ in range(len(available_query_classes))])

            logging.info(f"Selecting indexable columns on {len(available_query_classes)} query classes.")

            # load the workload for later indexable columns, choose one query per query class randomly.
            workload = self._workloads_from_tuples([(available_query_classes, query_class_frequencies)])[0]

            # return the sorted(by default) list of the indexable columns given the workload.
            indexable_columns = workload.indexable_columns(return_sorted=True)

            # : filter the indexes that do not appear in the query plan.
            if is_filter_utilized_cols:
                indexable_columns = self._only_utilized_indexes(indexable_columns)
        else:
            indexable_columns = self.schema_columns

        selected_columns = []
        global_column_id = 0
        # : to be optimized.
        for column in self.schema_columns:
            if column in indexable_columns:
                column.global_column_id = global_column_id
                global_column_id += 1

                selected_columns.append(column)

        return selected_columns

    def _workloads_from_tuples(self, tuples, unknown_query_probability=None):
        """
        Synthesize the workload based on the query class.
        Randomly choose a query per query_class in the `self.query_texts`.

        :param tuples: [(available_query_classes, query_class_frequencies)]
        :param unknown_query_probability: only add the `description` of the `number` of
            the previously_unseen_queries.
        :return:
        """
        # (0825): newly modified.
        query_cache = dict()

        workloads = list()
        unknown_query_probability = "" if unknown_query_probability is None else unknown_query_probability

        # : tuples is a list whose length is `1`? no,
        #  the call in `_generate_workloads()` to synthesize the training/validation/testing workload.
        # multiple workloads: len(tuples) = number of workloads/instances
        for tup in tuples:
            query_classes, query_class_frequencies = tup

            # single workload: len(query_classes/query_class_frequencies) = number of queries
            queries = list()  # select one query from one query_class
            for query_class, frequency in zip(query_classes, query_class_frequencies):
                # self.query_texts is list of lists.
                # Outer list for query classes, inner list for instances of this class.
                query_text = self.rnd.choice(self.query_texts[query_class - 1])
                query = Query(query_class, query_text, frequency=frequency)

                # (0825): newly modified.
                if self.is_query_cache and query.nr in query_cache.keys():
                    query.columns = query_cache[query.nr].columns
                else:
                    # retrieve the indexable columns(in the WHERE clause) given the query.
                    self._store_indexable_columns(query)
                    query_cache[query.nr] = query

                assert len(query.columns) > 0, f"Query columns should have length > 0: {query.text}"
                # try:
                #     assert len(query.columns) > 0, f"Query columns should have length > 0: {query.text}"
                # except:
                #     # self._store_indexable_columns(query)
                #     print(f"Query columns should have length > 0: {query.text}")

                queries.append(query)

            assert isinstance(queries, list), f"Queries is not of type list but of {type(queries)}"
            # : what's the function of `unknown_query_probability`?
            # the `number` of the previously_unseen_queries
            previously_unseen_queries = (round(unknown_query_probability * len(queries))
                                         if unknown_query_probability != "" else 0)
            workloads.append(Workload(queries,
                                      description=f"Contains {previously_unseen_queries} previously unseen queries."))

        return workloads

    def _store_indexable_columns(self, query):
        """
        Determine the indexable columns given the query.
        :param query:
        :return:
        """
        for column in self.schema_columns:
            column_tmp = [col for col in self.schema_columns if column.name == col.name]
            if len(column_tmp) == 1:
                if column.name in query.text.lower() and \
                        f"{column.table.name}" in query.text.lower():
                    query.columns.append(column)
            else:
                if column.table.name not in tpch_tables + tpcds_tables + list(job_table_alias.keys()):
                    if column.name in query.text.lower() and \
                            f"{column.table.name}" in query.text.lower():
                        query.columns.append(column)
                else:
                    if "." in query.text.lower().split("from")[0] or \
                            ("where" in query.text.lower() and (
                                    "." in query.text.lower().split("where")[0] or
                                    "." in query.text.lower().split("where")[-1].split(" ")[1])):
                        if str(column) in query.text.lower():
                            query.columns.append(column)
                        if " as " in query.text.lower():
                            tbl, col = str(column).split(".")
                            if f" {job_table_alias[tbl]}.{col}" in query.text.lower() \
                                    or f"({job_table_alias[tbl]}.{col}" in query.text.lower():
                                query.columns.append(column)

        # for column in self.schema_columns:
        #     # (0329): newly modified. for JOB,
        #     #  SELECT COUNT(*), too many candidates.
        #     if "." in query.text.lower().split("from")[0] or \
        #             ("where" in query.text.lower() and ("." in query.text.lower().split("where")[0] or
        #                                                 "." in query.text.lower().split("where")[-1].split(" ")[1])):
        #         if str(column) in query.text.lower():
        #             query.columns.append(column)
        #     else:
        #         # (0408): newly added. check?
        #         # if column.name in query.text:
        #         if column.name in query.text.lower() and \
        #                 f"{column.table.name}" in query.text.lower():
        #             query.columns.append(column)

        # if self.benchmark != "JOB":
        #     for column in self.schema_columns:
        #         if column.name in query.text.lower():  # : lower(completed), value.
        #             query.columns.append(column)
        # else:
        #     query_text = query.text.lower()  # : lower(completed), value.
        #     # assert "WHERE" in query_text, f"Query without WHERE clause encountered: {query_text} in {query.nr}"
        #     assert "where" in query_text, f"Query without WHERE clause encountered: {query_text} in {query.nr}"
        #
        #     # split = query_text.split("WHERE")
        #     #  (0329): newly added.
        #     split = query_text.split("where", 1)
        #     # assert len(split) == 2, f"Query split for JOB query contains subquery: {query_text} in {query.nr}"
        #     query_text_before_where = split[0]
        #     query_text_after_where = split[1]
        #
        #     for column in self.schema_columns:
        #         # : only column in where clause. why, existed?
        #         # if column.name in query_text_after_where and f" {column.table.name}" in query_text_before_where:
        #         #     query.columns.append(column)
        #         if column.name in query_text_after_where and f"{column.table.name}" in query_text_before_where:
        #             query.columns.append(column)

    def _only_utilized_indexes(self, indexable_columns):
        """
        todo: filter the indexes that do not appear in the query plan
              after the use of hypothetical index plugin?
        :param indexable_columns:
        :return:
        """
        frequencies = [1 for _ in range(len(self.available_query_classes))]
        workload_tuple = (self.available_query_classes, frequencies)
        workload = self._workloads_from_tuples([workload_tuple])[0]

        candidates = candidates_per_query(workload,
                                          max_index_width=1,
                                          candidate_generator=syntactically_relevant_indexes)

        # :
        # connector = PostgresDatabaseConnector(self.database_name, self.db_config?, autocommit=True)
        connector = PostgresDatabaseConnector(self.db_config, autocommit=True)
        connector.drop_indexes()
        cost_evaluation = CostEvaluation(connector)

        utilized_indexes, query_details = get_utilized_indexes(workload, candidates, cost_evaluation, True)

        columns_of_utilized_indexes = set()
        for utilized_index in utilized_indexes:
            column = utilized_index.columns[0]
            columns_of_utilized_indexes.add(column)

        output_columns = columns_of_utilized_indexes & set(indexable_columns)
        excluded_columns = set(indexable_columns) - output_columns
        logging.critical(f"Excluding columns based on utilization:\n   {excluded_columns}")

        return output_columns

    def _generate_workloads(self, train_instances, validation_instances, test_instances,
                            size, unknown_query_probability=None):
        """
        _generate_random_workload: tuples(query_class: int, frequency: int) -> workload.
        train_instances/validation_instances/test_instances: not overlapped.
        :param train_instances:
        :param validation_instances:
        :param test_instances:
        :param size:
        :param unknown_query_probability:
        :return:
        """
        required_unique_workloads = train_instances + validation_instances + test_instances

        unique_workload_tuples = set()
        while required_unique_workloads > len(unique_workload_tuples):
            # (workload_query_classes: ordered, query_class_frequencies)
            workload_tuple = self._generate_random_workload(size, unknown_query_probability)
            unique_workload_tuples.add(workload_tuple)

        validation_tuples = self.rnd.sample(unique_workload_tuples, validation_instances)
        unique_workload_tuples = unique_workload_tuples - set(validation_tuples)

        test_workload_tuples = self.rnd.sample(unique_workload_tuples, test_instances)
        unique_workload_tuples = unique_workload_tuples - set(test_workload_tuples)

        assert len(unique_workload_tuples) == train_instances
        train_workload_tuples = unique_workload_tuples

        assert (len(train_workload_tuples) + len(test_workload_tuples)
                + len(validation_tuples) == required_unique_workloads)

        # list(Object(Workload))
        validation_workloads = self._workloads_from_tuples(validation_tuples, unknown_query_probability)
        test_workloads = self._workloads_from_tuples(test_workload_tuples, unknown_query_probability)
        train_workloads = self._workloads_from_tuples(train_workload_tuples, unknown_query_probability)
        # set([str(sorted(list(map(str, work.queries)))) for work in train_workloads])
        return train_workloads, validation_workloads, test_workloads

    def _generate_random_workload(self, size, unknown_query_probability=None):
        """
        Determine the query_class of a single workload randomly
        (take `unknown_query_probability` into account).
        :param size: the number of queries in the workload.
        :param unknown_query_probability:
            the proportion of the unknown_query in the workload (for validation/testing).
        :return:
        """
        assert size <= self.number_of_query_classes, "Cannot generate workload with more queries than query classes"

        # 1) determine query class
        if unknown_query_probability is not None:
            number_of_unknown_queries = round(size * unknown_query_probability)  # default 0 digits
            number_of_known_queries = size - number_of_unknown_queries
            assert number_of_known_queries + number_of_unknown_queries == size

            known_query_classes = self.rnd.sample(self.known_query_classes, number_of_known_queries)
            unknown_query_classes = self.rnd.sample(self.unknown_query_classes, number_of_unknown_queries)

            query_classes = known_query_classes
            query_classes.extend(unknown_query_classes)
            workload_query_classes = tuple(query_classes)
            # (0820): to be sorted.
            # workload_query_classes = sorted(workload_query_classes)
            assert len(workload_query_classes) == size
        else:
            # (0820): to be sorted.
            workload_query_classes = tuple(self.rnd.sample(self.available_query_classes, size))
            # workload_query_classes = sorted(workload_query_classes)

        # 2) determine query frequencies
        if self.is_varying_frequencies:
            query_class_frequencies = tuple(list(self.np_rnd.integers(1, 10000, size)))
        else:
            query_class_frequencies = tuple([1 for _ in range(size)])

        workload_tuple = (workload_query_classes, query_class_frequencies)

        return workload_tuple

    def _generate_similar_workloads(self, instances, size):
        """
        # The core idea is to create workloads that are similar and only change slightly from one to another.
        # For the following workload, we:
        # 1) remove one random element,
        # 2) add another random one (available_classes - current_classes) with frequency, and
        # 3) randomly change the frequency of one element (including the new one).
        :param instances:
        :param size:
        :return:
        """
        assert size <= len(self.available_query_classes), \
            "Cannot generate workload with more queries than query classes"

        workload_tuples = []
        query_classes = self.rnd.sample(self.available_query_classes, size)
        # (0820): to be sorted.
        available_query_classes = self.available_query_classes - frozenset(query_classes)
        frequencies = list(self.np_rnd.zipf(1.5, size))

        workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        # : Remove a random element iteratively.
        for workload_idx in range(instances - 1):
            # Remove a random element
            idx_to_remove = self.rnd.randrange(len(query_classes))
            query_classes.pop(idx_to_remove)
            frequencies.pop(idx_to_remove)

            # Draw a new random element, the removed one is excluded
            query_classes.append(self.rnd.sample(available_query_classes, 1)[0])
            frequencies.append(self.np_rnd.zipf(1.5, 1)[0])

            frequencies[self.rnd.randrange(len(query_classes))] = self.np_rnd.zipf(1.5, 1)[0]

            available_query_classes = self.available_query_classes - frozenset(query_classes)
            workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        workloads = self._workloads_from_tuples(workload_tuples)

        return workloads

    # This version uses the same query id selection for `query_class_change_frequency` workloads.
    def _generate_similar_workloads_qccf(self, instances, size, query_class_change_frequency):
        """
        Change the query class in the workload every `query_class_change_frequency` round.
        :param instances:
        :param size:
        :param query_class_change_frequency:
        :return:
        """
        assert size <= len(
            self.available_query_classes
        ), "Cannot generate workload with more queries than query classes"

        workload_tuples = []

        while len(workload_tuples) < instances:
            if len(workload_tuples) % query_class_change_frequency == 0:
                query_classes = self.rnd.sample(self.available_query_classes, size)

            frequencies = list(self.np_rnd.integers(1, 10000, size))
            workload_tuples.append((copy.copy(query_classes), copy.copy(frequencies)))

        workloads = self._workloads_from_tuples(workload_tuples)

        return workloads
