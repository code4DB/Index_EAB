import os
import json
import random
import pickle
import numpy as np

import gzip
import logging
import datetime
import importlib

import gym

from index_advisor_selector.index_selection.swirl_selection.gym_db.common import EnvironmentType

from swirl_utils import swirl_com
from swirl_utils.schema import Schema
from swirl_utils.workload_generator import WorkloadGenerator
from swirl_utils.postgres_dbms import PostgresDatabaseConnector
from swirl_utils.configuration_parser import ConfigurationParser


class Experiment(object):
    def __init__(self, args):
        # 1) Record the experiment `start/end` time and the training `start/end` time.
        self._init_times()

        self.args = args
        self.id = args.exp_id

        # (0823): newly added.
        if "seed" not in self.args:
            self.args.seed = 60

        # (0418): newly added.
        # a) workload configurations.
        if "work_num" not in self.args:
            self.args.work_num = -1
        # (0822): newly added.
        if "temp_num" not in self.args:
            self.args.temp_num = None
        # (0821): newly added.
        if "class_num" not in self.args:
            self.args.class_num = -1
        if "work_size" not in self.args:
            self.args.work_size = 1
        if "work_gen" not in self.args:
            self.args.work_gen = "random"
        # (0825): newly added.
        if "is_query_cache" not in self.args:
            self.args.is_query_cache = False
        if "filter_workload_cols" not in self.args:
            self.args.filter_workload_cols = False
        # (0820): newly added.
        if "filter_utilized_columns" not in self.args:
            self.args.filter_utilized_columns = False
        # (0822): newly added.
        if "training_instances" not in self.args:
            self.args.training_instances = None
        if "validation_testing_instances" not in self.args:
            self.args.validation_testing_instances = None

        # b) database configurations.
        if "user" not in self.args:
            self.args.user = None
        if "password" not in self.args:
            self.args.password = None
        if "host" not in self.args:
            self.args.host = None
        if "db_name" not in self.args:
            self.args.db_name = None
        if "port" not in self.args:
            self.args.port = None

        # 2) load the configuration from file.
        cp = ConfigurationParser(args.exp_conf_file)
        self.exp_config = cp.config

        # (0823): newly added.
        if self.args.seed is not None:
            self.exp_config["random_seed"] = self.args.seed

        # (0805): newly added. for `constraint`.
        # c) constraint configurations.
        if "constraint" not in self.exp_config:
            if self.args.algo == "swirl":
                self.exp_config["constraint"] = "storage"
            elif self.args.algo == "drlinda" or self.args.algo == "dqn":
                self.exp_config["constraint"] = "number"

        if args.constraint is not None:
            self.exp_config["constraint"] = args.constraint

        # (0822): newly added.
        if args.training_instances is not None:
            self.exp_config["workload"]["training_instances"] = args.training_instances
        if args.validation_testing_instances is not None:
            self.exp_config["workload"]["validation_testing"]["number_of_workloads"] = args.validation_testing_instances

        self.exp_config["max_indexes"] = args.max_indexes

        self.schema = None
        self.schema_file = args.schema_file
        self.db_config_file = args.db_conf_file

        # 3) determine the baseline method, version 2.
        # config["rl_algorithm"]["stable_baselines_version"] = 2
        self._set_sb_version_specific_methods()

        self.model = None
        self.rnd = random.Random()
        self.rnd.seed(self.exp_config["random_seed"])

        self.comparison_performances = {
            "test": {"Extend": [], "DB2Adv": [], "Drop": []},
            "validation": {"Extend": [], "DB2Adv": [], "Drop": []}
        }
        self.comparison_indexes = {"Extend": set(), "DB2Adv": set(), "Drop": set()}

        self.workload_generator = None
        self.workload_embedder = None
        self.multi_validation_wl = []
        self.evaluated_workloads_strs = []

        self.globally_index_candidates = None
        self.single_column_flat_set = None
        self.globally_index_candidates_flat = None
        self.action_storage_consumptions = None

        self.number_of_features = None  # : 66? 96=50+1+1+4+40
        self.number_of_actions = None  # 3376: 40+336+3000

        # 4) Create the folder to store the experiment `exp_res`.
        self.EXPERIMENT_RESULT_PATH = args.res_save_path
        self._create_experiment_folder()
        log_file = args.log_file.format(args.exp_id)
        swirl_com.set_logger(log_file)

        # (0918): newly added.
        if "max_index_width" in self.args and self.args.max_index_width is not None:
            self.exp_config["max_index_width"] = self.args.max_index_width
        if "action_manager" in self.args and self.args.action_manager is not None:
            self.exp_config["action_manager"] = self.args.action_manager

        if "workload_embedder" in self.args and self.args.workload_embedder is not None:
            self.exp_config["workload_embedder"]["type"] = self.args.workload_embedder
        if "observation_manager" in self.args and self.args.observation_manager is not None:
            self.exp_config["observation_manager"] = self.args.observation_manager

        if "reward_calculator" in self.args and self.args.reward_calculator is not None:
            self.exp_config["reward_calculator"] = self.args.reward_calculator

    def _init_times(self):
        """
        Record the experiment start/end time
        and the training start/end time.
        :return:
        """
        self.start_time = datetime.datetime.now()

        self.end_time = None
        self.training_start_time = None
        self.training_end_time = None

    def _set_sb_version_specific_methods(self):
        if self.exp_config["rl_algorithm"]["stable_baselines_version"] == 2:
            from stable_baselines.common import set_global_seeds as set_global_seeds_sb2
            from stable_baselines.common.evaluation import evaluate_policy as evaluate_policy_sb2
            from stable_baselines.common.vec_env import DummyVecEnv as DummyVecEnv_sb2
            from stable_baselines.common.vec_env import VecNormalize as VecNormalize_sb2
            from stable_baselines.common.vec_env import sync_envs_normalization as sync_envs_normalization_sb2

            self.set_random_seed = set_global_seeds_sb2
            self.evaluate_policy = evaluate_policy_sb2
            self.DummyVecEnv = DummyVecEnv_sb2
            self.VecNormalize = VecNormalize_sb2
            self.sync_envs_normalization = sync_envs_normalization_sb2
        else:
            raise ValueError("There are only versions 2 of StableBaselines.")

    def _create_experiment_folder(self):
        """
        Create the folder to store the experiment exp_res.
        :return:
        """
        assert os.path.isdir(
            self.EXPERIMENT_RESULT_PATH
        ), f"Folder for experiment results should exist at: {self.EXPERIMENT_RESULT_PATH}"

        self.experiment_folder_path = f"{self.EXPERIMENT_RESULT_PATH}/{self.id}"
        assert os.path.isdir(self.experiment_folder_path) is False, (
            f"Experiment folder already exists at: {self.experiment_folder_path} - "
            "terminating here because we don't want to overwrite anything."
        )

        os.mkdir(self.experiment_folder_path)

    def prepare(self):
        """
        1) `Schema`: Load the information of the schema;
        2) `WorkloadGenerator`: Generate the workload(training, testing, validation);
        3) `globally_index_candidates`: Generate globally index candidates for action_space(Wmax=3);
        4) `workload_embedder_class`: Call the `workload_embedder` class.
        :return:
        """
        # 1) Schema information preparation, filter out certain columns.
        # db_config, tables, columns(filtered), database_name(for env register)
        # all_columns: 61 -> schema_columns: 47
        # region: {region.r_comment, region.r_regionkey, region.r_name}
        # nation: {nation.n_comment, nation.n_name, nation.n_regionkey, nation.n_nationkey}
        # supplier: {supplier.s_name, supplier.s_acctbal, supplier.s_comment, supplier.s_phone,
        #            supplier.s_address, supplier.s_nationkey, supplier.s_suppkey}
        self.schema = Schema(self.db_config_file,
                             self.schema_file,
                             # "column_filters": {"TableNumRowsFilter": 10000}
                             self.exp_config["column_filters"],
                             user=self.args.user, password=self.args.password,
                             host=self.args.host, db_name=self.args.db_name, port=self.args.port)

        # 2) Workload preparation
        # schema_columns: 47 -> workload_columns: 40
        # {lineitem.l_linenumber, partsupp.ps_comment, part.p_comment,
        # part.p_retailprice, part.p_mfgr, lineitem.l_comment, orders.o_clerk}

        # (0418): newly added. for multiple-query workload.
        self.exp_config["workload"]["size"] = self.args.work_size
        self.exp_config["workload"]["varying_frequencies"] = self.args.varying_frequencies
        self.workload_generator = WorkloadGenerator(work_config=self.exp_config["workload"],
                                                    work_type=self.args.work_type,
                                                    work_num=self.args.work_num,
                                                    temp_num=self.args.temp_num,
                                                    work_gen=self.args.work_gen,
                                                    work_file=self.args.work_file,
                                                    db_config=self.schema.db_config,
                                                    schema_columns=self.schema.columns,
                                                    random_seed=self.exp_config["random_seed"],
                                                    experiment_id=self.id,
                                                    # (0825): newly added.
                                                    is_query_cache=self.args.is_query_cache,
                                                    is_filter_workload_cols=self.args.filter_workload_cols,
                                                    # (0820): newly modified.
                                                    is_filter_utilized_cols=self.args.filter_utilized_columns)
        logging.info(f"Load the workload ({self.args.work_num}, {self.args.work_gen}) from `{self.args.work_file}`.")

        # randomly assign budget to each workload.
        # : newly added. specify the `storage` budget from the argument.
        if self.args.max_budgets is not None:
            self.exp_config["budgets"]["validation_and_testing"] = [int(self.args.max_budgets)]
        self._assign_budgets_to_workloads()
        # Save the workloads into `.pickle` file.
        self._pickle_workloads()

        # 3) Index candidates preparation
        # indexable columns appears in the workload.
        globally_indexable_columns = self.workload_generator.globally_indexable_columns

        # (0918): newly modified.
        if self.args.cand_gen is None:
            # [[single-column indexes]: 40, [2-column combinations]: 336, [3-column combinations]: 3000...]
            # (0822): newly modified.
            if self.args.algo == "dqn":
                if self.args.temp_expand:
                    temp_load = self.args.temp_load
                else:
                    temp_load = None

                # if work_load.endswith(".pickle"):
                #     self.globally_index_candidates = swirl_com.get_prom_index_candidates_from_pickle(self.args.temp_load,
                #                                                                                      self.args.colinfo_load,
                #                                                                                      globally_indexable_columns)
                # else:
                #     self.globally_index_candidates = swirl_com.get_prom_index_candidates(self.args.temp_load,
                #                                                                          self.args.colinfo_load,
                #                                                                          globally_indexable_columns)

                self.globally_index_candidates = swirl_com.get_prom_index_candidates_original(self.schema.db_config,
                                                                                              self.workload_generator.query_texts,
                                                                                              temp_load,
                                                                                              globally_indexable_columns)

                # self.single_column_flat_set = set(map(lambda x: x[0], globally_indexable_columns))
                # (0327): newly added. the max-width constraint.
                self.globally_index_candidates = self.globally_index_candidates[:self.exp_config["max_index_width"]]
                logging.info(
                    f"DQN: Generate the promising index candidates based on the query from `{self.args.temp_load}`.")
            else:  # set{(Column), ..., (Column)}; 54 + 502
                self.globally_index_candidates = swirl_com.create_column_permutation_indexes(
                    globally_indexable_columns, self.exp_config["max_index_width"])

        elif self.args.cand_gen == "permutation":
            self.globally_index_candidates = swirl_com.create_column_permutation_indexes(
                globally_indexable_columns, self.exp_config["max_index_width"])

        elif self.args.cand_gen == "dqn_rule":
            if self.args.temp_expand:
                temp_load = self.args.temp_load
            else:
                temp_load = None

            self.globally_index_candidates = swirl_com.get_prom_index_candidates_original(self.schema.db_config,
                                                                                          self.workload_generator.query_texts,
                                                                                          temp_load,
                                                                                          globally_indexable_columns)

            self.globally_index_candidates = self.globally_index_candidates[:self.exp_config["max_index_width"]]
            logging.info(
                f"DQN: Generate the promising index candidates based on the query from `{self.args.temp_load}`.")

        elif self.args.cand_gen == "openGauss":
            if self.args.temp_expand:
                temp_load = self.args.temp_load
            else:
                temp_load = None

            self.globally_index_candidates = swirl_com.get_prom_index_candidates_openGauss(self.schema.db_config,
                                                                                           self.workload_generator.query_texts,
                                                                                           temp_load,
                                                                                           globally_indexable_columns)

            # (0919): newly added.
            self.globally_index_candidates[0] = sorted([(column, ) for column in globally_indexable_columns])
            self.globally_index_candidates = self.globally_index_candidates[:self.exp_config["max_index_width"]]
            logging.info(
                f"openGauss: Generate the promising index candidates based on the query from `{self.args.temp_load}`.")

        self.single_column_flat_set = set(map(lambda x: x[0], self.globally_index_candidates[0]))

        # [(column,)....]
        self.globally_index_candidates_flat = [item for sublist in self.globally_index_candidates for item in sublist]
        logging.info(f"Feeding {len(self.globally_index_candidates_flat)} candidates into the environments.")

        # (0820): newly added. `args.algo`
        if self.args.algo != "swirl" or "NonMasking" in self.exp_config["action_manager"]:
            self.action_storage_consumptions = swirl_com.predict_index_sizes(
                self.globally_index_candidates_flat, self.schema.db_config, is_precond=False)
        else:  # `swirl` or `masking`
            self.action_storage_consumptions = swirl_com.predict_index_sizes(
                self.globally_index_candidates_flat, self.schema.db_config, is_precond=True)

        # 4) Workload embedding / representation.
        if (self.args.algo == "swirl" or self.args.algo == "dqn") \
                and "workload_embedder" in self.exp_config:
            workload_embedder_class = getattr(
                importlib.import_module("swirl_selection.swirl_utils.workload_embedder"),
                self.exp_config["workload_embedder"]["type"])
            workload_embedder_connector = PostgresDatabaseConnector(self.schema.db_config, autocommit=True)

            # : newly added, more SQL texts input for large LSI corpus.
            query_texts = self.workload_generator.query_texts
            if self.args.temp_expand:
                if self.args.temp_load.endswith(".json"):
                    with open(self.args.temp_load, "r") as rf:
                        sql_tokens = json.load(rf)

                    # query_texts = list()
                    for token in sql_tokens:
                        if isinstance(token, dict):
                            if "workload" in token.keys():
                                for item in token["workload"]:
                                    query_texts.append([item["sql"]])
                            elif "sql" in token.keys():
                                query_texts.append([token["sql"]])
                        elif isinstance(token, list):
                            for item in token:
                                # (0822): newly modified.
                                if isinstance(item[0], list):
                                    for it in item:
                                        query_texts.append([it[1]])
                                elif isinstance(item[0], int):
                                    query_texts.append([item[1]])
                                elif isinstance(item, str):
                                    query_texts.append([item])

                # (0817): newly added.
                elif self.args.temp_load.endswith(".sql"):
                    with open(self.args.temp_load, "r") as rf:
                        sql_list = rf.readlines()
                    for sql in sql_list:
                        query_texts.append([sql])

                # (0820): remove the duplicate classes.
                qno, query_list = list(), list()
                for no, queries in enumerate(query_texts):
                    query = self.rnd.sample(queries, 1)[0].strip("\n").strip(" ")
                    if query not in query_list:
                        query_list.append(query)
                        qno.append(no)
                query_texts = [texts for i, texts in enumerate(query_texts) if i in qno]

                # (0820): newly added. class num.
                logging.info(f"Expand the templates of the query ({len(query_texts)}) from `{self.args.temp_load}`.")

            # (0821): newly added.
            if self.args.class_num != -1:
                query_texts = self.rnd.sample(query_texts, self.args.class_num)
            logging.info(f"The number of the query class for the workload embedder is {len(query_texts)}.")

            self.workload_embedder = workload_embedder_class(query_texts,
                                                             self.exp_config["workload_embedder"][
                                                                 "representation_size"],
                                                             workload_embedder_connector,
                                                             self.globally_index_candidates)
            workload_embedder_connector.close()

        # class Q:
        #     def __init__(self, text):
        #         self.text = text
        # query = "SELECT * FROM region;"
        # plan = workload_embedder_connector.get_plan(Q(query))
        # self.workload_embedder.get_embeddings(plan)

        # : multi_validation, useless?
        if len(self.workload_generator.wl_validation) > 1:
            for workloads in self.workload_generator.wl_validation:
                self.multi_validation_wl.extend(self.rnd.sample(workloads, min(7, len(workloads))))

    def _assign_budgets_to_workloads(self):
        """
        Randomly assign the budgets to the `validation_and_testing` workloads.
        :return:
        """
        # : not add budget to training workload?
        for workload in self.workload_generator.wl_training:
            workload.budget = self.rnd.choice(self.exp_config["budgets"]["validation_and_testing"])

        for workload_list in self.workload_generator.wl_testing:
            for workload in workload_list:
                workload.budget = self.rnd.choice(self.exp_config["budgets"]["validation_and_testing"])

        for workload_list in self.workload_generator.wl_validation:
            for workload in workload_list:
                workload.budget = self.rnd.choice(self.exp_config["budgets"]["validation_and_testing"])

    def _pickle_workloads(self):
        """
        Save the workloads into `.pickle` file.
        :return:
        """
        with open(f"{self.experiment_folder_path}/testing_workloads.pickle", "wb") as handle:
            pickle.dump(self.workload_generator.wl_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{self.experiment_folder_path}/validation_workloads.pickle", "wb") as handle:
            pickle.dump(self.workload_generator.wl_validation, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def make_env(self, env_id, environment_type=EnvironmentType.TRAINING,
                 workloads_in=None, db_config=None):
        """
        1) Set up the `action_manager` class;
        2) Set up the `observation_manager` class;
        3) Set up the `reward_calculator` class;
        3) Register the environment.

        :param env_id:
        :param environment_type:
        :param workloads_in:
        :param db_config:
        :return:
        """

        def _init():
            # set up the `action_manager` (MultiColumnIndexActionManager) class.
            action_manager_class = getattr(
                importlib.import_module("swirl_selection.swirl_utils.action_manager"),
                self.exp_config["action_manager"])
            action_manager = action_manager_class(
                indexable_column_combinations=self.globally_index_candidates,
                action_storage_consumptions=self.action_storage_consumptions,
                sb_version=self.exp_config["rl_algorithm"]["stable_baselines_version"],
                max_index_width=self.exp_config["max_index_width"],
                max_index_num=self.exp_config["max_indexes"],
                reenable_indexes=self.exp_config["reenable_indexes"],
                # (0819): newly added.
                constraint=self.exp_config["constraint"]
            )
            if self.number_of_actions is None:
                self.number_of_actions = action_manager.number_of_actions

            # set up the `observation_manager` (SingleColumnIndexObservationManager) class.
            observation_manager_config = {
                "number_of_query_classes": self.workload_generator.number_of_query_classes,
                "workload_embedder": self.workload_embedder if "workload_embedder" in self.exp_config else None,
                "workload_size": self.exp_config["workload"]["size"]
            }
            observation_manager_class = getattr(
                importlib.import_module("swirl_selection.swirl_utils.observation_manager"),
                self.exp_config["observation_manager"]
            )
            observation_manager = observation_manager_class(
                action_manager.number_of_columns, observation_manager_config
            )
            if self.number_of_features is None:
                self.number_of_features = observation_manager.number_of_features

            # set up the `reward_calculator` (RelativeDifferenceRelativeToStorageReward) class.
            reward_calculator_class = getattr(
                importlib.import_module("swirl_selection.swirl_utils.reward_calculator"),
                self.exp_config["reward_calculator"]
            )

            if "SCALER" not in self.args or self.args.SCALER is None:
                reward_calculator = reward_calculator_class()
            else:
                reward_calculator = reward_calculator_class(self.args.SCALER)

            if environment_type == EnvironmentType.TRAINING:
                workloads = self.workload_generator.wl_training if workloads_in is None else workloads_in
            elif environment_type == EnvironmentType.TESTING:
                # : Selecting the hardest workload by default?
                workloads = self.workload_generator.wl_testing[-1] if workloads_in is None else workloads_in
            elif environment_type == EnvironmentType.VALIDATION:
                # : Selecting the hardest workload by default?
                workloads = self.workload_generator.wl_validation[-1] if workloads_in is None else workloads_in
            else:
                raise ValueError
                # try:
                #     raise ValueError
                # except:
                #     print(1)

            env = gym.make(
                f"DB-v{self.exp_config['gym_version']}",
                environment_type=environment_type,
                config={
                    "database_name": self.schema.database_name,
                    "globally_index_candidates": self.globally_index_candidates_flat,
                    "workloads": workloads,
                    "random_seed": self.exp_config["random_seed"] + env_id,
                    "max_steps_per_episode": self.exp_config["max_steps_per_episode"],
                    "action_manager": action_manager,
                    "observation_manager": observation_manager,
                    "reward_calculator": reward_calculator,
                    "env_id": env_id,
                    "constraint": self.exp_config["constraint"],
                    "similar_workloads": self.exp_config["workload"]["similar_workloads"],
                },
                db_config=db_config
            )
            return env

        self.set_random_seed(self.exp_config["random_seed"])

        return _init

    def set_model(self, model):
        self.model = model

    # : code duplication with validate_model
    def test_model(self, model):
        model_performances = list()
        for test_wl in self.workload_generator.wl_testing:
            test_env = self.DummyVecEnv(
                [self.make_env(0, EnvironmentType.TESTING, test_wl, db_config=self.schema.db_config)])
            test_env = self.VecNormalize(
                test_env, norm_obs=True, norm_reward=False,
                gamma=self.exp_config["rl_algorithm"]["gamma"], training=False
            )

            if self.model is not None and model != self.model:
                # if model != self.model:
                model.set_env(self.model.env)

            model_performance = self._evaluate_model(model, test_env, len(test_wl))
            model_performances.append(model_performance)
        # performances, run_type
        return model_performances, "test"

    def validate_model(self, model):
        model_performances = list()
        for validation_wl in self.workload_generator.wl_validation:
            validation_env = self.DummyVecEnv(
                [self.make_env(0, EnvironmentType.VALIDATION, validation_wl, db_config=self.schema.db_config)])
            validation_env = self.VecNormalize(
                validation_env,
                norm_obs=True,
                norm_reward=False,
                gamma=self.exp_config["rl_algorithm"]["gamma"],
                training=False,
            )

            if model != self.model:
                model.set_env(self.model.env)

            model_performance = self._evaluate_model(model, validation_env, len(validation_wl))
            model_performances.append(model_performance)
        # performances, run_type
        return model_performances, "validation"

    def _evaluate_model(self, model, evaluation_env, n_eval_episodes):
        # : ?
        training_env = model.get_vec_normalize_env()
        self.sync_envs_normalization(training_env, evaluation_env)

        self.evaluate_policy(model, evaluation_env, n_eval_episodes)

        episode_performances = evaluation_env.get_attr("episode_performances")[0]
        perfs = []
        for perf in episode_performances:
            perfs.append(round(perf["achieved_cost"], 2))

        mean_performance = np.mean(perfs)
        print(f"Mean performance: {mean_performance:.2f} ({perfs})")

        return episode_performances, mean_performance, perfs

    def _evaluate_comparison(self):
        for key, comparison_indexes in self.comparison_indexes.items():
            columns_from_indexes = set()
            for index in comparison_indexes:
                for column in index.columns:
                    columns_from_indexes |= set([column])

            impossible_index_columns = columns_from_indexes - self.single_column_flat_set
            logging.critical(f"{key} finds indexes on these not indexable columns:\n    {impossible_index_columns}")

            assert len(impossible_index_columns) == 0, "Found indexes on not indexable columns."

    def record_learning_start_time(self):
        """
        Set the `training_start_time`.
        :return:
        """
        self.training_start_time = datetime.datetime.now()

    def finish_learning_save_model(self, training_env, moving_average_model_step, best_mean_model_step):
        """
        1) record the `training_end_time`;
        2) save the model(`self.model`) and the environment(`training_env`);
        3) count the necessary information.
        :param training_env:
        :param moving_average_model_step:
        :param best_mean_model_step:
        :return:
        """
        self.training_end_time = datetime.datetime.now()

        self.moving_average_validation_model_at_step = moving_average_model_step
        self.best_mean_model_step = best_mean_model_step

        self.model.save(f"{self.experiment_folder_path}/final_model")
        training_env.save(f"{self.experiment_folder_path}/vec_normalize.pkl")

        self.evaluated_episodes = 0
        for number_of_resets in training_env.get_attr("number_of_resets"):
            self.evaluated_episodes += number_of_resets

        self.total_steps_taken = 0
        for total_number_of_steps in training_env.get_attr("total_number_of_steps"):
            self.total_steps_taken += total_number_of_steps

        self.cache_hits = 0
        self.cost_requests = 0
        self.costing_time = datetime.timedelta(0)
        for cache_info in training_env.env_method("get_cost_eval_cache_info"):
            self.cache_hits += cache_info[1]
            self.cost_requests += cache_info[0]
            self.costing_time += cache_info[2]
        self.costing_time /= self.exp_config["parallel_environments"]

        self.cache_hit_ratio = self.cache_hits / self.cost_requests * 100

        if self.exp_config["pickle_cost_estimation_caches"]:
            caches = []
            for cache in training_env.env_method("get_cost_eval_cache"):
                caches.append(cache)
            combined_caches = {}
            for cache in caches:
                combined_caches = {**combined_caches, **cache}
            with gzip.open(f"{self.experiment_folder_path}/caches.pickle.gzip", "wb") as handle:
                pickle.dump(combined_caches, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # (0825): newly added.
            cost_requests = training_env.venv.envs[0].env.env.env.cost_evaluation.cost_requests
            cache_hits = training_env.venv.envs[0].env.env.env.cost_evaluation.cache_hits

            estimation_duration = training_env.venv.envs[
                0].env.env.env.cost_evaluation.db_connector.cost_estimation_duration
            estimation_num = training_env.venv.envs[0].env.env.env.cost_evaluation.db_connector.cost_estimations

            simulation_num = training_env.venv.envs[0].env.env.env.cost_evaluation.db_connector.simulated_indexes
            simulation_duration = training_env.venv.envs[
                0].env.env.env.cost_evaluation.db_connector.index_simulation_duration

            with open(f"{self.experiment_folder_path}/cost_statistics.json", "w") as wf:
                json.dump({"self.cache_hits": self.cache_hits,
                           "self.cost_requests": self.cost_requests,
                           "self.costing_time": str(self.costing_time),
                           "self.cache_hit_ratio": self.cache_hit_ratio,
                           "cost_requests": cost_requests,
                           "cache_hits": cache_hits,
                           "estimation_duration": estimation_duration,
                           "estimation_num": estimation_num,
                           "simulation_num": simulation_num,
                           "simulation_duration": simulation_duration}, wf, indent=2)

    def finish_evaluation(self):
        """
        1) record the `experiment_end_time`;
        2) evaluation;
        3) write evaluation report into file `_write_report`.
        :return:
        """
        self.end_time = datetime.datetime.now()

        self.model.training = False
        self.model.env.norm_reward = False
        self.model.env.training = False

        # evaluate the final model
        self.test_fm = self.test_model(self.model)[0]
        self.vali_fm = self.validate_model(self.model)[0]

        # evaluate the moving average model
        self.moving_average_model = self.model_type.load(f"{self.experiment_folder_path}/moving_average_model.zip")
        self.moving_average_model.training = False
        self.test_ma = self.test_model(self.moving_average_model)[0]
        self.vali_ma = self.validate_model(self.moving_average_model)[0]
        # : useless?
        if len(self.multi_validation_wl) > 0:
            self.moving_average_model_mv = self.model_type.load(
                f"{self.experiment_folder_path}/moving_average_model_mv.zip"
            )
            self.moving_average_model_mv.training = False
            self.test_ma_mv = self.test_model(self.moving_average_model_mv)[0]
            self.vali_ma_mv = self.validate_model(self.moving_average_model_mv)[0]

        # : ?
        self.moving_average_model_3 = self.model_type.load(f"{self.experiment_folder_path}/moving_average_model_3.zip")
        self.moving_average_model_3.training = False
        self.test_ma_3 = self.test_model(self.moving_average_model_3)[0]
        self.vali_ma_3 = self.validate_model(self.moving_average_model_3)[0]
        # : useless?
        if len(self.multi_validation_wl) > 0:
            self.moving_average_model_3_mv = self.model_type.load(
                f"{self.experiment_folder_path}/moving_average_model_3_mv.zip"
            )
            self.moving_average_model_3_mv.training = False
            self.test_ma_3_mv = self.test_model(self.moving_average_model_3_mv)[0]
            self.vali_ma_3_mv = self.validate_model(self.moving_average_model_3_mv)[0]

        # evaluate the best mean reward model
        self.best_mean_reward_model = self.model_type.load(f"{self.experiment_folder_path}/best_mean_reward_model.zip")
        self.best_mean_reward_model.training = False
        self.test_bm = self.test_model(self.best_mean_reward_model)[0]
        self.vali_bm = self.validate_model(self.best_mean_reward_model)[0]
        # : useless?
        if len(self.multi_validation_wl) > 0:
            self.best_mean_reward_model_mv = self.model_type.load(
                f"{self.experiment_folder_path}/best_mean_reward_model_mv.zip"
            )
            self.best_mean_reward_model_mv.training = False
            self.test_bm_mv = self.test_model(self.best_mean_reward_model_mv)[0]
            self.vali_bm_mv = self.validate_model(self.best_mean_reward_model_mv)[0]

        self._write_report()

        logging.critical(
            (
                f"Finished training of ID {self.id}. Report can be found at "
                f"./{self.experiment_folder_path}/report_ID_{self.id}.txt"
            )
        )

    def _write_report(self):
        with open(f"{self.experiment_folder_path}/report_ID_{self.id}.txt", "w") as f:
            f.write(f"##### Report for Experiment with ID: {self.id} #####\n")
            f.write(f"Description: {self.exp_config['description']}\n")
            f.write("\n")

            f.write(f"Start:                         {self.start_time}\n")
            f.write(f"End:                           {self.end_time}\n")
            f.write(f"Duration:                      {self.end_time - self.start_time}\n")
            f.write("\n")
            f.write(f"Start Training:                {self.training_start_time}\n")
            f.write(f"End Training:                  {self.training_end_time}\n")
            f.write(f"Duration Training:             {self.training_end_time - self.training_start_time}\n")
            f.write(f"Moving Average model at step:  {self.moving_average_validation_model_at_step}\n")
            f.write(f"Mean reward model at step:     {self.best_mean_model_step}\n")
            # f.write(f"Git Hash:                      {subprocess.check_output(['git', 'rev-parse', 'HEAD'])}\n")
            f.write(f"Number of features:            {self.number_of_features}\n")
            f.write(f"Number of actions:             {self.number_of_actions}\n")
            f.write("\n")
            if self.exp_config["workload"]["unknown_queries"] > 0:
                f.write(f"Unknown Query Classes {sorted(self.workload_generator.unknown_query_classes)}\n")
                f.write(f"Known Queries: {self.workload_generator.known_query_classes}\n")
                f.write("\n")
            probabilities = len(self.exp_config["workload"]["validation_testing"]["unknown_query_probabilities"])
            for idx, unknown_query_probability in enumerate(
                    self.exp_config["workload"]["validation_testing"]["unknown_query_probabilities"]
            ):
                f.write(f"Unknown query probability: {unknown_query_probability}:\n")
                f.write("    Final mean performance test:\n")
                test_fm_perfs, self.performance_test_final_model, self.test_fm_details = self.test_fm[idx]
                vali_fm_perfs, self.performance_vali_final_model, self.vali_fm_details = self.vali_fm[idx]

                _, self.performance_test_moving_average_model, self.test_ma_details = self.test_ma[idx]
                _, self.performance_vali_moving_average_model, self.vali_ma_details = self.vali_ma[idx]
                _, self.performance_test_moving_average_model_3, self.test_ma_details_3 = self.test_ma_3[idx]
                _, self.performance_vali_moving_average_model_3, self.vali_ma_details_3 = self.vali_ma_3[idx]
                _, self.performance_test_best_mean_reward_model, self.test_bm_details = self.test_bm[idx]
                _, self.performance_vali_best_mean_reward_model, self.vali_bm_details = self.vali_bm[idx]

                if len(self.multi_validation_wl) > 0:
                    _, self.performance_test_moving_average_model_mv, self.test_ma_details_mv = self.test_ma_mv[idx]
                    _, self.performance_vali_moving_average_model_mv, self.vali_ma_details_mv = self.vali_ma_mv[idx]
                    _, self.performance_test_moving_average_model_3_mv, self.test_ma_details_3_mv = self.test_ma_3_mv[
                        idx
                    ]
                    _, self.performance_vali_moving_average_model_3_mv, self.vali_ma_details_3_mv = self.vali_ma_3_mv[
                        idx
                    ]
                    _, self.performance_test_best_mean_reward_model_mv, self.test_bm_details_mv = self.test_bm_mv[idx]
                    _, self.performance_vali_best_mean_reward_model_mv, self.vali_bm_details_mv = self.vali_bm_mv[idx]

                self.test_fm_wl_budgets = self._get_wl_budgets_from_model_perfs(test_fm_perfs)
                self.vali_fm_wl_budgets = self._get_wl_budgets_from_model_perfs(vali_fm_perfs)

                f.write(
                    (
                        "        Final model:               "
                        f"{self.performance_test_final_model:.2f} ({self.test_fm_details})\n"
                    )
                )
                f.write(
                    (
                        "        Moving Average model:      "
                        f"{self.performance_test_moving_average_model:.2f} ({self.test_ma_details})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Moving Average model (MV): "
                            f"{self.performance_test_moving_average_model_mv:.2f} ({self.test_ma_details_mv})\n"
                        )
                    )
                f.write(
                    (
                        "        Moving Average 3 model:    "
                        f"{self.performance_test_moving_average_model_3:.2f} ({self.test_ma_details_3})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Moving Average 3 mod (MV): "
                            f"{self.performance_test_moving_average_model_3_mv:.2f} ({self.test_ma_details_3_mv})\n"
                        )
                    )
                f.write(
                    (
                        "        Best mean reward model:    "
                        f"{self.performance_test_best_mean_reward_model:.2f} ({self.test_bm_details})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Best mean reward mod (MV): "
                            f"{self.performance_test_best_mean_reward_model_mv:.2f} ({self.test_bm_details_mv})\n"
                        )
                    )
                for key, value in self.comparison_performances["test"].items():
                    if len(value) < 1:
                        continue
                    f.write(f"        {key}:                    {np.mean(value[idx]):.2f} ({value[idx]})\n")
                f.write("\n")
                f.write(f"        Budgets:                   {self.test_fm_wl_budgets}\n")
                f.write("\n")
                f.write("    Final mean performance validation:\n")
                f.write(
                    (
                        "        Final model:               "
                        f"{self.performance_vali_final_model:.2f} ({self.vali_fm_details})\n"
                    )
                )
                f.write(
                    (
                        "        Moving Average model:      "
                        f"{self.performance_vali_moving_average_model:.2f} ({self.vali_ma_details})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Moving Average model (MV): "
                            f"{self.performance_vali_moving_average_model_mv:.2f} ({self.vali_ma_details_mv})\n"
                        )
                    )
                f.write(
                    (
                        "        Moving Average 3 model:    "
                        f"{self.performance_vali_moving_average_model_3:.2f} ({self.vali_ma_details_3})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Moving Average 3 mod (MV): "
                            f"{self.performance_vali_moving_average_model_3_mv:.2f} ({self.vali_ma_details_3_mv})\n"
                        )
                    )
                f.write(
                    (
                        "        Best mean reward model:    "
                        f"{self.performance_vali_best_mean_reward_model:.2f} ({self.vali_bm_details})\n"
                    )
                )
                if len(self.multi_validation_wl) > 0:
                    f.write(
                        (
                            "        Best mean reward mod (MV): "
                            f"{self.performance_vali_best_mean_reward_model_mv:.2f} ({self.vali_bm_details_mv})\n"
                        )
                    )
                for key, value in self.comparison_performances["validation"].items():
                    if len(value) < 1:
                        continue
                    f.write(f"        {key}:                    {np.mean(value[idx]):.2f} ({value[idx]})\n")
                f.write("\n")
                f.write(f"        Budgets:                   {self.vali_fm_wl_budgets}\n")
                f.write("\n")
                f.write("\n")
            f.write("Overall Test:\n")

            def final_avg(values, probabilities):
                val = 0
                for res in values:
                    val += res[1]
                return val / probabilities

            f.write("        Final model:               " f"{final_avg(self.test_fm, probabilities):.2f}\n")
            f.write("        Moving Average model:      " f"{final_avg(self.test_ma, probabilities):.2f}\n")
            if len(self.multi_validation_wl) > 0:
                f.write("        Moving Average model (MV): " f"{final_avg(self.test_ma_mv, probabilities):.2f}\n")
            f.write("        Moving Average 3 model:    " f"{final_avg(self.test_ma_3, probabilities):.2f}\n")
            if len(self.multi_validation_wl) > 0:
                f.write("        Moving Average 3 mod (MV): " f"{final_avg(self.test_ma_3_mv, probabilities):.2f}\n")
            f.write("        Best mean reward model:    " f"{final_avg(self.test_bm, probabilities):.2f}\n")
            if len(self.multi_validation_wl) > 0:
                f.write("        Best mean reward mod (MV): " f"{final_avg(self.test_bm_mv, probabilities):.2f}\n")
            f.write(
                (
                    "        Extend:                    "
                    f"{np.mean(self.comparison_performances['test']['Extend']):.2f}\n"
                )
            )
            f.write(
                (
                    "        DB2Adv:                    "
                    f"{np.mean(self.comparison_performances['test']['DB2Adv']):.2f}\n"
                )
            )
            f.write("\n")
            f.write("Overall Validation:\n")
            f.write("        Final model:               " f"{final_avg(self.vali_fm, probabilities):.2f}\n")
            f.write("        Moving Average model:      " f"{final_avg(self.vali_ma, probabilities):.2f}\n")
            if len(self.multi_validation_wl) > 0:
                f.write("        Moving Average model (MV): " f"{final_avg(self.vali_ma_mv, probabilities):.2f}\n")
            f.write("        Moving Average 3 model:    " f"{final_avg(self.vali_ma_3, probabilities):.2f}\n")
            if len(self.multi_validation_wl) > 0:
                f.write("        Moving Average 3 mod (MV): " f"{final_avg(self.vali_ma_3_mv, probabilities):.2f}\n")
            f.write("        Best mean reward model:    " f"{final_avg(self.vali_bm, probabilities):.2f}\n")
            if len(self.multi_validation_wl) > 0:
                f.write("        Best mean reward mod (MV): " f"{final_avg(self.vali_bm_mv, probabilities):.2f}\n")
            f.write(
                (
                    "        Extend:                    "
                    f"{np.mean(self.comparison_performances['validation']['Extend']):.2f}\n"
                )
            )
            f.write(
                (
                    "        DB2Adv:                    "
                    f"{np.mean(self.comparison_performances['validation']['DB2Adv']):.2f}\n"
                )
            )
            f.write("\n")
            f.write("\n")
            f.write(f"Evaluated episodes:            {self.evaluated_episodes}\n")
            f.write(f"Total steps taken:             {self.total_steps_taken}\n")
            f.write(
                (
                    f"CostEval cache hit ratio:      "
                    f"{self.cache_hit_ratio:.2f} ({self.cache_hits} of {self.cost_requests})\n"
                )
            )
            training_time = self.training_end_time - self.training_start_time
            f.write(
                f"Cost eval time (% of total):   {self.costing_time} ({self.costing_time / training_time * 100:.2f}%)\n"
            )
            # f.write(f"Cost eval time:                {self.costing_time:.2f}\n")

            f.write("\n\n")
            f.write("Used configuration:\n")
            json.dump(self.exp_config, f)
            f.write("\n\n")
            f.write("Evaluated test workloads:\n")
            for evaluated_workload in self.evaluated_workloads_strs[: (len(self.evaluated_workloads_strs) // 2)]:
                f.write(f"{evaluated_workload}\n")
            f.write("Evaluated validation workloads:\n")
            # fmt: off
            for evaluated_workload in self.evaluated_workloads_strs[
                                      (len(self.evaluated_workloads_strs) // 2):]:  # noqa: E203, E501
                f.write(f"{evaluated_workload}\n")
            # fmt: on
            f.write("\n\n")

    def _get_wl_budgets_from_model_perfs(self, perfs):
        wl_budgets = []
        for perf in perfs:
            assert perf["evaluated_workload"].budget == perf["available_budget"], "Budget mismatch!"
            wl_budgets.append(perf["evaluated_workload"].budget)
        return wl_budgets
