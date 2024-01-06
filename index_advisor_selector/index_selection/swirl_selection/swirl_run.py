# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: run_inference
# @Author: Wei Zhou
# @Time: 2022/6/19 15:06

import time
import json
import pickle
import random
import logging
import configparser

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # default: "0"

# from index_eab.eab_algo.swirl_selection.gym_db.common import EnvironmentType
from index_advisor_selector.index_selection.swirl_selection.gym_db.common import EnvironmentType
from index_advisor_selector.index_selection.swirl_selection.swirl_utils import swirl_com
from index_eab.eab_algo.swirl_selection.swirl_utils.swirl_com import get_parser
from index_eab.eab_algo.swirl_selection.swirl_utils.workload import Query, Workload
from index_eab.eab_algo.swirl_selection.swirl_utils.cost_evaluation import CostEvaluation
from index_eab.eab_algo.swirl_selection.swirl_utils.postgres_dbms import PostgresDatabaseConnector
from index_eab.eab_algo.swirl_selection.stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


# https://github.com/hill-a/stable-baselines/issues/599


def pre_infer_obj(exp_load, model_load, env_load, db_conf=None):
    # : 1. Load the `experiment` object.
    # load the configuration, create the experiment folder?
    with open(exp_load, "rb") as rf:
        swirl_exp = pickle.load(rf)

    # (0925): newly added.
    if db_conf is not None and db_conf["postgresql"]["database"] != swirl_exp.schema.db_config["postgresql"]["database"]:
        if swirl_exp.args.algo != "swirl" or "NonMasking" in swirl_exp.exp_config["action_manager"]:
            swirl_exp.action_storage_consumptions = swirl_com.predict_index_sizes(
                swirl_exp.globally_index_candidates_flat, db_conf, is_precond=False)
        else:  # `swirl` or `masking`
            swirl_exp.action_storage_consumptions = swirl_com.predict_index_sizes(
                swirl_exp.globally_index_candidates_flat, db_conf, is_precond=True)

    if "max_indexes" not in swirl_exp.exp_config.keys():
        swirl_exp.exp_config["max_indexes"] = 5
    if "max_budgets" not in swirl_exp.exp_config.keys():
        swirl_exp.exp_config["max_budgets"] = 500
    if "constraint" not in swirl_exp.exp_config.keys():
        if "swirl" in exp_load:
            swirl_exp.exp_config["constraint"] = "storage"
        elif "drlinda" in exp_load or "dqn" in exp_load:
            swirl_exp.exp_config["constraint"] = "number"
    if db_conf is not None:
        swirl_exp.schema.db_config = db_conf

    # : 2. Prepare the `model` and `env` objects. env=
    swirl_model = swirl_exp.model_type.load(model_load)
    swirl_model.training = False

    ParallelEnv = SubprocVecEnv if swirl_exp.exp_config["parallel_environments"] > 1 else DummyVecEnv
    # ParallelEnv = DummyVecEnv
    # training_env.envs[0].env.env.env.connector
    training_env = ParallelEnv([swirl_exp.make_env(env_id,
                                                   environment_type=EnvironmentType.TRAINING,
                                                   workloads_in=None,
                                                   db_config=swirl_exp.schema.db_config)
                                # for env_id in range(1)])
                                for env_id in range(swirl_exp.exp_config["parallel_environments"])])
    swirl_model.set_env(VecNormalize.load(env_load, training_env))
    swirl_model.env.training = True  # False
    # swirl_model.env.norm_obs = True
    # swirl_model.env.norm_reward = False

    return swirl_exp, swirl_model


def get_swirl_res(args, work_list, swirl_exp=None, swirl_model=None):
    random.seed(args.seed)

    if swirl_exp is None or swirl_model is None:
        db_conf = configparser.ConfigParser()
        db_conf.read(args.db_conf_file)
        swirl_exp, swirl_model = pre_infer_obj(args.rl_exp_load, args.rl_model_load,
                                               args.rl_env_load, db_conf=db_conf)

    # : workload with only a single/multiple query.
    # query_text = experiment.workload_generator._load_no_temp_workload(eval_load)

    # : 3. Prepare the evaluated workload.
    if args.work_file is None:
        eval_workload = swirl_exp.workload_generator.wl_testing[0]
    elif args.work_file.endswith(".pickle"):
        with open(args.work_file, "rb") as rf:
            eval_workload = pickle.load(rf)[0]
    else:
        queries = list()
        for qid, sql in enumerate(work_list):
            if isinstance(sql, str):
                if args.varying_frequencies:
                    freq = random.randomint(1, 1000)
                else:
                    freq = 1

                query = Query(qid, sql, frequency=freq)
                # assign column value to `query` object.
                swirl_exp.workload_generator._store_indexable_columns(query)
                queries.append(query)
            elif isinstance(sql, list):
                if args.varying_frequencies:
                    freq = sql[-1]
                else:
                    freq = 1

                query = Query(sql[0], sql[1], frequency=freq)
                # assign column value to `query` object.
                swirl_exp.workload_generator._store_indexable_columns(query)
                queries.append(query)

        eval_workload = Workload(queries, description="")
        eval_workload.budget = args.max_budgets
        eval_workload = [eval_workload]

    n_eval_episodes = len(eval_workload)
    # res = experiment.test_model(trained_model)[0]

    # : 4. Do the evaluation.
    evaluation_env = swirl_exp.DummyVecEnv(
        [swirl_exp.make_env(0, EnvironmentType.TESTING,
                            workloads_in=eval_workload,
                            db_config=swirl_exp.schema.db_config)])
    evaluation_env = swirl_exp.VecNormalize(
        evaluation_env, norm_obs=True, norm_reward=False,
        gamma=swirl_exp.exp_config["rl_algorithm"]["gamma"], training=False
    )

    training_env = swirl_model.get_vec_normalize_env()
    # : Sync eval and train environments when using VecNormalize
    swirl_exp.sync_envs_normalization(training_env, evaluation_env)

    time_start = time.time()

    # VecNormalize: evaluation_env.reset() -> DummyVecEnv: evaluation_env.venv.reset()
    # -> OrderEnforcing<DBEnvV1<DB-v1>>: evaluation_env.venv.envs[env_idx].reset()
    # -> DBEnvV1 (id:DB-v1): evaluation_env.venv.envs[env_idx].env.reset()
    logging.disable(logging.WARNING)
    swirl_exp.evaluate_policy(swirl_model, evaluation_env, n_eval_episodes)
    logging.disable(logging.INFO)

    time_end = time.time()

    performances = evaluation_env.get_attr("episode_performances")[0]
    # np.mean([item["achieved_cost"] for item in performances])

    performances[0]["time_duration"] = time_end - time_start

    no_cost, ind_cost = list(), list()
    total_no_cost, total_ind_cost = 0, 0

    # (0926): newly modified.
    # swirl_exp.schema.db_config → swirl_exp.workload_generator.db_config
    connector = PostgresDatabaseConnector(swirl_exp.schema.db_config, autocommit=True)
    connector.drop_indexes()
    cost_evaluation = CostEvaluation(connector)

    for query in queries:
        work = Workload([query], description="")
        cost = cost_evaluation.calculate_cost(work, "")
        no_cost.append(cost)
        total_no_cost += cost

    for query in queries:
        work = Workload([query], description="")
        cost = cost_evaluation.calculate_cost(work, performances[0]["indexes"])
        ind_cost.append(cost)
        total_ind_cost += cost

    indexes_pre = list()
    for index in performances[0]["indexes"]:
        index_pre = f"{index.columns[0].table.name}#{','.join([col.name for col in index.columns])}"
        indexes_pre.append(index_pre)
    indexes_pre.sort()

    data = {"workload": work_list,
            "indexes": indexes_pre,
            "no_cost": no_cost,
            "total_no_cost": total_no_cost,
            "ind_cost": ind_cost,
            "total_ind_cost": total_ind_cost,
            "sel_info": {"time_duration": performances[0]["time_duration"]}}
    return data


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    args.seed = 666
    args.max_budgets = 500

    args.db_conf_file = "/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf"

    args.rl_exp_load = "/data/wz/index/index_eab/eab_algo/swirl_selection/exp_res/dqn_h_w18_num10w_n800_s333_v1/experiment_object.pickle"
    args.rl_model_load = "/data/wz/index/index_eab/eab_algo/swirl_selection/exp_res/dqn_h_w18_num10w_n800_s333_v1/best_mean_reward_model.zip"
    args.rl_env_load = "/data/wz/index/index_eab/eab_algo/swirl_selection/exp_res/dqn_h_w18_num10w_n800_s333_v1/vec_normalize.pkl"

    args.rl_exp_load = "/data/wz/index/index_eab/eab_algo/swirl_selection/exp_res/dqn_h_query_sto10w_n800_s666_v1/experiment_object.pickle"
    args.rl_model_load = "/data/wz/index/index_eab/eab_algo/swirl_selection/exp_res/dqn_h_query_sto10w_n800_s666_v1/best_mean_reward_model.zip"
    args.rl_env_load = "/data/wz/index/index_eab/eab_algo/swirl_selection/exp_res/dqn_h_query_sto10w_n800_s666_v1/vec_normalize.pkl"

    args.work_file = "/data/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql"
    args.work_file = "/data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_n100_test.json"
    args.work_file = "/data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_query_temp_multi_c100_n1321_test.json"

    data = get_swirl_res(args)
    print(data)
