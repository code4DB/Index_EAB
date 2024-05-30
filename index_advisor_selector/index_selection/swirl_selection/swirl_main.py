import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # default: "0"

import torch

import copy
import json
import pickle

import logging
import importlib
import numpy as np

from experiment import Experiment
from gym_db.common import EnvironmentType

from swirl_utils.swirl_com import set_logger, get_parser
from swirl_utils.workload import Query, Workload
from swirl_utils.workload_generator import WorkloadGenerator
from swirl_utils.configuration_parser import ConfigurationParser

# import sys
# sys.path.append("/data/wz/index")

# https://github.com/hyrise/rl_index_selection
# https://stable-baselines.readthedocs.io/en/master/


def train_swirl(args):
    logging.info(f"The training mode is `{args.train_mode}`.")
    with open(args.rl_exp_load, "rb") as rf:
        experiment = pickle.load(rf)
    parallel_environments = experiment.exp_config["parallel_environments"]

    experiment.id = args.exp_id
    cp = ConfigurationParser(args.exp_conf_file)
    experiment.exp_config = cp.config
    experiment.exp_config["parallel_environments"] = parallel_environments
    logging.info(f"The value of `parallel_environments` is `{experiment.exp_config['parallel_environments']}`.")

    experiment._create_experiment_folder()

    log_file = args.log_file.format(args.exp_id)
    set_logger(log_file)

    experiment.comparison_performances = {
        "test": {"Extend": [], "DB2Adv": []},
        "validation": {"Extend": [], "DB2Adv": []}
    }
    experiment.comparison_indexes = {"Extend": set(), "DB2Adv": set()}

    with open(args.work_file, "r") as rf:
        query_text = rf.readlines()

    eval_workload = list()
    for qid, sql in enumerate(query_text):
        query = Query(qid, sql, frequency=1)
        # assign column value to `query` object.
        experiment.workload_generator._store_indexable_columns(query)
        workload = Workload([query], description="the adversarial training")
        eval_workload.append(workload)

    experiment.workload_generator.wl_training = eval_workload
    experiment.workload_generator.wl_validation = [eval_workload]
    experiment.workload_generator.wl_testing = [eval_workload]

    experiment.exp_config["workload"]["validation_testing"]["number_of_workloads"] = len(eval_workload)

    # randomly assign budget to each workload.
    experiment._assign_budgets_to_workloads()

    # Save the workloads into `.pickle` file.
    experiment._pickle_workloads()
    if True:
        # 1) Record the experiment time; 2) Load the configuration;
        # 3) Specify the stable_baselines version; 4) Create the experiment folder.
        experiment = Experiment(args)
        logging.info(f"The value of `parallel_environments` is `{experiment.exp_config['parallel_environments']}`.")

    # only stable_baselines2 supported.
    if experiment.exp_config["rl_algorithm"]["stable_baselines_version"] == 2:
        from stable_baselines.common.callbacks import EvalCallbackWithTBRunningAverage
        from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

        # <class 'stable_baselines.ppo2.ppo2.PPO2'>
        algorithm_class = getattr(
            importlib.import_module("swirl_selection.stable_baselines"),
            experiment.exp_config["rl_algorithm"]["algorithm"])
    else:
        raise ValueError

    if True:
        # 1) Schema information preparation; 2) Workload preparation;
        # 3) Index candidates preparation; 4) Workload embedding / representation.
        experiment.prepare()

        eval_workload = None
        if args.eval_file is not None:
            cp = ConfigurationParser(args.exp_conf_file)
            exp_config = cp.config

            # : number
            if args.max_budgets is not None:
                exp_config["budgets"]["validation_and_testing"] = [int(args.max_budgets)]
            # is_varying_frequencies = exp_config["workload"]["varying_frequencies"]
            is_varying_frequencies = args.varying_frequencies
            if is_varying_frequencies:
                np_rnd = np.random.default_rng(seed=exp_config["random_seed"])

            eval_workload = list()
            if args.eval_file.endswith(".sql"):
                with open(args.eval_file, "r") as rf:
                    query_text = rf.readlines()
            elif args.eval_file.endswith(".json"):
                with open(args.eval_file, "r") as rf:
                    query_text = json.load(rf)

            # (0818): newly added.
            # query_text = experiment.workload_generator._preprocess_queries(query_text)

            # (0822): newly modified.  [["", ""]]; [[(), ()]]
            # [Q1, Q2, ...]
            if isinstance(query_text[0], str):
                sql_list = list()
                for qid, sql in enumerate(query_text):
                    if is_varying_frequencies:
                        frequency = np_rnd.integers(1, 10000, 1)[0]
                    else:
                        frequency = 1
                    query = Query(qid, sql, frequency=frequency)
                    # assign column value to `query` object.
                    experiment.workload_generator._store_indexable_columns(query)
                    sql_list.append(query)

                workload = Workload(sql_list, description="the evaluated workload")
                workload.budget = experiment.rnd.choice(experiment.exp_config["budgets"]["validation_and_testing"])
                eval_workload.append(workload)

            # [[qid1, Q1, freq1], [qid2, Q2, freq2], ...]
            elif isinstance(query_text[0], list) and \
                    isinstance(query_text[0][0], int):
                sql_list = list()
                for item in query_text:
                    if is_varying_frequencies:
                        frequency = item[-1]
                    else:
                        frequency = 1
                    query = Query(item[0], item[1], frequency=frequency)
                    # assign column value to `query` object.
                    experiment.workload_generator._store_indexable_columns(query)
                    sql_list.append(query)

                workload = Workload(sql_list, description="the evaluated workload")
                workload.budget = experiment.rnd.choice(experiment.exp_config["budgets"]["validation_and_testing"])
                eval_workload.append(workload)

            # [[Q1, Q2, ...], [Q1, Q2, ...], ...]
            elif isinstance(query_text[0][0], str):
                for item in query_text:
                    sql_list = list()
                    for qid, it in enumerate(item):
                        if is_varying_frequencies:
                            frequency = np_rnd.integers(1, 10000, 1)[0]
                        else:
                            frequency = 1
                        query = Query(qid, it, frequency=frequency)
                        # assign column value to `query` object.
                        experiment.workload_generator._store_indexable_columns(query)
                        sql_list.append(query)

                    workload = Workload(sql_list, description="the evaluated workload")
                    workload.budget = experiment.rnd.choice(
                        experiment.exp_config["budgets"]["validation_and_testing"])
                    eval_workload.append(workload)

            # [[[qid1, Q1, freq1], [qid2, Q2, freq2], ...], [[qid1, Q1, freq1], [qid2, Q2, freq2], ...], ...]
            elif isinstance(query_text[0][0], list) and \
                    isinstance(query_text[0][0][0], int):
                for item in query_text:
                    sql_list = list()
                    for it in item:
                        if is_varying_frequencies:
                            frequency = it[-1]
                        else:
                            frequency = 1
                        query = Query(it[0], it[1], frequency=frequency)
                        # assign column value to `query` object.
                        experiment.workload_generator._store_indexable_columns(query)
                        sql_list.append(query)

                    workload = Workload(sql_list, description="the evaluated workload")
                    workload.budget = experiment.rnd.choice(
                        experiment.exp_config["budgets"]["validation_and_testing"])
                    eval_workload.append(workload)

        ParallelEnv = SubprocVecEnv if experiment.exp_config["parallel_environments"] > 1 else DummyVecEnv
        # : register the Env, workloads_in
        training_env = ParallelEnv([experiment.make_env(env_id,
                                                        environment_type=EnvironmentType.TRAINING,
                                                        workloads_in=None,
                                                        db_config=experiment.schema.db_config)
                                    for env_id in range(experiment.exp_config["parallel_environments"])])
        training_env = VecNormalize(training_env, norm_obs=True, norm_reward=True,
                                    gamma=experiment.exp_config["rl_algorithm"]["gamma"], training=True)

        # Normalization is applied to improve the network’s learning behavior.
        # save the `experiment` object.
        experiment.model_type = algorithm_class
        with open(f"{experiment.experiment_folder_path}/experiment_object.pickle", "wb") as handle:
            # pickle.dump(experiment, handle, protocol=pickle.DEFAULT_PROTOCOL)
            pickle.dump(experiment, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if args.algo == "swirl":
            model = algorithm_class(policy=experiment.exp_config["rl_algorithm"]["policy"],  # MlpPolicy by default
                                    env=training_env,
                                    # learning_rate=2.5e-3,
                                    verbose=2,
                                    seed=experiment.exp_config["random_seed"],
                                    gamma=experiment.exp_config["rl_algorithm"]["gamma"],
                                    tensorboard_log=args.logdir.format(args.exp_id),  # "tensor_log",
                                    policy_kwargs=copy.copy(
                                        experiment.exp_config["rl_algorithm"]["model_architecture"]
                                    ),  # This is necessary because SB modifies the passed dict.
                                    **experiment.exp_config["rl_algorithm"]["args"])
        elif args.algo == "drlinda" or args.algo == "dqn":
            model = algorithm_class(policy=experiment.exp_config["rl_algorithm"]["policy"],  # MlpPolicy by default
                                    env=training_env,
                                    # learning_rate=2.5e-3,
                                    verbose=2,
                                    seed=experiment.exp_config["random_seed"],
                                    gamma=experiment.exp_config["rl_algorithm"]["gamma"],
                                    tensorboard_log=args.logdir.format(args.exp_id))

        logging.warning(
            f"Creating model with NN architecture(value/policy): {experiment.exp_config['rl_algorithm']['model_architecture']}")

        experiment.set_model(model)

    # : ?  callback_test_env.venv.envs[0].unwrapped.workloads
    callback_test_env = VecNormalize(
        DummyVecEnv([experiment.make_env(0, environment_type=EnvironmentType.TESTING,
                                         # : workloads_in
                                         workloads_in=eval_workload,
                                         db_config=experiment.schema.db_config)]),
        norm_obs=True,
        norm_reward=False,
        gamma=experiment.exp_config["rl_algorithm"]["gamma"],
        training=False)
    test_callback = EvalCallbackWithTBRunningAverage(
        # The number of episodes to test the agent
        n_eval_episodes=experiment.exp_config["workload"]["validation_testing"]["number_of_workloads"],
        # Evaluate the agent every eval_freq call of the callback
        eval_freq=round(experiment.exp_config["validation_frequency"] / experiment.exp_config["parallel_environments"]),
        # The environment used for initialization
        eval_env=callback_test_env,
        verbose=1,
        name="test",
        # Whether the evaluation should use a stochastic or deterministic actions
        deterministic=True,
        comparison_performances=experiment.comparison_performances["test"])

    callback_validation_env = VecNormalize(
        DummyVecEnv([experiment.make_env(0, environment_type=EnvironmentType.VALIDATION,
                                         # : workloads_in
                                         workloads_in=eval_workload,
                                         db_config=experiment.schema.db_config)]),
        norm_obs=True,
        norm_reward=False,
        gamma=experiment.exp_config["rl_algorithm"]["gamma"],
        training=False)
    validation_callback = EvalCallbackWithTBRunningAverage(
        n_eval_episodes=experiment.exp_config["workload"]["validation_testing"]["number_of_workloads"],
        eval_freq=round(experiment.exp_config["validation_frequency"] / experiment.exp_config["parallel_environments"]),
        eval_env=callback_validation_env,
        best_model_save_path=experiment.experiment_folder_path,
        verbose=1,
        name="validation",
        deterministic=True,
        comparison_performances=experiment.comparison_performances["validation"])

    # explain_callbacks = [
    #     IntegratedGradientsCallback(
    #         validation_data=(np.zeros((64, 1)), 1),
    #         class_index=0,
    #         n_steps=1000,
    #         output_dir=args.logdir.format(args.exp_id)
    #     )
    # ]

    callbacks = [validation_callback, test_callback]
    # callbacks = [test_callback]

    # for `continuous` train_mode validation.
    # qtext_list = "/data/wz/index/attack/swirl_selection/exp_res/tpcds_1gb_test_env_par50w/testing_workloads.pickle"
    # evaluation_env = get_eval_env(experiment, model, qtext_list, budget=None)
    # experiment.evaluate_policy(model, evaluation_env, len(eval_workload))
    # np.mean([item["achieved_cost"] for item in evaluation_env.get_attr("episode_performances")[0]])

    # set the `training_start_time`.
    experiment.record_learning_start_time()
    # : newly added `experiment.exp_config["timesteps"]`.
    model.learn(total_timesteps=args.timesteps,
                callback=callbacks,
                tb_log_name=experiment.id)  # the name of the run for tensorboard log
    experiment.finish_learning_save_model(model.get_env(),  # training_env,
                                          validation_callback.moving_average_step * experiment.exp_config[
                                              "parallel_environments"],
                                          validation_callback.best_model_step * experiment.exp_config[
                                              "parallel_environments"])
    experiment.finish_evaluation()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    train_swirl(args)
