# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: heu_run
# @Author: Wei Zhou
# @Time: 2023/8/16 14:42

import json
import configparser
from tqdm import tqdm

from index_advisor_selector.index_selection.heu_selection.heu_utils import heu_com
from index_advisor_selector.index_selection.heu_selection.heu_utils.index import Index
from index_advisor_selector.index_selection.heu_selection.heu_utils.workload import Workload
from index_advisor_selector.index_selection.heu_selection.heu_utils.heu_com import get_parser
from index_advisor_selector.index_selection.heu_selection.heu_utils.postgres_dbms import PostgresDatabaseConnector

from index_advisor_selector.index_selection.heu_selection.heu_algos.auto_admin_algorithm import AutoAdminAlgorithm
from index_advisor_selector.index_selection.heu_selection.heu_algos.db2advis_algorithm import DB2AdvisAlgorithm, IndexBenefit
from index_advisor_selector.index_selection.heu_selection.heu_algos.drop_algorithm import DropAlgorithm
from index_advisor_selector.index_selection.heu_selection.heu_algos.extend_algorithm import ExtendAlgorithm
from index_advisor_selector.index_selection.heu_selection.heu_algos.relaxation_algorithm import RelaxationAlgorithm
from index_advisor_selector.index_selection.heu_selection.heu_algos.anytime_algorithm import AnytimeAlgorithm
from index_advisor_selector.index_selection.heu_selection.heu_algos.cophy_algorithm import CoPhyAlgorithm

ALGORITHMS = {
    "auto_admin": AutoAdminAlgorithm,
    "db2advis": DB2AdvisAlgorithm,
    "drop": DropAlgorithm,
    "extend": ExtendAlgorithm,
    "relaxation": RelaxationAlgorithm,
    "anytime": AnytimeAlgorithm,
    "cophy": CoPhyAlgorithm
}


class IndexEncoder(json.JSONEncoder):
    def default(self, obj):
        # 👇️ if passed in object is instance of Decimal
        # convert it to a string

        # db2advis
        if isinstance(obj, IndexBenefit):
            return str(obj)
        if isinstance(obj, Index):
            return str(obj)
        if "Workload" in str(obj.__class__):
            return str(obj)
        if "Index" in str(obj.__class__):
            return str(obj)
        if "Column" in str(obj.__class__):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)

        # 👇️ otherwise use the default behavior
        return json.JSONEncoder.default(self, obj)


def get_heu_result(args, algos, work_list):
    # if args.algo is not None:
    #     algos = [args.algo]
    # else:
    #     algos = ALGORITHMS.keys()

    db_conf = configparser.ConfigParser()
    db_conf.read(args.db_conf_file)

    _, columns = heu_com.get_columns_from_schema(args.schema_file)

    connector = PostgresDatabaseConnector(db_conf, autocommit=True, host=args.host, port=args.port,
                                          db_name=args.db_name, user=args.user, password=args.password)

    res_data = dict()
    for algo in tqdm(algos):
        # indexes, no_cost, total_no_cost, ind_cost, total_ind_cost, sel_info
        exp_conf_file = args.exp_conf_file.format(algo)

        with open(exp_conf_file, "r") as rf:
            exp_config = json.load(rf)

        configs = heu_com.find_parameter_list(exp_config["algorithms"][0],
                                                params=args.sel_params)
        # (0824): newly modified.
        workload = Workload(heu_com.read_row_query(work_list, exp_config, columns, type="",
                                                     varying_frequencies=args.varying_frequencies, seed=args.seed))

        data = list()
        for config in tqdm(configs):
            connector.drop_hypo_indexes()

            # (0818): newly added.
            if args.constraint is not None:
                config["parameters"]["constraint"] = args.constraint
            if args.budget_MB is not None:
                config["parameters"]["budget_MB"] = args.budget_MB
            if args.max_indexes is not None:
                config["parameters"]["max_indexes"] = args.max_indexes

            # (0926): newly added.
            if "max_index_width" in args and args.max_index_width is not None:
                config["parameters"]["max_index_width"] = args.max_index_width

            # (0918): newly added.
            if algo == "drop" and "multi_column" in args:
                config["parameters"]["multi_column"] = args.multi_column

            # (1211): newly added. for `cophy`
            if algo == "cophy":
                config["parameters"]["ampl_bin_path"] = args.ampl_bin_path
                config["parameters"]["ampl_mod_path"] = args.ampl_mod_path
                config["parameters"]["ampl_dat_path"] = args.ampl_dat_path
                config["parameters"]["ampl_solver"] = args.ampl_solver

            algorithm = ALGORITHMS[algo](connector, config["parameters"], args.process,
                                         args.cand_gen, args.is_utilized, args.sel_oracle)

            # return algorithm.get_index_candidates(workload, db_conf=db_conf, columns=columns)

            if not args.process and not args.overhead:
                sel_info = ""
                indexes = algorithm.calculate_best_indexes(workload, overhead=args.overhead,
                                                           db_conf=db_conf, columns=columns)
            else:
                indexes, sel_info = algorithm.calculate_best_indexes(workload, overhead=args.overhead,
                                                                     db_conf=db_conf, columns=columns)

            indexes = [str(ind) for ind in indexes]
            cols = [ind.split(",") for ind in indexes]
            cols = [list(map(lambda x: x.split(".")[-1], col)) for col in cols]
            indexes = [f"{ind.split('.')[0]}#{','.join(col)}" for ind, col in zip(indexes, cols)]

            no_cost, ind_cost = list(), list()
            total_no_cost, total_ind_cost = 0, 0

            # # (0916): newly added.
            # freq_list = [1 for _ in work_list]
            # if isinstance(work_list[0], list):
            #     work_list = [item[1] for item in work_list]
            #     if args.varying_frequencies:
            #         freq_list = [item[-1] for item in work_list]
            #
            # # (0916): newly modified.
            # for sql, freq in zip(work_list, freq_list):
            #     no_cost_ = connector.get_ind_cost(sql, "") * freq
            #     total_no_cost += no_cost_
            #     no_cost.append(no_cost_)
            #
            #     ind_cost_ = connector.get_ind_cost(sql, indexes) * freq
            #     total_ind_cost += ind_cost_
            #     ind_cost.append(ind_cost_)

            # (0916): newly modified.
            freq_list = list()
            for query in workload.queries:
                no_cost_ = connector.get_ind_cost(query.text, "") * query.frequency
                total_no_cost += no_cost_
                no_cost.append(no_cost_)

                ind_cost_ = connector.get_ind_cost(query.text, indexes) * query.frequency
                total_ind_cost += ind_cost_
                ind_cost.append(ind_cost_)

                freq_list.append(query.frequency)

            # (0916): newly added.
            if args.varying_frequencies:
                data.append({"config": config["parameters"],
                             "workload": [work_list, freq_list],
                             "indexes": indexes,
                             "no_cost": no_cost,
                             "total_no_cost": total_no_cost,
                             "ind_cost": ind_cost,
                             "total_ind_cost": total_ind_cost,
                             "sel_info": sel_info})
            else:
                data.append({"config": config["parameters"],
                             "workload": work_list,
                             "indexes": indexes,
                             "no_cost": no_cost,
                             "total_no_cost": total_no_cost,
                             "ind_cost": ind_cost,
                             "total_ind_cost": total_ind_cost,
                             "sel_info": sel_info})

        if len(data) == 1:
            data = data[0]

        res_data[algo] = data
    return res_data


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    algos = ["drop"]

    args.constraint = "storage"
    args.budget_MB = 500

    # args.constraint = "number"
    args.max_indexes = 5

    args.multi_column = True

    if args.work_file.endswith(".sql"):
        with open(args.work_file, "r") as rf:
            work_list = rf.readlines()
    elif args.work_file.endswith(".json"):
        with open(args.work_file, "r") as rf:
            work_list = json.load(rf)

    data = get_heu_result(args, algos, work_list)

    if args.res_save is not None:
        with open(args.res_save, "w") as wf:
            json.dump(data, wf, indent=2, cls=IndexEncoder)
