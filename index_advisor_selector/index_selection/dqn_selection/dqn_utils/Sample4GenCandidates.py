import json
import pickle
from configparser import ConfigParser

import psqlparse

# import sys
# sys.path.append("..")

import Encoding as en
import ParserForIndex as pi


def gen_work_cands(workload, parser):
    added_cands = set()
    for no, query in enumerate(workload):
        try:
            b = psqlparse.parse_dict(query)
            parser.parse_stmt(b[0])
            parser.gain_candidates()
        except:
            b = psqlparse.parse_dict(query)
            parser.parse_stmt(b[0])
            parser.gain_candidates()

    f_cands = parser.index_candidates | added_cands
    f_cands = list(f_cands)
    f_cands.sort()

    return f_cands


def run_gen_cands(conf_load, data_load, data_save):
    enc = en.encoding_schema(conf_load)
    parser = pi.Parser(enc["attr"])

    # path to your tpch directory/dbgen
    # work_dir = "/Users/lanhai/XProgram/PycharmProjects/2.18.0_rc2/dbgen"
    # w_size = 14
    # wd_generator = ds.TPCH(work_dir, w_size)
    # workload = wd_generator.gen_workloads()

    if data_load.endswith(".pickle"):
        with open(data_load, "rb") as rf:
            workload = pickle.load(rf)
    elif data_load.endswith(".sql"):
        with open(data_load, "r") as rf:
            workload = rf.readlines()
    # (0822): newly added.
    elif data_load.endswith(".json"):
        with open(data_load, "r") as rf:
            data = json.load(rf)

        workload = list()
        for item in data:
            if isinstance(item, dict):
                if "workload" in item.keys():
                    workload.extend(item["workload"])
                elif "sql" in item.keys():
                    workload.append(item["sql"])
            elif isinstance(item, list):
                # (0822): newly modified. data:[item:[info:[]]]
                if isinstance(item[0], list):
                    workload.extend([info[1] for info in item])
                elif isinstance(item[0], str):
                    workload.extend(item)
                elif isinstance(item[0], int):
                    workload.append(item[1])
            elif isinstance(item, str):
                workload.append(item)

    workload = list(set(workload))

    # workload = ["select char_name.id, complete_cast.status_id, cast_info.role_id, title.season_nr, sum(complete_cast.movie_id) from char_name JOIN cast_info ON cast_info.person_role_id = char_name.id JOIN title ON title.id = cast_info.movie_id JOIN complete_cast ON complete_cast.movie_id = title.id where char_name.surname_pcode = 'A13' AND complete_cast.subject_id > 2 group by char_name.id, complete_cast.status_id, title.season_nr, cast_info.role_id having sum(complete_cast.movie_id) = 2427152"]

    workload = [workload[3]]

    cands = gen_work_cands(workload, parser)

    if data_save is not None:
        with open(data_save, "wb") as df:
            pickle.dump(list(cands), df, protocol=0)

    return cands


if __name__ == "__main__":
    conf_load = "/data/wz/index/code_aidb/IndexAdvisor/configure.ini"
    conf_load = "/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf"

    conf_load = "/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_job.conf"
    config_raw = ConfigParser()
    config_raw.read(conf_load)


    data_load = "/data/wz/index/code_aidb/IndexAdvisor/Entry/workload.pickle"
    data_load = "/data/wz/index/attack/data_resource/bench_template/job_template_113.sql"
    # data_load = "/data/wz/index/attack/data_resource/bench_template/tpcds_template_99.sql"
    data_load = "/data/wz/index/index_eab/eab_olap/bench_temp/job/job_template_33_multi_work.json"

    data_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_temp_multi_query_n3000.json"
    data_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_duplicate_multi_w18_n1000.json"

    data_save = "/data/wz/index/code_aidb/IndexAdvisor/Entry/cands_tpch.pickle"

    data_load = "/data/wz/index/index_eab/eab_oltp/bench_temp/tpcc/tpcc_1gb_read_work_n5_test.json"

    data_load = "/data/wz/index/index_eab/eab_olap/bench_random/job/job_work_random_duplicate_multi_w33_n3000.json"

    data_load = "/data/wz/index/index_eab/eab_olap/bench_random/job/job_work_multi_w33_n100_test.json"
    run_gen_cands(config_raw, data_load, data_save)
