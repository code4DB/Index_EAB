## TPC-H
--constraint storage --max_memory 500
--constraint number --max_count 5
```
python mab_run.py 

--exp_id mab_sto5_ep100 --bench tpch --rounds 100

--constraint storage --max_memory 500

--workload_file /data/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql

--exp_file /data/wz/index/index_eab/eab_algo/mab_selection/config/exp.conf
--db_file /data/wz/index/index_eab/eab_algo/mab_selection/config/db.conf
--schema_file /data/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json

--params_load /data/wz/index/index_eab/eab_algo/mab_selection/experiments/new_exp_shell_storage_model/model/mab_90.pt

--save_gap 10

```