## TPC-H

--algo drop

--constraint storage --budget_MB 500
--constraint number --max_indexes 5

```
python heu_run.py

--res_save /data/wz/index/index_eab/eab_olap/bench_result/tpch/tpch_template18_num5_work_index.json

--process --overhead

--sel_params parameters
--exp_conf_file /data/wz/index/index_eab/eab_data/heu_run_conf/{}_config_tpch.json

--constraint number --max_indexes 5

--work_file /data/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql
--db_conf_file /data/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf
--schema_file /data/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json

--host 10.26.42.166
--db_name dsb_1gb103
--port 5432
--user wz
--password ai4db2021

```

## TPC-H Skew

--algo drop

--constraint storage --budget_MB 500
--constraint number --max_indexes 5

```
python heu_run.py

--res_save /data1/wz/index/index_eab/eab_olap/bench_result/tpch/tpch_skew_template18_sto500_work_index.json

--process --overhead

--sel_params parameters
--exp_conf_file /data1/wz/index/index_eab/eab_data/heu_run_conf/{}_config_tpch.json

--constraint storage --budget_MB 500

--work_file /data1/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql
--db_conf_file /data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf
--schema_file /data1/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json

--host 10.26.42.166
--db_name tpch_1gb103_skew
--port 5432
--user wz
--password ai4db2021

```

## TPC-DS

--algo drop

--constraint storage --budget_MB 500
--constraint number --max_indexes 5

```
python heu_run.py

--res_save /data2/wz/index/index_eab/eab_olap/bench_result/tpcds/tpcds_template83_num5_work_index.json

--process --overhead

--sel_params parameters
--exp_conf_file /data2/wz/index/index_eab/eab_data/heu_run_conf/{}_config_tpch.json

--constraint number --max_indexes 5

--work_file /data2/wz/index/index_eab/eab_olap/bench_temp/tpcds_template_83.sql
--db_conf_file /data2/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpcds_1gb.conf
--schema_file /data2/wz/index/index_eab/eab_data/db_info_conf/schema_tpcds_1gb.json

--host 10.26.42.166
--db_name dsb_1gb103
--port 5432
--user wz
--password ai4db2021

```

## DSB

--algo drop

--constraint storage --budget_MB 500
--constraint number --max_indexes 5

```
python heu_run.py

--res_save /data1/wz/index/index_eab/eab_olap/bench_result/dsb/dsb_template53_sto500_work_index.json

--process --overhead

--sel_params parameters
--exp_conf_file /data1/wz/index/index_eab/eab_data/heu_run_conf/{}_config_tpch.json

--constraint storage --budget_MB 500

--work_file /data1/wz/index/index_eab/eab_olap/bench_temp/dsb_template_53.sql
--db_conf_file /data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpcds_1gb.conf
--schema_file /data1/wz/index/index_eab/eab_data/db_info_conf/schema_tpcds_1gb.json

--host 10.26.42.166
--db_name dsb_1gb103
--port 5432
--user wz
--password ai4db2021

```

## JOB

--algo drop

--constraint storage --budget_MB 500
--constraint number --max_indexes 5

```
python heu_run.py

--res_save /data/wz/index/index_eab/eab_olap/bench_result/job/job_template33_num5_work_index.json

--process --overhead

--sel_params parameters
--exp_conf_file /data/wz/index/index_eab/eab_data/heu_run_conf/{}_config_tpch.json

--constraint number --max_indexes 5

--work_file /data/wz/index/index_eab/eab_olap/bench_temp/job_template_33.sql
--db_conf_file /data/wz/index/index_eab/eab_data/db_info_conf/local_db103_job.conf
--schema_file /data/wz/index/index_eab/eab_data/db_info_conf/schema_job.json

--host 10.26.42.166
--db_name dsb_1gb103
--port 5432
--user wz
--password ai4db2021

```
