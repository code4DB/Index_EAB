
## TPC-H

### Train
--varying_frequencies

--constraint storage --max_budgets 500
--constraint number --max_indexes 5

--temp_expand
--temp_load /data1/wz/index/index_eab/eab_olap/bench_temp/tpch_template_21.sql

```
python swirl_main.py

--algo swirl --exp_id swirl_h_query_sto10w_n800_s333_v1 
--timesteps 100000 --seed 333

--constraint storage --max_budgets 500

--exp_conf_file /data/wz/index/index_eab/eab_data/rl_run_conf/swirl_tpch_1gb.json

--work_size 1 --work_gen load 
--work_type not_template --temp_num 22

--training_instances 800 --validation_testing_instances 21

--work_file /data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_query_temp_multi_n1000.json
--eval_file /data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_query_temp_multi_n21_eval.json

--db_conf_file /data/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf
--schema_file /data/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json
--colinfo_load /data/wz/index/index_eab/eab_data/db_info_conf/colinfo_tpch_1gb.json

--train_mode scratch

```


### Infer
```
python swirl_run.py

--work_file /data1/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql

--rl_exp_load /data1/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/experiment_object.pickle
--rl_model_load /data1/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/best_mean_reward_model.zip
--rl_env_load /data1/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/vec_normalize.pkl

--db_conf_file /data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf

```

## TPC-H Skew

### Train
--varying_frequencies

--constraint storage --max_budgets 500
--constraint number --max_indexes 5
```
python swirl_main.py

--algo swirl --exp_id swirl_hskew_query_sto10w_n800_s999_v1 
--timesteps 100000 --seed 999

--constraint storage --max_budgets 500

--exp_conf_file /data1/wz/index/index_eab/eab_data/rl_run_conf/swirl_tpch_1gb.json

--work_size 1 --work_gen load 
--work_type not_template --temp_num 22

--training_instances 800 --validation_testing_instances 21

--work_file /data1/wz/index/index_eab/eab_olap/bench_temp/tpch_skew/tpch_skew_query_temp_multi_n1000.json
--eval_file /data1/wz/index/index_eab/eab_olap/bench_temp/tpch_skew/tpch_skew_query_temp_multi_n21_eval.json

--db_conf_file /data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf
--schema_file /data1/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json
--colinfo_load /data1/wz/index/index_eab/eab_data/db_info_conf/colinfo_tpch_1gb.json

--train_mode scratch

--host 10.26.42.166
--db_name tpch_1gb103_skew
--port 5432
--user wz
--password ai4db2021

```


### Infer
```
python swirl_run.py

--work_file /data1/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql

--rl_exp_load /data1/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/experiment_object.pickle
--rl_model_load /data1/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/best_mean_reward_model.zip
--rl_env_load /data1/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/vec_normalize.pkl

--db_conf_file /data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf

```

## TPC-DS

### Train
--varying_frequencies

--constraint storage --max_budgets 500
--constraint number --max_indexes 5
```
python swirl_main.py

--algo swirl --exp_id swirl_ds_query_sto10w_n3k_s999_v1 
--timesteps 100000 --seed 999

--constraint storage --max_budgets 500

--exp_conf_file /data2/wz/index/index_eab/eab_data/rl_run_conf/swirl_tpcds_1gb.json

--work_size 1 --work_gen load 
--work_type not_template --temp_num 99

--training_instances 3000 --validation_testing_instances 99

--work_file /data2/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_query_temp_multi_n5000.json
--eval_file /data2/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_query_temp_multi_n99_eval.json

--db_conf_file /data2/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpcds_1gb.conf
--schema_file /data2/wz/index/index_eab/eab_data/db_info_conf/schema_tpcds_1gb.json
--colinfo_load /data2/wz/index/index_eab/eab_data/db_info_conf/colinfo_tpcds_1gb.json

--train_mode scratch

```


### Infer
```
python swirl_run.py

--work_file /data1/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql

--rl_exp_load /data1/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/experiment_object.pickle
--rl_model_load /data1/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/best_mean_reward_model.zip
--rl_env_load /data1/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/vec_normalize.pkl

--db_conf_file /data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf

```

## DSB

### Train
--varying_frequencies

--constraint storage --max_budgets 500
--constraint number --max_indexes 5
```
python swirl_main.py

--algo swirl --exp_id swirl_dsb_query_sto10w_n2k_s999_v1 
--timesteps 100000 --seed 999

--constraint storage --max_budgets 500

--exp_conf_file /data1/wz/index/index_eab/eab_data/rl_run_conf/swirl_tpcds_1gb.json

--work_size 1 --work_gen load 
--work_type not_template --temp_num 53

--training_instances 2000 --validation_testing_instances 53

--work_file /data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_query_temp_multi_n3000.json
--eval_file /data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_query_temp_multi_n53_eval.json

--db_conf_file /data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpcds_1gb.conf
--schema_file /data1/wz/index/index_eab/eab_data/db_info_conf/schema_tpcds_1gb.json
--colinfo_load /data1/wz/index/index_eab/eab_data/db_info_conf/colinfo_tpcds_1gb.json

--train_mode scratch

--host 10.26.42.166
--db_name dsb_1gb103
--port 5432
--user wz
--password ai4db2021

```


### Infer
```
python swirl_run.py

--work_file /data1/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql

--rl_exp_load /data1/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/experiment_object.pickle
--rl_model_load /data1/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/best_mean_reward_model.zip
--rl_env_load /data1/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/vec_normalize.pkl

--db_conf_file /data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf

```

## JOB

### Train
--varying_frequencies

--constraint storage --max_budgets 500
--constraint number --max_indexes 5 --max_budgets 50000

--temp_expand
--temp_load /data/wz/index/index_eab/eab_olap/bench_temp/job_template_33.sql

/data/wz/index/index_eab/eab_olap/bench_temp/job_template_33.sql
```
python swirl_main.py

--algo swirl --exp_id swirl_job_query_sto10w_n100_s999_v1 
--timesteps 100000 --seed 999

--constraint storage --max_budgets 500

--exp_conf_file /data/wz/index/index_eab/eab_data/rl_run_conf/swirl_job.json

--work_size 1 --work_gen load 
--work_type not_template --temp_num 33

--training_instances 100 --validation_testing_instances 33

--work_file /data/wz/index/index_eab/eab_olap/bench_temp/job/job_query_temp_multi_n113.json
--eval_file /data/wz/index/index_eab/eab_olap/bench_temp/job/job_query_temp_multi_n33_eval.json

--db_conf_file /data/wz/index/index_eab/eab_data/db_info_conf/local_db103_job.conf
--schema_file /data/wz/index/index_eab/eab_data/db_info_conf/schema_job.json
--colinfo_load /data/wz/index/index_eab/eab_data/db_info_conf/colinfo_job.json

--train_mode scratch

```


### Infer
```
python swirl_run.py

--work_file /data1/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql

--rl_exp_load /data1/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/experiment_object.pickle
--rl_model_load /data1/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/best_mean_reward_model.zip
--rl_env_load /data1/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/vec_normalize.pkl

--db_conf_file /data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf

```
