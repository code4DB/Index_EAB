
## TPC-H

### Train
--varying_frequencies

--constraint number --max_indexes 5
--constraint storage --max_budgets 500
```
python swirl_main.py

--algo dqn --exp_id dqn_h_w18_num17_10w_n800_s999_v1 
--timesteps 100000 --seed 999

--constraint number --max_indexes 17

--exp_conf_file /data/wz/index/index_eab/eab_data/rl_run_conf/dqn_tpch_1gb.json

--work_size 18 --work_gen load 
--work_type not_template --temp_num 22

--training_instances 800 --validation_testing_instances 10

--work_file /data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_duplicate_multi_w18_n1000.json
--eval_file /data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_n10_eval.json

--db_conf_file /data/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf
--schema_file /data/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json
--colinfo_load /data/wz/index/index_eab/eab_data/db_info_conf/colinfo_tpch_1gb.json

--train_mode scratch

--temp_expand
--temp_load /data/wz/index/attack/data_resource/visrew_spj_data/bak/server152/tpch_1gb/s152_1gb_4all00_tblcol_index_all.json

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

--constraint number --max_indexes 5
--constraint storage --max_budgets 500
```
python swirl_main.py

--algo dqn --exp_id dqn_hskew_w18_sto900_10w_n800_s999_v1 
--timesteps 100000 --seed 999

--constraint storage --max_budgets 900

--exp_conf_file /data1/wz/index/index_eab/eab_data/rl_run_conf/dqn_tpch_1gb.json

--work_size 18 --work_gen load 
--work_type not_template --temp_num 22

--training_instances 800 --validation_testing_instances 1

--work_file /data1/wz/index/index_eab/eab_olap/bench_temp/tpch_skew/tpch_skew_work_temp_duplicate_multi_w18_n1000.json
--eval_file /data1/wz/index/index_eab/eab_olap/bench_temp/tpch_skew/tpch_skew_work_temp_multi_w18_n1_eval.json

--db_conf_file /data1/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf
--schema_file /data1/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json
--colinfo_load /data1/wz/index/index_eab/eab_data/db_info_conf/colinfo_tpch_1gb.json

--train_mode scratch

--host 10.26.42.166
--db_name tpch_1gb103_skew
--port 5432
--user wz
--password ai4db2021

--temp_expand
--temp_load /data1/wz/index/index_eab/eab_data/s152_1gb_4all00_tblcol_index_all.json

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

--constraint number --max_indexes 5
--constraint storage --max_budgets 500
```
python swirl_main.py

--algo dqn --exp_id dqn_ds_w79_sto10w_n2k_e10_s666_v1 
--timesteps 100000 --seed 666

--constraint storage --max_budgets 500

--exp_conf_file /data2/wz/index/index_eab/eab_data/rl_run_conf/dqn_tpcds_1gb.json

--work_size 79 --work_gen load --work_num 2002
--work_type not_template --temp_num 99
--is_query_cache

--training_instances 2000 --validation_testing_instances 10

--work_file /data2/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_work_temp_duplicate_multi_w79_n5000.json
--eval_file /data2/wz/index/index_eab/eab_olap/bench_temp/tpcds/tpcds_work_temp_multi_w79_n10_eval.json

--db_conf_file /data2/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpcds_1gb.conf
--schema_file /data2/wz/index/index_eab/eab_data/db_info_conf/schema_tpcds_1gb.json
--colinfo_load /data2/wz/index/index_eab/eab_data/db_info_conf/colinfo_tpcds_1gb.json

--train_mode scratch

--host 10.26.42.166
--db_name tpcds_1gb103
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

## DSB

### Train
--varying_frequencies

--constraint number --max_indexes 5
--constraint storage --max_budgets 500
```
python swirl_main.py

--algo dqn --exp_id dqn_dsb_w53_sto10w_n2k_s444_v1 
--timesteps 100000 --seed 444

--constraint storage --max_budgets 500

--exp_conf_file /data1/wz/index/index_eab/eab_data/rl_run_conf/dqn_tpcds_1gb.json

--work_size 53 --work_gen load --work_num 2002
--work_type not_template --temp_num 53
--is_query_cache

--training_instances 2000 --validation_testing_instances 1

--work_file /data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_work_temp_duplicate_multi_w53_n3000.json
--eval_file /data1/wz/index/index_eab/eab_olap/bench_temp/dsb/dsb_work_temp_multi_w53_n1_eval.json

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

--constraint number --max_indexes 5
--constraint storage --max_budgets 500

--temp_expand 
--temp_load /data/wz/index/index_eab/eab_olap/bench_temp/job_template_33.sql

```
python swirl_main.py

--algo dqn --exp_id dqn_job_w33_sto10w_n2k_e10_s999_v1 
--timesteps 100000 --seed 999

--constraint storage --max_budgets 500

--exp_conf_file /data2/wz/index/index_eab/eab_data/rl_run_conf/dqn_job.json

--work_size 33 --work_gen load 
--work_type not_template --temp_num 33

--training_instances 2000 --validation_testing_instances 10

--work_file /data2/wz/index/index_eab/eab_olap/bench_temp/job/job_work_temp_duplicate_multi_w33_n3000.json
--eval_file /data2/wz/index/index_eab/eab_olap/bench_temp/job/job_work_temp_multi_w33_n10_eval.json

--db_conf_file /data2/wz/index/index_eab/eab_data/db_info_conf/local_db103_job.conf
--schema_file /data2/wz/index/index_eab/eab_data/db_info_conf/schema_job.json
--colinfo_load /data2/wz/index/index_eab/eab_data/db_info_conf/colinfo_job.json

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
