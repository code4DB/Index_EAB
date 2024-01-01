
## TPC-H
### Train

--pre_create --is_dnn 

--constraint number --max_count 5
--constraint storage --max_storage 500

``` 
python dqn_run.py 

--exp_id dqn_sto500_ep1k 

--action_mode train --epoch 1000

--constraint storage --max_storage 500

--is_ps --is_double

--work_load /data/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql
--conf_load /data/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf

--cand_load empty
--model_load empty

--save_gap 50

```

### Infer

``` --pre_create --is_dnn 
python dqn_run.py 

--exp_id new_dqn_pre_test1k 

--action_mode infer

--constraint number --max_count 5

--pre_create
--is_ps --is_double

--work_load /data/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql
--conf_load /data/wz/index/index_eab/eab_algo/dqn_selection/dqn_data/configure.ini

--cand_load empty
--model_load empty

```
