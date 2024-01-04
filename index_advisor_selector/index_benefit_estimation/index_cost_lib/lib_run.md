
## TPC-H

## Train

```
python lib_train.py

--exp_id exp_lib_tpch_tgt_ep500_bat2048 
--gpu_no 1 --seed 666

--epoch_num 500 --batch_size 2048 

--train_data_file /data/wz/index/index_eab/eab_benefit/index_cost_lib/data/tpch/lib_tpch_cost_data_tgt_train.json
--valid_data_file /data/wz/index/index_eab/eab_benefit/index_cost_lib/data/tpch/lib_tpch_cost_data_tgt_valid.json

```

## Infer

```
python lib_infer.py

--test_data_file /data/wz/index/index_eab/eab_benefit/index_cost_lib/data/tpch/lib_tpch_cost_data_tgt_test.json
--model_load /data/wz/index/index_eab/eab_benefit/index_cost_lib/exp_res/exp_lib_tpch_tgt_ep500_bat2048/model/lib_LIB_200.pt

```

## TPC-DS

## Train

```
python lib_train.py

--exp_id exp_lib_tpcds_tgt_ep500_bat2048 
--gpu_no 4 --seed 666

--epoch_num 500 --batch_size 2048 

--train_data_file /data1/wz/index/index_eab/eab_benefit/index_cost_lib/data/tpcds/lib_tpcds_cost_data_tgt_train.json
--valid_data_file /data1/wz/index/index_eab/eab_benefit/index_cost_lib/data/tpcds/lib_tpcds_cost_data_tgt_valid.json

```

## Infer

```
python lib_infer.py

--test_data_file /data1/wz/index/index_eab/eab_benefit/index_cost_lib/data/tpcds/lib_tpcds_cost_data_tgt_test.json
--model_load /data1/wz/index/index_eab/eab_benefit/index_cost_lib/exp_res/exp_lib_tpcds_tgt_ep500_bat2048/model/lib_LIB_200.pt

```


## JOB

## Train

```
python lib_train.py

--exp_id exp_lib_job_tgt_ep500_bat1024 
--gpu_no 1 --seed 666

--epoch_num 500 --batch_size 1024 

--train_data_file /data2/wz/index/index_eab/eab_benefit/index_cost_lib/data/job/lib_job_cost_data_tgt_train.json
--valid_data_file /data2/wz/index/index_eab/eab_benefit/index_cost_lib/data/job/lib_job_cost_data_tgt_valid.json

```

## Infer

```
python lib_infer.py

--test_data_file /data2/wz/index/index_eab/eab_benefit/index_cost_lib/data/job/lib_job_cost_data_tgt_test.json
--model_load /data2/wz/index/index_eab/eab_benefit/index_cost_lib/exp_res/exp_lib_job_tgt_ep500_bat1024/model/lib_LIB_200.pt

```
