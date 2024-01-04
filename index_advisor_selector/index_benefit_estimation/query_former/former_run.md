
## TPC-H

## Train

```
python former_train.py

--exp_id exp_former_tpch_tgt_ep500_bat1024 
--gpu_no 0 --seed 666

--epoch_num 500 --batch_size 1024

--train_data_file /data/wz/index/index_eab/eab_benefit/cost_data/tpch/tpch_cost_data_tgt_train.json
--valid_data_file /data/wz/index/index_eab/eab_benefit/cost_data/tpch/tpch_cost_data_tgt_valid.json

--encoding_load /data/wz/index/index_eab/eab_benefit/query_former/data/tpch/encoding_tpch.pt
--cost_norm_load /data/wz/index/index_eab/eab_benefit/query_former/data/tpch/cost_norm_tpch.pt

```

## Infer

```
python former_infer.py

--test_data_file /data/wz/index/index_eab/eab_benefit/cost_data/tpch/tpch_cost_data_tgt_test.json
--model_load /data/wz/index/index_eab/eab_benefit/query_former/exp_res/exp_former_tpch_tgt_ep500_bat1024/model/former_FORMER_200.pt

--encoding_load /data/wz/index/index_eab/eab_benefit/query_former/data/tpch/encoding_tpch_v2.pt
--cost_norm_load /data/wz/index/index_eab/eab_benefit/query_former/data/tpch/cost_norm_tpch_v2.pt

```

## TPC-DS

## Train

```
python former_train.py

--exp_id exp_former_tpcds_tgt_ep500_bat1024 
--gpu_no 5 --seed 666

--epoch_num 500 --batch_size 1024

--train_data_file /data1/wz/index/index_eab/eab_benefit/cost_data/tpcds/tpcds_cost_data_tgt_train.json
--valid_data_file /data1/wz/index/index_eab/eab_benefit/cost_data/tpcds/tpcds_cost_data_tgt_valid.json

--encoding_load /data1/wz/index/index_eab/eab_benefit/query_former/data/tpcds/encoding_tpcds.pt
--cost_norm_load /data1/wz/index/index_eab/eab_benefit/query_former/data/tpcds/cost_norm_tpcds.pt

```

## Infer

```
python former_infer.py

--test_data_file /data1/wz/index/index_eab/eab_benefit/cost_data/tpcds/tpcds_cost_data_tgt_test.json
--model_load /data1/wz/index/index_eab/eab_benefit/query_former/exp_res/exp_former_tpcds_tgt_ep500_bat1024/model/former_FORMER_200.pt

--encoding_load /data1/wz/index/index_eab/eab_benefit/query_former/data/tpcds/encoding_tpcds_v2.pt
--cost_norm_load /data1/wz/index/index_eab/eab_benefit/query_former/data/tpcds/cost_norm_tpcds_v2.pt

```

## JOB

## Train

```
python former_train.py

--exp_id exp_former_job_tgt_ep500_bat1024 
--gpu_no 0 --seed 666

--epoch_num 500 --batch_size 1024

--train_data_file /data2/wz/index/index_eab/eab_benefit/cost_data/job/job_cost_data_tgt_train.json
--valid_data_file /data2/wz/index/index_eab/eab_benefit/cost_data/job/job_cost_data_tgt_valid.json

--encoding_load /data2/wz/index/index_eab/eab_benefit/query_former/data/job/encoding_job.pt
--cost_norm_load /data2/wz/index/index_eab/eab_benefit/query_former/data/job/cost_norm_job.pt

```

## Infer

```
python former_infer.py

--test_data_file /data2/wz/index/index_eab/eab_benefit/cost_data/job/job_cost_data_tgt_test.json
--model_load /data2/wz/index/index_eab/eab_benefit/query_former/exp_res/exp_former_tpcds_tgt_ep500_bat1024/model/former_FORMER_200.pt

--encoding_load /data2/wz/index/index_eab/eab_benefit/query_former/data/job/encoding_job_v2.pt
--cost_norm_load /data2/wz/index/index_eab/eab_benefit/query_former/data/job/cost_norm_job_v2.pt

```
