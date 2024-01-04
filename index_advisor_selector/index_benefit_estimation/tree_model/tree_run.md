
## TPC-H

## Train

```
python tree_cost_main.py

--exp_id exp_xgb_cost_tpch_tgt_round5k 
--model_type XGBoost --feat_chan cost --seed 666 

--num_rounds 5000

--train_data_load /data/wz/index/index_eab/eab_benefit/tree_model/data/tpch/tree_tpch_cost_data_tgt_train.json
--valid_data_load /data/wz/index/index_eab/eab_benefit/tree_model/data/tpch/tree_tpch_cost_data_tgt_valid.json

```

```
python tree_cost_main.py

--exp_id exp_lgb_row_tpch_tgt_round5k 
--model_type LightGBM --feat_chan row --seed 666 

--num_rounds 5000

--train_data_load /data/wz/index/index_eab/eab_benefit/tree_model/data/tpch/tree_tpch_cost_data_tgt_train.json
--valid_data_load /data/wz/index/index_eab/eab_benefit/tree_model/data/tpch/tree_tpch_cost_data_tgt_valid.json

```

```
python tree_cost_main.py

--exp_id exp_rf_tpch_tgt_round5k 
--model_type RandomForest --seed 666 

--num_rounds 5000

--train_data_load /data/wz/index/index_eab/eab_benefit/tree_model/data/tpch/tree_tpch_cost_data_tgt_train.json
--valid_data_load /data/wz/index/index_eab/eab_benefit/tree_model/data/tpch/tree_tpch_cost_data_tgt_valid.json

```

## Infer

```
python tree_cost_infer.py

--model_type XGBoost --feat_chan cost_row

--test_data_load /data/wz/index/index_eab/eab_benefit/tree_model/data/tpch/tree_tpch_cost_data_tgt_test.json
--model_load /data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_tpch_tgt_round5k/model/reg_xgb_cost.xgb.model
--scale_load /data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_tpch_tgt_round5k/data/train_scale_data.pt

```

```
python tree_cost_infer.py

--model_type LightGBM --feat_chan cost_row

--test_data_load /data/wz/index/index_eab/eab_benefit/tree_model/data/tpch/tree_tpch_cost_data_tgt_test.json
--model_load /data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_lgb_tpch_tgt_round5k/model/reg_lgb_cost.lgb.model
--scale_load /data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_lgb_tpch_tgt_round5k/data/train_scale_data.pt

```

## TPC-DS

## Train

```
python tree_cost_main.py

--exp_id exp_xgb_tpcds_tgt_round5k 
--model_type XGBoost --seed 666 

--num_rounds 5000

--train_data_load /data1/wz/index/index_eab/eab_benefit/tree_model/data/tpcds/tree_tpcds_cost_data_tgt_train.json
--valid_data_load /data1/wz/index/index_eab/eab_benefit/tree_model/data/tpcds/tree_tpcds_cost_data_tgt_valid.json

```

```
python tree_cost_main.py

--exp_id exp_lgb_row_tpcds_tgt_round5k 
--model_type LightGBM --feat_chan row --seed 666 

--num_rounds 5000

--train_data_load /data1/wz/index/index_eab/eab_benefit/tree_model/data/tpcds/tree_tpcds_cost_data_tgt_train.json
--valid_data_load /data1/wz/index/index_eab/eab_benefit/tree_model/data/tpcds/tree_tpcds_cost_data_tgt_valid.json

```

```
python tree_cost_main.py

--exp_id exp_rf_tpcds_tgt_round5k 
--model_type RandomForest --seed 666 

--num_rounds 5000

--train_data_load /data1/wz/index/index_eab/eab_benefit/tree_model/data/tpcds/tree_tpcds_cost_data_tgt_train.json
--valid_data_load /data1/wz/index/index_eab/eab_benefit/tree_model/data/tpcds/tree_tpcds_cost_data_tgt_valid.json

```

## Infer

```
python tree_cost_infer.py

--model_type XGBoost --feat_chan cost_row

--test_data_load /data1/wz/index/index_eab/eab_benefit/tree_model/data/tpcds/tree_tpcds_cost_data_tgt_test.json
--model_load /data1/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_tpcds_tgt_round5k/model/reg_xgb_cost.xgb.model
--scale_load /data1/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_tpcds_tgt_round5k/data/train_scale_data.pt

```

```
python tree_cost_infer.py

--model_type LightGBM --feat_chan cost_row

--test_data_load /data1/wz/index/index_eab/eab_benefit/tree_model/data/tpcds/tree_tpcds_cost_data_tgt_test.json
--model_load /data1/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_lgb_tpcds_tgt_round5k/model/reg_lgb_cost.lgb.model
--scale_load /data1/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_lgb_tpcds_tgt_round5k/data/train_scale_data.pt

```

## JOB

## Train

```
python tree_cost_main.py

--exp_id exp_xgb_row_job_tgt_round5k 
--model_type XGBoost --feat_chan row --seed 666 

--num_rounds 5000

--train_data_load /data2/wz/index/index_eab/eab_benefit/tree_model/data/job/tree_job_cost_data_tgt_train.json
--valid_data_load /data2/wz/index/index_eab/eab_benefit/tree_model/data/job/tree_job_cost_data_tgt_valid.json

```

```
python tree_cost_main.py

--exp_id exp_lgb_row_job_tgt_round5k 
--model_type LightGBM --feat_chan row --seed 666 

--num_rounds 5000

--train_data_load /data2/wz/index/index_eab/eab_benefit/tree_model/data/job/tree_job_cost_data_tgt_train.json
--valid_data_load /data2/wz/index/index_eab/eab_benefit/tree_model/data/job/tree_job_cost_data_tgt_valid.json

```

```
python tree_cost_main.py

--exp_id exp_rf_job_tgt_round5k 
--model_type RandomForest --seed 666 

--num_rounds 5000

--train_data_load /data2/wz/index/index_eab/eab_benefit/tree_model/data/job/tree_job_cost_data_tgt_train.json
--valid_data_load /data2/wz/index/index_eab/eab_benefit/tree_model/data/job/tree_job_cost_data_tgt_valid.json

```

## Infer

```
python tree_cost_infer.py

--model_type XGBoost --feat_chan cost_row

--test_data_load /data/wz/index/index_eab/eab_benefit/tree_model/data/job/tree_job_cost_data_tgt_test.json
--model_load /data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_job_tgt_round5k/model/reg_xgb_cost.xgb.model
--scale_load /data/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_xgb_job_tgt_round5k/data/train_scale_data.pt

```

```
python tree_cost_infer.py

--model_type LightGBM --feat_chan cost_row

--test_data_load /data2/wz/index/index_eab/eab_benefit/tree_model/data/job/tree_job_cost_data_tgt_test.json
--model_load /data2/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_lgb_job_tgt_round5k/model/reg_lgb_cost.lgb.model
--scale_load /data2/wz/index/index_eab/eab_benefit/tree_model/cost_exp_res/exp_lgb_job_tgt_round5k/data/train_scale_data.pt

```
