
## TPC-H

## Train

```
python distill_main.py

--exp_id exp_xgb_tpch_round5k 
--model_type XGBoost --seed 666 

--num_rounds 5000

--train_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpch_train.json
--valid_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpch_valid.json

```

```
python distill_main.py

--exp_id exp_lgb_tpch_round5k 
--model_type LightGBM --seed 666 

--num_rounds 5000

--train_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpch_train.json
--valid_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpch_valid.json

```

```
python distill_main.py

--exp_id exp_rf_tpch_round5k 
--model_type RandomForest --seed 666 

--num_rounds 5000

--train_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpch_train.json
--valid_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpch_valid.json

```

## Infer

```
python distill_infer.py

--model_type XGBoost 

--test_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpch_test.json

--model_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_xgb_tpch_round5k/model/reg_xgb_cost.xgb.model
--scale_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_xgb_tpch_round5k/data/train_scale_data.pt

```

```
python distill_infer.py

--model_type LightGBM

--test_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpch_test.json

--model_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_lgb_tpch_round5k/model/reg_lgb_cost.lgb.model
--scale_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_lgb_tpch_round5k/data/train_scale_data.pt

```

```
python distill_infer.py

--model_type RandomForest

--test_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpch_test.json

--model_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_rf_tpch_round5k/model/reg_rf_cost.rf.joblib
--scale_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_lgb_tpch_round5k/data/train_scale_data.pt

```

## TPC-DS

## Train

```
python distill_main.py

--exp_id exp_xgb_tpcds_round5k 
--model_type XGBoost --seed 666 

--num_rounds 5000

--train_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpcds_train.json
--valid_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpcds_valid.json

```

```
python distill_main.py

--exp_id exp_lgb_tpcds_round5k 
--model_type LightGBM --seed 666 

--num_rounds 5000

--train_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpcds_train.json
--valid_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpcds_valid.json

```

```
python distill_main.py

--exp_id exp_rf_tpcds_round5k 
--model_type RandomForest --seed 666 

--num_rounds 5000

--train_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpcds_train.json
--valid_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpcds_valid.json

```

## Infer

```
python distill_infer.py

--model_type XGBoost 

--test_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpcds_test.json

--model_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_xgb_tpcds_round5k/model/reg_xgb_cost.xgb.model
--scale_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_xgb_tpcds_round5k/data/train_scale_data.pt

```

```
python distill_infer.py

--model_type LightGBM 

--test_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpcds_test.json

--model_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_lgb_tpcds_round5k/model/reg_lgb_cost.lgb.model
--scale_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_lgb_tpcds_round5k/data/train_scale_data.pt

```

```
python distill_infer.py

--model_type RandomForest 

--test_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_tpcds_test.json

--model_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_rf_tpcds_round5k/model/reg_rf_cost.rf.joblib
--scale_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_rf_tpcds_round5k/data/train_scale_data.pt

```

## JOB

## Train

```
python distill_main.py

--exp_id exp_xgb_job_round5k 
--model_type XGBoost --seed 666 

--num_rounds 5000

--train_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_job_train.json
--valid_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_job_valid.json

```

```
python distill_main.py

--exp_id exp_lgb_job_round5k 
--model_type LightGBM --seed 666 

--num_rounds 5000

--train_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_job_train.json
--valid_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_job_valid.json

```

```
python distill_main.py

--exp_id exp_rf_job_round5k 
--model_type RandomForest --seed 666 

--num_rounds 5000

--train_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_job_train.json
--valid_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_job_valid.json

```

## Infer

```
python distill_infer.py

--model_type XGBoost

--test_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_job_test.json

--model_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_xgb_job_round5k/model/reg_xgb_cost.xgb.model
--scale_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_xgb_job_round5k/data/train_scale_data.pt

```

```
python distill_infer.py

--model_type LightGBM 

--test_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_job_test.json

--model_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_lgb_job_round5k/model/reg_lgb_cost.lgb.model
--scale_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_lgb_job_round5k/data/train_scale_data.pt

```

```
python distill_infer.py

--model_type RandomForest 

--test_data_load /data1/wz/index/index_eab/eab_other/distill_model/distill_data/plan_raw_feat_job_test.json

--model_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_rf_job_round5k/model/reg_rf_cost.rf.joblib
--scale_load /data1/wz/index/index_eab/eab_other/distill_model/cost_exp_res/exp_rf_job_round5k/data/train_scale_data.pt

```
