## TPC-H
--is_trace

--constraint storage --cardinality 5 --storage 500
--constraint number --cardinality 5 --storage 500
```
python mcts_run.py --exp_id mcts_sto500_bce_ep1k

--budget 1000

--process --overhead --is_trace
--mcts_seed 666

--work_file /data/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql
--schema_file /data/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json
--db_file /data/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf

--constraint storage --cardinality 5 --storage 500

--max_index_width 2 
--select_policy UCT --roll_num 1 --best_policy BCE

--model_load empty

```

## TPC-H Skew
--is_trace

--constraint storage --cardinality 5 --storage 500
--constraint number --cardinality 5 --storage 500
```
python mcts_run.py 

--exp_id mcts_hskew_w18_bce_sto1k_s666_v1

--budget 1000

--process --overhead --is_trace
--mcts_seed 666

--work_file /data1/wz/index/index_eab/eab_olap/bench_temp/tpch_skew/tpch_skew_work_temp_multi_w18_n10_eval.json
--schema_file /data/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json
--db_file /data/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf

--constraint storage --cardinality 5 --storage 500

--max_index_width 2 
--select_policy UCT --roll_num 1 --best_policy BCE

--model_load empty

```
