# Index Advisor [Experiment, Analysis & Benchmark]

This is the code respository of the testbed proposed in the **Index Advisor (EA&B)** paper, which conducts a comprehensive assessment of the heuristic-based and the learning-based index advisors.

***Note that we have released a python package about this testbed at the official open-source website: *[index-eab · PyPI](https://pypi.org/project/index-eab/0.1.0/)*. Therefore, you can use the follwoing command to download it with little effort!**

```shell
pip install index-eab==0.1.0
```

![image-20240107023318952](/pypi_package.png)



## Project Structure

Specifically, the testbed is comprised of three modules. 

- **(1) Configuration Loader:** initializes a series of evaluation settings, including the benchmark, the index advisor, and the database;
- **(2) Workload Generator:** supports three methods for generating workloads with diverse features (e.g., query changes due to typical workload drifts) to simulate the requirements posed by various scenarios;
- **(3) Index Advisor Selector:** implements existing index advisors, including seven heuristic-based index advisors and ten learning-based index advisors.

The overall code structure of our **Index Advisor (EA&B)** project, where the critical files are marked with additional comments.

```
Index_EAB/
├── configuration_loader			 											# Module 1: the evaluation settings                 
│   ├── becnhmark
│   ├── index_advisor
│   │   ├── heu_run_conf
│   │   │   ├──xxxx_config.json  # configurations of heuristic-based index advisors
│   │   │   └── ...
│   │   └── rl_run_conf
│   │   │   ├──xxxx_config.json  # configurations of learning-based index advisors
│   │   │   └── ...
│   └── database
│   │   ├── db_con.conf  # configurations of database connection
│   │   └── ...
├── workload_generator				 											# Module 2: the testing workloads
│   ├── template_based  # workload from template-based generation
│   ├── perturbation_based  # workload from perturbation-based generation
│   │   ├── perturb_utils
│   │   └── ...
│   ├── random  # workload from random generation
│   └── gen_workload.py
├── index_advisor_selector			 											# Module 3: the implemented index advisors
│   ├── index_candidate_generation
│   │   └── distill_model
│   │       ├── distill_utils
│   │       │	├── distill_com.py  # parameters of learned filter model
│   │       │	└── ...
│   │       └── ...
│   ├── index_selection
│   │   ├── heu_selection
│   │   │   ├── heu_utils
│   │   │   │   ├── heu_com.py  # parameters of heuristic-based index advisors
│   │   │   │   └── ... 
│   │   │   ├── heu_algos
│   │   │   ├── heu_run.py  # entrance of heuristic-based index advisors
│   │   │   └── ...
│   │   ├── dqn_selection
│   │   │   └── dqn_utils
│   │   │   ├── dqn_run.py  # inference entrance of learning-based index advisors
│   │   │   └── ...
│   │   ├── swirl_selection
│   │   │   ├── swirl_utils
│   │   │   │   ├── swirl_com.py  # parameters of learning-based index advisors
│   │   │   │   └── ... 
│   │   │   ├── gym_db
│   │   │   ├── stable_baselines
│   │   │   ├── swirl_main.py  # training entrance of learning-based index advisors
│   │   │   ├── swirl_run.py  # inference entrance of learning-based index advisors
│   │   │   └── ...
│   │   ├── mab_selection
│   │   │   ├── bandits
│   │   │   ├── simualtion
│   │   │   ├── shared
│   │   │   │   ├── mab_com.py  # parameters of learning-based index advisors
│   │   │   │   └── ...
│   │   │   ├── database
│   │   │   ├── mab_run.py  # inference entrance of learning-based index advisors
│   │   │   └── ... 
│   │   └── mcts_selection
│   │       ├── mcts_utils
│   │       │   ├── mcts_com.py  # parameters of learning-based index advisors
│   │       │   └── ... 
│   │       ├── mcts_run.py  # inference entrance of learning-based index advisors
│   │       └── ... 
└── ├── index_benefit_estimation
    └── ├── benefit_utils
        ├── optmizer_cost
        │   ├── optimizer_utils
        │   │   ├── optimizer_com.py  # parameters of statistic-based method
        │   │   └── ... 
        │   ├── optimizer_train.py  # training entrance of statistic-based method
        │   ├── optimizer_infer.py  # inference entrance of statistic-based method
        │   └── ... 
        ├── tree_model
        │   ├── tree_cost_utils
        │   │   ├── tree_cost_com.py  # parameters of learned estimation model
        │   │   └── ... 
        │   ├── tree_cost_main.py  # training entrance of learned estimation model
        │   ├── tree_cost_infer.py  # inference entrance of learned estimation model
        │   └── ... 
        ├── index_cost_lib
        │   ├── lib_train.py  # training entrance of learned estimation model
        │   ├── lib_infer.py  # inference entrance of learned estimation model
        │   └── ... 
        └── query_former
            ├── former_train.py  # training entrance of learned estimation model
            ├── former_infer.py  # inference entrance of learned estimation model
            └── ... 
```



## Setup

We introduce the indispensable step, i.e., experiment setup for the experiment evaluations, you should check the following things:

- Create **the database instance** according to the provided toolkit;
- Create **the HypoPG extension** on the database instance for the usage of hypothetical index according to [HypoPG/hypopg: Hypothetical Indexes for PostgreSQL (github.com)](https://github.com/HypoPG/hypopg);
- Create **the python virtual environment**. Specifically, you can utilize the following script and the corresponding file `requirements.txt` is provided under the main directory. Please check the packages required are properly installed.

```shell
# Create the virtualenv `TRAP`
conda create -n Index_EAB python=3.7		 	

# Activate the virtualenv `TRAP`
conda activate Index_EAB				

# Install requirements with pip
while read requirement; do pip install $requirement; done < requirements.txt	
```



## Testbed Workflow

### 1.  Configuration Setup

Please Specify the **configuration** about the benchmark, the index advisor, and the database.

- **Benchmark:** set the vocabulary (already provided) to generate workloads of *Query Perturbation*

- **Index Advisor:** 
  - set the configurations for the **heuristic-based** index advisors at `/configuration_loader/heu_run_conf`
  - set the configurations for the **learning-based** index advisors at `/configuration_loader/rl_run_conf`

| Parameter       | Description                                                  |
  | --------------- | ------------------------------------------------------------ |
  | constraint      | The constraint of the budget type (`storage` by default)     |
  | budget_MB       | The constraint of the storage budget (MB) (`500` by default, valid when `constraint = storage`) |
  | max_indexes     | The constraint of the maximum allowable number (`5` by default, valid when `constraint = number`) |
  | max_index_width | The constraint of the index width over the considered index candidates (`2` by default) |

The parameters above are the basic configurations of index advisors. More illustrations about the fine-grained parameters (e.g., the method utilized in each underlying building block) are presented in the running script in ***Step 3. Index Advisor Evaluation***.

- **Database:** set the configurations at `configuration_loader/databse/db_con.conf` for the connection **to your own database instance**

```
host = -- your host --
port = -- your port --
user = -- your user --
password = -- your password --
database = -- your database --
```

Apart from that, we provide some files including the statistics of the database / benchmark in the directory. For example, the file `configuration_loader/database/schema_tpch.json` stores the schema information of the *TPC-H* benchmark.



### 2.  Data Preparation

The workload  data provided in `/workload_generator` has already been preprocessed, which involves three types of the workloads,  i.e., **(1) template-based**, **(2) perturbation-based**, and **(3) random**. These data can be utilized for direct evaluation and you can generate your own workload data organized in the following format. 

```sql
[
	[
        1,		# query ID
        "SELECT MIN(mc.note) AS production_note, MIN(t.title) AS movie_title ...",	# query frequency
        666		# query frequency
    ],
    ...
]
```

For example, you can generate your own perturbation-based workload, i.e., conduct query changes (e.g., add a new selection predicate) over the given workloads using the file `/workload_generator/gen_workload.py`. It currently supports three perturbation manners with different amplitudes that simulate the typical workload drifts introduced in our paper:

- **Value Only Perturbation:** modifications on the predicate values of the query templates with placeholders;
- **Column Consistent Perturbation:** modifications on the values and the set same of columns (e.g., change the column order in *GROUP BY* clause);
- **Shared Table Perturbation:** modifications on the SQL tokens of the same table (e.g., add a new selection predicate).



### 3.  Index Advisor Evaluation

With the specified configurations in ***Step 1. Configuration Setup*** and the prepared data in ***Step 2. Data Preparation***, we next proceed to the evaluation of different index advisors. 

For example, we can evaluate heuristic-based index advisors with the following script:

| Parameter | Descritpiton                                                 |
| --------- | ------------------------------------------------------------ |
| cand_gen  | The methods utilized in ***Index Candidate Generation*** building block (`"permutation" / "dqn_rule" / "openGauss"`) |
| est_model | The methods utilized in ***Index Benefit Estimation*** building block (`"optimizer" / "tree" / "lib" / "queryformer"`) |
| process   | The ddetails of the overall process of ***Index Selection*** building block |
| overhead  | The time overhead spent on each building block               |

```shell
python heu_run.py

--res_save /index_advisor_selector/index_selection/heu_res.json

--process --overhead

--sel_params parameters
--exp_conf_file /configuration_loader/index_advisor/heu_run_conf/{}_config.json

--constraint storage --budget_MB 500

--cand_gen permutation --est_model optimizer

--work_file /workload_generator/template_based/tpch_work_temp_multi.json
--db_conf_file /configuration_loader/database/db_con.conf
--schema_file /configuration_loader/database/schema_tpch.json
```



Besides, we can evaluate learning-based index advisors with the following script:

| Parameter           | Descritpiton                                                 |
| ------------------- | ------------------------------------------------------------ |
| exp_id              | The experiment ID specified to store the result under `/index_advisor_selector/index_selection/swirl_selection/exp_res` |
| algo                | The learning-based index advisor to be assessed, i.e., `"swirl", "drlinda", "dqn"` |
| workload_embedder   | The class of the workload representation of the learning-based index advisors |
| observation_manager | The class of the state representation of the learning-based index advisors |
| action_manager      | The class of the action space of the learning-based index advisors |
| reward_calculator   | The class of the reward function of the learning-based index advisors |
| rl_exp_load         | The configurations of the trained learning-based index advisors |
| rl_model_load       | The agent of the trained learning-based index advisors       |
| rl_env_load         | The environment of the trained learning-based index advisors |

```shell
# Training

python swirl_main.py

--algo swirl --exp_id swirl_tpch_v1 
--timesteps 100000 --seed 666

--constraint storage --max_budgets 500

--exp_conf_file /configuration_loader/index_advisor/rl_run_conf/swirl_config.json

--work_size 18 --work_gen load 
--work_type not_template --temp_num 22

--training_instances 80 --validation_testing_instances 5

--work_file /workload_generator/template_based/tpch_work_temp_multi.json
--eval_file /workload_generator/template_based/tpch_work_temp_multi.json

--db_conf_file /configuration_loader/database/db_con.conf
--schema_file /configuration_loader/database/schema_tpch.json
--colinfo_load /configuration_loader/database/colinfo_tpch.json

-------------------------------------------------------------------------------------------

# Inference

python swirl_run.py

--algo swirl --seed 666

--constraint storage --max_budgets 500

--rl_exp_load /index_advisor_selector/index_selection/swirl_selection/exp_res/swirl_tpch_v1/experiment_object.pickle
--rl_model_load /index_advisor_selector/index_selection/swirl_selection/exp_res/swirl_tpch_v1/best_mean_reward_model.zip
--rl_env_load /index_advisor_selector/index_selection/swirl_selection/exp_res/swirl_tpch_v1/vec_normalize.pkl

--work_file /workload_generator/template_based/tpch_work_temp_multi.json

--db_conf_file /configuration_loader/database/db_con.conf

```



## Reference

**We sincerely appreciate the authors of the following work for their efforts over the research of index advisors assessed in our work !**

[1] Rainer Schlosser, Jan Kossmann, and Martin Boissier. 2019. *Efficient Scalable Multi-attribute Index Selection Using Recursive Strategies*. In ICDE. 1238–1249.

[2] Gary Valentin, Michael Zuliani, Daniel C. Zilio, Guy M. Lohman, and Alan Skelley. 2000. *DB2 Advisor: An Optimizer Smart Enough to Recommend Its Own Indexes*. In ICDE. 101–110.

[3] Surajit Chaudhuri and Vivek R. Narasayya. 1997. An Efficient Cost-Driven Index Selection Tool for Microsoft SQL Server. In VLDB. 146–155.

[4] Kyu-Young Whang. 1987. Index Selection in Relational Databases. Foundations of Data Organization (1987), 487–500.

[5] Nicolas Bruno and Surajit Chaudhuri. 2005. Automatic Physical Database Tuning: A Relaxation-based Approach. In SIGMOD Conference. 227–238.

[6] S. Chaudhuri and V. Narasayya. 2020. Anytime Algorithm of Database Tuning Advisor for Microsoft SQL Server. https://www.microsoft.com/en-us/research/publication/.

[7] Alberto Caprara, Matteo Fischetti, and Dario Maio. 1995. Exact and Approximate Algorithms for the Index Selection Problem in Physical Database Design. IEEE Trans. Knowl. Data Eng. 7, 6 (1995), 955–967.

[8] Debabrata Dash, Neoklis Polyzotis, and Anastasia Ailamaki. 2011. CoPhy: A Scalable, Portable, and Interactive Index Advisor for Large Workloads. Proc. VLDB Endow. 4, 6 (2011), 362–372.

[9] Jan Kossmann, Alexander Kastius, and Rainer Schlosser. 2022. SWIRL: Selection of Workload-aware Indexes using Reinforcement Learning. In EDBT. 2:155–2:168. 

[10] Zahra Sadri, Le Gruenwald, and Eleazar Leal. 2020. DRLindex: deep reinforcement learning index advisor for a cluster database. In IDEAS. 11:1–11:8.

[11] Hai Lan, Zhifeng Bao, and Yuwei Peng. 2020. An Index Advisor Using Deep Reinforcement Learning. In CIKM. 2105–2108.

[12] R. Malinga Perera, Bastian Oetomo, Benjamin I. P. Rubinstein, and Renata Borovica-Gajic. 2021. DBA bandits: Self-driving index tuning under ad-hoc, analytical workloads with safety guarantees. In ICDE. 600–611.

[13] Xuanhe Zhou, Luyang Liu, Wenbo Li, Lianyuan Jin, Shifu Li, Tianqing Wang, and Jianhua Feng. 2022. AutoIndex: An Incremental Index Management System for Dynamic Workloads. In ICDE. 2196–2208.

[14] Wentao Wu, Chi Wang, Tarique Siddiqui, Junxiong Wang, Vivek R. Narasayya, Surajit Chaudhuri, and Philip A. Bernstein. 2022. Budget-aware Index Tuning with Reinforcement Learning. In SIGMOD Conference. 1528–1541.

[15] Bailu Ding, Sudipto Das, Ryan Marcus, Wentao Wu, Surajit Chaudhuri, and Vivek R. Narasayya. 2019. AI Meets AI: Leveraging Query Executions to Improve Index Recommendations. In SIGMOD Conference. 1241–1258.

[16] Tarique Siddiqui, Wentao Wu, Vivek R. Narasayya, and Surajit Chaudhuri. 2022. DISTILL: Low-Overhead Data-Driven Techniques for Filtering and Costing Indexes for Scalable Index Tuning. Proc. VLDB Endow. 15, 10 (2022), 2019–2031.

[17] Jiachen Shi, Gao Cong, and Xiaoli Li. 2022. Learned Index Benefits: Machine Learning Based Index Performance Estimation. Proc. VLDB Endow. 15, 13 (2022), 3950–3962.

[18] Yue Zhao, Gao Cong, Jiachen Shi, and Chunyan Miao. 2022. QueryFormer: A Tree Transformer Model for Query Plan Representation. Proc. VLDB Endow. 15, 8 (2022), 1658–1670.

