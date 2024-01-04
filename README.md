# Index Advisor [Experiment, Analysis & Benchmark]

This is the code respository of the testbed proposed in the **Index Advisor (EA&B)** paper, which conducts a comprehensive assessment of the heuristic-based and learning-based index advisors.

Specifically, the testbed is comprised of three modules. 

- **(1) Configuration Loader:** initializes a series of evaluation settings, including the benchmark, the index advisor, and the database;
- **(2) Workload Generator:** supports three methods for generating workloads with diverse features (e.g., query changes due to typical workload drifts) to simulate the requirements posed by various scenarios;
- **(3) Index Advisor Selector:** implements existing index advisors, including seven heuristic-based index advisors and ten learning-based index advisors.



## Project Structure

```
Index_EAB/
├── configuration_loader              # Module 1: the evaluation settings                 
│   ├── becnhmark
│   ├── index_advisor
│   │   ├── heu_run_conf
│   │   └── rl_run_conf
│   └── database
├── workload_generator				 # Module 2: the testing workloads
│   ├── template_based
│   ├── perturbation_based
│   │   └── perturb_utils
│   └── random
├── index_advisor_selector			 # Module 3: the implemented index advisors
│   ├── index_candidate_generation
│   │   └── distill_model
│   │       └── distill_utils
│   ├── index_selection
│   │   ├── heu_selection
│   │   │   ├── heu_utils
│   │   │   └── heu_algos
│   │   ├── dqn_selection
│   │   │   └── dqn_utils
│   │   ├── swirl_selection
│   │   │   ├── swirl_utils
│   │   │   ├── gym_db
│   │   │   └── stable_baselines
│   │   ├── mab_selection
│   │   │   ├── bandits
│   │   │   ├── simualtion
│   │   │   ├── shared
│   │   │   └── database
│   │   └── mcts_selection
│   │       └── mcts_utils
└── ├── index_benefit_estimation
    └── ├── benefit_utils
        ├── optmizer_cost
        ├── tree_model
        ├── index_cost_lib
        └── query_former
```



## Setup

We introduce the indispensable step, i.e., experiment setup for the experiment evaluations, you should check the following things:

- Create **the database instance** according to the toolkit provided;
- Create **the HypoPG extension** on the TPC-H/TPC-DS database instance for the usage of hypothetical index according to [HypoPG/hypopg: Hypothetical Indexes for PostgreSQL (github.com)](https://github.com/HypoPG/hypopg);
- Create **the python virtual environment**. Specifically, you can utilize the following script and the corresponding file `requirements.txt` is provided under the main directory. Please check the packages required are properly installed.

```shell
# Create the virtualenv `TRAP`
conda create -n Index_EAB python=3.7		 	

# Activate the virtualenv `TRAP`
conda activate Index_EAB				

# Install requirements with pip
while read requirement; do pip install $requirement; done < requirements.txt	
```

