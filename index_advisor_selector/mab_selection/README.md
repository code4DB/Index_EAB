# DBA Bandits
We propose a self-driving approach to online index selection that eschews the DBA and query optimiser, and instead learns the benefits of viable structures through strategic exploration and direct performance observation.

**Cite:** R Malinga Perera, Bastian Oetomo, Benjamin IP Rubinstein, and Renata Borovica-Gajic. DBA bandits: Self-driving index tuning under ad-hoc, analytical workloads with safety guarantees. In 2021 IEEE 37th International Conference on Data Engineering (ICDE), pages 600–611. IEEE, 2021.

## How to run
1. Setting up the environment 
    1. You need to generate data for your benchmark 
    2. You need to generate queries for you benchmark
    3. You need to create a workload file for your benchmark (example workload files can be found in `resources/workloads` folder)
    4. Notice that we have included the predicates and payload of those queries in the workload file
    5. Add DB connection details to `config/db.conf`
2. Setting up your experiment. Our framework allows you easily setup experiments in `config/exp.conf`
    1. See the examples in `config/exp.conf`
    2. Check the explanation under 'Experiment Config Explained' below
3. Program starting point is `simulation/sim_run_experiment.py`. This is the file that you will be running. Simply add a 
list of experiments you need to run and execute the file
4. Results
    1. Results will be saved under in a sub-folder under `experiments` folder. Sub-folder name will be the name you provided 
    for the experiment.
    2. Results will include graphs, CSV of main results, pickle fill of all the data.
    
### Experiment Config Explained

#### Example 
1. [tpc_h_static_10_MAB]
2. reps = 1
3. rounds = 25
4. hyp_rounds = 0
5. workload_shifts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
6. queries_start = [0, 21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 231, 252, 273, 294, 315, 336, 357, 378, 399, 420, 441, 462, 483, 504]
7. queries_end = [21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 231, 252, 273, 294, 315, 336, 357, 378, 399, 420, 441, 462, 483, 504, 525]
8. ta_runs = [1]
9. ta_workload = optimal
10. workload_file = \resources\workloads\tpc_h_static_100.json
11. config_shifts = [0]
12. config_start = [0]
13. config_end = [20]
14. max_memory = 25000
15. input_alpha = 1
16. input_lambda = 0.5
17. time_weight = 5
18. memory_weight = 0
19. components = ["MAB"]
20. mab_versions = ["simulation.sim_c3ucb_vR"]

#### Explanation

1. Name of the experiment
2. Number of times we run the experiment
3. Number of bandit rounds in the experiment 
4. Ignore
5. Places where the workload shifts (even in static we run different templates in rounds, so it is simply 1 to number of rounds)0
6. For each round we run a set of queries from workload file starting from `queries_start` index to ...
7. `queries_end` index
8. Rounds where we invoke PDTool
9. `optimal` for static and dynamic shifting, `last_run` for dynamic random
10. Location of the workload file
11. Ignore
12. Ignore
13. Ignore
14. Memory limit in MB
15. Bandit parameter `alpha`
16. Bandit parameter `lambda`
17. Ignore
18. Ignore
19. Components you need to compare in this experiment (subset of `MAB`, `TA_OPTIMAL`, `NO_INDEX`, ...) 
20. MAB version we are running (name of the mab file)