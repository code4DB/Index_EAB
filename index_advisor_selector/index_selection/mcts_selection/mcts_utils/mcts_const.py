# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: mcts_const
# @Author: Wei Zhou
# @Time: 2023/7/21 21:21

# TPC-H schema
tpch_tables = ["region", "nation", "supplier", "part", "customer", "orders", "partsupp", "lineitem"]

# TPC-DS schema
tpcds_tables = ["dbgen_version", "warehouse", "promotion", "web_page", "item", "income_band", "time_dim",
                "catalog_page", "web_site", "ship_mode", "customer_address", "reason", "store", "customer_demographics",
                "store_returns", "catalog_sales", "inventory", "store_sales", "web_returns", "web_sales", "customer",
                "household_demographics", "catalog_returns", "call_center", "date_dim"]

# JOB schema
job_table_alias = {"aka_title": "at",
                   "char_name": "cn",
                   "role_type": "rt",
                   "comp_cast_type": "cct",
                   "movie_link": "ml",
                   "link_type": "lt",
                   "cast_info": "ci",
                   "complete_cast": "cc",
                   "title": "t",
                   "aka_name": "an",
                   "movie_companies": "mc",
                   "kind_type": "kt",
                   "name": "n",
                   "company_type": "ct",
                   "movie_keyword": "mk",
                   "movie_info": "mi",
                   "person_info": "pi",
                   "company_name": "cn",
                   "keyword": "k",
                   "movie_info_idx": "mi_idx",
                   "info_type": "it"}
