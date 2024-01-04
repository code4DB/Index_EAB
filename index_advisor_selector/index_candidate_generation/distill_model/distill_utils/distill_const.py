# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: benefit_const
# @Author: Wei Zhou
# @Time: 2023/10/5 14:09

# node types
int_ops = ["Seq Scan", "Hash Join", "Nested Loop", "CTE Scan",
           "Index Only Scan", "Index Scan", "Merge Join", "Sort"]

ops = ["Aggregate", "BitmapAnd", "Hash Join", "Gather",
       "Nested Loop", "Gather Merge", "Materialize", "Bitmap Heap Scan",
       "Index Scan", "Merge Join", "Seq Scan", "Sort", "Hash",
       "Index Only Scan", "Bitmap Index Scan"]

ops = {"Gather": 0, "Hash Join": 1, "Seq Scan": 2,
       "Hash": 3, "Bitmap Heap Scan": 4, "Bitmap Index Scan": 5,
       "Nested Loop": 6, "Index Scan": 7, "Merge Join": 8,
       "Gather Merge": 9, "Materialize": 10, "BitmapAnd": 11, "Sort": 12}

ops_dict = {"Aggregate": 0, "Bitmap Heap Scan": 1, "Bitmap Index Scan": 2,
            "BitmapAnd": 3, "BitmapOr": 4, "Gather": 5, "Gather Merge": 6,
            "Hash": 7, "Hash Join": 8, "Index Only Scan": 9, "Index Scan": 10, "Materialize": 11,
            "Merge Join": 12, "Nested Loop": 13, "Result": 14, "Seq Scan": 15, "Sort": 16}

ops_join_dict = {"Hash Join": 0, "Merge Join": 1, "Nested Loop": 2}
ops_sort_dict = {"Sort": 0}
ops_group_dict = {"Aggregate": 0}
ops_scan_dict = {"Bitmap Heap Scan": 0, "Bitmap Index Scan": 1, "Index Scan": 2, "Index Only Scan": 3, "Seq Scan": 4}

ind_ops_dict = {"Hash Join": 0, "Merge Join": 1, "Nested Loop": 2,
                "Sort": 3,
                "Aggregate": 4,
                "Bitmap Heap Scan": 5, "Bitmap Index Scan": 6, "Index Scan": 7, "Index Only Scan": 8, "Seq Scan": 9}

ops_range = {">": 0, "<": 1, ">=": 2, "<=": 3, "!=": 4}
ops_equal = {"=": 0}

rule_dict = {"order-by": 0, "group-by": 1, "selection": 2, "join": 3}

alias2table_tpch = {"n": "nation", "c": "customer", "o": "orders", "l": "lineitem",
                    "ps": "partsupp", "r": "region", "p": "part", "s": "supplier"}

table2alias_tpch = {"nation": "n", "customer": "c", "orders": "o", "lineitem": "l",
                    "partsupp": "ps", "region": "r", "part": "p", "supplier": "s"}

alias2table_tpcds = {"dv": "dbgen_version", "ca": "customer_address", "c": "customer", "cd": "customer_demographics",
                     "d": "date_dim", "hd": "household_demographics", "cc": "call_center", "cp": "catalog_page",
                     "cr": "catalog_returns", "i": "item", "r": "reason", "sm": "ship_mode", "t": "time_dim",
                     "w": "warehouse", "cs": "catalog_sales", "p": "promotion", "ib": "income_band",
                     "inv": "inventory", "s": "store", "sr": "store_returns", "ss": "store_sales",
                     "web": "web_site", "wp": "web_page", "wr": "web_returns", "ws": "web_sales"}

table2alias_tpcds = {"dbgen_version": "dv", "customer_address": "ca", "customer": "c", "customer_demographics": "cd",
                     "date_dim": "d", "household_demographics": "hd", "call_center": "cc", "catalog_page": "cp",
                     "catalog_returns": "cr", "item": "i", "reason": "r", "ship_mode": "sm", "time_dim": "t",
                     "warehouse": "w", "catalog_sales": "cs", "promotion": "p", "income_band": "ib",
                     "inventory": "inv", "store": "s", "store_returns": "sr", "store_sales": "ss",
                     "web_site": "web", "web_page": "wp", "web_returns": "wr", "web_sales": "ws"}

table2alias_job = {"aka_title": "at", "char_name": "cn", "role_type": "rt", "comp_cast_type": "cct", "movie_link": "ml",
                   "link_type": "lt", "cast_info": "ci", "complete_cast": "cc", "title": "t", "aka_name": "an",
                   "movie_companies": "mc", "kind_type": "kt", "name": "n", "company_type": "ct", "movie_keyword": "mk",
                   "movie_info": "mi", "person_info": "pi", "company_name": "cn", "keyword": "k",
                   "movie_info_idx": "mi_idx", "info_type": "it"}

alias2table_job = {"at": "aka_title", "cn": "company_name", "rt": "role_type", "cct": "comp_cast_type",
                   "ml": "movie_link", "lt": "link_type", "ci": "cast_info", "cc": "complete_cast", "t": "title",
                   "an": "aka_name", "mc": "movie_companies", "kt": "kind_type", "n": "name", "ct": "company_type",
                   "mk": "movie_keyword", "mi": "movie_info", "pi": "person_info", "k": "keyword",
                   "mi_idx": "movie_info_idx", "it": "info_type"}
