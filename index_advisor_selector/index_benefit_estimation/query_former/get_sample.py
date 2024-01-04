import numpy as np
import pandas as pd

import psycopg2
from sqlalchemy import create_engine

# 1. generate 1000 sample points for each table
# 2. duplicate database schema from full db
#       pg_dump -h localhost imdb_load103 -U postgres -s -O > imdb_schema.sql
# 3. create small base by in psql
#       create database imdb_sample
# 4. create schema using imdb_schema.sql (remove all PK,FK constraints)
#       psql -h localhost -U postgres -d imdb_load103_sample -p 5432 -f imdb_schema.sql
# 5. load the sample data using pandas and sqlalchemy
# 6. query the small base to get sample bitmaps for each predicate


imdb_schema = {'title': ['t.id', 't.kind_id', 't.production_year'],
               'movie_companies': ['mc.id',
                                   'mc.company_id',
                                   'mc.movie_id',
                                   'mc.company_type_id'],
               'cast_info': ['ci.id', 'ci.movie_id', 'ci.person_id', 'ci.role_id'],
               'movie_info_idx': ['mi_idx.id', 'mi_idx.movie_id', 'mi_idx.info_type_id'],
               'movie_info': ['mi.id', 'mi.movie_id', 'mi.info_type_id'],
               'movie_keyword': ['mk.id', 'mk.movie_id', 'mk.keyword_id']}
t2alias = {'title': 't', 'movie_companies': 'mc', 'cast_info': 'ci',
           'movie_info_idx': 'mi_idx', 'movie_info': 'mi', 'movie_keyword': 'mk'}
alias2t = {}
for k, v in t2alias.items():
    alias2t[v] = k

conn = psycopg2.connect(database="imdb_load103", host="127.0.0.1", port="5432",
                        user="postgres", password="dmai4db2021.")
conn.set_session(autocommit=True)
cur = conn.cursor()

# sampling extension
# cmd = 'CREATE EXTENSION tsm_system_rows'
# cur.execute(cmd)

# create the sampling database
# cmd = 'CREATE DATABASE imdb_load103_sample'
# cur.execute(cmd)

tables = list(imdb_schema.keys())
sample_data = {}
for table in tables:
    cur.execute("SELECT * FROM {} LIMIT 0".format(table))
    colnames = [desc[0] for desc in cur.description]

    ts = pd.DataFrame(columns=colnames)

    # zw: table size fewer than 1k etc.
    for num in range(1000):
        cmd = 'SELECT * FROM {} TABLESAMPLE SYSTEM_ROWS(1)'.format(table)
        cur.execute(cmd)
        samples = cur.fetchall()
        for i, row in enumerate(samples):
            ts.loc[num] = row

    sample_data[table] = ts

# SQLAlchemy is an open-source Python library that provides a set of tools and abstractions
# for working with databases using the SQL (Structured Query Language) language.
# It serves as an Object-Relational Mapping (ORM) tool, allowing developers to
# interact with databases in a more Pythonic and object-oriented manner.
# zw: establish a database connection?

engine = create_engine('postgresql://postgres:dmai4db2021.@localhost:5432/imdb_load103_sample')

conn = psycopg2.connect(database="imdb_load103_sample", host="127.0.0.1", port="5432",
                        user="postgres", password="dmai4db2021.")
conn.set_session(autocommit=True)
cur = conn.cursor()

for k, v in sample_data.items():
    v['sid'] = list(range(1000))
    cmd = 'alter table {} add column sid integer'.format(k)
    cur.execute(cmd)

    # cmd = 'alter table {} drop column sid'.format(k)
    # cur.execute(cmd)

    # zw: write the contents of a DataFrame directly to a SQL database, if_exists='replace'?
    v.to_sql(name=k, con=engine, if_exists='append', index=False)

query_file = pd.read_csv('data/imdb/workloads/synthetic.csv', sep='#', header=None)
# zw: cast_info ci; nan; ci.person_id,=,172968; 838
#     title t,movie_info mi; t.id=mi.movie_id; t.kind_id,<,3,t.production_year,=,2008,mi.info_type_id,>,2; 297013
query_file.columns = ['table', 'join', 'predicate', 'card']

table_samples = []
for no, row in query_file.iterrows():
    table_sample = {}
    preds = row['predicate'].split(',')
    for i in range(0, len(preds), 3):
        # zw: fetch the qualified tuple for each predicate.
        left, op, right = preds[i:i + 3]
        alias, col = left.split('.')
        table = alias2t[alias]
        pred_string = ''.join((col, op, right))
        q = 'select sid from {} where {}'.format(table, pred_string)
        cur.execute(q)

        sps = np.zeros(1000).astype('uint8')
        sids = cur.fetchall()
        sids = np.array(sids).squeeze()
        # zw: set qualified position to 1.
        if sids.size > 1:
            sps[sids] = 1

        if table in table_sample:
            table_sample[table] = table_sample[table] & sps
        else:
            table_sample[table] = sps
    table_samples.append(table_sample)
