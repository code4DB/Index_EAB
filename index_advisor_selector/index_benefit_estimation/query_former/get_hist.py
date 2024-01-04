import numpy as np
import pandas as pd
import psycopg2

# partial schema: table + column?
# hist, pg_stats
# non-numeric column

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


def to_vals(data_list):
    for dat in data_list:
        val = dat[0]
        if val is not None:
            break
    try:
        float(val)
        return np.array(data_list, dtype=float).squeeze()
    except:
        #         print(val)
        res = []
        for dat in data_list:
            try:
                # zw: returns a floating-point value representing
                # the number of seconds since the Unix epoch (January 1, 1970)
                mi = dat[0].timestamp()
            except:
                mi = 0
            res.append(mi)
        return np.array(res)


hist_file = pd.DataFrame(columns=['table', 'column', 'bins', 'table_column'])
for table, columns in imdb_schema.items():
    for column in columns:
        cmd = 'select {} from {} as {}'.format(column, table, t2alias[table])
        cur.execute(cmd)
        col = cur.fetchall()
        col_array = to_vals(col)
        # zw: compute the specified percentile of an array, ignoring any NaN values.
        hists = np.nanpercentile(col_array, range(0, 101, 2), axis=0)
        res_dict = {
            'table': table,
            'column': column,
            'table_column': '.'.join((table, column)),
            'bins': hists
        }
        hist_file = hist_file.append(res_dict, ignore_index=True)
