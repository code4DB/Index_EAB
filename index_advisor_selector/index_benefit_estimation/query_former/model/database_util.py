import numpy as np
import pandas as pd
import csv
import torch

from index_advisor_selector.index_benefit_estimation.benefit_utils.benefit_const import alias2table_job


def floyd_warshall_rewrite(adjacency_matrix, dist_inf=60):
    """
    The Floyd-Warshall algorithm: a dynamic programming algorithm,
    used to find the shortest paths between all pairs of vertices in a weighted directed graph.
    :param adjacency_matrix:
    :param dist_inf:
    :return:
    """
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    M = adjacency_matrix.copy().astype('long')
    for i in range(nrows):
        for j in range(ncols):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = dist_inf

    for k in range(nrows):
        for i in range(nrows):
            for j in range(nrows):
                M[i][j] = min(M[i][j], M[i][k] + M[k][j])
    return M


def get_job_table_sample(workload_file_name, num_materialized_samples=1000):
    tables = []
    samples = []

    # Load queries
    with open(workload_file_name + ".csv", "r") as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        # 'table', 'join', 'predicate', 'card'
        for row in data_raw:
            tables.append(row[0].split(","))

            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
    print("Loaded queries with len ", len(tables))

    # Load bitmaps
    # zw: uses bit manipulation to divide `(num_materialized_samples + 7)` by 8(equivalent to right-shifting by 3)
    #     The value represents the number of bytes needed to store a bitmap for the materialized_samples.
    num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    with open(workload_file_name + ".bitmaps", "rb") as f:
        for i in range(len(tables)):
            # zw: read 4 bytes at a time, assuming that these bytes represent an integer value.
            four_bytes = f.read(4)
            if not four_bytes:
                print("Error while reading 'four_bytes'")
                exit(1)
            # zw: converts the previously read 4 bytes into an integer value
            #     'little': to interpret the bytes as a little-endian integer.
            num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
            # zw: creates an empty NumPy array called bitmaps with
            #     the specified shape (num_bitmaps_curr_query, num_bytes_per_bitmap * 8).
            bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
            for j in range(num_bitmaps_curr_query):
                # Read bitmap
                bitmap_bytes = f.read(num_bytes_per_bitmap)
                if not bitmap_bytes:
                    print("Error while reading 'bitmap_bytes'")
                    exit(1)
                # zw: unpacks the bitmap_bytes into a NumPy array of uint8 type.
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            samples.append(bitmaps)
    print("Loaded bitmaps")

    # zw: to dict()
    table_sample = []
    for ts, ss in zip(tables, samples):
        d = {}
        for t, s in zip(ts, ss):
            tf = t.split(" ")[0]  # remove alias
            d[tf] = s
        table_sample.append(d)

    return table_sample


def get_hist_file(hist_path, bin_number=50):
    # ['table', 'column', 'freq', 'bins', 'table_column']
    hist_file = pd.read_csv(hist_path)
    for i in range(len(hist_file)):
        freq = hist_file['freq'][i]
        freq_np = np.frombuffer(bytes.fromhex(freq), dtype=np.float)
        hist_file['freq'][i] = freq_np

    table_column = []
    for i in range(len(hist_file)):
        table = hist_file['table'][i]
        col = hist_file['column'][i]
        table_alias = ''.join([tok[0] for tok in table.split('_')])
        if table == 'movie_info_idx':
            table_alias = 'mi_idx'
        combine = '.'.join([table_alias, col])
        table_column.append(combine)
    hist_file['table_column'] = table_column

    for rid in range(len(hist_file)):
        hist_file['bins'][rid] = \
            [int(i) for i in hist_file['bins'][rid][1:-1].split(' ') if len(i) > 0]

    if bin_number != 50:  # zw: len(hist_file['bins'][rid]) != bin_number? 51?
        hist_file = re_bin(hist_file, bin_number)

    return hist_file


def re_bin(hist_file, target_number):
    for i in range(len(hist_file)):
        freq = hist_file['freq'][i]
        bins = freq2bin(freq, target_number)
        hist_file['bins'][i] = bins
    return hist_file


def freq2bin(freqs, target_number):
    freq = freqs.copy()
    maxi = len(freq) - 1

    step = 1. / target_number
    mini = 0
    while freq[mini + 1] == 0:
        mini += 1
    pointer = mini + 1
    cur_sum = 0
    res_pos = [mini]
    residue = 0
    while pointer < maxi + 1:
        cur_sum += freq[pointer]
        freq[pointer] = 0
        if cur_sum >= step:
            cur_sum -= step
            res_pos.append(pointer)
        else:
            pointer += 1

    if len(res_pos) == target_number:
        res_pos.append(maxi)

    return res_pos


class Batch:
    def __init__(self, attn_bias, rel_pos, heights, x, y=None):
        super(Batch, self).__init__()

        self.heights = heights
        self.x, self.y = x, y
        self.attn_bias = attn_bias
        self.rel_pos = rel_pos

    def to(self, device):
        self.heights = self.heights.to(device)
        self.x = self.x.to(device)

        self.attn_bias, self.rel_pos = self.attn_bias.to(device), self.rel_pos.to(device)

        return self

    def __len__(self):
        return self.in_degree.size(0)


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    # dont know why add 1, comment out first
    #    x = x + 1 # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype) + 1
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    """
    [1, 1, -inf, -inf,
     1, 1, -inf, -inf,
     0, 0, -inf, -inf,
     0, 0, -inf, -inf]
    """
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def collator(small_set):
    y = small_set[1]
    xs = [s['x'] for s in small_set[0]]

    num_graph = len(y)
    x = torch.cat(xs)
    attn_bias = torch.cat([s['attn_bias'] for s in small_set[0]])
    rel_pos = torch.cat([s['rel_pos'] for s in small_set[0]])
    heights = torch.cat([s['heights'] for s in small_set[0]])

    return Batch(attn_bias, rel_pos, heights, x), y


def filterDict2Hist(hist_file, filterDict, encoding):
    buckets = len(hist_file['bins'][0])
    empty = np.zeros(buckets - 1)
    ress = np.zeros((3, buckets - 1))
    for i in range(len(filterDict['colId'])):
        colId = filterDict['colId'][i]
        col = encoding.idx2col[colId]
        if col == 'NA':
            ress[i] = empty
            continue
        bins = hist_file.loc[hist_file['table_column'] == col, 'bins'].item()

        opId = filterDict['opId'][0]
        op = encoding.idx2op[opId]

        val = filterDict['val'][0]
        mini, maxi = encoding.column_min_max_vals[col]
        val_unnorm = val * (maxi - mini) + mini

        left = 0
        right = len(bins) - 1
        for j in range(len(bins)):
            if bins[j] < val_unnorm:
                left = j
            if bins[j] > val_unnorm:
                right = j
                break

        res = np.zeros(len(bins) - 1)
        # : ?
        if op == '=':
            res[left:right] = 1
        elif op == '<':
            res[:left] = 1
        elif op == '>':
            res[right:] = 1
        ress[i] = res

    ress = ress.flatten()
    return ress


def formatJoin(json_node):
    join = None
    if 'Hash Cond' in json_node:
        join = json_node['Hash Cond']
    elif 'Join Filter' in json_node:
        join = json_node['Join Filter']
    # : index cond
    # elif 'Index Cond' in json_node and not json_node['Index Cond'][-2].isnumeric() and \
    #         not json_node['Index Cond'][-3].isnumeric() and " = " in str(json_node['Index Cond']):
    #     join = json_node['Index Cond']

    # sometimes no alias, say t.id
    # remove repeat (both way are the same)
    if join is not None:
        twoCol = join[1:-1].split(' = ')

        # print(json_node.keys())
        # print(json_node["Node Type"])
        # print(join)
        #
        # print(twoCol)
        # print(len(twoCol[0].split('.')) == 1)

        # (1006): to be removed.
        try:
            twoCol = [json_node['Alias'] + '.' + col
                      if len(col.split('.')) == 1 else col for col in twoCol]
            join = ' = '.join(sorted(twoCol))
        except:
            import traceback
            traceback.print_exc()

            join = None

    return join


def formatFilter(plan):
    alias = None
    if 'Alias' in plan:
        alias = plan['Alias']
    else:
        pl = plan
        while 'parent' in pl:
            pl = pl['parent']
            if 'Alias' in pl:
                alias = pl['Alias']
                break

    filters = []
    if 'Filter' in plan:
        filters.append(plan['Filter'])
    if 'Index Cond' in plan and plan['Index Cond'][-2].isnumeric():
        filters.append(plan['Index Cond'])
    if 'Recheck Cond' in plan:
        filters.append(plan['Recheck Cond'])

    return filters, alias


class Encoding:
    def __init__(self, column_min_max_vals, col2idx, op2idx={'>': 0, '=': 1, '<': 2, 'NA': 3}):
        self.column_min_max_vals = column_min_max_vals
        self.col2idx = col2idx
        self.op2idx = op2idx

        idx2col = {}
        for k, v in col2idx.items():
            idx2col[v] = k
        self.idx2col = idx2col

        idx2op = {}
        for k, v in op2idx.items():
            idx2op[v] = k
        self.idx2op = idx2op
        # self.idx2op = {0: '>', 1: '=', 2: '<', 3: 'NA'}

        self.type2idx = {}
        self.idx2type = {}
        self.join2idx = {}
        self.idx2join = {}

        self.table2idx = {'NA': 0}
        self.idx2table = {0: 'NA'}

    def normalize_val(self, column, val, log=False):
        mini, maxi = self.column_min_max_vals[column]

        val_norm = 0.0
        if maxi > mini:
            # (1006): newly modified.
            # val_norm = (val - mini) / (maxi - mini)
            # (1006): to be removed. try
            try:
                val = val.split("::")[0]
                val, mini, maxi = float(val), float(mini), float(maxi)
                val_norm = (val - mini) / (maxi - mini)
            except:
                import traceback
                traceback.print_exc()

        return val_norm

    def encode_filters(self, filters=[], alias=None, alias2tbl=None):
        # filters: list of dict

        #        print(filt, alias)
        if len(filters) == 0:
            return {'colId': [self.col2idx['NA']],
                    'opId': [self.op2idx['NA']],
                    'val': [0.0]}

        res = {'colId': [], 'opId': [], 'val': []}
        for filt in filters:
            filt = ''.join(c for c in filt if c not in '()')
            # (1006): newly added.
            filt = filt.replace("(", "").replace(")", "")

            # : zw: OR? disjunctive predicates, unsupported.
            fs_temp = filt.split(' AND ')
            fs = list()
            for temp in fs_temp:
                fs.extend(temp.split(' OR '))

            for f in fs:
                #           print(filters)
                # (1006): newly added. maxsplit=2
                col, op, num = f.split(' ', maxsplit=2)
                col = col.split("::")[0]

                if op == "<>":
                    op = "NA"

                if alias in alias2table_job.keys():
                    column = alias2table_job[alias] + '.' + col
                else:
                    if alias is not None:
                        column = alias + '.' + col
                    else:
                        column = col
                #            print(f)

                # (1006): newly added.
                # sum, min, max, avg
                if column[:3] in ["sum", "min", "max", "avg"]:
                    column = column[3:]
                # count
                if column[:5] in ["count"]:
                    column = column[5:]

                # (1006): to be removed. try
                try:
                    if "." not in column:
                        column = f"{alias2tbl[column.split('_')[0]]}.{column}"

                    res['colId'].append(self.col2idx[column])
                    res['opId'].append(self.op2idx[op])
                    # (1006): newly modified.
                    # res['val'].append(self.normalize_val(column, float(num)))
                    res['val'].append(self.normalize_val(column, num))
                except:
                    import traceback
                    traceback.print_exc()

                    return {'colId': [self.col2idx['NA']],
                            'opId': [self.op2idx['NA']],
                            'val': [0.0]}

        return res

    def encode_join(self, join):
        if join is not None:
            # (1006): newly added.
            # (1006): to be removed. try
            try:
                col1, col2 = join.split(" = ")
                tbl1, name1 = col1.split(".")
                tbl2, name2 = col2.split(".")
                if tbl1 in alias2table_job.keys():
                    tbl1 = alias2table_job[tbl1]
                if tbl2 in alias2table_job.keys():
                    tbl2 = alias2table_job[tbl2]
                join = " = ".join(sorted([f"{tbl1}.{name1}", f"{tbl2}.{name2}"]))
            except:
                join = None

                import traceback
                traceback.print_exc()

        if join not in self.join2idx:
            # (1007): to be removed.
            join = None
            # self.join2idx[join] = len(self.join2idx)
            # self.idx2join[self.join2idx[join]] = join

        return self.join2idx[join]

    def encode_table(self, table):
        if table not in self.table2idx:
            self.table2idx[table] = len(self.table2idx)
            self.idx2table[self.table2idx[table]] = table
        return self.table2idx[table]

    def encode_type(self, nodeType):
        """
        {'Gather': 0, 'Hash Join': 1, 'Seq Scan': 2, 'Hash': 3,
         'Bitmap Heap Scan': 4, 'Bitmap Index Scan': 5, 'Nested Loop': 6,
         'Index Scan': 7, 'Merge Join': 8, 'Gather Merge': 9,
         'Materialize': 10, 'BitmapAnd': 11, 'Sort': 12}
         'Memoize'

        :param nodeType:
        :return:
        """
        if nodeType not in self.type2idx:
            # (1007): to be removed.
            nodeType = "Result"
            # self.type2idx[nodeType] = len(self.type2idx)
            # self.idx2type[self.type2idx[nodeType]] = nodeType
        return self.type2idx[nodeType]


class TreeNode:
    def __init__(self, nodeType, typeId, filt, card, join, join_str, filterDict):
        self.nodeType = nodeType
        self.typeId = typeId
        self.filter = filt

        self.table = 'NA'
        self.table_id = 0
        self.query_id = None  # so that sample bitmap can recognise

        self.join = join
        self.join_str = join_str
        self.card = card  # 'Actual Rows', do not use due to erroneous estimates
        self.children = []
        self.rounds = 0

        self.filterDict = filterDict

        self.parent = None

        self.feature = None

    def addChild(self, treeNode):
        self.children.append(treeNode)

    def __str__(self):
        #        return TreeNode.print_nested(self)
        return '{} with {}, {}, {} children'.format(self.nodeType, self.filter, self.join_str, len(self.children))

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def print_nested(node, indent=0):
        print('--' * indent + '{} with {} and {}, {} childs'.format(node.nodeType, node.filter, node.join_str,
                                                                    len(node.children)))
        for k in node.children:
            TreeNode.print_nested(k, indent + 1)
