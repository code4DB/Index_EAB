import json
from collections import deque
from torch.utils.data import Dataset

from .database_util import *


# max_pred_num

class PlanTreeDataset(Dataset):
    def __init__(self, json_df: pd.DataFrame, train: pd.DataFrame, encoding,
                 hist_file, card_norm, cost_norm, to_predict, table_sample, alias2tbl=None):
        self.table_sample = table_sample
        self.encoding = encoding
        self.hist_file = hist_file

        # (1006): newly added.
        self.alias2tbl = alias2tbl

        self.length = len(json_df)
        # train = train.loc[json_df['id']]

        # zw: json.loads(): parse a JSON string into a Python object
        # nodes = [json.loads(plan)['Plan'] for plan in json_df['json']]
        nodes = [item["w/ plan"] for item in json_df]

        # (1005): newly modified.
        # self.cards = [node['Actual Rows'] for node in nodes]
        # self.costs = [json.loads(plan)['Execution Time'] for plan in json_df['json']]
        self.costs = [item["w/ actual cost"] for item in json_df]

        # (1005): newly modified.
        # self.card_labels = torch.from_numpy(card_norm.normalize_labels(self.cards))
        self.cost_labels = torch.from_numpy(cost_norm.normalize_labels(self.costs))

        self.to_predict = to_predict
        if to_predict == 'cost':
            self.gts = self.costs
            self.labels = self.cost_labels
        # (1005): newly modified.
        # elif to_predict == 'card':
        #     self.gts = self.cards
        #     self.labels = self.card_labels
        # elif to_predict == 'both':  # try not to use, just in case
        #     self.gts = self.costs
        #     self.labels = self.cost_labels
        else:
            raise Exception('Unknown to_predict type')

        # (1005): newly modified.
        idxs = list(range(len(json_df)))
        # idxs = list(json_df['id'])

        self.treeNodes = []  # for mem collection
        self.collated_dicts = [self.js_node2dict(i, node) for i, node in zip(idxs, nodes)]

    def js_node2dict(self, idx, node):
        treeNode = self.traversePlan(node, idx, self.encoding)
        # {'features': (node_num, 1165), 'heights': (node_num,), 'adjacency_list': (edge_num, 2)}
        _dict = self.node2dict(treeNode)
        # padded, {'x'(features): (1, 30, 1165), 'attn_bias': (1, 31, 31), 'rel_pos': (1, 30, 30), 'heights': (1, 30)}
        collated_dict = self.pre_collate(_dict)

        self.treeNodes.clear()
        del self.treeNodes[:]

        return collated_dict

    def __len__(self):
        return self.length

    # (1006): newly modified.
    def __getitem__(self, idx):
        return self.collated_dicts[idx], (self.cost_labels[idx], self.cost_labels[idx])

    # def __getitem__(self, idx):
    #     return self.collated_dicts[idx], (self.cost_labels[idx], self.card_labels[idx])

    def old_getitem(self, idx):
        return self.dicts[idx], (self.cost_labels[idx], self.card_labels[idx])

    # pre-process first half of old collator
    def pre_collate(self, the_dict, max_node=30, rel_pos_max=20):
        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])  # : zw: len(the_dict['features']) = the_dict['features'].size()[0]?
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # : zw: 1 for super node?

        edge_index = the_dict['adjacency_list'].t()
        if len(edge_index) == 0:
            shortest_path_result = np.array([[0]])
            path = np.array([[0]])
            adj = torch.tensor([[0]]).bool()
        else:
            adj = torch.zeros([N, N], dtype=torch.bool)
            adj[edge_index[0, :], edge_index[1, :]] = True

            shortest_path_result = floyd_warshall_rewrite(adj.numpy())

        rel_pos = torch.from_numpy((shortest_path_result)).long()

        attn_bias[1:, 1:][rel_pos >= rel_pos_max] = float('-inf')
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1)
        rel_pos = pad_rel_pos_unsqueeze(rel_pos, max_node)

        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)

        return {
            'x': x,
            'attn_bias': attn_bias,
            'rel_pos': rel_pos,
            'heights': heights
        }

    def node2dict(self, treeNode):
        adj_list, num_child, features = self.topo_sort(treeNode)
        heights = self.calculate_height(adj_list, len(features))

        return {
            # 'features': torch.FloatTensor(features),
            # 'heights': torch.LongTensor(heights),
            'features': torch.FloatTensor(np.array(features)),
            'heights': torch.LongTensor(np.array(heights)),
            'adjacency_list': torch.LongTensor(np.array(adj_list))
        }

    def topo_sort(self, root_node):
        #        nodes = []
        adj_list = []  # from parent to children
        num_child = []
        features = []

        toVisit = deque()
        toVisit.append((0, root_node))
        next_id = 1
        while toVisit:
            idx, node = toVisit.popleft()
            #            nodes.append(node)
            features.append(node.feature)
            num_child.append(len(node.children))
            for child in node.children:
                toVisit.append((next_id, child))
                adj_list.append((idx, next_id))
                next_id += 1

        return adj_list, num_child, features

    def traversePlan(self, plan, idx, encoding):  # bfs accumulate plan
        nodeType = plan['Node Type']
        typeId = encoding.encode_type(nodeType)
        card = None  # plan['Actual Rows']
        filters, alias = formatFilter(plan)
        join = formatJoin(plan)
        joinId = encoding.encode_join(join)
        filters_encoded = encoding.encode_filters(filters, alias, self.alias2tbl)

        root = TreeNode(nodeType, typeId, filters, card, joinId, join, filters_encoded)

        self.treeNodes.append(root)

        if 'Relation Name' in plan:
            root.table = plan['Relation Name']
            root.table_id = encoding.encode_table(plan['Relation Name'])
        root.query_id = idx

        # zw: (type_join, filts, mask, hists, table, sample)
        root.feature = node2feature(root, encoding, self.hist_file, self.table_sample)

        #    print(root)
        if "Plans" in plan:
            for subplan in plan['Plans']:
                subplan['parent'] = plan
                node = self.traversePlan(subplan, idx, encoding)
                node.parent = root
                root.addChild(node)

        return root

    def calculate_height(self, adj_list, tree_size):
        if tree_size == 1:
            return np.array([0])

        adj_list = np.array(adj_list)
        node_ids = np.arange(tree_size, dtype=int)
        node_order = np.zeros(tree_size, dtype=int)
        uneval_nodes = np.ones(tree_size, dtype=bool)

        parent_nodes = adj_list[:, 0]
        child_nodes = adj_list[:, 1]

        n = 0
        while uneval_nodes.any():
            uneval_mask = uneval_nodes[child_nodes]
            unready_parents = parent_nodes[uneval_mask]

            node2eval = uneval_nodes & ~np.isin(node_ids, unready_parents)
            node_order[node2eval] = n
            uneval_nodes[node2eval] = False
            n += 1
        return node_order


def node2feature(node, encoding, hist_file, table_sample,
                 max_pred_num=8, sample_num=1000):
    """
    1) operator(E_o): 1;
    2) join(E_j): 1;
    3) table(E_t): 1; only one table involved in one node? plan['Relation Name']
    4) predicates(E_p): 3*3, col op val, number=3
       mask: 3;
    5) histogram(E_h): 3*50;
    6) sample bitmap(E_s): 1000.

    :return:
    """

    # type, join, filter123, mask123(for pred, hist embedding)
    # 1, 1, 3x3=9, 3
    type_join = np.array([node.typeId, node.join])

    # : add sample (or so-called table)
    num_filter = len(node.filterDict['colId'])
    # : zw: at most 3 predicates?
    # (1006): newly modified.
    pad = np.zeros((3, max(0, max_pred_num - num_filter)))
    # pad = np.zeros((3, max_pred_num - num_filter))
    filts = np.array(list(node.filterDict.values()))  # cols, ops, vals
    # 3x3 -> 9, get back with reshape 3,3
    # (1006): newly modified. [:, :max_pred_num]
    filts = np.concatenate((filts, pad), axis=1)[:, :max_pred_num].flatten()

    # zw: for pred, hist embedding(average non-mask ones)
    mask = np.zeros(max_pred_num)
    mask[:num_filter] = 1

    # zw: (3*50, )
    # (1006): newly modified.
    if hist_file is not None:
        hists = filterDict2Hist(hist_file, node.filterDict, encoding)

    # table, bitmap, 1 + 1000 bits
    table = np.array([node.table_id])
    # (1006): newly modified.
    if table_sample is not None:
        if node.table_id == 0:
            sample = np.zeros(sample_num)
        else:
            sample = table_sample[node.query_id][node.table]

    # (1006): newly modified.
    if hist_file is not None and table_sample is not None:
        # return np.concatenate((type_join, filts, mask))
        return np.concatenate((type_join, filts, mask, hists, table, sample))
    else:
        return np.concatenate((type_join, filts, mask, table))
