import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from index_advisor_selector.index_benefit_estimation.query_former.model.database_util import collator
from index_advisor_selector.index_benefit_estimation.query_former.model.dataset import PlanTreeDataset

import logging

tf_step = 0
summary_writer = None


def set_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # log to file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


def add_summary_value(key, value, step=None):
    if step is None:
        summary_writer.add_scalar(key, value, tf_step)
    else:
        summary_writer.add_scalar(key, value, step)


class Normalizer:
    def __init__(self, mini=None, maxi=None):
        self.mini = mini
        self.maxi = maxi

    def normalize_labels(self, labels, reset_min_max=False):
        # added 0.001 for numerical stability
        labels = np.array([np.log(float(l) + 0.001) for l in labels])
        if self.mini is None or reset_min_max:
            self.mini = labels.min()
            print("min log(label): {}".format(self.mini))
        if self.maxi is None or reset_min_max:
            self.maxi = labels.max()
            print("max log(label): {}".format(self.maxi))

        labels_norm = (labels - self.mini) / (self.maxi - self.mini)
        # Threshold labels <-- but why...
        labels_norm = np.minimum(labels_norm, 1)
        labels_norm = np.maximum(labels_norm, 0.001)

        return labels_norm

    def unnormalize_labels(self, labels_norm):
        labels_norm = np.array(labels_norm, dtype=np.float32)
        labels = (labels_norm * (self.maxi - self.mini)) + self.mini

        #         return np.array(np.round(np.exp(labels) - 0.001), dtype=np.int64)
        return np.array(np.exp(labels) - 0.001)


def seed_everything(seed):
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False


def normalize_data(val, column_name, column_min_max_vals):
    min_val = column_min_max_vals[column_name][0]
    max_val = column_min_max_vals[column_name][1]
    val = float(val)
    val_norm = 0.0
    if max_val > min_val:
        val_norm = (val - min_val) / (max_val - min_val)
    return np.array(val_norm, dtype=np.float32)


def get_corr(ps, ls):
    # unnormalised
    ps = np.array(ps)
    ls = np.array(ls)
    corr, _ = pearsonr(np.log(ps), np.log(ls))

    return corr


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    e_50, e_90 = np.median(qerror), np.percentile(qerror, 90)
    e_mean = np.mean(qerror)
    print("Median: {}".format(e_50))
    print("90th percentile: {}".format(e_90))
    print("Mean: {}".format(e_mean))
    return


def evaluate(model, ds, bs, norm, device):
    model.eval()
    cost_predss = np.empty(0)

    with torch.no_grad():
        for i in range(0, len(ds), bs):
            batch, batch_labels = collator(list(zip(*[ds[j] for j in range(i, min(i + bs, len(ds)))])))

            batch = batch.to(device)

            cost_preds, _ = model(batch)
            cost_preds = cost_preds.squeeze()

            cost_predss = np.append(cost_predss, cost_preds.cpu().detach().numpy())

    print_qerror(norm.unnormalize_labels(cost_predss), ds.costs)
    corr = get_corr(norm.unnormalize_labels(cost_predss), ds.costs)
    print('Corr: ', corr)

    return


def eval_workload(workload, methods):
    get_table_sample = methods['get_sample']

    workload_file_name = './data/imdb/workloads/' + workload
    table_sample = get_table_sample(workload_file_name)
    plan_df = pd.read_csv('./data/imdb/{}_plan.csv'.format(workload))
    workload_csv = pd.read_csv('./data/imdb/workloads/{}.csv'.format(workload), sep='#', header=None)
    workload_csv.columns = ['table', 'join', 'predicate', 'cardinality']

    ds = PlanTreeDataset(plan_df, workload_csv,
                         methods['encoding'], methods['hist_file'], methods['cost_norm'],
                         methods['cost_norm'], 'cost', table_sample)

    evaluate(methods['model'], ds, methods['bs'], methods['cost_norm'], methods['device'])
    return
