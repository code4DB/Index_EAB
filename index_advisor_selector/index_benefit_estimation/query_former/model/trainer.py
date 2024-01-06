import time
import logging
from tqdm import tqdm

import torch
import pandas as pd

import numpy as np
from scipy.stats import pearsonr

from .database_util import collator
from .dataset import PlanTreeDataset


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def print_qerror(preds_unnorm, labels_unnorm, prints=False):
    qerror = list()
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    e_mean = np.mean(qerror)
    e_50, e_90 = np.median(qerror), np.percentile(qerror, 90)
    e_95, e_max = np.percentile(qerror, 95), np.max(qerror)

    if prints:
        logging.info("Mean: {}, Median: {}, 90th: {}, 95th: {}, Max: {}".format(e_mean, e_50, e_90, e_95, e_max))

    res = {
        'q_mean': round(e_mean, 4),
        'q_median': round(e_50, 4),
        'q_90': round(e_90, 4),
        "q_95": round(e_95, 4),
        "q_max": round(e_max, 4)
    }

    return qerror, res


def get_corr(ps, ls):  # unnormalised
    ps = np.array(ps)
    ls = np.array(ls)
    corr, _ = pearsonr(np.log(ps), np.log(ls))  # : why unnormalised?

    return corr


def train(model, train_ds, val_ds, crit, cost_norm, args,
          optimizer=None, scheduler=None, device="cpu"):
    to_pred, bs, epochs, clip_size = \
        args.to_predict, args.batch_size, args.epoch_num, args.clip_size
    lr = args.lr

    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    t0 = time.time()
    rng = np.random.default_rng()

    best_prev = 999999
    for epoch in tqdm(range(epochs)):
        losses = 0
        cost_predss = np.empty(0)

        model.train()
        train_idxs = rng.permutation(len(train_ds))
        cost_labelss = np.array(train_ds.costs)[train_idxs]
        for idxs in tqdm(chunks(train_idxs, bs)):
            optimizer.zero_grad()

            batch, batch_labels = collator(list(zip(*[train_ds[j] for j in idxs])))

            # cost_label, cardinality_label
            l, r = zip(*batch_labels)

            batch = batch.to(device)
            batch_cost_label = torch.FloatTensor(l).to(device)

            cost_preds, _ = model(batch)
            cost_preds = cost_preds.squeeze()

            loss = crit(cost_preds, batch_cost_label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_size)
            optimizer.step()

            losses += loss.item()
            cost_predss = np.append(cost_predss, cost_preds.detach().cpu().numpy())

        if (epoch + 1) % args.model_save_gap == 0:
            model_state_dict = model.state_dict()
            model_source = {
                "settings": args,
                "model": model_state_dict,
            }
            torch.save(model_source, args.model_save.format(
                args.exp_id, "FORMER_" + str(epoch + 1)))

        logging.info('Epoch: {}  Avg Loss: {}, Time: {}'.format(epoch, losses / len(train_ds), time.time() - t0))
        _, train_scores = print_qerror(cost_norm.unnormalize_labels(cost_predss), cost_labelss, True)

        if epoch > 40:
            y_pred, qerror, test_scores, corrs = evaluate(model, val_ds, bs, cost_norm, device, prints=False)

            if test_scores['q_mean'] < best_prev:  # mean mse
                best_model_path = logging_func(args, epoch, test_scores,
                                               filename='log.txt', save_model=True, model=model)
                best_prev = test_scores['q_mean']

        scheduler.step()

    return model, best_model_path


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

    y_pred, qerror, eval_score, corr = evaluate(methods['model'], ds, methods['bs'], methods['cost_norm'], methods['device'], True)

    return eval_score, ds


def evaluate(model, ds, bs, norm, device, prints=False):
    model.eval()
    cost_predss = np.empty(0)

    time_start = time.time()

    with torch.no_grad():
        for i in tqdm(range(0, len(ds), bs)):
            batch, batch_labels = collator(list(zip(*[ds[j] for j in range(i, min(i + bs, len(ds)))])))

            batch = batch.to(device)

            cost_preds, _ = model(batch)
            cost_preds = cost_preds.squeeze()

            cost_predss = np.append(cost_predss, cost_preds.cpu().detach().numpy())

    time_end = time.time()
    print(f"The time overhead ({len(cost_predss)}) is {time_end - time_start}.")

    qerror, scores = print_qerror(norm.unnormalize_labels(cost_predss), ds.costs, prints)
    corr = get_corr(norm.unnormalize_labels(cost_predss), ds.costs)
    if prints:
        logging.info('Corr: ', corr)

    return cost_predss, qerror, scores, corr


def logging_func(args, epoch, qscores, filename=None, save_model=False, model=None):
    # (1004): newly modified.
    arg_keys = [attr for attr in dir(args) if not attr.startswith('__') and not attr.startswith('_')]
    arg_vals = [getattr(args, attr) for attr in arg_keys]

    res = dict(zip(arg_keys, arg_vals))
    model_checkpoint = str(hash(tuple(arg_vals))) + '.pt'

    res['epoch'] = epoch
    res['model'] = model_checkpoint

    res = {**res, **qscores}

    # filename = args.newpath + filename
    # if filename is not None:
    #     if os.path.isfile(filename):
    #         df = pd.read_csv(filename)
    #         df = df.append(res, ignore_index=True)
    #         df.to_csv(filename, index=False)
    #     else:
    #         df = pd.DataFrame(res, index=[0])
    #         df.to_csv(filename, index=False)

    model_checkpoint = args.model_save.format(args.exp_id, model_checkpoint)
    if save_model:
        torch.save({
            'model': model.state_dict(),
            'args': args
        }, model_checkpoint)

    return res['model']
