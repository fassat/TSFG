import random

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, mutual_info_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor  # 用于多线程

def sample(args,df,df_c,df_d,init_c_seq,init_d_seq,ppo, workers_c,workers_d,device,processData_lists,max_seq_len,c_vocab,d_vocab,best_temp_c,best_temp_d):
    init_c_df = torch.from_numpy(df_c.values).float().transpose(0, 1).to(device)
    init_d_df = torch.from_numpy(df_d.values).float().transpose(0, 1).to(device)
    if df_c.shape[1] > 1:
        stop_c_index = c_vocab.index('STOP')
        for num, worker in enumerate(workers_c):
            # init_c_df = torch.from_numpy(worker.new_c_data.values).float().transpose(0, 1).to(device)
            actions_c, log_c_prob, action_softmax_c,new_c_seq = ppo.choose_action_c(init_c_df,init_c_seq,best_temp_c)
            if len(new_c_seq) < max_seq_len:
                new_c_seq.extend([stop_c_index] * (max_seq_len - len(new_c_seq)))
            new_c_data = processData_lists[num].generate_newdata_phase2(new_c_seq,mode='c')
            worker.new_c_data = new_c_data
            worker.actions_c = actions_c
            worker.log_c_prob = log_c_prob
            worker.action_softmax_c = action_softmax_c
            worker.new_c_seq = new_c_seq
    if df_d.shape[1] > 1:
        stop_d_index = d_vocab.index('STOP')
        for num, worker in enumerate(workers_d):

            actions_d, log_d_prob, action_softmax_d, new_d_seq = ppo.choose_action_d(init_d_df, init_d_seq,best_temp_d)
            if len(new_d_seq) < max_seq_len:
                new_d_seq.extend([stop_d_index] * (max_seq_len - len(new_d_seq)))
            new_d_data = processData_lists[num].generate_newdata(new_d_seq,mode='d')
            worker.new_d_data = new_d_data
            worker.actions_d = actions_d
            worker.log_d_prob = log_d_prob
            worker.action_softmax_d = action_softmax_d
            worker.new_d_seq = new_d_seq
    return workers_c,workers_d




def process_worker(worker_c, worker_d, data_info, mode, predictor=None):

    if mode == 'train':
        X_c = worker_c.new_c_data
        X_d = worker_d.new_d_data
        X = pd.concat([X_c, X_d], axis=1)
        X_re = remove_duplication(X)
        X_re = np.where(X_re > 1e15, 1e15, X_re)

        y = data_info['dataframe'].loc[:, data_info['label_name']]
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        if X_re.shape[0] == 0:
            worker_c.new_score = 0
            worker_d.new_score = 0
            return worker_c, worker_d

        if data_info['type'] == "cls":
            clf = RandomForestClassifier(n_estimators=10, random_state=0)
            scores = cross_val_score(clf, X_re, y, scoring='f1_micro', cv=5)
        else:
            model = RandomForestRegressor(n_estimators=10, random_state=0)
            rae_score1 = make_scorer(rae, greater_is_better=True)
            scores = cross_val_score(model, X_re, y, cv=5, scoring=rae_score1)

        worker_c.new_score = np.array(scores).mean()
        worker_d.new_score = np.array(scores).mean()
        worker_c.new_data = X
        worker_d.new_data = X

    else:
        new_seq = np.concatenate([worker_c.new_seq_c, worker_d.new_seq_d], axis=0)
        score = predictor.eval_predictor(new_seq)
        worker_c.new_score = score.cpu().detach().numpy()[0]
        worker_d.new_score = score.cpu().detach().numpy()[0]

    return worker_c, worker_d


def get_reward(args, workers_c, workers_d, data_info, predictor_c,predictor_d, mode):
    with ThreadPoolExecutor() as executor:  # 使用线程池进行并行处理
        futures = []

        for worker_c, worker_d in zip(workers_c, workers_d):
            futures.append(executor.submit(process_worker, worker_c, worker_d, data_info, mode, predictor_c))

        results = [future.result() for future in futures]


        for worker_c, worker_d in results:
            pass
    return workers_c, workers_d


def rae(y, y_hat):
    y = np.array(y).reshape(-1)
    y_hat = np.array(y_hat).reshape(-1)
    y_mean = np.mean(y)
    absolute_errors = np.abs(y_hat - y)
    mean_errors = np.abs(y_mean - y)
    rae = np.sum(absolute_errors) / np.sum(mean_errors)
    res = 1 - rae
    return res


def remove_duplication(data):
    _, idx = np.unique(data, axis=1, return_index=True)
    y = data.iloc[:, np.sort(idx)]
    return y