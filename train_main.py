import argparse
import logging
import os
import random
import sys
import time
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

from TSFG.embedding_policy_network import TransformerEncoder, TransformerDecoder, Predictor, Pretrainer
from TSFG.runPhaseOne import runPhaseOne, evaluate
from TSFG.training_phase2 import sample, get_reward
from TSFG.worker import Worker
from utils import get_binning_df, get_action, log_b, get_init_seq
from utils import datainfos
from TSFG.process_data.process_data import Pipeline
from TSFG.PPO import PPO
from TSFG.data import data_split, subset_to_dataframe

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
warnings.filterwarnings("ignore")


class TransFG:
    def __init__(self, input_data, args):
        log_path = fr"./logs/{args.file_name}"
        log_b(log_path)
        data_info = datainfos[args.file_name]
        self.target = data_info['target']
        args.target = self.target
        self.type = data_info['type']
        if self.type == 'cls':
            self.metric = 'f1'
        elif self.type == 'reg':
            self.metric = 'rae'
        self.v_columns = data_info['v_columns']
        self.d_columns = data_info['d_columns']
        logging.info(f'File name: {args.file_name}')
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        random.seed(1)
        np.random.seed(1)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        self.ori_df = input_data
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def main_function(self, args):

        dataset = self.ori_df
        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(dataset[self.target])
        dataset[self.target] = target
        train_dataset, val_dataset, test_dataset = data_split(dataset)
        train_df = subset_to_dataframe(train_dataset)
        val_df = subset_to_dataframe(val_dataset)
        test_df = subset_to_dataframe(test_dataset)
        train_df_val_df = pd.concat([train_df, val_df], axis=0)

        target, type, metric = self.target, self.type, self.metric
        score_ori, scores_ori = self.data_scores(train_df_val_df, args, type, metric)
        acc = self.data_scores_test(train_df, val_df, test_df, args, type, metric)
        v_columns, d_columns = self.v_columns, self.d_columns
        logging.info(f'score_ori={score_ori}')
        logging.info(f'score_acc={acc}')
        self.best_score = 0

        if type == "cls":
            d_out = len(dataset[target].unique())
        new_train_df, new_train_c_df, new_train_d_df, new_v_columns, new_d_columns, v_columns, d_columns, df = get_binning_df(
            args, train_df,
            v_columns,
            d_columns,
            type)
        new_val_df, new_val_c_df, new_val_d_df, new_v_columns, new_d_columns, v_columns, d_columns, df = get_binning_df(
            args, val_df,
            v_columns,
            d_columns,
            type)
        new_test_df, new_test_c_df, new_test_d_df, new_v_columns, new_d_columns, v_columns, d_columns, df = get_binning_df(
            args, test_df,
            v_columns,
            d_columns,
            type)
        d_in = len(new_d_columns)
        c_in = len(new_v_columns)
        train_loader_c, val_loader_c, test_loader_c = get_train_loader(new_train_c_df, new_val_c_df, new_test_c_df)
        train_loader_d, val_loader_d, test_loader_d = get_train_loader(new_train_d_df, new_val_d_df, new_test_d_df)

        n_features_c, n_features_d = len(new_v_columns), len(new_d_columns)
        c_vocab, d_vocab = get_action(n_features_c, n_features_d)
        init_c_seq, init_d_seq = get_init_seq(n_features_c, n_features_d, c_vocab, d_vocab)
        score_b, scores_b = self.data_scores(new_train_df, args, type, metric)
        df_c, df_d = new_train_df.loc[:, new_v_columns + [target]], new_train_df.loc[:, new_d_columns + [target]]
        ori_continuous_data = df.loc[:, v_columns]
        ori_discrete_data = df.loc[:, d_columns]

        df_c_test, df_d_test = new_test_df.loc[:, new_v_columns + [target]], new_test_df.loc[:,
                                                                             new_d_columns + [target]]
        ori_continuous_data_test = new_test_df.loc[:, v_columns]
        ori_discrete_data_test = new_test_df.loc[:, d_columns]

        df_c_val, df_d_val = new_val_df.loc[:, new_v_columns + [target]], new_val_df.loc[:, new_d_columns + [target]]
        ori_continuous_data_val = new_val_df.loc[:, v_columns]
        ori_discrete_data_val = new_val_df.loc[:, d_columns]

        df_label = new_train_df.loc[:, target]
        feature_nums = n_features_c + n_features_d
        data_nums_train = new_train_df.shape[0]
        data_nums_val = new_val_df.shape[0]
        data_nums_test = new_test_df.shape[0]
        data_nums_train_val = new_train_df.shape[0] + new_val_df.shape[0]
        operations_c = len(c_vocab)
        operations_d = len(d_vocab)
        d_model = args.d_model
        batch_size = args.batch_size
        hidden_size = args.hidden_size
        max_seq_len = max(len(c_vocab) + len(init_c_seq), len(d_vocab) + len(init_d_seq))
        max_seq_len = min(max_seq_len, 100)
        max_len_vocab = max(len(c_vocab), len(d_vocab))
        data_infomation_train = {'dataframe': new_train_df,
                                 'continuous_columns': new_v_columns,
                                 'discrete_columns': new_d_columns,
                                 'continuous_data': df_c,
                                 'discrete_data': df_d,
                                 'label_name': target,
                                 'type': type,
                                 'ori_continuous_data': ori_continuous_data,
                                 'ori_discrete_data': ori_discrete_data
                                 }
        data_infomation_val = {'dataframe': new_val_df,
                               'continuous_columns': new_v_columns,
                               'discrete_columns': new_d_columns,
                               'continuous_data': df_c_val,
                               'discrete_data': df_d_val,
                               'label_name': target,
                               'type': type,
                               'ori_continuous_data': ori_continuous_data_val,
                               'ori_discrete_data': ori_discrete_data_val
                               }
        data_infomation_test = {'dataframe': new_test_df,
                                'continuous_columns': new_v_columns,
                                'discrete_columns': new_d_columns,
                                'continuous_data': df_c_test,
                                'discrete_data': df_d_test,
                                'label_name': target,
                                'type': type,
                                'ori_continuous_data': ori_continuous_data_test,
                                'ori_discrete_data': ori_discrete_data_test
                                }
        seads = []
        self.num_layers = args.nums_layers
        self.num_hiddens = args.hidden_size
        self.vocab_size = max_len_vocab
        self.num_features_c = n_features_c
        self.num_features_d = n_features_d
        num_heads = 8
        hidden_dim = 128
        num_layers = 1
        self.Encoder_c = TransformerEncoder(n_features_c, d_model, num_heads, hidden_dim, num_layers).to(self.device)
        self.Decoder_c = TransformerDecoder(self.vocab_size, d_model, num_heads, self.num_hiddens, self.num_layers,
                                            self.device).to(
            self.device)
        self.Encoder_d = TransformerEncoder(n_features_d, d_model, num_heads, hidden_dim, num_layers).to(self.device)
        self.Decoder_d = TransformerDecoder(self.vocab_size, d_model, num_heads, self.num_hiddens, self.num_layers,
                                            self.device).to(self.device)

        if n_features_c > 0:
            if type == 'reg':
                d_out = 1
            self.predictor_c = Predictor(n_features_c, d_out, self.device).to(self.device)
            self.pretrainer_c = Pretrainer(self.predictor_c, self.device).to(self.device)
            self.pretrainer_c.fit(train_loader_c, val_loader_c, c_in, type)
        if n_features_d > 0:
            if type == 'reg':
                d_out = 1
            self.predictor_d = Predictor(n_features_d, d_out, self.device).to(self.device)
            self.pretrainer_d = Pretrainer(self.predictor_d, self.device).to(self.device)
            self.pretrainer_d.fit(train_loader_d, val_loader_d, d_in, type)

        self.pipeline_train = Pipeline(data_infomation_train, c_vocab, n_features_c, d_vocab, n_features_d,
                                       data_nums_train)
        self.pipeline_val = Pipeline(data_infomation_val, c_vocab, n_features_c, d_vocab, n_features_d, data_nums_val)
        self.pipeline_test = Pipeline(data_infomation_val, c_vocab, n_features_c, d_vocab, n_features_d, data_nums_test)
        mode = 'train'
        max_len_vocab = max(len(c_vocab), len(d_vocab))
        vocab_size = max_len_vocab
        seeds_c = []
        seeds_d = []
        best_temp_c = None
        best_temp_d = None
        if n_features_c > 0:
            Encoder_c, Decoder_c, predictor, seeds_c, best_temp_c = runPhaseOne(args, train_loader_c, val_loader_c,
                                                                                self.Encoder_c,
                                                                                self.Decoder_c, self.predictor_c, df_c,
                                                                                df_d,
                                                                                self.device, max_seq_len,
                                                                                self.pipeline_train,
                                                                                self.pipeline_val, type, vocab_size,
                                                                                init_c_seq,
                                                                                init_c_seq, c_vocab, d_vocab, train_df,
                                                                                val_df, df_c_val, 'c',
                                                                                d_out)

        if n_features_d > 0:
            Encoder_d, Decoder_d, predictor, seeds_d, best_temp_d = runPhaseOne(args, train_loader_d, val_loader_d,
                                                                                self.Encoder_d,
                                                                                self.Decoder_d, self.predictor_d, df_c,
                                                                                df_d,
                                                                                self.device, max_seq_len,
                                                                                self.pipeline_train,
                                                                                self.pipeline_val, type, vocab_size,
                                                                                init_d_seq,
                                                                                init_d_seq, c_vocab, d_vocab, train_df,
                                                                                val_df, df_d_val, 'd',
                                                                                d_out)


        new_df = pd.concat([new_train_df, new_val_df], axis=0)
        new_c_df = pd.concat([new_train_c_df, new_val_c_df], axis=0)
        new_d_df = pd.concat([new_train_d_df, new_val_d_df], axis=0)
        data_infomation_train_val = {'dataframe': new_df,
                                     'continuous_columns': new_v_columns,
                                     'discrete_columns': new_d_columns,
                                     'continuous_data': df_c,
                                     'discrete_data': df_d,
                                     'label_name': target,
                                     'type': type,
                                     'ori_continuous_data': new_c_df,
                                     'ori_discrete_data': new_d_df
                                     }
        input_dim = new_df.shape[0]
        train_rewards = []
        rewards = []
        self.ppo = PPO(args, input_dim, max_len_vocab, d_model, input_dim, self.device)
        for epoch in tqdm(range(args.epochs)):

            workers_c = []
            workers_d = []
            processData_list = []

            for i in range(args.episodes):
                worker = Worker(args)
                if len(seeds_c) > 0:
                    worker.new_c_data = seeds_c[i]
                if len(seeds_d) > 0:
                    worker.new_d_data = seeds_d[i]
                worker.scores_b = scores_b
                worker.best_score = 0
                workers_c.append(worker)
                workers_d.append(worker)
                processData_list.append(
                    Pipeline(data_infomation_train_val, c_vocab, n_features_c, d_vocab, n_features_d,
                             data_nums_train_val))

            workers_c, workers_d = sample(args, new_df, new_c_df, new_d_df, init_c_seq.copy(), init_d_seq.copy(),
                                          self.ppo, workers_c, workers_d, self.device, processData_list, max_seq_len,
                                          c_vocab, d_vocab, best_temp_c, best_temp_d)

            workers_c, workers_d = get_reward(args, workers_c, workers_d, data_infomation_train_val, self.predictor_c,
                                              self.predictor_c, mode)



            for i, worker in enumerate(workers_c):
                rewards.append(worker.new_score)

                train_rewards.append([worker.new_score, worker.new_seq, worker.new_data])

                if np.mean(worker.new_score) > self.best_score:
                    self.best_score = np.mean(worker.new_score)
                    logging.info(f"epoch:{epoch}_new_best_score{self.best_score}")
                    best_seq_c = worker.new_c_seq
                    best_seq_d = worker.new_d_seq


            if new_c_df.shape[1] > 1:
                self.ppo.update_c(workers_c, new_c_df, init_c_seq.copy(),best_temp_c)


            if new_d_df.shape[1] > 1:
                self.ppo.update_d(workers_d, new_d_df, init_d_seq.copy(),best_temp_d)


        logging.info(rewards)
        pipeline_train = Pipeline(data_infomation_train, c_vocab, n_features_c, d_vocab, n_features_d, data_nums_train)
        pipeline_test = Pipeline(data_infomation_test, c_vocab, n_features_c, d_vocab, n_features_d, data_nums_test)
        pipeline_val = Pipeline(data_infomation_val, c_vocab, n_features_c, d_vocab, n_features_d, data_nums_val)
        best_train_data_c = None
        best_test_data_c = None
        best_val_data_c = None
        best_train_data_d = None
        best_test_data_d = None
        best_val_data_d = None
        if n_features_c > 0:
            logging.info(best_seq_c)
            best_train_data_c = pipeline_train.generate_newdata_phase2(best_seq_c, mode='c')
            best_test_data_c = pipeline_test.generate_newdata_phase2(best_seq_c, mode='c')
            best_val_data_c = pipeline_val.generate_newdata_phase2(best_seq_c, mode='c')
        if n_features_d > 0:
            logging.info(best_seq_d)
            best_train_data_d = pipeline_train.generate_newdata_phase2(best_seq_d, mode='d')
            best_test_data_d = pipeline_test.generate_newdata_phase2(best_seq_d, mode='d')
            best_val_data_d = pipeline_val.generate_newdata_phase2(best_seq_d, mode='d')

        best_train_data = pd.concat([best_train_data_c, best_train_data_d], axis=1)
        best_train_data[self.target] = train_df[self.target].reset_index(drop=True)

        best_test_data = pd.concat([best_test_data_c, best_test_data_d], axis=1)
        best_test_data[self.target] = test_df[self.target].reset_index(drop=True)

        best_val_data = pd.concat([best_val_data_c, best_val_data_d], axis=1)
        best_val_data[self.target] = val_df[self.target].reset_index(drop=True)

        acc = self.data_scores_test(best_train_data, best_val_data, best_test_data, args, type, metric)

        logging.info(f"acc:{acc}")

    def data_scores(self, df: pd.DataFrame, args, type, metric):
        target = args.target
        X = df.drop(columns=[target])
        y = df[target]
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        if type == "cls":
            clf = RandomForestClassifier(n_estimators=10, random_state=0)
            scores = cross_val_score(clf, X, y, scoring='f1_micro', cv=5)
        else:
            model = RandomForestRegressor(n_estimators=10, random_state=0)
            rae_score1 = make_scorer(rae, greater_is_better=True)
            scores = cross_val_score(model, X, y, cv=5, scoring=rae_score1)
        return np.array(scores).mean(), scores

    def data_scores_test(self, df_train, df_val, df_test, args, type, metric):
        target = args.target
        X_train = df_train.drop(columns=[target])  # Assuming 'target' is the column to predict
        y_train = df_train[target]

        X_val = df_val.drop(columns=[target])
        y_val = df_val[target]  # Use transform here to avoid refitting

        X_test = df_test.drop(columns=[target])
        y_test = df_test[target]  # Use transform here to avoid refitting

        # Add labels to track data source
        X_train['dataset'] = 'train'
        X_val['dataset'] = 'val'
        X_test['dataset'] = 'test'


        X_combined = pd.concat([X_train, X_val, X_test], axis=0)
        y_combined = pd.concat([y_train, y_val, y_test], axis=0)
        X_combined = remove_duplication(X_combined)


        test_size = len(X_test)

        X_train_val_combined = X_combined[X_combined['dataset'] != 'test'].drop(columns=['dataset'])
        y_train_val_combined = y_combined[X_combined['dataset'] != 'test']

        X_test_final = X_combined[X_combined['dataset'] == 'test'].drop(columns=['dataset'])
        y_test_final = y_combined[X_combined['dataset'] == 'test']


        if type == "cls":
            # Initialize and train the RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=10, random_state=42)
            rf.fit(X_train_val_combined, y_train_val_combined)

            # Make predictions on the test data
            y_pred = rf.predict(X_test_final)

            # Calculate the accuracy
            # accuracy = accuracy_score(y_test, y_pred)
            accuracy = f1_score(y_test_final, y_pred, average='weighted')
        else:
            # Initialize the RandomForestClassifier
            rf = RandomForestRegressor(n_estimators=10, random_state=0)
            # Train the model on the training data
            rf.fit(X_train_val_combined, y_train_val_combined)

            # Make predictions on the test data
            y_pred = rf.predict(X_test_final)

            # Calculate the accuracy
            # accuracy = accuracy_score(y_test, y_pred)
            accuracy = rae(y_test_final, y_pred)
        return accuracy


def rae(y, y_hat):
    y = np.array(y).reshape(-1)
    y_hat = np.array(y_hat).reshape(-1)
    y_mean = np.mean(y)
    absolute_errors = np.abs(y_hat - y)
    mean_errors = np.abs(y_mean - y)
    rae = np.sum(absolute_errors) / np.sum(mean_errors)
    res = 1 - rae
    return res


def get_final_reward(sorted_list, data_info):
    y = data_info['dataframe'].loc[:, data_info['label_name']]
    for list in sorted_list:
        X = list[2]
        X = np.where(X > 1e15, 1e15, X)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        if X.shape[0] == 0:
            continue
        if data_info['type'] == "cls":
            clf = RandomForestClassifier(n_estimators=10, random_state=0)
            scores = cross_val_score(clf, X, y, scoring='f1_micro', cv=5)
        else:
            model = RandomForestRegressor(n_estimators=10, random_state=0)
            rae_score1 = make_scorer(rae, greater_is_better=True)
            scores = cross_val_score(model, X, y, cv=5, scoring=rae_score1)
        logging.info(f"real_score:{np.mean(scores)}_pre_score{list[0]}")


def get_train_loader(new_train_c_df, new_val_c_df, new_test_c_df):
    x_data_c = torch.tensor(new_train_c_df.iloc[:, :-1].values, dtype=torch.float32)  # 特征
    y_data_c = torch.tensor(new_train_c_df.iloc[:, -1].values, dtype=torch.float32)  # 标签
    # 创建 TensorDataset
    dataset_train_c = TensorDataset(x_data_c, y_data_c)

    x_data_c = torch.tensor(new_val_c_df.iloc[:, :-1].values, dtype=torch.float32)  # 特征
    y_data_c = torch.tensor(new_val_c_df.iloc[:, -1].values, dtype=torch.float32)  # 标签
    # 创建 TensorDataset
    dataset_val_c = TensorDataset(x_data_c, y_data_c)

    x_data_c = torch.tensor(new_test_c_df.iloc[:, :-1].values, dtype=torch.float32)  # 特征
    y_data_c = torch.tensor(new_test_c_df.iloc[:, -1].values, dtype=torch.float32)  # 标签
    # 创建 TensorDataset
    dataset_test_c = TensorDataset(x_data_c, y_data_c)

    train_loader_c = DataLoader(dataset_train_c, batch_size=1024, shuffle=True, pin_memory=True, drop_last=False)
    val_loader_c = DataLoader(dataset_val_c, batch_size=1024, pin_memory=True)
    test_loader_c = DataLoader(dataset_test_c, batch_size=1024, pin_memory=True)
    return train_loader_c, val_loader_c, test_loader_c


def remove_duplication(data):

    dataset_column = data['dataset'] if 'dataset' in data.columns else None

    data_without_dataset = data.drop(columns=['dataset']) if dataset_column is not None else data

    _, idx = np.unique(data_without_dataset, axis=1, return_index=True)

    data_without_duplicates = data_without_dataset.iloc[:, np.sort(idx)]

    if dataset_column is not None:
        data_without_duplicates['dataset'] = dataset_column

    return data_without_duplicates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='cpu')
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--ppo_epochs", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--file_name", type=str, default='')
    parser.add_argument("--seed", type=int, default=1, help='seed')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nums_layers", type=int, default=1)
    parser.add_argument("--entropy_weight", type=float, default=1e-4)
    parser.add_argument("--baseline_weight", type=float, default=0.95)

    args = parser.parse_args()
    dataset_path = f"{BASE_DIR}\\data\\{args.file_name}.csv"
    df = pd.read_csv(dataset_path)
    autofe = TransFG(df, args)
    autofe.main_function(args)
