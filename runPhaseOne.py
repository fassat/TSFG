import random
from copy import deepcopy

import torch.nn as nn
import numpy as np
import pandas as pd
from torch.distributions import Categorical
from tqdm import tqdm
import torch
import torch.optim as optim

from TSFG.trian_utils import restore_parameters



def runPhaseOne(args, train_loader, val_loader, Encoder_c, Decoder_c, predictor, df_c, df_d,
                device, max_seq_len, pipeline_train, pipeline_val, type, vocab_size, init_c_seq, init_d_seq, c_vocab,
                d_vocab, train_df, val_df,df_val, type_c_or_d,n_cls=None):
    patience = 2
    min_lr = 1e-6
    lr = 1e-4
    val_loss_fn = None
    val_loss_mode = None
    verbose = True
    factor = 0.2
    nepochs = 100
    early_stopping_epochs = 5
    if type == 'reg':
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.CrossEntropyLoss()
    if val_loss_fn is None:
        val_loss_fn = loss_fn
        val_loss_mode = 'min'
    total_epochs = 0
    num_bad_epochs = 0
    loss_num = 1
    seeds = []
    n_features_c = df_c.shape[1]
    n_features_d = df_d.shape[1]
    if type_c_or_d == 'c':
        n_features = n_features_c
        vocab = c_vocab
        init_seq = init_c_seq
    else:
        n_features = n_features_d
        vocab = d_vocab
        init_seq = init_d_seq

    start_temp = 1.0  # 起始温度
    end_temp = 0.1  # 最终温度
    temp_steps = 5  # 温度衰减的步数（通常为总训练轮数）
    best_temp = None
    temp_feature = {
        'temp_feature': [],
        'val_loss': 0,
        'temp_data': pd.DataFrame()
    }
    # 使用 np.geomspace 生成温度序列
    for temp in np.geomspace(start_temp, end_temp, temp_steps):
        if verbose:
            print(f'Starting training with temp = {temp:.4f}\n')
        opt = optim.Adam(
            list(predictor.parameters()) + list(Encoder_c.parameters()) + list(Decoder_c.parameters()),
            lr=lr
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=val_loss_mode, factor=factor, patience=patience,
            min_lr=min_lr, verbose=verbose)
        best_Encoder_c = deepcopy(Encoder_c)
        best_Decoder_c = deepcopy(Decoder_c)
        best_predictor = deepcopy(predictor)
        for epoch in tqdm(range(nepochs), desc='Training', unit='epoch'):
            # Switch models to training mode.
            Encoder_c.train()
            Decoder_c.train()
            predictor.train()
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                emb_df = Encoder_c(x)
                max_features = 0
                new_seq = init_seq.copy()
                data_new = df_c.iloc[:, :-1].copy()
                temp_actions = []
                new_features = 0
                for i in range(max_seq_len):
                    new_action_softmax = Decoder_c(new_seq, emb_df, temp)
                    dist = Categorical(new_action_softmax)
                    action = dist.sample()
                    int_action = action.item()
                    temp_actions.append(int_action)
                    iscombine_type = 'c'

                    if action == vocab_size - 1:
                        break
                    elif vocab[action] == 'EOS':
                        new_features += 1
                        if new_features > 3 * n_features:
                            break
                        # TODO 根据temp_actions 生成新的data_new
                        if len(temp_actions) <= 1:
                            temp_actions = []
                            continue
                        feature_new = pipeline_train.generate_newdata(temp_actions, iscombine_type)
                        data_new = data_new.reset_index(drop=True)
                        feature_new = feature_new.reset_index(drop=True)
                        data_new = pd.concat([data_new, feature_new], axis=1)
                        data_new = data_new.clip(lower=-1e15, upper=1e15)

                        temp_actions = []
                        max_features += 1
                        pred = predictor(data_new)
                        if type == 'cls':
                            y = y.long()
                        else:
                            y = y.float()
                        loss = loss_fn(pred, y)
                        opt.zero_grad()
                        (loss / max_features).backward()
                        opt.step()
                        loss_num += 1
                    else:
                        temp_actions.append(int_action)
                        new_seq.append(int_action)

            # Calculate validation loss.
            Encoder_c.eval()
            Decoder_c.eval()
            predictor.eval()
            data_new = df_val.iloc[:, :-1].copy()
            temp_features = []

            with torch.no_grad():
                pred_list = []
                label_list = []

                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    emb_df = Encoder_c(x)
                    new_seq = init_seq.copy()

                    temp_actions = []
                    new_features = 0
                    for i in range(max_seq_len):
                        new_action_softmax = Decoder_c(new_seq, emb_df, temp)
                        dist = Categorical(new_action_softmax)
                        action = dist.sample()
                        int_action = action.item()
                        temp_actions.append(int_action)
                        temp_features.append(int_action)
                        iscombine_type = 'c'

                        if action == vocab_size - 1:
                            break
                        elif vocab[action] == 'EOS':
                            new_features += 1
                            if new_features > 3 * n_features:
                                break
                            if (len(temp_actions) <= 1):
                                temp_actions = []
                                continue
                            feature_new = pipeline_val.generate_newdata(temp_actions, iscombine_type)
                            data_new = data_new.reset_index(drop=True)
                            feature_new = feature_new.reset_index(drop=True)
                            data_new = pd.concat([data_new, feature_new], axis=1)

                            temp_actions = []
                            pred = predictor(data_new)
                            pred_list.append(pred.cpu())
                            if type == 'cls':
                                y = y.long()
                            else:
                                y = y.float()
                            label_list.append(y.cpu())

                if (len(pred_list) == 0):
                    continue
                pred = torch.cat(pred_list, 0)
                y = torch.cat(label_list, 0)
                val_loss = val_loss_fn(pred, y)

            if verbose:
                print(f'{"-" * 8}Epoch {epoch + 1} ({epoch + 1 + total_epochs} total){"-" * 8}')
                print(f'Val loss = {val_loss:.4f}')
            scheduler.step(val_loss)
            if len(seeds) < args.episodes:
                temp_feature['temp_feature'] = init_seq + temp_features
                temp_feature['val_loss'] = val_loss
                temp_feature['temp_data'] = data_new
                seeds.append(temp_feature)
            else:
                max_loss_index = max(range(len(seeds)), key=lambda i: seeds[i]['val_loss'])
                if val_loss < seeds[max_loss_index]['val_loss']:

                    temp_feature['temp_feature'] = temp_features
                    temp_feature['val_loss'] = val_loss
                    temp_feature['temp_data'] = data_new
                    seeds[max_loss_index] = temp_feature

            if abs(val_loss - scheduler.best) < 1e-6:
                best_Encoder_c = deepcopy(Encoder_c)
                best_Decoder_c = deepcopy(Decoder_c)
                best_predictor = deepcopy(predictor)
                num_bad_epochs = 0
                best_temp = temp
            else:
                num_bad_epochs += 1
            if num_bad_epochs > early_stopping_epochs:
                break
        restore_parameters(Encoder_c, best_Encoder_c)
        restore_parameters(Decoder_c, best_Decoder_c)
        restore_parameters(predictor, best_predictor)
    return Encoder_c, Decoder_c, predictor, seeds,best_temp
def evaluate(test_loader, Encoder_c, Decoder_c, predictor, df_c, df_d,
             device, max_seq_len, pipeline_test, type, vocab_size, init_c_seq, init_d_seq, c_vocab, d_vocab, test_df,
             auroc_metric, acc_metric):
    Encoder_c.eval()
    Decoder_c.eval()
    predictor.eval()
    pred_list = []
    label_list = []
    data_new = test_df.iloc[:, :-1]
    with torch.no_grad():
        for x, y in test_loader:
            # Move to device.
            x = x.to(device)
            y = y.to(device)
            emb_df = Encoder_c(x)
            new_seq = init_c_seq.copy()
            temp_actions = []
            for i in range(max_seq_len):
                new_action_softmax = Decoder_c(new_seq, emb_df, 1)
                dist = Categorical(new_action_softmax)
                action = dist.sample()
                int_action = action.item()
                temp_actions.append(int_action)
                iscombine_type = 'c'
                if type == 'cls' and action == vocab_size - 1:
                    break
                elif c_vocab[action] == 'EOS':
                    if (len(temp_actions) <= 1):
                        temp_actions = []
                        continue
                    feature_new = pipeline_test.generate_newdata(temp_actions, iscombine_type)
                    data_new = data_new.reset_index(drop=True)
                    feature_new = feature_new.reset_index(drop=True)
                    data_new = pd.concat([data_new, feature_new], axis=1)
                    temp_actions = []
                elif c_vocab[action] == 'STOP':
                    break

            pred = predictor(data_new)
            pred_list.append(pred.cpu())
            if type == 'cls':
                y = y.long()
            else:
                y = y.float()
            label_list.append(y.cpu())

    pred = torch.cat(pred_list, 0)
    y = torch.cat(label_list, 0)
    score_auroc = auroc_metric(pred, y)
    score_acc = acc_metric(pred, y)

    return score_auroc, score_acc


def remove_duplication(data):
    _, idx = np.unique(data, axis=1, return_index=True)
    y = data.iloc[:, np.sort(idx)]
    return y
