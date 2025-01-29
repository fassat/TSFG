import logging
import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.tree import DecisionTreeClassifier


def log_b(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    log_format = '%(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, force=True)
    file = logging.FileHandler(os.path.join(dir, "log.txt"))
    logging.getLogger().addHandler(file)


Operations = {
    "doubles": ["add", "subtract", "multiply", "divide"],
    "single": ["rabs", 'square', 'reciprocal', 'log', 'sqrt', 'power3'],
    "discrete": ["cross", 'nunique'],
    "specials": ["EOS", "STOP"]
}


def get_init_seq(n_features_c, n_features_d, c_vocab, d_vocab):
    init_c_seq = []
    init_d_seq = []
    if len(c_vocab) > 1:
        c_index_EOS = c_vocab.index('EOS')
        c_index_STOP = c_vocab.index('STOP')
        for i in range(n_features_c):
            init_c_seq.append(i)
            init_c_seq.append(c_index_EOS)
        init_c_seq.append(c_index_STOP)
    if len(d_vocab) > 1:
        d_index_EOS = d_vocab.index('EOS')
        d_index_STOP = d_vocab.index('STOP')
        for i in range(n_features_d):
            init_d_seq.append(i)
            init_d_seq.append(d_index_EOS)
        init_d_seq.append(d_index_STOP)
    return init_c_seq, init_d_seq


def get_action(n_features_c, n_features_d):
    c_generation = []
    d_generation = []
    doubles = Operations["doubles"]
    single_c = Operations["single"]
    discrete = Operations["discrete"]
    specials = Operations['specials']
    if n_features_c == 0:
        c_generation = []
    elif n_features_c == 1:
        c_generation.extend(single_c)
    else:
        for i in range(4):
            op = doubles[i]
            for j in range(n_features_c):
                c_generation.append(op)
        c_generation.extend(single_c)
    if n_features_d != 0:
        for i in range(len(discrete)):
            op = discrete[i]
            for j in range(n_features_d):
                d_generation.append(op)
    else:
        d_generation = []

    max_len = max(len(c_generation), len(d_generation))
    if len(d_generation) < max_len and len(d_generation) != 0:
        d_ori_generation = d_generation
        while len(d_generation) < max_len:
            d_generation.extend(d_ori_generation)
        d_generation = d_generation[:max_len]
    if len(c_generation) < max_len and len(c_generation) != 0:
        c_ori_generation = c_generation
        while len(c_generation) < max_len:
            c_generation.extend(c_ori_generation)
        c_generation = c_generation[:max_len]

    for _ in range(int(len(c_generation) / 1)):
        c_generation.append('EOS')
    c_generation.append("STOP")
    for _ in range(int(len(d_generation) / 1)):
        d_generation.append('EOS')
    d_generation.append("STOP")
    return c_generation, d_generation


def get_binning_df(args, df, v_columns, d_columns, type):
    if df.shape[1] > 1000:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        selector = SelectKBest(score_func=mutual_info_regression, k=100)
        X_new = selector.fit_transform(X, y)
        X_new = pd.DataFrame(X_new)
        new_df = pd.concat([X_new, y], axis=1)
        df.columns = df.columns.astype(str)
        new_df.columns = new_df.columns.astype(str)
        new_v_columns = [str(name) for name in X_new.columns]
        new_d_columns = []
        new_c_df = new_df.loc[:, [new_v_columns, args.target]]
        new_d_df = new_df.loc[:, [new_d_columns, args.target]]
    else:
        new_df = pd.DataFrame()
        new_v_columns = []
        new_d_columns = []
        label = df.loc[:, args.target]
        if type == 'cls':
            for col in v_columns:
                new_df[col] = df[col]
                new_v_columns.append(col)
            for col in v_columns:
                ori_fe = np.array(df[col])
                label = np.array(label)
                new_fe = binning(ori_fe, label)
                new_name = 'bin_' + col
                new_df[new_name] = new_fe
                new_d_columns.append(new_name)
            for col in d_columns:
                new_df[col] = df[col]
                new_d_columns.append(col)
        else:
            for col in v_columns:
                new_df[col] = df[col]
                new_v_columns.append(col)
            for col in d_columns:
                new_df[col] = df[col]
                new_v_columns.append(col)
        new_df[args.target] = label
        new_c_df = new_df.loc[:, new_v_columns]
        new_c_df[args.target] = label
        new_d_df = new_df.loc[:, new_d_columns]
        new_d_df[args.target] = label
    return new_df, new_c_df, new_d_df, new_v_columns, new_d_columns, v_columns, d_columns, df


def get_pos_emb(input_data, con_or_dis):
    position = np.array(con_or_dis).reshape(-1, 1)
    div_term = 10 * np.exp(np.arange(0, 128, 1) * -(np.log(10.0) / 128))
    pos_encoding = np.sin(position * div_term) / 10
    return pos_encoding


def binning(ori_fe, label):
    boundaries = []
    clf = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=6, min_samples_leaf=0.05)
    fe = ori_fe.reshape(-1, 1)
    clf.fit(fe, label.astype("int"))
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold
    for i in range(n_nodes):
        if children_left[i] != children_right[i]:
            boundaries.append(threshold[i])
    boundaries.sort()

    def assign_bin(value, boundaries):
        for i, boundary in enumerate(boundaries):
            if value <= boundary:
                return i
        return len(boundaries)

    if boundaries:
        new_fe = np.array([assign_bin(x, boundaries) for x in ori_fe])
    else:
        new_fe = ori_fe
    return new_fe


datainfos = {
    'wine_red': {"type": "cls",
                 'v_columns': ['fixed acidity', 'volatile acidity', 'citric acid',
                               'residual sugar', 'chlorides', 'free sulfur dioxide',
                               'total sulfur dioxide', 'density', 'pH', 'sulphates',
                               'alcohol'],
                 'd_columns': [],
                 'target': 'quality', }
}
