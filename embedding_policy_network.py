import math
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from TSFG.trian_utils import restore_parameters
from tqdm import tqdm


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(d_model, hidden_dim)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        return self.dropout(self.dense2(self.relu(self.dense1(X))))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout=0.1):
        super(AddNorm, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, Y):
        return self.ln(X + self.dropout(Y))


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(Attention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.liner_q = nn.Linear(d_model, d_model)
        self.liner_k = nn.Linear(d_model, d_model)
        self.liner_v = nn.Linear(d_model, d_model)
        self.scale_factor = math.sqrt(self.d_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.liner_out = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        b_size = Q.shape[0]
        qs = self.liner_q(Q).view(b_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        ks = self.liner_k(K).view(b_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        vs = self.liner_v(V).view(b_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(qs, ks.transpose(-1, -2)) / self.scale_factor
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        context = torch.matmul(attn, vs)
        context = context.transpose(1, 2).contiguous().view(b_size, -1, self.num_heads * self.d_head)
        return self.liner_out(context)



class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.attention = Attention(d_model, num_heads, dropout)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.ffn = PositionWiseFFN(d_model, hidden_dim, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)

    def forward(self, X):
        Y = self.addnorm1(X, self.attention(X, X, X))
        return self.addnorm2(Y, self.ffn(Y))


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            Encoder(d_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, X):
        X = self.input_projection(X).unsqueeze(dim=0)
        for layer in self.layers:
            X = layer(X)
        return X.squeeze(0)



class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super(Decoder, self).__init__()
        self.attention1 = Attention(d_model, num_heads, dropout)
        self.addnorm1 = AddNorm(d_model, dropout)
        self.attention2 = Attention(d_model, num_heads, dropout)
        self.addnorm2 = AddNorm(d_model, dropout)
        self.ffn = PositionWiseFFN(d_model, hidden_dim, dropout)
        self.addnorm3 = AddNorm(d_model, dropout)

    def forward(self, X, enc_outputs):
        X = X.unsqueeze(dim=0)
        enc_outputs = enc_outputs.unsqueeze(dim=0)
        X2 = self.attention1(X, X, X)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z))


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, hidden_dim, num_layers, device, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Decoder(d_model, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.device = device

    def forward(self, X, enc_outputs, temperature=None):
        X = torch.tensor(X).to(self.device)
        X = self.embedding(X)
        for layer in self.layers:
            X = layer(X, enc_outputs)
        new_state = self.output_projection(X).squeeze(0)
        new_action = torch.mean(new_state, dim=0)
        action_probabilities = new_action / temperature
        return F.softmax(action_probabilities)



class Predictor(nn.Module):
    def __init__(self, d_in, d_out, device):
        super(Predictor, self).__init__()
        hidden = 128
        dropout = 0.3
        self.target_dim = d_in * 10
        self.device = device
        self.network = nn.Sequential(
            nn.Linear(d_in * 10, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_out)
        )

    def forward(self, x):
        if isinstance(x, pd.DataFrame):
            standardizer = StandardScaler()
            x = standardizer.fit_transform(x.values)
            x = torch.from_numpy(x).float().to(self.device)
        if x.shape[1] < self.target_dim:
            padding = torch.zeros((x.shape[0], self.target_dim - x.shape[1]), device=self.device, dtype=x.dtype)
            new_data = torch.cat([x, padding], dim=1)
            x = new_data
        return self.network(x)


class Pretrainer(nn.Module):

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device

    def fit(self,
            train_loader,
            val_loader,
            d_in,
            type,
            patience=2,
            min_lr=1e-6,
            ):

        self.target_dim = d_in * 4
        lr = 1e-2
        val_loss_fn = None
        val_loss_mode = None
        verbose = True
        factor = 0.2
        nepochs = 250
        early_stopping_epochs = 5
        if type == 'cls':
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.MSELoss()
        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = 'min'
        else:
            if val_loss_mode is None:
                raise ValueError('must specify val_loss_mode (min or max) when validation_loss_fn is specified')

        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = 'min'
        else:
            if val_loss_mode is None:
                raise ValueError('must specify val_loss_mode (min or max) when validation_loss_fn is specified')


        model = self.model
        opt = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=val_loss_mode, factor=factor, patience=patience,
            min_lr=min_lr, verbose=verbose)


        best_model = None
        num_bad_epochs = 0
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1

        for epoch in range(nepochs):

            model.train()

            for x, y in train_loader:

                x = x.to(self.device)
                y = y.to(self.device)


                for num_features_to_mask in tqdm(range(1, x.size(1) * 2)):
                    new_tensor = generate_new_features(x)
                    combined_tensor = torch.cat((x, new_tensor), dim=1)

                    padding = torch.zeros((combined_tensor.shape[0], self.target_dim - combined_tensor.shape[1]),
                                          device=self.device, dtype=x.dtype)
                    new_data = torch.cat([combined_tensor, padding], dim=1)
                    pred = model(new_data)
                    if type == 'cls':
                        y = y.long()
                    if type == 'reg':
                        y = y.float()
                    loss = loss_fn(pred, y)
                    loss.backward()
                    opt.step()
                    model.zero_grad()
            model.eval()
            with torch.no_grad():
                # For mean loss.
                pred_list = []
                label_list = []

                for x, y in val_loader:
                    x = x.to(self.device)
                    if type == 'cls':
                        y = y.long()
                    if type == 'reg':
                        y = y.float()

                    # Generate newdata.
                    new_tensor = generate_new_features(x)
                    combined_tensor = torch.cat((x, new_tensor), dim=1)

                    padding = torch.zeros((combined_tensor.shape[0], self.target_dim - combined_tensor.shape[1]),
                                          device=self.device, dtype=x.dtype)
                    new_data = torch.cat([combined_tensor, padding], dim=1)

                    pred = model(new_data)
                    pred_list.append(pred.cpu())
                    label_list.append(y.cpu())

                # Calculate loss.
                y = torch.cat(label_list, 0)
                pred = torch.cat(pred_list, 0)
                val_loss = val_loss_fn(pred, y).item()

            # Print progress.
            if verbose:
                print(f'{"-" * 8}Epoch {epoch + 1}{"-" * 8}')
                print(f'Val loss = {val_loss:.4f}\n')
            scheduler.step(val_loss)

            if val_loss == scheduler.best:
                best_model = deepcopy(model)
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1

            # Early stopping.
            if num_bad_epochs > early_stopping_epochs:
                if verbose:
                    print(f'Stopping early at epoch {epoch + 1}')
                break
        restore_parameters(model, best_model)



def generate_new_features(tensor, num_columns=3, num_operations=2):
    operations = [sqrt, power3, reciprocal, square, rabs, log, add, multiply, subtract, divide]
    selected_columns = random.sample(range(tensor.shape[1]), num_columns)
    new_features = []
    device = tensor.device

    for _ in range(num_operations):
        operation = random.choice(operations)

        if operation in [add, multiply, subtract, divide]:
            col1, col2 = random.sample(selected_columns, 2)
            new_feature = operation(tensor[:, col1], tensor[:, col2])
        else:
            col = random.choice(selected_columns)
            new_feature = operation(tensor[:, col])
        new_features.append(new_feature)
    new_features_tensor = torch.stack(new_features, dim=1).to(device)

    return new_features_tensor


def sqrt(feature):
    sqrt_feature = torch.sqrt(torch.abs(feature))
    return sqrt_feature


def power3(feature):
    new_feature = torch.pow(feature, 3)
    return new_feature


def reciprocal(feature):
    new_feature = torch.where(feature != 0, 1 / feature, feature)
    return new_feature


def square(feature):
    new_feature = torch.square(feature)
    return new_feature


def rabs(feature):
    new_feature = torch.abs(feature)
    return new_feature


def log(feature):
    log_feature = torch.where(torch.abs(feature) > 0, torch.log(torch.abs(feature)),
                              torch.log(torch.tensor(1.0).to(feature.device)))
    return log_feature


def add(feature1, feature2):
    return feature1 + feature2


def multiply(feature1, feature2):
    return feature1 * feature2


def subtract(feature1, feature2):
    return torch.abs(feature1 - feature2)


def divide(feature1, feature2):
    feature_d1 = torch.where(feature2 != 0, feature1 / feature2, torch.tensor(1.0).to(feature1.device))
    return feature_d1


def get_nunique_feature(feature1, feature2):
    feature1 = feature1.unsqueeze(-1)
    feature2 = feature2.unsqueeze(-1)
    feature = torch.cat([feature1, feature2], dim=1)
    new_fe = torch.zeros(feature.size(0), device=feature.device)
    for i in range(feature.size(0)):
        x = feature[i, 0] + feature[i, 1]
        new_fe[i] = x
    return new_fe


def generate_cross_fe(ori_fe1, ori_fe2, feasible_values):
    k = len(ori_fe1)
    new_fe = torch.zeros(k, device=ori_fe1.device)
    for i in range(k):
        cross_feature_value = str(int(ori_fe1[i].item())) + str(int(ori_fe2[i].item()))
        ind = feasible_values[cross_feature_value]
        new_fe[i] = ind
    return new_fe
