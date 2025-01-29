import torch
import torch.optim as optim
import os

from torch.distributions import RelaxedOneHotCategorical
from torch.distributions.categorical import Categorical
import logging
from TSFG.embedding_policy_network import TransformerEncoder, TransformerDecoder,Predictor
import numpy as np
import torch.nn.functional as F


class PPO(object):
    def __init__(self, args, input_dim,len_vocab, d_model, data_nums, device):
        self.args = args
        self.entropy_weight = args.entropy_weight
        self.ppo_epochs = args.ppo_epochs
        self.num_layers = args.nums_layers
        self.outputs_num = len_vocab
        self.device = device
        self.vocab_size = len_vocab
        self.num_hiddens = args.hidden_size
        self.data_nums = data_nums
        self.num_heads = 8
        self.Encoder_c = TransformerEncoder(input_dim,d_model, self.num_heads, self.num_hiddens, self.num_layers).to(self.device)
        self.Decoder_c = TransformerDecoder(self.vocab_size, d_model,self.num_heads,self.num_hiddens, self.num_layers, device).to(
            self.device)
        self.policy_opt_c = optim.Adam(list(self.Encoder_c.parameters()) + list(self.Decoder_c.parameters()),
                                       lr=args.lr)

        self.Encoder_d = TransformerEncoder(input_dim,d_model, self.num_heads, self.num_hiddens, self.num_layers).to(self.device)
        self.Decoder_d = TransformerDecoder(self.vocab_size, d_model,self.num_heads,self.num_hiddens, self.num_layers, device).to(
            self.device)
        self.policy_opt_d = optim.Adam(list(self.Encoder_d.parameters()) + list(self.Decoder_d.parameters()),
                                       lr=args.lr)

        self.baseline = None
        self.baseline_weight = args.baseline_weight
        self.clip_epsion = 0.2
        self.max_seq_len = 100

    def choose_action_c(self, init_df, init_seq, best_temp):
        actions = []
        log_probs = []
        action_softmaxs = []
        new_seq = init_seq.copy()
        emb_df = self.Encoder_c(init_df.to(self.device))
        feature_nums = 0
        for i in range(self.max_seq_len):
            # while 1:
            new_action_softmax = self.Decoder_c(new_seq, emb_df, best_temp)
            dist = Categorical(new_action_softmax)
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(dim=0).float()
            if action == self.vocab_size - 1:
                action = action.int().item()
                actions.append(action)
                log_probs.append(log_prob)
                action_softmaxs.append(new_action_softmax)
                new_seq.append(action)
                break
            if feature_nums >= self.data_nums * 2:
                break
            action = action.int().item()
            actions.append(action)
            log_probs.append(log_prob)
            action_softmaxs.append(new_action_softmax)
            new_seq.append(action)
        return actions, log_probs, action_softmaxs, new_seq

    def choose_action_d(self, init_df, init_seq, best_temp):
        actions = []
        log_probs = []
        action_softmaxs = []
        new_seq = init_seq.copy()
        feature_nums = 0
        emb_df = self.Encoder_d(init_df.to(self.device))
        for i in range(self.max_seq_len):
            new_action_softmax = self.Decoder_d(new_seq, emb_df, best_temp)
            dist = Categorical(new_action_softmax)
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(dim=0).float()
            if action == self.vocab_size - 1:
                action = action.int().item()
                actions.append(action)
                log_probs.append(log_prob)
                action_softmaxs.append(new_action_softmax)
                new_seq.append(action)
                break

            if feature_nums >= self.data_nums * 2:
                break
            action = action.int().item()
            actions.append(action)
            log_probs.append(log_prob)
            action_softmaxs.append(new_action_softmax)
            new_seq.append(action)
        return actions, log_probs, action_softmaxs, new_seq

    def update_c(self, workers, df, init_seq,best_temp):

        sample_score = [worker.new_score for worker in workers]
        if self.baseline is None:
            self.baseline = np.mean(sample_score)
        else:
            for worker in workers:
                self.baseline = self.baseline * self.baseline_weight + \
                                worker.new_score * (1 - self.baseline_weight)

        score_lists = [worker.new_score for worker in workers]
        score_reward = [score_list - self.baseline for score_list in score_lists]

        score_reward_abs = [abs(reward) for reward in score_reward]
        min_reward = min(score_reward_abs)
        max_reward = max(score_reward_abs)
        score_reward_minmax = []
        for reward in score_reward:
            if reward < 0:
                min_max_reward = -(abs(reward) - min_reward) / (max_reward - min_reward + 1e-6)
            elif reward >= 0:
                min_max_reward = (abs(reward) - min_reward) / (max_reward - min_reward + 1e-6)
            else:
                min_max_reward = None
                print('error')
            score_reward_minmax.append(min_max_reward)
        score_reward_minmax = [i / 2 if i is not None else 0 for i in score_reward_minmax]
        normal_reward = score_reward_minmax

        init_df = torch.from_numpy(df.values).float().transpose(0, 1).to(self.device)
        for epoch in range(self.ppo_epochs):
            total_loss = 0
            total_loss_actor = 0
            total_loss_entorpy = 0
            emb_df = self.Encoder_c(init_df.to(self.device))
            for num, worker in enumerate(workers):
                entropys = []
                new_log_probs = []
                new_seq = init_seq.copy()
                actions = worker.actions_c
                old_log_probs = worker.log_c_prob
                for i in range(len(actions)):
                    action_softmax = self.Decoder_c(new_seq, emb_df,best_temp)
                    dist = Categorical(action_softmax)
                    new_action = dist.sample()
                    new_seq.append(new_action.int().item())
                    entropy = dist.entropy()
                    entropys.append(entropy.unsqueeze(dim=0))
                    action = actions[i]
                    action = torch.tensor(action).to(self.device)
                    new_log_prob = dist.log_prob(action).unsqueeze(dim=0).float()
                    new_log_probs.append(new_log_prob)
                if len(new_log_probs) == 0:
                    continue
                new_log_probs = torch.cat(new_log_probs)
                old_log_probs = torch.cat(old_log_probs).detach()
                entropys = torch.cat(entropys)
                # ppo
                prob_ratio = new_log_probs.exp() / old_log_probs.exp()
                weighted_probs = normal_reward[num] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.clip_epsion, 1 + self.clip_epsion) * \
                                         normal_reward[num]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs)
                actor_loss = actor_loss.sum()

                entropy_loss = entropys.sum()
                total_loss_actor += actor_loss
                total_loss_entorpy += (- self.args.entropy_weight * entropy_loss)
                total_loss += (actor_loss - self.args.entropy_weight * entropy_loss)

            total_loss /= len(workers)
            if isinstance(total_loss, int):
                continue
            self.policy_opt_c.zero_grad()
            total_loss.backward()
            self.policy_opt_c.step()

    def update_d(self, workers, df, init_seq,best_temp):

        sample_score = [worker.new_score for worker in workers]
        if self.baseline is None:
            self.baseline = np.mean(sample_score)
        else:
            for worker in workers:
                self.baseline = self.baseline * self.baseline_weight + \
                                worker.new_score * (1 - self.baseline_weight)

        score_lists = [worker.new_score for worker in workers]
        score_reward = [score_list - self.baseline for score_list in score_lists]

        score_reward_abs = [abs(reward) for reward in score_reward]
        min_reward = min(score_reward_abs)
        max_reward = max(score_reward_abs)
        score_reward_minmax = []
        for reward in score_reward:
            if reward < 0:
                min_max_reward = -(abs(reward) - min_reward) / (max_reward - min_reward + 1e-6)
            elif reward >= 0:
                min_max_reward = (abs(reward) - min_reward) / (max_reward - min_reward + 1e-6)
            else:
                min_max_reward = None
            score_reward_minmax.append(min_max_reward)
        score_reward_minmax = [i / 2 for i in score_reward_minmax]
        normal_reward = score_reward_minmax
        init_df = torch.from_numpy(df.values).float().transpose(0, 1).to(self.device)
        for epoch in range(self.ppo_epochs):
            total_loss = 0
            total_loss_actor = 0
            total_loss_entorpy = 0
            emb_df = self.Encoder_d(init_df.to(self.device))
            for num, worker in enumerate(workers):
                entropys = []
                new_log_probs = []
                new_seq = init_seq.copy()
                actions = worker.actions_d
                old_log_probs = worker.log_d_prob
                for i in range(len(actions)):
                    action_softmax = self.Decoder_d(new_seq, emb_df,best_temp)
                    dist = Categorical(action_softmax)
                    new_action = dist.sample()
                    new_seq.append(new_action.int().item())
                    entropy = dist.entropy()
                    entropys.append(entropy.unsqueeze(dim=0))
                    action = actions[i]
                    action = torch.tensor(action).to(self.device)
                    new_log_prob = dist.log_prob(action).unsqueeze(dim=0).float()
                    new_log_probs.append(new_log_prob)
                if len(new_log_probs) == 0:
                    continue
                new_log_probs = torch.cat(new_log_probs)
                old_log_probs = torch.cat(old_log_probs).detach()
                entropys = torch.cat(entropys)
                # ppo
                prob_ratio = new_log_probs.exp() / old_log_probs.exp()
                weighted_probs = normal_reward[num] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.clip_epsion, 1 + self.clip_epsion) * \
                                         normal_reward[num]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs)
                actor_loss = actor_loss.sum()
                entropy_loss = entropys.sum()
                total_loss_actor += actor_loss
                total_loss_entorpy += (- self.args.entropy_weight * entropy_loss)
                total_loss += (actor_loss - self.args.entropy_weight * entropy_loss)

            total_loss /= len(workers)
            if isinstance(total_loss, int):
                continue
            self.policy_opt_d.zero_grad()
            total_loss.backward()
            self.policy_opt_d.step()
