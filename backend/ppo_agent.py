# ppo_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import ShogiModel

class PPOAgent:
    def __init__(self, action_size, device):
        self.device = device
        self.model = ShogiModel(action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.eps_clip = 0.2
        self.K_epoch = 3
        self.data = []
    
    def put_data(self, transition):
        self.data.append(transition)
    
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        
        s = torch.tensor(np.array(s_lst), dtype=torch.float).to(self.device)
        a = torch.tensor(a_lst).to(self.device)
        r = torch.tensor(r_lst).to(self.device)
        s_prime = torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(self.device)
        done_mask = torch.tensor(done_lst, dtype=torch.float).to(self.device)
        old_log_prob = torch.tensor(prob_a_lst, dtype=torch.float).to(self.device)
        self.data = []
        return s, a, r, s_prime, done_mask, old_log_prob
    
    def train_net(self):
        s, a, r, s_prime, done_mask, old_log_prob = self.make_batch()
        
        for i in range(self.K_epoch):
            policy, value = self.model(s)
            policy = F.softmax(policy, dim=1)
            dist = torch.distributions.Categorical(policy)
            log_prob = dist.log_prob(a.squeeze())
            entropy = dist.entropy()
            
            policy_prime, value_prime = self.model(s_prime)
            value = value.squeeze()
            value_prime = value_prime.squeeze()
            td_target = r.squeeze() + 0.99 * value_prime * done_mask.squeeze()
            delta = td_target - value
            advantage = delta.detach()
            
            ratio = torch.exp(log_prob - old_log_prob.squeeze())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(value, td_target.detach()) - 0.01 * entropy.mean()
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
