
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
from utils.transformer import PositionalEncoding, LearnablePositionalEncoding, TransformerLayer
from torch.distributions import Normal
import os
EPS = 0.003

class TransformerEncoder(nn.Module):
    def __init__(self, d_in, d_model, d_attention, nhead, dim_feedforward, seq_len=18):
        super(TransformerEncoder, self).__init__()
        self.inp_embedding = nn.Linear(d_in, d_model)
        self.positional_encoding = PositionalEncoding(d_model, seq_len)
        self.transformer_layer = TransformerLayer(d_model, d_attention, nhead, dim_feedforward, dropout=0.1, only_last_state=True)

        

    def forward(self, src):
        x = src
        x = self.inp_embedding(x)
        #x = x * self.embedding_scale
        x = self.pos_embedding(x)
        x = self.encoder(x)  # batch, seq, emb
        x = x[:, -1]
        return x

class Critic(nn.Module):
    def __init__(self, state_dim=24, action_dim=4):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.state_encoder = TransformerEncoder(d_in=self.state_dim,
            d_model=96, d_attention=32, nhead=4, dim_feedforward=192)

        self.fc2 = nn.Linear(96 + self.action_dim, 192)
        self.fc_out = nn.Linear(192, 1, bias=False)
        self.act = nn.Tanh()

        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_nn = os.path.join(dir_current, 'trans_net_critc.pth')
        if os.path.exists(self.path_nn):
            self.load_state_dict(torch.load(self.path_nn, map_location='cpu'))
        else:
            self.__init_parameters()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        """
        returns Value function Q(s,a) obtained from critic network
        :param state: Input state (Torch Variable : [n,state_dim] )
        :param action: Input Action (Torch Variable : [n,action_dim] )
        :return: Value function : Q(S,a) (Torch Variable : [n,1] )
        """
        s = self.state_encoder(state)
        x = torch.cat((s,action),dim=1)
        x = self.act(self.fc2(x))
        x = self.fc_out(x)*10
        return x
    
    def __init_parameters(self):
        nn.init.xavier_uniform_(self.state_encoder.inp_embedding.weight)
        nn.init.zeros_(self.state_encoder.inp_embedding.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.uniform_(self.fc_out.weight, -0.003, +0.003)

    def load_state_dict(self):
        state_dict = torch.load(self.path_nn)
        self.load_state_dict(state_dict, strict=False)


class Actor(nn.Module):
    """
    Actor network for continuous action spaces.
    :param state_dim: Dimension of input state (int)
    :param action_dim: Dimension of output action (int)
    :param stochastic: If True, outputs a distribution over actions
    :return: Actor model
    """
    def __init__(self, state_dim=24, action_dim=4, stochastic=False):
        """
        :param state_dim: Dimension of input state (int)
        :param action_dim: Dimension of output action (int)
        :param action_lim: Used to limit action in [-action_lim,action_lim]
        :return:
        """
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.stochastic = stochastic
        
        self.state_encoder = TransformerEncoder(d_in=self.state_dim,
            d_model=96, d_attention=32, nhead=4, dim_feedforward=192)

        self.fc = nn.Linear(96, action_dim, bias=False)
        #nn.init.zeros_(self.fc.bias)
        if self.stochastic:
            self.log_std = nn.Linear(96, action_dim, bias=False)
            #nn.init.zeros_(self.log_std.bias)  
            
        self.tanh = nn.Tanh()

        dir_current = os.path.dirname(os.path.abspath(__file__))
        self.path_nn = os.path.join(dir_current, 'trans_net_actor.pth')
        if os.path.exists(self.path_nn):
            self.load_state_dict(torch.load(self.path_nn, map_location='cpu'))
        else:
            self.__init_parameters()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, explore=True):
        """
        returns either:
        - deterministic policy function mu(s) as policy action.
        - stochastic action sampled from tanh-gaussian policy, with its entropy value.
        this function returns actions lying in (-1,1) 
        :param state: Input state (Torch Variable : [n,state_dim] )
        :return: Output action (Torch Variable: [n,action_dim] )
        """
        s = self.state_encoder(state)
        if self.stochastic:
            means = self.fc(s)
            log_stds = self.log_std(s)
            log_stds = torch.clamp(log_stds, min=-10.0, max=2.0)
            stds = log_stds.exp()
            dists = Normal(means, stds)
            if explore:
                x = dists.rsample()
            else:
                x = means
            actions = self.tanh(x)
            log_probs = dists.log_prob(x) - torch.log(1-actions.pow(2) + 1e-6)
            entropies = -log_probs.sum(dim=1, keepdim=True)
            return actions, entropies

        else:
            actions = self.tanh(self.fc(s))
            return actions

    def __init_parameters(self):
        nn.init.xavier_uniform_(self.state_encoder.inp_embedding.weight)
        nn.init.zeros_(self.state_encoder.inp_embedding.bias)
        nn.init.uniform_(self.fc.weight, -0.003,+0.003)
        nn.init.uniform_(self.log_std.weight, -0.003,+0.003)

    def load_state_dict(self):
        state_dict = torch.load(self.path_nn)
        self.load_state_dict(state_dict, strict=False)

if __name__ == "__main__":
    # Test the Actor and Critic models
    state_dim = 24
    action_dim = 4

    critic = Critic(state_dim, action_dim)
    print(critic)