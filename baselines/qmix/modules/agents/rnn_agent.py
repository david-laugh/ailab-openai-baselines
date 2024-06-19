import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim, n_actions):
        super(RNNAgent, self).__init__()
        self.input_shape = input_shape
        self.rnn_hidden_dim = rnn_hidden_dim
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_shape, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def initial_state(self):
        # sample
        return torch.zeros(1, self.input_shape)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
