import torch
import torch.nn as nn
from torch.autograd import Variable


class SoccerRNN(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, output_size, n_layers, dropout):
        super(SoccerRNN, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        output, hidden = self.rnn(input.view(self.seq_len, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))


class SoccerFCN(nn.Module):
    def __init__(self, topology):
        super(SoccerFCN, self).__init__()
        self.topology = topology
        m = len(topology)
        self.layers = torch.nn.ModuleList()
        for i in range(0,m-1):
            self.layers.append(torch.nn.Linear(topology[i], topology[i+1]))

    def forward(self, X):
        i = 0
        while i < len(self.layers)-1:
            X = nn.functional.relu(self.layers[i](X))
            i += 1
        out = self.layers[-1](X)
        return out
