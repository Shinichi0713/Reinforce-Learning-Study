import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn import LSTMCell

class Encoder(nn.Module):
    def __init__(self, n_neurons=128, batch_size=4, seq_length=10):
        super().__init__()
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.enc_rec_cell = nn.LSTMCell(n_neurons, n_neurons)
        self.bilstm = nn.LSTM(n_neurons, n_neurons, batch_first=True, bidirectional=True)

    def forward(self, inputs):
        # inputs: [batch, seq_len, n_neurons]
        outputs, (h_n, c_n) = self.bilstm(inputs)
        # outputs: [batch, seq_len, 2*n_neurons]
        input_list = outputs.transpose(0, 1)  # [seq_len, batch, 2*n_neurons]
        enc_outputs = []
        enc_states = []
        state = self._get_initial_state(inputs.device, inputs.size(0))
        for input in input_list:
            output, state = self.enc_rec_cell(input, state)
            enc_outputs.append(output)
            enc_states.append(state)
        enc_outputs = torch.stack(enc_outputs, dim=0)  # [seq_len, batch, n_neurons]
        enc_outputs = enc_outputs.transpose(0, 1)      # [batch, seq_len, n_neurons]
        enc_state = enc_states[-1]
        return enc_outputs, enc_state

    def _get_initial_state(self, device, batch_size):
        h = torch.zeros(batch_size, self.n_neurons, device=device)
        c = torch.zeros(batch_size, self.n_neurons, device=device)
        return (h, c)




class ActorDecoder(nn.Module):
    def __init__(self, n_neurons=128, batch_size=4, seq_length=10):
        super().__init__()
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.infty = 1e8

        self.dec_first_input = nn.Parameter(torch.zeros(1, n_neurons))
        self.W_ref = nn.Parameter(torch.randn(1, n_neurons, n_neurons))
        self.W_out = nn.Parameter(torch.randn(n_neurons, n_neurons))
        self.v = nn.Parameter(torch.randn(n_neurons))
        self.dec_rec_cell = nn.LSTMCell(n_neurons, n_neurons)

    def forward(self, enc_outputs, enc_state, seed=None):
        output_list = enc_outputs.transpose(0, 1)  # [seq_len, batch, n_neurons]
        batch_size = enc_outputs.size(0)
        mask = torch.zeros(batch_size, self.seq_length, device=enc_outputs.device)
        locations, log_probs = [], []
        input = self.dec_first_input.expand(batch_size, -1)
        state = enc_state
        for step in range(self.seq_length):
            output, state = self.dec_rec_cell(input, state)
            masked_scores = self._pointing(enc_outputs, output, mask)
            probs = F.softmax(masked_scores, dim=1)
            dist = torch.distributions.Categorical(probs)
            location = dist.sample()
            locations.append(location)
            logp = dist.log_prob(location)
            log_probs.append(logp)
            mask[torch.arange(batch_size), location] = 1
            input = output_list[location, torch.arange(batch_size), :]
        first_location = locations[0]
        locations.append(first_location)
        tour = torch.stack(locations, dim=1)
        log_prob = torch.stack(log_probs, dim=1).sum(1)
        return log_prob, tour

    def _pointing(self, enc_outputs, dec_output, mask):
        # enc_outputs: [batch, seq_length, n_neurons]
        enc_term = F.conv1d(enc_outputs.transpose(1,2), self.W_ref, padding=0).transpose(1,2)
        dec_term = torch.matmul(dec_output, self.W_out).unsqueeze(1)
        scores = torch.sum(self.v * torch.tanh(enc_term + dec_term), dim=-1)
        masked_scores = scores - self.infty * mask
        return masked_scores


class CriticDecoder(nn.Module):
    def __init__(self, n_neurons=128, batch_size=4, seq_length=10):
        super().__init__()
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.W_ref_g = nn.Parameter(torch.randn(1, n_neurons, n_neurons))
        self.W_q_g = nn.Parameter(torch.randn(n_neurons, n_neurons))
        self.v_g = nn.Parameter(torch.randn(n_neurons))
        self.fc1 = nn.Linear(n_neurons, n_neurons)
        self.fc2 = nn.Linear(n_neurons, 1)

    def forward(self, enc_outputs, enc_state):
        # enc_outputs: [batch, seq_length, n_neurons]
        # enc_state: (h, c)
        frame = enc_state[0]  # h, [batch, n_neurons]
        enc_ref_g = F.conv1d(enc_outputs.transpose(1,2), self.W_ref_g, padding=0).transpose(1,2)
        enc_q_g = torch.matmul(frame, self.W_q_g).unsqueeze(1)
        scores_g = torch.sum(self.v_g * torch.tanh(enc_ref_g + enc_q_g), dim=-1)
        attention_g = F.softmax(scores_g, dim=1)
        glimpse = (enc_outputs * attention_g.unsqueeze(2)).sum(dim=1)
        hidden = F.relu(self.fc1(glimpse))
        baseline = self.fc2(hidden)
        return baseline

