import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=True):
        super(Encoder, self).__init__()

        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, inputs):
        # Forward pass through RNN
        outputs, hidden = self.rnn(inputs)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size*2, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)

        # (batch_size, 1 (unsqueezed), hidden_size)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)

        # 1st line of Eq.(3) in the paper
        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)

        return F.softmax(u_i, dim=-1)


class Placer(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size, bidirectional=True, batch_first=True, C=5, T=10, lr=0.1):
        super(Placer, self).__init__()

        self.input_dim = input_dim
        # (Decoder) hidden size
        self.hidden_size = hidden_size
        # Bidirectional Encoder
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = 1
        self.batch_first = batch_first

        # We use an embedding layer for more complicate application usages later, e.g., word sequences.
        self.encoder = Encoder(embedding_dim=input_dim, hidden_size=hidden_size, num_layers=self.num_layers,
                               bidirectional=bidirectional, batch_first=batch_first)
        self.decoding_rnn = nn.LSTMCell(input_size=output_size, hidden_size=hidden_size)
        self.proj_fn = nn.Linear(hidden_size * 3, output_size)

        self.attn = Attention(hidden_size=hidden_size)

        self.C = C
        self.T = T

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.optim = torch.optim.Adam(list(self.encoder.parameters())+
                                      list(self.decoding_rnn.parameters())+
                                      list(self.proj_fn.parameters()),
                                      lr=lr)

    def forward(self, input_seq, mask=None):

        if self.batch_first:
            batch_size = input_seq.size(0)
            max_seq_len = input_seq.size(1)
        else:
            batch_size = input_seq.size(1)
            max_seq_len = input_seq.size(0)


        # (batch_size, max_seq_len, embedding_dim)

        # encoder_output => (batch_size, max_seq_len, hidden_size) if batch_first else (max_seq_len, batch_size, hidden_size)
        # hidden_size is usually set same as embedding size
        # encoder_hidden => (num_layers * num_directions, batch_size, hidden_size) for each of h_n and c_n
        encoder_outputs, encoder_hidden = self.encoder(input_seq)

        # if self.bidirectional:
            # Optionally, Sum bidirectional RNN outputs
            # encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size:]
        input_lengths = torch.tensor([max_seq_len])
        encoder_h_n, encoder_c_n = encoder_hidden
        encoder_h_n = encoder_h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        encoder_c_n = encoder_c_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)

        # Lets use zeros as an intial input for sorting example
        decoder_input = encoder_outputs.new_zeros(torch.Size((batch_size, self.output_size)))
        decoder_hidden = (encoder_h_n[-1, 0, :, :],  encoder_c_n[-1, 0, :, :])

        log_prob = torch.zeros(batch_size)
        ret = []

        for i in range(max_seq_len):

            # h, c: (batch_size, hidden_size)
            h_i, c_i = self.decoding_rnn(decoder_input, decoder_hidden)

            # next hidden
            decoder_hidden = (h_i, c_i)

            # Get a pointer distribution over the encoder outputs using attention
            # (batch_size, max_seq_len)
            a = self.attn(h_i, encoder_outputs).unsqueeze(1)

            emb = torch.bmm(a, encoder_outputs).squeeze(1)
            emb = F.tanh(self.proj_fn(torch.cat((emb, h_i), dim=-1)) / self.T) * self.C
            if mask is None:
                m = [1 for _ in range(self.output_size)]
            else:
                m = mask[i]
            emb_masked = emb.clone()
            emb_masked[:,m==0] = -float('inf')
            probs = F.softmax( emb_masked, dim=-1)
            m = torch.distributions.Categorical(probs=probs)
            action = m.sample()
            log_prob += m.log_prob(action)
            ret.append(action)

            decoder_input = F.one_hot(action, num_classes=self.output_size).float()

        return ret, log_prob

    def update(self, log_prob, R):
        loss =  torch.mean(- log_prob * R)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()



input_size = 5
output_size = 20
seq_lenth = 10
hidden_dim = 36
s = torch.rand(2, seq_lenth, input_size)

agent = Placer(input_size, hidden_dim, output_size)
ret, log_prob = agent(s)
agent.update(log_prob, torch.zeros(2))