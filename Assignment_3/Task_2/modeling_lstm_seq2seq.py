import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Config:
    def __init__(
        self,
        input_size,
        embedding_size,
        hidden_size,
        vocab_size,
        num_layers,
        dropout,
        device,
        max_length,
    ):
        self.input_size = input_size
        self.emb_dim = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.max_length = max_length

class Attention(nn.Module):
    def __init__(
        self,
        config: Config,
        scoring_function: str = "dot",
        alignment: str = "global",
        window_size: int = 10,
    ):
        super(Attention, self).__init__()

        if scoring_function not in ["dot", "general", "concat", "location"]:
            raise ValueError("Invalid scoring function! Must be one of 'dot', 'general', 'concat', 'location'.")

        if alignment not in ["global", "local-m", "local-p"]:
            raise ValueError("Invalid alignment! Must be one of 'global', 'local-m', 'local-p'.")

        self.scoring_function = scoring_function
        self.alignment = alignment
        self.window = window_size

        if scoring_function == "general":
            self.W = nn.Linear(config.hidden_size, config.hidden_size)
        elif scoring_function == "concat":
            self.W = nn.Linear(config.hidden_size * 2, config.hidden_size)
            self.v = nn.Linear(config.hidden_size, 1, bias=False)
        elif scoring_function == "location":
            self.W = nn.Linear(config.hidden_size, config.hidden_size)

        if alignment == "local-m" or alignment == "local-p":
            self.W_p = nn.Linear(config.hidden_size, config.hidden_size)
            self.v_p = nn.Linear(config.hidden_size, 1, bias=False)
            

        if config.device == "cuda":
            self.W = self.cuda()
            if scoring_function == "concat" or scoring_function == "location":
                self.v = self.cuda()
            if alignment == "local-m" or alignment == "local-p":
                self.v = self.cuda()

    def forward(self, hidden, encoder_outputs):
        if self.scoring_function == "dot":
            energy = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)
        elif self.scoring_function == "general":
            energy = torch.bmm(encoder_outputs, self.W(hidden).unsqueeze(2)).squeeze(2)
        elif self.scoring_function == "concat":
            energy = torch.bmm(encoder_outputs, self.v(torch.tanh(self.W(torch.cat((hidden, encoder_outputs), dim=2))))).squeeze(2)
        elif self.scoring_function == "location":
            energy = torch.bmm(encoder_outputs, self.v(torch.tanh(self.W(encoder_outputs)))).squeeze(2)

        if self.alignment == "local-m" or self.alignment == "local-p":
            energy = self._local_alignment(energy)

        attention = F.softmax(energy, dim=1)
        context = torch.bmm(encoder_outputs.transpose(1, 2), attention.unsqueeze(2)).squeeze(2)

        return context, attention
    
    def _local_alignment(self, energy):
        batch_size, seq_length = energy.size()
        energy = energy.unsqueeze(1).repeat(1, seq_length, 1)
        energy_p = self.v_p(torch.tanh(self.W_p(energy)))
        energy_p = energy_p.squeeze(1)
        energy_p = energy_p.view(batch_size, seq_length, seq_length)

        if self.alignment == "local-m":
            energy_p = self._local_masking(energy_p)
        elif self.alignment == "local-p":
            energy_p = self._local_padding(energy_p)

        return energy_p
    
    def _local_masking(self, energy_p):
        batch_size, seq_length, _ = energy_p.size()
        for i in range(batch_size):
            for j in range(seq_length):
                for k in range(max(0, j - self.window), min(seq_length, j + self.window)):
                    energy_p[i, j, k] = -1e10
        return energy_p
    
    def _local_padding(self, energy_p):
        batch_size, seq_length, _ = energy_p.size()
        for i in range(batch_size):
            for j in range(seq_length):
                for k in range(max(0, j - self.window), min(seq_length, j + self.window)):
                    energy_p[i, j, k] = energy_p[i, j, k] - energy_p[i, j, j]
        return energy_p


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        config: Config
    ):
        super(LSTMEncoder, self).__init__()

        self.embedding = nn.Embedding(config.input_size, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_size, config.num_layers, dropout=config.dropout, batch_first=True)

        if config.device == "cuda":
            self.lstm = self.cuda()

    def forward(self, x):
        embedding = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedding)
        return outputs, hidden, cell

class LSTMDecoder(nn.Module):
    def __init__(
        self,
        config: Config,
        attention: bool = False,
        scoring_function: str = "dot",
        alignment: str = "global",
    ):
        super(LSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(config.input_size, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_size, config.num_layers, dropout=config.dropout, batch_first=True)

        if attention:
            self.attention = Attention(config=config, scoring_function=scoring_function, alignment=alignment)

        if config.device == "cuda":
            self.lstm = self.cuda()

    def forward(self, x, hidden, cell, encoder_outputs):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        if self.attention:
            context, att = self.attention(hidden, encoder_outputs)
            outputs += context
        return outputs, hidden, cell

class LSTMSeq2Seq(nn.Module):
    def __init__(
        self,
        config: Config,
        attention: bool = False,
        alignment: str = "global",
        scoring_function: str = "dot",
    ):
        super(LSTMSeq2Seq, self).__init__()

        self.config = config
        self.encoder = LSTMEncoder(config=config)
        self.decoder = LSTMDecoder(config=config, attention=attention, scoring_function=scoring_function, alignment=alignment)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def forward(self, source, target, teacher_forcing_ratio=1.0):
        batch_size = target.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.config.vocab_size
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.config.device)
        hidden, cell = self.encoder(source)
        x = target[:, 0]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t] = output
            best_guess = output.argmax(1)
            x = target[:, t] if random.random() < teacher_forcing_ratio else best_guess
        return outputs
    