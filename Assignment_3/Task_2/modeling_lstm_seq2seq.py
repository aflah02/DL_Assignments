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
        scoring_function: str = "location",
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
            self.W = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
        elif scoring_function == "concat":
            self.W = nn.Linear(2 * config.hidden_size, config.hidden_size, bias = False)
            self.v = nn.Linear(config.hidden_size, 1, bias=False)
        elif scoring_function == "location":
            self.W = nn.Linear(config.hidden_size, config.hidden_size)

        if alignment == "local-m" or alignment == "local-p":
            self.W_p = nn.Linear(config.hidden_size, config.hidden_size, bias = False)
            self.v_p = nn.Linear(config.hidden_size, 1, bias=False)

        if config.device == "cuda":
            if scoring_function != "dot":
                self.W = self.W.cuda()
                if scoring_function == "concat":
                    self.v = self.v.cuda()
                if alignment == "local-m" or alignment == "local-p":
                    self.W_p = self.W_p.cuda()
                    self.v_p = self.v_p.cuda()
    
    # bmm: B,N,M x B,M,P = B,N,P
    def forward(self, decoder_outputs, encoder_outputs, pos = None):
        seq_length = encoder_outputs.shape[1]
        if self.alignment == "local-m" or self.alignment == "local-p":
            pos = self._local_alignment(decoder_outputs, pos)
        
        if self.scoring_function == "dot":
            # print(encoder_outputs.shape, decoder_outputs.shape)
            energy = torch.bmm(encoder_outputs, decoder_outputs.unsqueeze(1).transpose(1, 2))
            # print(energy.shape)
        elif self.scoring_function == "general":
            projected_decoder_outputs = self.W(decoder_outputs)
            # print(decoder_outputs.shape, encoder_outputs.shape, projected_decoder_outputs.shape)
            energy = torch.bmm(encoder_outputs, projected_decoder_outputs.unsqueeze(2))
            # print(energy.shape)
        elif self.scoring_function == "concat":
            # projected_decoder_outputs = self.W(decoder_outputs).unsqueeze(1)
            energy = torch.cat([encoder_outputs, decoder_outputs.unsqueeze(1).expand(encoder_outputs.shape)], dim=2)
            energy = self.W(energy)
            energy = nn.Tanh()(energy)
            energy = self.v(energy)
        elif self.scoring_function == "location":
            energy = self.W(decoder_outputs)

        if self.alignment == "local-m":
            align_wts = torch.ones(energy.shape, device = energy.device)
            for i in range(energy.shape[0]):
                for j in range(energy.shape[1]):
                    if j<max(0, pos[i] - self.window) or j>min(seq_length-1, pos[i]+self.window-1):
                        # energy[i, j, :] *= -1e10
                        align_wts[i, j] *= -1e10
            energy = torch.mul(align_wts, energy)
        elif self.alignment == "local-p":
            align_wts = torch.ones(energy.shape, device = energy.device)
            for i in range(energy.shape[0]):
                x = torch.Tensor(range(1, energy.shape[1]+1), device = energy.device)
                # for j in range(energy.shape[1]):
                gaussian_factor = torch.exp(-2*((j+1 - pos[i])**2)/self.window**2)
                align_wts[i, j] *=  gaussian_factor
            energy = torch.mul(align_wts, energy)
            
        # print(energy.shape)

        attention = F.softmax(energy, dim=1)
        context = torch.bmm(attention.transpose(1, 2), encoder_outputs)

        # print(context.shape)

        return context, attention
    
    def _local_alignment(self, decoder_outputs, pos):
        seq_length = decoder_outputs.shape[1]
        if self.alignment == "local-p":
            partial_aligned_decoder_outputs = self.W_p(decoder_outputs)
            partial_aligned_decoder_outputs = nn.Tanh()(self.v_p(partial_aligned_decoder_outputs))
            posi = seq_length * nn.Sigmoid()(partial_aligned_decoder_outputs.squeeze(1))
        elif self.alignment == "local-m":
            posi = [pos for i in range(decoder_outputs.shape[0])]
        return posi


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        config: Config
    ):
        super(LSTMEncoder, self).__init__()

        self.embedding = nn.Embedding(config.input_size, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_size, config.num_layers, dropout=config.dropout, batch_first=True)

        if config.device == "cuda":
            self.embedding = self.embedding.cuda()
            self.lstm = self.lstm.cuda()

    def forward(self, x):
        embedding = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedding)
        return outputs, hidden, cell

class LSTMDecoder(nn.Module):
    def __init__(
        self,
        config: Config,
        attention: bool = False,
        scoring_function: str = "location",
        alignment: str = "global",
    ):
        super(LSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(config.input_size, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim, config.hidden_size, config.num_layers, dropout=config.dropout, batch_first=True)
        self.attention = attention

        if attention:
            self.attention = Attention(config=config, scoring_function=scoring_function, alignment=alignment)
            if config.device == "cuda":
                self.attention = self.attention.cuda()

        if config.device == "cuda":
            self.embedding = self.embedding.cuda()
            self.lstm = self.lstm.cuda()

    def forward(self, x, hidden, cell, encoder_outputs, t):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        if self.attention:
            context, att = self.attention(hidden[-1], encoder_outputs, t)
            outputs = torch.cat([context, hidden[-1].unsqueeze(1)], dim=2)
        return outputs, hidden, cell

class LSTMSeq2Seq(nn.Module):
    def __init__(
        self,
        config: Config,
        attention: bool = False,
        alignment: str = "global",
        scoring_function: str = "location",
    ):
        super(LSTMSeq2Seq, self).__init__()

        self.config = config
        self.encoder = LSTMEncoder(config=config)
        self.decoder = LSTMDecoder(config=config, attention=attention, scoring_function=scoring_function, alignment=alignment)

        if attention:
            self.lm_head = nn.Linear(2 * config.hidden_size, config.vocab_size)
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # if config.device == "cuda":
        #     self.encoder = self.encoder.cuda()
        #     self.decoder = self.decoder.cuda()
        #     self.lm_head = self.lm_head.cuda()

    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def forward(self, source, target, teacher_forcing_ratio=1.0):
        batch_size = target.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.config.vocab_size
        outputs = torch.zeros(batch_size, target_len, target_vocab_size, device = torch.device(self.config.device))
        encoder_outputs, hidden, cell = self.encoder(source)
        # print(encoder_outputs.shape, hidden.shape, cell.shape)
        x = target[:, 0]
        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell, encoder_outputs, t)
            logits = self.lm_head(output)
            softmax_logits = F.softmax(logits, dim=2)
            softmax_logits = softmax_logits.squeeze(dim = 1)
            outputs[:, t] = softmax_logits
            # best_guess = output.argmax(1)
            x = target[:, t]
        return outputs
    