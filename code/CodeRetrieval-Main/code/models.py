from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F


class BOWEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, config):
        super(BOWEncoder, self).__init__()
        self.emb_size = emb_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.config = config

    def forward(self, input):
        batch_size, seq_len = input.size()
        embedded = self.embedding(input)  # input: [batch_sz x seq_len x 1]  embedded: [batch_sz x seq_len x emb_sz]
        embedded = F.dropout(embedded, self.config['bow_dropout'], self.training)  # [batch_size x seq_len x emb_size]
        output_pool = F.max_pool1d(embedded.transpose(1, 2), seq_len).squeeze(2)  # [batch_size x emb_size]
        encoding = torch.tanh(output_pool)
        return encoding


class SeqEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, config):
        super(SeqEncoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.config = config

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        for w in self.lstm.parameters():  # initialize the gate weights with orthogonal
            if w.dim() > 1:
                weight_init.xavier_normal_(w)

    def forward(self, input):
        batch_size, seq_len = input.size()
        embedded = self.embedding(input)  # input: [batch_sz x seq_len]  embedded: [batch_sz x seq_len x emb_sz]
        embedded = F.dropout(embedded, self.config['seqenc_dropout'], self.training)
        rnn_output, hidden = self.lstm(embedded)  # out:[b x seq x hid_sz*2](biRNN)
        rnn_output = F.dropout(rnn_output, self.config['seqenc_dropout'], self.training)
        output_pool = F.max_pool1d(rnn_output.transpose(1, 2), seq_len).squeeze(2)  # [batch_size x hid_size*2]
        encoding = torch.tanh(output_pool)

        return encoding


class JointEmbeder(nn.Module):
    def __init__(self, config):
        super(JointEmbeder, self).__init__()
        self.name = "default"

        self.conf = config
        self.margin = config['margin']
        self.qt_encoder = SeqEncoder(config['qt_n_words'], config['emb_size'], config['lstm_dims'], self.conf)

        if self.conf['use_anno']:
            self.cand_encoder = SeqEncoder(config['qt_n_words'], config['emb_size'], config['lstm_dims'], self.conf)

        else:
            if self.conf['code_encoder'] == "bilstm":
                self.cand_encoder = SeqEncoder(config['code_n_words'], config['emb_size'], config['lstm_dims'],
                                               self.conf)  # Bi-LSTM
            else:
                self.cand_encoder = BOWEncoder(config['code_n_words'], 2 * config['lstm_dims'], self.conf)  # MLP

    def cand_encoding(self, cand):
        code_repr = self.cand_encoder(cand)
        return code_repr

    def qt_encoding(self, qt):
        qt_repr = self.qt_encoder(qt)
        return qt_repr

    def scoring(self, qt_repr, cand_repr):
        sim = F.cosine_similarity(qt_repr, cand_repr)
        return sim

    def forward(self, qt, good_cand, bad_cand):
        good_cand_repr = self.cand_encoding(good_cand)
        bad_cand_repr = self.cand_encoding(bad_cand)

        qt_repr = self.qt_encoding(qt)

        good_sim = self.scoring(qt_repr, good_cand_repr)
        bad_sim = self.scoring(qt_repr, bad_cand_repr)

        loss = (self.margin - good_sim + bad_sim).clamp(min=1e-6).mean()
        return loss, good_sim, bad_sim