from __future__ import absolute_import

import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.checkpoint


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len=13):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe


class Model_transformer(nn.Module):
    def __init__(self, seq_len=51,
                 signal_len=15,
                 num_layers1=3,
                 num_layers2=2,
                 num_classes=2,
                 dropout_rate=0.1,
                 hidden_signal=128,
                 hidden_size=1024,
                 nhead=8,
                 vocab_size=16,
                 embedding_size=124):
        super(Model_transformer, self).__init__()
        self.sigfea_num = 4
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.hidden_signal = hidden_signal
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.position_encoder_seq = PositionalEncoding(embedding_size + self.sigfea_num, seq_len)
        self.position_encoder_signal = PositionalEncoding(hidden_signal, seq_len)

        self.fc_signal = nn.Linear(signal_len, hidden_signal)
        self.seq_encoder_layer = nn.TransformerEncoderLayer(embedding_size + self.sigfea_num,
                                                            nhead=nhead,
                                                            dim_feedforward=hidden_size,
                                                            dropout=dropout_rate,
                                                            activation='relu',
                                                            batch_first=True,
                                                            norm_first=True)
        self.signal_encoder_layer = nn.TransformerEncoderLayer(hidden_signal,
                                                               nhead=nhead,
                                                               dim_feedforward=hidden_size,
                                                               dropout=dropout_rate,
                                                               activation='relu',
                                                               batch_first=True,
                                                               norm_first=True)
        self.combine_encoder_layer = nn.TransformerEncoderLayer(embedding_size + self.sigfea_num + hidden_signal,
                                                                nhead=nhead,
                                                                dim_feedforward=hidden_size,
                                                                dropout=dropout_rate,
                                                                activation='relu',
                                                                batch_first=True,
                                                                norm_first=True)
        self.seq_encoder_layer_norm = nn.LayerNorm(embedding_size + self.sigfea_num)
        self.signal_encoder_layer_norm = nn.LayerNorm(hidden_signal)
        self.combine_encoder_layer_norm = nn.LayerNorm(embedding_size + self.sigfea_num + hidden_signal)
        self.seq_encoder = nn.TransformerEncoder(self.seq_encoder_layer, num_layers2, self.seq_encoder_layer_norm, enable_nested_tensor=False)
        self.signal_encoder = nn.TransformerEncoder(self.signal_encoder_layer, num_layers2, self.signal_encoder_layer_norm, enable_nested_tensor=False)
        self.combine_encoder = nn.TransformerEncoder(self.combine_encoder_layer, num_layers1, self.combine_encoder_layer_norm, enable_nested_tensor=False)
        # self.center_weight = take_weight(seq_len).to(device)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear((embedding_size + self.sigfea_num + hidden_signal) * 1, num_classes)

    def forward(self, kmer, signals):
        batch_size = signals.shape[0]
        kmer_embed = self.embed(kmer.long())
        signals = signals.reshape(signals.shape[0], signals.shape[2], signals.shape[3])

        out_seq = torch.cat((kmer_embed, signals[:, :, :4]), 2)
        out_seq = self.position_encoder_seq(out_seq)
        out_seq = self.seq_encoder(out_seq)

        out_signal = signals[:, :, 4:]
        out_signal = self.fc_signal(out_signal)
        out_signal = self.position_encoder_signal(out_signal)
        out_signal = self.signal_encoder(out_signal)
        out = torch.cat((out_seq, out_signal), 2)
        out = self.combine_encoder(out)
        # out = (out * self.center_weight).reshape(batch_size, -1)
        # out = out.reshape(batch_size, -1)
        out = out[:, self.seq_len // 2, :]
        out = self.dropout(out)
        out = self.fc(out)
        logits = self.softmax(out)
        return out, logits


class Model_transformer_ed(nn.Module):
    def __init__(self,
                 seq_len=13,
                 signal_len=15,
                 signal_encoder_layer_num=6,
                 seq_decoder_layer_num=6,
                 num_classes=2,
                 dropout_rate=0.1,
                 hidden_size=1024,
                 hidden_signal=128,
                 nhead_seq=8,
                 nhead_signal=8,
                 vocab_size=16,
                 embedding_size=124):
        super(Model_transformer_ed, self).__init__()
        self.sigfea_num = 4
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.fc_signal = nn.Linear(signal_len, hidden_signal)
        self.position_encoder_seq = PositionalEncoding(embedding_size + self.sigfea_num, seq_len)
        self.position_encoder_signal = PositionalEncoding(hidden_signal, seq_len)

        self.seq_decoder_layer = nn.TransformerDecoderLayer(embedding_size + self.sigfea_num,
                                                            nhead=nhead_seq,
                                                            dim_feedforward=hidden_size,
                                                            dropout=dropout_rate,
                                                            activation='gelu',
                                                            batch_first=True,
                                                            norm_first=True)

        self.signal_encoder_layer = nn.TransformerEncoderLayer(hidden_signal,
                                                               nhead=nhead_signal,
                                                               dim_feedforward=hidden_size,
                                                               dropout=dropout_rate,
                                                               activation='gelu',
                                                               batch_first=True,
                                                               norm_first=True)

        self.seq_decoder_layer_norm = nn.LayerNorm(embedding_size + self.sigfea_num)
        self.signal_encoder_layer_norm = nn.LayerNorm(hidden_signal)

        self.signal_encoder = nn.TransformerEncoder(self.signal_encoder_layer, signal_encoder_layer_num, self.signal_encoder_layer_norm, enable_nested_tensor=False)
        self.seq_decoder = nn.TransformerDecoder(self.seq_decoder_layer, seq_decoder_layer_num, self.seq_decoder_layer_norm)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(hidden_signal, num_classes)

    def forward(self, kmer, signals):
        batch_size = signals.shape[0]
        kmer_embed = self.embed(kmer.long())
        signals = signals.reshape(signals.shape[0], signals.shape[2], signals.shape[3])
        out_seq = torch.cat((kmer_embed, signals[:, :, :4]), 2)
        out_seq = self.position_encoder_seq(out_seq)

        out_signal = signals[:, :, 4:]
        out_signal = self.fc_signal(out_signal)
        out_signal = self.position_encoder_signal(out_signal)
        out_signal = self.signal_encoder(out_signal)
        out = self.seq_decoder(out_seq, out_signal)
        out = out[:, self.seq_len // 2, :]
        # out = out.reshape(batch_size, -1)
        out = self.dropout(out)
        out = self.fc(out)
        logits = self.softmax(out)
        return out, logits
