import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from functions import *
from utils import *

#configuring variational droput
class VariationalDropout(nn.Module):
    def __init__(self, log_alpha=-3.):
        super(VariationalDropout, self).__init__()
        self.max_log_alpha = 0.0
        self.log_alpha = nn.Parameter(torch.Tensor([log_alpha]))

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    def forward(self, x):
        if self.train():
            normal_noise = torch.randn_like(x)
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_log_alpha)
            random_tensor = 1. + normal_noise * torch.sqrt(self.alpha)
            x *= random_tensor
        return x
    
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1,tie_weights=False):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        self.v_dropout = VariationalDropout() #applying the configured variational dropout
        #tying weights if hidden size and embedded size are not equal
        if tie_weights:
            if hidden_size != emb_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.embedding.weight = self.output.weight
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        lstm_out, _  = self.lstm(emb)
        output = self.output(lstm_out).permute(0,2,1)
        final_output = self.v_dropout(output)
        return final_output
    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()
    
    def get_most_similar(self, vector, top_k=10):
        embs = self.embedding.weight.detach().cpu().numpy()
        #Our function that we used before
        scores = []
        for i, x in enumerate(embs):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))
        # Take ids of the most similar tokens 
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]  
        top_scores = scores[indexes]
        return (indexes, top_scores)    
    