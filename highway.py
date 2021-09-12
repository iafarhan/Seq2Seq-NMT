#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
class Highway(nn.Module):
    def __init__(self,word_embed_size,dropout_rate=0.1):
        super(Highway,self).__init__()
        self.dropout_rate = dropout_rate
        self.proj = nn.Sequential(
            nn.Linear(in_features=word_embed_size,out_features=word_embed_size),
            nn.ReLU())
        self.gate = nn.Sequential(
            nn.Linear(in_features=word_embed_size,out_features=word_embed_size),
            nn.Sigmoid())
    def forward(self,x):
        x_proj = self.proj(x)
        x_gate = self.gate(x)
        x_highway = torch.mul(x_gate , x_proj) + torch.mul((1. - x_gate), x)
        x_dropout = nn.functional.dropout(x_highway,p=self.dropout_rate)
        return x_dropout

if __name__=='__main__':

    word_embed_size = 20
    batch_size = 10
    x = torch.rand((batch_size,word_embed_size))
    highway = Highway(word_embed_size)
    print(highway(x))