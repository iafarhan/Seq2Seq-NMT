#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,e_char,word_embed_size,k=5,pad=1):
        """
        @param e_char (int) : char embedding size. used as Cin in convolution.
        @param word_embed_size (int) : size of emb for a word 
        """
        super(CNN,self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=e_char,out_channels=word_embed_size,kernel_size=k,padding=pad),
            nn.ReLU()
            )


    def forward(self,x):
        # convolve over the char embeddings to get word embedding for a word.
        conv_out =  self.conv_block(x)
        pool_out = nn.functional.max_pool1d(conv_out,conv_out.shape[-1])
        return pool_out
# if __name__ == "__main__":
#     e_char = 10
#     m_word = 11 # len of max word in the vocabulary
#     x_reshaped = torch.rand((30,10,11))
#     e_word = 3
#     cnn_block = CNN(e_char,e_word)
#     print(cnn_block(x_reshaped))