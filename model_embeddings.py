#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab,e_char=50,dropout_rate = 0.3):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """

        super(ModelEmbeddings, self).__init__()
        self.word_embed_size = word_embed_size
        self.vocab = vocab
        self.e_char = e_char # char embedding dim
        self.dropout_rate = dropout_rate
        pad_tkn_idx = vocab.char_pad # padding token index 
        num_chars = len(self.vocab.char2id) # No. of characters in the vocabulary

        self.char_embs = nn.Embedding(num_embeddings=num_chars,embedding_dim= self.e_char,padding_idx=pad_tkn_idx)
        self.cnn = CNN(self.e_char,self.word_embed_size)
        self.highway = Highway(self.word_embed_size,dropout_rate=self.dropout_rate)
        
    def forward(self, input_):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        #print("inside model embedding",input_.shape)
        sent_len, batch, max_word_len = input_.shape
        char_embeddings = self.char_embs(input_) # (sent_len, batch, max_word_len, e_char)
        
        # self.cnn expects (batch, e_char , max_word_length) so we reshape our tensor
        char_embeddings = char_embeddings.permute(0,1,3,2).\
                                          reshape(-1,self.e_char,max_word_len) # (batch * sent_len , max_word_len , e_char)
        x_conv_out = self.cnn(char_embeddings) # batch * sent_len , word_embed_size

        x_conv_out = x_conv_out.squeeze(-1)
        x_word_emb = self.highway(x_conv_out)
        x_word_emb = x_word_emb.reshape(sent_len,batch,self.word_embed_size)
        return x_word_emb