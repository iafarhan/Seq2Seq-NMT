#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.target_vocab = target_vocab
        size_vocab = len(self.target_vocab.char2id)
        self.pad_idx = self.target_vocab.char_pad
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size,size_vocab )
        self.decoderCharEmb = nn.Embedding(size_vocab, char_embedding_size,
                                           padding_idx=self.pad_idx)
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.pad_idx,reduction='sum')

    def forward(self, inpt, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        Given a sequence of integers we look up their char embeddings and pass these as input to the LSTM to obtain hidden and cell state

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        
        """
        char_embs = self.decoderCharEmb(inpt)
        # dec_ hidden (len,batch,hidden) , hn (1, batch, hidden)
        dec_hidden, (hn,cn) = self.charDecoder(char_embs,dec_hidden)

        dec_hidden_reshaped = dec_hidden.reshape(dec_hidden.size(0)*dec_hidden.size(1),dec_hidden.size(2)) # len * batch , hidden
        scores = self.char_output_projection(dec_hidden_reshaped)
        scores = scores.reshape(dec_hidden.size(0),dec_hidden.size(1),-1) # len, batch , vocab 
        return scores,(hn,cn)
    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss
        input_sequence = char_sequence[:-1,:]
        output_sequence = char_sequence[1:,:]

        scores,(hn,cn) = self(input_sequence,dec_hidden)
        
        scores = scores.permute(1,2,0)
        output_sequence = output_sequence.permute(1,0)
        loss = self.cross_entropy(scores,output_sequence)
        return loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=5):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """
        batch_size = initialStates[0].size(1)
        curr_batch = torch.tensor([self.target_vocab.start_of_word] * batch_size,device=device)
        decoded_words = [[] for _ in range(batch_size)]
        dec_states = initialStates
        end_idx = self.target_vocab.end_of_word
        start_idx = self.target_vocab.start_of_word
        decoded = []
    
        for i in range(max_length):
            scores, dec_states = self(curr_batch.view(1,-1),dec_states)
            scores = scores.squeeze(0)
            char_idxs = scores.argmax(dim=1).flatten()
            for i,idx in enumerate(char_idxs):
                idx = idx.item()
                if idx == start_idx:
                    continue
                if idx != end_idx and i not in decoded:
                    char = self.target_vocab.id2char[idx]
                    decoded_words[i] += char
                    #print(decoded_words[i])
                else:
                    decoded.append(i)
            curr_batch = char_idxs
        decoded_words = ["".join(x) for x in decoded_words]
        #print(decoded_words)
        return decoded_words


