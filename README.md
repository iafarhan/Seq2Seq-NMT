# Seq2Seq-Neural-machine-translation
Designed a sequence transduction model based on Recurrent, Convolutional Neural Networks and Attention
mechanism. Seq2Seq Model used a Bidirectional LSTM Encoder and a Unidirectional LSTM Decoder.
Attention is used at each decoder step to jointly provide alignment and translation.
Used a CNN and Highway Network to deal with Out-of-vocabulary (OOV) words. It was Inspired by work in
Character-Aware Neural Language Models. This is used at Embedding lookup stage.
A Character-based decoder is triggered to generate target word if the word-based decoder produces an
unknown token (UNK). Achieved subtle BLEU scores for both Cherokee-to-English and Spanish-to-English
translation task.
