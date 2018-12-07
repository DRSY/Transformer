# This is the implementation of Google Transformer model
## Transfomer

transformer is the model presented in th paper "attention is all you need".It discard the traditional recurrent and convolutional structure and instead use only self-attention mechanism.

train:
  train.py is used for seq2seq task.
  BERT.py is used for pre-trainning with masked language model and next sentence prediction tasks.
  
model:
  myTransformer.py
  Multiheadattention.py
  utils.py
