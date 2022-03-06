from d2l import torch as d2l
from Transformer import *
num_hiddens,num_layers,dropout,batch_size,num_steps=32,2,0.1,200,20
lr,num_epochs,device=0.005,200,d2l.try_gpu()
ffn_num_input,ffn_num_hiddens,num_heads=32,64,4
key_size,query_size,value_size=32,32,32
norm_shape=[32]
train_iter,src_vocab,tgt_vocab=d2l.load_data_nmt(batch_size,num_steps,num_examples=600)

# print(src_vocab)
# encoder=Encoder(len(src_vocab),query_size,num_hiddens,norm_shape,
#                 ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout)
# decoder=Decoder(len(tgt_vocab),query_size,num_hiddens,norm_shape,
#                 ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout)
# transformer=EncoderDecoder(encoder,decoder)
# d2l.train_seq2seq(transformer,train_iter,lr,num_epochs,tgt_vocab,device)
# pred,_=d2l.predict_seq2seq(transformer,"I am Lasy .",src_vocab,tgt_vocab,num_steps,device)
# print(pred)