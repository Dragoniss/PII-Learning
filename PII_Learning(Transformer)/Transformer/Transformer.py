import math

import numpy as np
import pandas as pd
import torch
from d2l import torch as d2l  # Use PyTorch as the backend
from torch import nn
# 这里的X是经过线性层变换后的W_q,W_k,W_v,最后一维的维度从word_embedding_len转到了num_hidden
# ===================================================================================================
# num_hidden可能是设成了word_embedding_len的num_head倍，因此这一步是将最后一个维度的num_head×embedding_len
# 中的num_head提取出来，与batch_size合并，最终形成了(batch_size×num_head,sequence_len,word_embedding_len)
#=====================================================================================================
# 纠正：但是发现其实最终embedding_len和hidden_size竟然设置的是一样的，相当于是一种下采样了
# 比如整个词嵌入长度为24，经全连接操作后仍然是24，不过这里变为了长度为3的八头向量
# 这一步是相当于是一个卷积操作，目的是生成多个通道(在这里是多个head),
def transpose_qkv(X,num_head):
    X = X.reshape(X.shape[0],X.shape[1],num_head,-1)
    X = X.permute(0,2,1,3)
    return X.reshape(-1,X.shape[2],X.shape[3])
def transpose_output(X,num_heads):
    X=X.reshape(-1,num_heads,X.shape[1],X.shape[2])
    X=X.permute(0,2,1,3)
    return X.reshape(X.shape[0],X.shape[1],-1)

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:`sec_attention-scoring-functions`"""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    def __init__(self,dropout):
        super(DotProductAttention,self).__init__()
        self.dropout=nn.Dropout(dropout)
    def forward(self,queries,keys,values,valid_lens):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

# 加入了时间信息，可以理解为越靠近的词加的激励越大
class PositionalEncoding(nn.Module):
    def __init__(self,embedding_size,dropout,max_len=1000):
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.P=np.zeros((1,max_len,embedding_size))
        X=np.arange(0,max_len)
        # print('X.arange:',X.shape)
        X=X.reshape(-1,1)
        # print('X.reshape:',X.shape)
        Y=np.arange(0,embedding_size,2)/embedding_size
        # print('Y.reshape:',Y.shape)
        X=X/np.power(10000,Y)
        # print('X.power:',X.shape)
        self.P[:,:,0::2]=np.sin(X)
        self.P[:,:,1::2]=np.cos(X)
        self.P=torch.FloatTensor(self.P)
        # print('P:',self.P.shape)
    def forward(self,X):
        if X.is_cuda and not self.P.is_cuda:
            self.P=self.P.cuda()
        X = X + self.P[:,:X.shape[1],:]
        return self.dropout(X)

# X=torch.ones((2,100,32))
# position=PositionalEncoding(32,0.1,1000)
# print(position.eval())
# print(position(X).shape)


class MultiHeadAttention(nn.Module):
    def __init__(self,key_size,value_size,query_size,num_hidden,num_head,dropout,bias=False):
        super(MultiHeadAttention,self).__init__()
        self.num_head=num_head
        self.attention=DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size,num_hidden,bias=bias)
        self.W_k = nn.Linear(key_size,num_hidden,bias=bias)
        self.W_v = nn.Linear(value_size,num_hidden,bias=bias)
        self.W_o = nn.Linear(num_hidden,query_size,bias=bias)
    def forward(self,queries,keys,values,mask=None):
        queries = transpose_qkv(self.W_q(queries),self.num_head)
        keys = transpose_qkv(self.W_k(keys),self.num_head)
        values = transpose_qkv(self.W_v(values),self.num_head)

        if mask is not None:
            # 因为变成了多通道，并且将num_head与batch_size合并
            # 对于每一个样本 原来的mask是下三角的1矩阵，现在成多个通道了，所以矩阵的每一行都要重复多次以对应num_head个通道
            # 举个例子, 假设原来的 mask 是[[1,0],[1,1]],原样本是[['hello','world'],['hello','world']](当然是embedding后)
            # 假设num_head=2 那么经过多头操作就变成了 [['hello','world'],['hello','world'],['hello','world'],['hello','world']]
            # 因此 mask 也要变成[[1,0],[1,0],[1,1],[1,1]]
            mask =  torch.repeat_interleave(mask,repeats=self.num_head,dim=0)
        # print('queries.shape:',queries.shape)
        # print('keys.shape:',keys.shape)
        # print('values.shape:',values.shape)
        output = self.attention(queries,keys,values,mask)
        output_concat =  transpose_output(output,self.num_head)
        return self.W_o(output_concat)

# attention测试
# X=torch.ones((2,4,50))
# mask= (1 - torch.triu(
#         torch.ones((10, 8, 4), device=X.device), diagonal=1))
# print(mask)
# attentionLayer=MultiHeadAttention(50,50,50,100,5,0.5)
# attention=attentionLayer(X,X,X)
# print(attentionLayer.eval())
# print(attention.shape)


# repeat_interleave 重复张量元素
# mask = torch.tensor([[1,0],[1,1]],dtype=torch.float32)
# mask =  torch.repeat_interleave(mask,repeats=8,dim=0)
# print(mask)


class FeedForward(nn.Module):
    #由两个全连接层组成
    #保证输入输出维度一致，因此ffn_num_input == ffn_num_output , ffn_num_hidden随意
    #输入向量维度应该是(batch,sequence_len,wordembeddding_len)
    #由于sequence_len并不是我们要的feature(事实上，最后一个维度才包含着每一个单词的feature)
    #因此在进入Linear层之前先将前两个维度合并,变为(batch×sequence_len,word_embedding_len)，经过两个Linear层变换后再变回原规模
    #而Pytorch的Linear层仅提供两个参数(input_dim,output_dim),对于维度大于2的输入向量，会自动将前面n-1个维度拉伸成一个维度
    #因此都不需要我们人为的将前面两个维度合并了
    def __init__(self,ffn_num_input,ffn_num_hidden,ffn_num_output):
        super(FeedForward,self).__init__()
        self.dense1 = nn.Linear(ffn_num_input,ffn_num_hidden)
        self.relu=nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hidden,ffn_num_output)

    def forward(self,X):
        return self.dense2(self.relu(self.dense1(X)))

#FeedForward测试
# ffn=FeedForward(4,4,8)
# print(ffn.eval())
# print(ffn(torch.ones((2,3,4))))

class AddNorm(nn.Module):
    # residual + layerNorm
    # nn.LayerNorm()中的参数为normalized_shape 是默认最后几维的
    # 比如你的输入向量是3维的 你的normalized_shape=[xx,xx]那么意味着将对后面两个维度进行LN
    # 这里我们只对每一个样本进行LN,因此normalized_shape=[sequence_len,word_embedding_len]
    def __init__(self,normalized_shape,dropout):
        super(AddNorm,self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layerNorm = nn.LayerNorm(normalized_shape)

    def forward(self,X,Y):
        # Y就是X经过一定的神经网络层后的结果
        return self.layerNorm(self.dropout(Y)+X)

#AddNorm 测试
# add_norm= AddNorm([3,4],0.5)
# print(add_norm.eval())
# print(add_norm(X=torch.ones((2,3,4)),Y=torch.ones((2,3,4))).shape)

#LN和BN的对比
# layerNorm=nn.LayerNorm(2)
# batchNorm=nn.BatchNorm1d(2)
# X = torch.tensor([[1,2],[2,3]],dtype=torch.float32)
# print(layerNorm(X))
# print(batchNorm(X))

class EncoderUnit(nn.Module):
    def __init__(self,seq_size,num_hidden,norm_shape,ffn_num_input,ffn_num_hiddens,num_head,dropout,bias=False):
        super(EncoderUnit,self).__init__()
        self.multiHeadAttention=MultiHeadAttention(seq_size,seq_size,seq_size,num_hidden,num_head,dropout,bias=bias)
        self.addNorm1=AddNorm(norm_shape,dropout)
        self.feedForward=FeedForward(ffn_num_input,ffn_num_hiddens,num_hidden)
        self.addNorm2=AddNorm(norm_shape,dropout)
    def forward(self,X,mask=None):
        Y=self.addNorm1(X,self.multiHeadAttention(X,X,X,mask))
        return self.addNorm2(Y,self.feedForward(Y))

# EncoderUnit 测试
# X=torch.ones((2,100,24))
# encoderUnit=EncoderUnit(24,24,[100,24],24,48,8,0.1)
# print(encoderUnit.eval())
# print(encoderUnit(X).shape)

class Encoder(nn.Module):
    def __init__(self,vocab_size,seq_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hidden,num_head,num_layer,dropout,use_bias=False):
        super(Encoder,self).__init__()
        self.num_hiddens=num_hiddens
        self.embedding=nn.Embedding(vocab_size,num_hiddens)
        self.pos_encoding=PositionalEncoding(num_hiddens,dropout)
        self.encoders=nn.Sequential()
        for i in range(num_layer):
            self.encoders.add_module("block"+str(i),
                                     EncoderUnit(seq_size,num_hiddens,norm_shape,
                                                 ffn_num_input,ffn_num_hidden,num_head,dropout,bias=use_bias))
    def forward(self,X,mask=None):
        X=self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self.attention_weights=[None]*len(self.encoders)
        for i,encoderLayer in enumerate(self.encoders):
            X=encoderLayer(X,mask)
            self.attention_weights[i]=encoderLayer.multiHeadAttention.attention.attention_weights
        return X

# Encoder 测试
X=torch.ones((2,100),dtype=torch.long)
encoder=Encoder(200,24,24,[100,24],24,48,8,6,0.1)
# print(encoder.eval())
# print(encoder(X).shape)
X=encoder(X)

class DecoderUnit(nn.Module):
    def __init__(self,seq_size,num_hidden,norm_shape,ffn_num_input,ffn_num_hidden,num_head,dropout,i):
        super(DecoderUnit,self).__init__()
        self.i=i
        self.training=True
        self.mask_attention=MultiHeadAttention(seq_size,seq_size,seq_size,num_hidden,num_head,dropout)
        self.addNorm1=AddNorm(norm_shape,dropout)
        self.attention=MultiHeadAttention(seq_size,seq_size,seq_size,num_hidden,num_head,dropout)
        self.addNorm2=AddNorm(norm_shape,dropout)
        self.feedForward=FeedForward(ffn_num_input,ffn_num_hidden,num_hidden)
        self.addNorm3=AddNorm(norm_shape,dropout)
    def forward(self,X,state):
        encode_output,encode_mask=state[0],state[1]
        if state[2][self.i] is None:
            key_values=X
        else:
            key_values=torch.cat((state[2][self.i],X),axis=1)

        state[2][self.i]=key_values
        if self.training:
            batch_size,num_steps,_=X.shape
            decode_mask = torch.arange(1,num_steps+1,device=X.device).repeat(batch_size,1)
        else:
            decode_mask=None
        X2=self.mask_attention(X,key_values,key_values,decode_mask)
        Y=self.addNorm1(X,X2)
        Y2=self.attention(Y,encode_output,encode_output,encode_mask)
        Z=self.addNorm2(Y,Y2)
        return self.addNorm3(Z,self.feedForward(Z)),state

# encodeUnit=EncoderUnit(24,24,[100,24],24,48,8,0.1)
# decodeUnit=DecoderUnit(24,24,[100,24],24,48,8,0.1,0)
# print(decodeUnit.eval())
# X=torch.ones(2,100,24)
# state=[encodeUnit(X),None,[None]]
# print(decodeUnit(X,state)[0].shape)

class Decoder(nn.Module):
    def __init__(self,vocab_size,seq_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hidden,num_head,num_layer,dropout,use_bias=False):
        super(Decoder,self).__init__()
        self.num_hiddens=num_hiddens
        self.num_layers=num_layer
        self.embedding=nn.Embedding(vocab_size,num_hiddens)
        self.pos_encoding=PositionalEncoding(num_hiddens,dropout)
        self.decoders=nn.Sequential()
        for i in range(num_layer):
            self.decoders.add_module("block"+str(i),
                                     DecoderUnit(seq_size,num_hiddens,norm_shape,
                                                 ffn_num_input,ffn_num_hidden,num_head,dropout,i))
        self.dense=nn.Linear(num_hiddens,vocab_size)

    def init_state(self,encode_output,encode_mask=None):
        return [encode_output,encode_mask,[None]*self.num_layers]
    def forward(self,X,state):
        X=self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        #存attention_weight方便可视化
        self._attention_weights=[[None]*len(self.decoders) for _ in range(2)]
        for i,decoderLayer in enumerate(self.decoders):
            X,state=decoderLayer(X,state)
            self._attention_weights[0][i]=decoderLayer.mask_attention.attention.attention_weights
            self._attention_weights[1][i]=decoderLayer.attention
        return self.dense(X),state

    @property
    def attention_weights(self):
        return self._attention_weights
#Decoder 测试
# Y=torch.ones((2,100),dtype=torch.long)
# decoder=Decoder(200,24,24,[100,24],24,48,8,6,0.1)
# decoder_state=decoder.init_state(X)
# print(decoder.eval())
# print(decoder(Y,decoder_state)[0].shape)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encode_X, decode_X,valid_len=None):
        encode_outputs = self.encoder(encode_X,valid_len)
        decode_state = self.decoder.init_state(encode_outputs)
        return self.decoder(decode_X, decode_state)

# 测试 Transformer
# X=torch.ones((2,100),dtype=torch.long)
# Y=torch.ones((2,100),dtype=torch.long)
encoder=Encoder(200,24,24,[100,24],24,48,8,6,0.1)
decoder=Decoder(200,24,24,[100,24],24,48,8,6,0.1)
transformer=EncoderDecoder(encoder,decoder)


# print(transformer.eval())
# print(transformer(X,Y))

def sequence_mask(X,valid_len,value=0):
    max_len=X.size(1)
    mask=torch.arange(max_len,device=X.device).reshape(1,-1) < valid_len.reshape(-1,1)
    X[~mask]=value
    return X

X=torch.ones((2,3))
valid_len=torch.tensor([1,2])
print(sequence_mask(X,valid_len))
class Mask_CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self,pred,labels,valid_len):
        self.reduction='none'
        #调用基类的forward方法
        unweighted_loss=super(Mask_CrossEntropyLoss,self).forward(pred.permute(0,2,1),labels)
        weights = torch.ones_like(labels)
        weights = sequence_mask(weights,valid_len)
        weighted_loss= (unweighted_loss*weights).mean(dim=1)
        return weighted_loss
def train(net,data_iter,num_epochs,lr,num_layer,target_vocab,device):
    net.to(device)
    loss = Mask_CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    net.train()
    for epoch in range(num_epochs):
        time=d2l.Timer()
        d2l.train_seq2seq()
        return