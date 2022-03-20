import os
import pickle
import nltk
import re
import torch
import wordninja
from d2l import torch as d2l
from Transformer import Transformer
def read_psw(filename):
    a=[]
    with open(filename,'r') as f:
       file=f.readlines()
       for row in file:
            a.append(row.strip('\n').split(','))
    return a


alpha = re.compile(r'[a-zA-z]')
digit = re.compile(r'[0-9]+')
def cutword(sent):
    type='x'
    cut_sent=[]
    cur_word=""
    for ch in sent:
        if ch.isalpha():
            if type=='L':
                cur_word+=ch
                continue
            else:
                type='L'
                if cur_word !='':
                    cut_sent.append(cur_word)
                cur_word=ch
        elif ch.isdigit():
            if type=='D':
                cur_word+=ch
                continue
            else:
                type='D'
                if cur_word!='':
                    cut_sent.append(cur_word)
                cur_word=ch
        else:
            type='x'
            if cur_word !='':
                cut_sent.append(cur_word)
            cur_word=''
    cut_sent.append(cur_word)
    cut_after_ninja=[]
    for word in cut_sent:
        cut_after_ninja+=wordninja.split(word)

    print("cut",cut_sent)
    print("cut_after_ninja",cut_after_ninja)


    return cut_sent
def make_username(username):
    #print(username)
    username=cutword(username)
    #print(username)
    return username
def make_email(email):
    email=email.split('@')[0] #只要@前的信息
    email=cutword(email)
    return email
def make_name(name):
    name=name.strip().split(' ')
    name=[n.lower() for n in name]
    acronym = ""
    for n in name:
        if n != '':
            acronym += n[0]
    name.append(acronym)
    return name
def make_birth(birth):
    birth=birth.split('/')
    if len(birth)<3:
        return
    Y=int(birth[0])
    M=int(birth[1])
    D=int(birth[2])
    birthPattern = ['%s%s%s' % (Y, M, D), '%s' % (Y), '%s%s' % (M, D), '%s%s' % (D, M), '%s%s%s' % (D, M, Y),
                    '%s%s%s' % (M, D, Y), '%s%s' % (D, Y), '%s%s' % (Y, D),
                    '%s%s%s' % (Y % 100, M, D), '%s' % (Y % 100), '%s%s' % (M, D), '%s%s' % (D, M),
                    '%s%s%s' % (D, M, Y % 100), '%s%s%s' % (M, D, Y % 100), '%s%s' % (D, Y % 100),
                    '%s%s' % (Y % 100, D),
                    '%s%02d%02d' % (Y, M, D), '%02d%02d' % (M, D), '%02d%02d' % (D, M), '%02d%02d%s' % (D, M, Y),
                    '%02d%02d%s' % (M, D, Y), '%02d%02d' % (Y, D), '%02d%02d' % (D, Y), '%02d' % (D),
                    '%s%02d%02d' % (Y % 100, M, D), '%02d%02d' % (M, D), '%02d%02d' % (D, M),
                    '%02d%02d%s' % (D, M, Y % 100), '%02d%02d%s' % (M, D, Y % 100), '%02d%02d' % (Y % 100, D),
                    '%02d%02d' % (D, Y % 100)]
    return birthPattern
def make_password(format,password):
    password_list=[]
    begin=0
    num_list=map(int,digit.findall(format))
    for num in num_list:
        password_list.append(password[begin:begin+num])
        begin+=num
    return password_list

def concat(array_list):
    ans=[]
    for array in array_list:
        if array != None:
            ans.extend(array)
    # print(ans)
    return ans

def load_data(data):
    source=[]
    target=[]
    PII=[]
    print("get dict...")
    num=6000
    for i,datum in enumerate(data):
        if i>num:
            break
        username=make_username(datum[0])
        email=make_email(datum[1])
        name=make_name(datum[2])
        PII.append(concat([username,email,name,[datum[3]]]))
        birth=make_birth(datum[3])
        password=make_password(datum[5],datum[6])
        source.append(concat([username,email,name,birth,password]))
        target.append(password)
        # print('username:',username,' email:',email,' name:',name," birth:",birth,' password:',password)
    print("Building Vocab...")
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])


    batch_size, num_steps = 200, 30



    src_array, src_valid_len = d2l.build_array_nmt(PII, src_vocab,num_steps)
    tgt_array, tgt_valid_len = d2l.build_array_nmt(target, src_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    filename= 'data/iter_vocab.p'
    with open(filename,'wb') as file:
        pickle.dump((data_iter,src_vocab),file)
    return data_iter,src_vocab,PII

if __name__=='__main__':

    print(torch.cuda.is_available())
    preprocess_data="/data/iter_vocab.p"
    train_iter=None
    src_vocab=None
    data = read_psw('../data/pcfg_format_password+password.csv')
    train_iter,src_vocab,PIIs=load_data(data)
    print(PIIs)
    tgt_vocab=src_vocab

    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 30
    lr, num_epochs, device = 0.005, 30, d2l.try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]


    print("start training")

    encoder=Transformer.Encoder(len(src_vocab),query_size,num_hiddens,norm_shape,
                    ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout)
    decoder=Transformer.Decoder(len(tgt_vocab),query_size,num_hiddens,norm_shape,
                    ffn_num_input,ffn_num_hiddens,num_heads,num_layers,dropout)
    transformer=Transformer.EncoderDecoder(encoder,decoder)
    d2l.train_seq2seq(transformer,train_iter,lr,num_epochs,tgt_vocab,device)
    pred,_=d2l.predict_seq2seq(transformer,PIIs[0:5],src_vocab,tgt_vocab,num_steps,device)
    print(zip(PIIs[0:5],pred))