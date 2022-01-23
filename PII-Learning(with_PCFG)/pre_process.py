import numpy as np
import unidecode
import string
import random
import time
import math
import numpy
import json
import os
import re
import pickle
import  zipfile
from config import max_token_length,start_word,end_word,batch_size
from tqdm import tqdm

dict={}
def read_file(filename):
    data=[]
    file = open(filename).readlines()
    for row in file:
        row=unidecode.unidecode(row)
        data.append(row[0:-1])
    return data

def build_samples(usage):
    passwords=read_file('../data/pcfg_format_password.csv')
    code={}
    letterType={'U':0,'E':1,'B':2,'N':3,'L':4,'D':5,'S':6}
    letter_format = re.compile(r'[a-zA-z][0-9]+')
    for password in passwords:
        letter_list=letter_format.findall(password)
        for letter in letter_list:
            if letter not in code:
                code[letter]=letterType[letter[0]]*max_token_length+int(letter[1:])
    code[start_word]=6*max_token_length+1
    code[end_word]=6*max_token_length+2
    # for i in sorted(code):
    #     print(i,code[i])
    samples=[]
    print('Generating Sample...')
    for i in range(len(passwords)):
        PII_id = i
        password_sample = passwords[i]
        input = []
        last_word = start_word
        letter_list=letter_format.findall(password_sample)
        for letter in letter_list:
            input.append(code[last_word])
            samples.append({'PII_id': PII_id, 'input': list(input), 'output': code[letter]})
            last_word = letter
        input.append(code[last_word])
        samples.append({'PII_id': PII_id, 'input': list(input), 'output': code[end_word]})

    print(samples)
    filename = '../data/pcfg_samples_{}.p'.format(usage)
    with open(filename, 'wb') as f:
        pickle.dump(samples, f)

def encode_PII(usage):
    PII=read_file('../data/pcfg_format_PII.csv')
    encoding={}
    all_characters = string.printable
    n_characters = len(all_characters)
    code = {}
    for index, letter in enumerate(all_characters):
        code[letter] = index
    code[start_word] = n_characters
    code[end_word] = n_characters + 1

    num_batches=int(np.ceil(len(PII)/float(batch_size)))
    print('Find out the max size of PII...')
    maxlen=0
    for datum in PII:
            maxlen=max(maxlen,len(datum))
    print('The maxsize of PII is %d'%(maxlen))

    print('encoding {} PIIs...'.format(usage))
    for idx in tqdm(range(num_batches)):
        i=idx*batch_size
        length = min(batch_size,len(PII)-i)
        for i_batch in range(length):
            PII_data=PII[i+i_batch]
            PII_array=[]
            for chr in PII_data:
                PII_array.append(code[chr])
            encoding[i+i_batch]=PII_array

    filename= '../data/pcfg_encoded_{}_PII.p'.format(usage)
    with open(filename,'wb') as encoded_pickle:
        pickle.dump(encoding,encoded_pickle)

if __name__=='__main__':

    # print(passwords)
    # print(PII)
    build_samples('train')
    encode_PII('train')