import numpy as np
import unidecode
import string
import random
import time
import math
import numpy
import json
import os
import pickle
import  zipfile
from tqdm import tqdm
from config import start_word,end_word
from config import PII_path
from config import batch_size,PII_size

def read_file(filename):
    data=[]
    file = open(filename).readlines()
    for row in file:
        row=unidecode.unidecode(row)
        data.append(row[0:-2])
    return data

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def encode_PII(usage):
    encoding={}
    PII=read_file('data/clean_PII.csv')
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
        if(len(datum)>maxlen):
            maxlen=len(datum)

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

    filename= 'data/encoded_{}_PII.p'.format(usage)
    with open(filename,'wb') as encoded_pickle:
        pickle.dump(encoding,encoded_pickle)

def build_samples(usage):

    password=read_file('data/clean_password.csv')
    all_characters = string.printable
    n_characters = len(all_characters)
    code = {}
    for index, letter in enumerate(all_characters):
        code[letter] = index
    code[start_word] = n_characters
    code[end_word] = n_characters + 1

    samples=[]
    print('Generating Sample...')
    for i in range(len(password)):
        PII_id=i
        password_sample=password[i]
        input=[]
        last_word=start_word
        for chr in password_sample:
            input.append(code[last_word])
            samples.append({'PII_id':PII_id,'input':list(input),'output':code[chr]})
            last_word=chr
        input.append(code[last_word])
        samples.append({'PII_id':PII_id,'input':list(input),'output':code[end_word]})

    print(samples)
    filename = 'data/samples_{}.p'.format(usage)
    with open(filename,'wb') as f:
        pickle.dump(samples,f)

if __name__ =='__main__':
    #parameters
    ensure_folder('data')

    if not os.path.isfile('data/samples_train.p'):
        build_samples('train')

    encode_PII('train')