import pickle
import time
import numpy as np

max_ngrams=4 #ngram size
num_generate=1000 #number of passwords to generate
result=open("result.txt",'w+')
begin_word='`'
#generate a single new password using a stats dict created during the training phase
def gen_password(n):
    output=begin_word*n
    for i in range(100):
        output+=gen_char(output[i:i+n])
        if output[-1]=='\n':
            return output[0:-1].replace(begin_word,''[0:-1])

#Sample a char if the ngram appeards in the stats dict.
#Otherwise recursively decrement n to try smaller grams in
#hopes to find a match('e.g. "off" becomes "of").
#This is a deviation from a vanilla markov text generator
#which one n-size. This generator uses all values<=n
#preferencing higher vales of n first.
def gen_char(ngram):
    if ngram in stats:
        return np.random.choice(list(stats[gram].keys()),p=list(stats[ngram].values()))
    else:
        return gen_char(ngram[0:-1])



with open('{}-gram.pickle'.format(max_ngrams),"rb+") as file:
    stats=pickle.load(file)
#start=time.time()
for num in range(num_generate):
    password=gen_password(max_ngrams)
    if pw is not None:
        result.write(pw+'\n')

