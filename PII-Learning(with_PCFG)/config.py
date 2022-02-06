
import string

PII_size = 113
n_epochs=10
hidden_size=100
embedding_size=50
patience = 50
learning_rate=0.01
batch_size=200
num_train_samples=537072
num_valid_samples=537072
max_token_length=18
vocab_size=114
PII_path='data/'


start_word='<start>'
end_word='<end>'
all_characters = string.printable
n_characters = len(all_characters)
code = {}
decode={}
for index, letter in enumerate(all_characters):
    code[letter] = index
    decode[index]=letter
code[start_word] = n_characters
code[end_word] = n_characters + 1
decode[n_characters]=start_word
decode[n_characters+1]=end_word
letterType={'U':0,'E':1,'B':2,'N':3,'L':4,'D':5,'S':6}
pcfg_code={}
pcfg_decode={}
pcfg_list=[]
for key in letterType:
    for i in range(1,17):
        string=key+str(i)
        pcfg_list.append(string)
for index, letter in enumerate(pcfg_list):
    pcfg_code[letter] = index
    pcfg_decode[index]=letter
pcfg_code[start_word] = len(pcfg_list)
pcfg_code[end_word] = len(pcfg_list) + 1
pcfg_decode[len(pcfg_list)]=start_word
pcfg_decode[len(pcfg_list)]=end_word
# print(pcfg_code)