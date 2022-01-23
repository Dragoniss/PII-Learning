import pickle
import nltk

stats={}
max_ngrams=4
begin_word='`'
filename='john.txt'
#create a list of ngrams from a single line in the train data

with open(filename) as file:
    for password in file:
        #add ngrams to the stats dict for all n<=max_ngrams
        for key_length in range(max_ngrams):
            for index,next_char in enumerate(password):
                if index<=key_length:
                    preword=begin_word*(key_length-index+1)
                    preword+=password[0:index]
                else:
                    preword+=password[index:key_length-index]
                if not preword in stats:
                    stats[preword]={}
                if not next_char in stats[preword]:
                    stats[preword][next_char]=0
                stats[preword][next_char]+=1

#convert frequency count to probabilities
for preword in stats:
    total=sum(list(stats[preword].values()))
    for next_char in stats[preword]:
        stats[preword][next_char]/=float(total)

with open('{}-gram.pickle'.format(max_ngrams),'wb') as file:
    pickle.dump(stats,file)