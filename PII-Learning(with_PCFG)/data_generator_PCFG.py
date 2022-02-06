import pickle
import random
import keras
import string
import numpy as np
from keras.preprocessing import sequence
from tensorflow.keras.utils import Sequence,to_categorical
import string
from config import batch_size, max_token_length,PII_size
from config import start_word,end_word,pcfg_code
from pre_process import read_file
class DataGenSequence(Sequence):
    def __init__(self,usage):
        self.usage = usage
        self.code = pcfg_code

        filename = 'data/pcfg_encoded_{}_PII.p'.format(usage)
        self.PII_encoding=pickle.load(open(filename,'rb'))
        if usage == 'train':
            samples_path = 'data/pcfg_samples_train.p'

        samples = pickle.load(open(samples_path,'rb'))
        self.samples = samples
        np.random.shuffle(self.samples)

    def __len__(self):
        return int(np.ceil(len(self.samples)/float(batch_size)))

    def __getitem__(self,idx):
        i=idx*batch_size
        length=min(batch_size,(len(self.samples)-i))
        batch_y=[]
        password_input=[]
        PII_input=[]
        for i_batch in range(length):
            sample = self.samples[i+i_batch]
            PII_id = sample['PII_id']
            temp=[]
            for item in self.PII_encoding[PII_id]:
                temp.append(item)

            PII_input.append(temp)
            password_input.append(sample['input'])

            y=[]
            y_index=sample['output']
            # print(len(self.code))
            for dat in range(len(self.code)):
                if dat!=y_index:
                    y.append(0)
                else:
                    y.append(1)
            batch_y.append(y)
        # print(len(batch_y))
        # batch_y=np.array(batch_y)

        for PII in PII_input:
            padding_num = PII_size - len(PII)
            for i in range(padding_num):
                PII.append(0)

        for password in password_input:
            padding_num = max_token_length - len(password)
            for i in range(padding_num):
                password.append(0)
        PII_input=np.array(PII_input).astype(np.float64)
        password_input=np.array(password_input)
        # print(password_input)
        password_input=password_input.astype(np.float64)
        X=[PII_input,password_input]
        batch_y=np.array(batch_y).astype(np.float64)
        # print(X)
        return X,batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.samples)

def train_gen():
     return DataGenSequence('train')


def valid_gen():
    return DataGenSequence('train')

if __name__=="__main__":
    train_generator = DataGenSequence('train')  # Start Fine-tuning
    tempx, tempy = train_generator.__getitem__(random.randint(0, 2000))
    print("X:", tempx)
    print("Y:", tempy)
    print(tempx[0].shape,tempx[1].shape,tempy.shape)