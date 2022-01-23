import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense, LSTM,CuDNNLSTM, Concatenate,Embedding,RepeatVector,TimeDistributed,Dropout
from keras.models import Model
from tensorflow.keras.utils import plot_model
from pydot import *
from config import max_token_length
from config import vocab_size, embedding_size,PII_size

def build_model():
    #word embedding
    password_input = Input(shape=(max_token_length,),dtype='int32')
    x = Embedding(input_dim=vocab_size,output_dim=embedding_size)(password_input)
    x = LSTM(256, return_sequences=True)(x)
    password_embedding = TimeDistributed(Dense(embedding_size))(x)

    # PII embedding
    PII_input =Input(shape=(PII_size,))
    x=Dense(embedding_size,activation='relu',name='PII_embedding')(PII_input)
    # the PII I is only input once
    PII_embedding =RepeatVector(1)(x)

    # language model
    x = [PII_embedding,password_embedding]
    x = Concatenate(axis=1)(x)
    x = Dropout(0.1)(x)
    x = LSTM(512,return_sequences=True,name='language_lstm_1')(x)
    x = Dropout(0.2)(x)
    x = LSTM(512,name='language_lstm_2')(x)
    x = Dropout(0.4)(x)
    output = Dense(vocab_size,activation='softmax',name='output')(x)

    inputs=[PII_input,password_input]
    model = Model(inputs=inputs,outputs=output)
    return model

if __name__ == '__main__':
    with tf.device("/cpu:0"):
        model =  build_model()
    model.summary()
    # plot_model(model, to_file='model.svg', show_layer_names=True,show_shapes=True)

    K.clear_session()