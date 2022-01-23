import argparse
import keras
import random

import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from keras.utils import multi_gpu_model
from config import patience, n_epochs, num_train_samples, num_valid_samples, batch_size
from model import build_model
from data_generator import *
import matplotlib.pyplot as plt
def draw_trainplot(history):
    historyLoss=history.history['loss']
    epoch_num=range(len(historyLoss))
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("Epochs-Loss Graph")
    plt.plot(epoch_num,historyLoss)
    plt.show()

if __name__ =='__main__':

    checkpoint_models_path = 'models/'
    train_generator = DataGenSequence('train')  # Start Fine-tuning
    valid_generator = DataGenSequence('train')

    # Callbacks
    # tensor_board = keras.callbacks.TensorBoard(log_dir='.logs', histogram_freq=0, write_graph=True, write_images=True)
    # model_names = checkpoint_models_path + 'model.{epoch:02d}-{val_loss:4f}.hdf5'
    # model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    # early_stop = EarlyStopping('val_loss', patience=patience)
    # reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 5), verbose=1)
    # class MyCallback(keras.callbacks.Callback):
    #     def __init__(self,model):
    #         keras.callbacks.Callback.__init__(self)
    #         self.model_to_save=model
    #
    #     def on_epoch_end(self,epoch,logs=None):
    #         fmt=checkpoint_models_path+'model.%2d-%.4f.hdf5'
    #         self.model_to_save.save(fmt % (epoch,logs['val_loss']))


    new_model = build_model()

    adam=tf.keras.optimizers.Adam(learning_rate=5e-5)
    new_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    new_model.summary()
    x1=[]
    x2=[]
    y=[]
    for i in range(n_epochs):
        tempx,tempy=train_generator.__getitem__(random.randint(0,2000))
        # print("X:",tempx)
        # print("Y:",tempy)

        x1.extend(tempx[0])
        x2.extend(tempx[1])
        y.extend(tempy)
        # print(y.shape)

    x1=np.array(x1).astype(np.float64)
    x2=np.array(x2).astype(np.float64)
    y=np.array(y).astype(np.float64)
    print(y.shape)
    #Final callbacks
    # callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]
    history=new_model.fit(
                  [x1,x2],y,
                  steps_per_epoch = num_train_samples//batch_size,
                  # validation_data=valid_generator,
                  # validation_steps=num_valid_samples//batch_size,
                  epochs=n_epochs,
                  verbose=1,
                  # callbacks=callbacks,
                 )
    model.save('adam-v1.0.h5')
    draw_trainplot(history)