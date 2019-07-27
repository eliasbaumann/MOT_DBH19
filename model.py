import tensorflow as tf
import numpy as np

class feature_extractor(object):
    def __init__(self,data,batchnorm=True):
        self.batchnorm = batchnorm
        self.bn = tf.keras.layers.BatchNormalization() if self.batchnorm else lambda x: x
        self.input = data


    def create_lstm_model(self):
        x = tf.keras.layers.BatchNormalization()(self.input) # TODO we can also do this on the data     
        # cell1 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
        #                                     input_shape=[None,None,3],
        #                                     output_channels=64,
        #                                     kernel_shape=3)
        # x = tf.keras.layers.RNN(cell=cell1,return_sequences=True,dty)(x)#(cell=cell1,inputs=x,dtype=tf.float32)                    
        x = tf.keras.layers.ConvLSTM2D(64,kernel_size=(3,3),padding='same',return_sequences=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ConvLSTM2D(32,kernel_size=(3,3),padding='same',return_sequences=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ConvLSTM2D(16,kernel_size=(3,3),padding='same',return_sequences=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(8)(x)
        x = tf.keras.layers.Dense(2)(x)
        return x

        




    

