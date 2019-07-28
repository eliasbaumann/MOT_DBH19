import tensorflow as tf
import numpy as np

class feature_extractor(object):
    def __init__(self,batchnorm=True):
        self.batchnorm = batchnorm
        self.bn = tf.keras.layers.BatchNormalization() if self.batchnorm else lambda x: x
        self.training = tf.placeholder(tf.bool,shape=None, name='is_training')
        # self.input = data
        


    def create_lstm_model(self,data):
        x = tf.keras.layers.BatchNormalization()(data,training = self.training) 
        x = tf.keras.layers.ConvLSTM2D(64,kernel_size=(3,3),padding='same',return_sequences=True)(x,training = self.training)
        x = tf.keras.layers.BatchNormalization()(x,training = self.training)
        x = tf.keras.layers.ConvLSTM2D(32,kernel_size=(3,3),padding='same',return_sequences=True)(x,training = self.training)
        x = tf.keras.layers.BatchNormalization()(x,training = self.training)
        x = tf.keras.layers.ConvLSTM2D(16,kernel_size=(3,3),padding='same',return_sequences=False)(x,training = self.training)
        x = tf.keras.layers.BatchNormalization()(x,training = self.training)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(8)(x)
        x = tf.keras.layers.Dense(2)(x)
        return x,x

    def create_3dconv_model(self,data):
        x = tf.keras.layers.BatchNormalization()(data,training = self.training) 
        x = tf.keras.layers.Conv3D(filters=64,kernel_size=3,padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x,training = self.training)
        x = tf.keras.layers.Conv3D(filters=32,kernel_size=3,padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x,training = self.training)
        x = tf.keras.layers.Conv3D(filters=16,kernel_size=3,padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x,training = self.training)
        y = tf.keras.layers.GlobalAveragePooling3D()(x)
        x = tf.keras.layers.Dense(12)(y)
        x = tf.keras.layers.Dense(2)(x)
        return x,y
        

    def small_model(self,data):
        x = tf.keras.layers.BatchNormalization()(data,training = self.training) 
        x = tf.keras.layers.GlobalAveragePooling3D()(x)
        y = tf.keras.layers.Dense(128,activation=tf.nn.leaky_relu)(x)
        x = tf.keras.layers.Dense(32)(y)
        x = tf.keras.layers.Dense(2)(x)
        return x,y


    



        




    

