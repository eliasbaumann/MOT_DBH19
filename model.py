import tensorflow as tf

#from feature_extractor import Feature_extractor
# TODO we need two different parts:
# 1. YOLO
# 2. The no bells and whistles Tracktor

# Also all the necessary helper methods and potentially an extra file for regression

class Tracktor(object):
    
    def __init__(self):
        with tf.variable_scope('tracktor'):
            #self.img_in= tf.keras.Input(shape=(None,None))
            #self.box_in = tf.keras.Input(shape=(None,None))

            # self.feature_extractor = Feature_extractor() # TODO how do we get something here...

            self.features = 0 # here i want the features

