import tensorflow as tf
import numpy as np

import os
# maybe its more sensible to just use yolo pretrained and transferlearn with new classes and stuff?
# from https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-2/
class CfgParser(object):
    def __init__(self,path):
        self.path = path

    def get_config(self):
        file = open(self.path, 'r')
        lines = file.read().split('\n')                        # store the lines in a list
        lines = [x for x in lines if len(x) > 0]               # get read of the empty lines 
        lines = [x for x in lines if x[0] != '#']              # get rid of comments
        lines = [x.rstrip().lstrip() for x in lines] 

        block = {}
        blocks = []

        for line in lines:
            if line[0] == "[":               # This marks the start of a new block
                if len(block) != 0:          # If block is not empty, implies it is storing values of previous block.
                    blocks.append(block)     # add it the blocks list
                    block = {}               # re-init the block
                block["type"] = line[1:-1].rstrip()     
            else:
                key,value = line.split("=") 
                block[key.rstrip()] = value.lstrip()
        blocks.append(block)

        return blocks

class Feature_extractor(object):

    def __init__(self):
        mp = tf.keras.layers.MaxPool2D
        config = CfgParser(path=os.path.join(os.getcwd(),'yolov3_tiny.cfg')).get_config()
        layers = []
        with tf.variable_scope('Feature_Extractor'):
            self.img_in = tf.keras.Input(shape=(None,None,3)) # TODO: shape, dtype set correctly
            self.box_in = tf.keras.Input(shape=(None,None,3)) # TODO: shape, dtype set correctly
            
            for index,x in enumerate(config):
                index = index-1
                print(index,x['type'])
                
                if len(layers)==0:
                    inp = self.img_in #TODO
                else:
                    inp = layers[index-1]

                

                if x['type'] == 'convolutional':
                    if x['activation'] == 'leaky':
                        activ = tf.nn.leaky_relu
                    else:
                        activ = None
                    
                    try:
                        batch_normalize = int(x["batch_normalize"])
                    except:
                        batch_normalize = 0

                    layers.append(self.conv_layer(inp,
                                                  int(x['filters']),
                                                  int(x['size']),
                                                  int(x['stride']),
                                                  activ,
                                                  int(x['pad']),batch_normalize
                                                  ))
                elif x['type'] == 'maxpool':
                    size = int(x['size'])
                    try:
                        stride = int(x['stride'])
                    except:
                        stride = size
                    layers.append(mp(pool_size=size,strides=stride)(inp))
                    
                elif x['type'] == 'upsample':
                    layers.append(self.upsample(inp))
                elif x['type'] == 'route':
                    x['layers'] = x['layers'].split(',')
                    start = int(x['layers'][0])
                    try:
                        stop = int(x['layers'[1]])
                    except: 
                        stop = 0
                    if stop == 0:
                        layers.append(tf.keras.layers.concatenate([layers[index+start],layers[index-1]],axis=3)) #AXIS TODO
                    else:
                        layers.append(tf.keras.layers.concatenate([layers[index+start],layers[stop]],axis=3))
                    
                elif x['type'] == 'shortcut':
                    index2 = int(x['from'])
                    layers.append(layers[index-1]+layers[index2])
                elif x['type'] == 'yolo':
                    mask = x["mask"].split(",")
                    mask = [int(x) for x in mask]

                    anchors = x["anchors"].split(",")
                    anchors = [int(a) for a in anchors]
                    anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
                    anchors = [anchors[i] for i in mask]
                    layers.append(self.yolo(inp,anchors))
                elif x['type'] == 'net':
                    print('TODO, overall config grabber not yet implemented')

            print(layers)



             

    


    def conv_layer(self,x,filters,kernel_size,strides,activation,pad,batchnorm = True):
        bn = tf.keras.layers.BatchNormalization() if batchnorm else lambda x:x
        bias = False if batchnorm else True
        padding = 'valid' if pad else 'same'
        x = bn(tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,activation=activation,use_bias=bias,padding=padding)(x))
        return x
    
    def upsample(self,x):
        x = tf.keras.layers.UpSampling2D(size=2,interpolation='bilinear')(x)
        return x

    def yolo(self,x,anchors):
        detection = Detection_Layer(anchors)(x)
        return detection
    

class Detection_Layer(tf.keras.layers.Layer):
    def __init__(self,anchors,trainable=True,name=None,activity_regularizer=None,**kwargs):
        super(Detection_Layer,self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
        **kwargs)
        self.anchors = anchors

    # def residual(self,x):
    #     pass #TODO if we want it (probably)

    # def conv_net(self,x,nl=tf.nn.leaky_relu,batchnorm=False):
    #     bn = tf.keras.layers.BatchNormalization if batchnorm else lambda x:x
        
    #     x = bn(tf.keras.layers.Conv3D(filters=32,kernel_size=8, strides = (4,4,4),activation= nl)(x)) #arbitrary values for all, but need something to start
    #     x = bn(tf.keras.layers.Conv3D(filters=64, kernel_size=4, strides=(2, 2,2), activation=nl)(x))
    #     x = bn(tf.keras.layers.Conv3D(filters=64, kernel_size=3, strides=(1, 1,1), activation=nl)(x))
    #     x = bn(tf.keras.layers.Dense(units=100,activation=None)(x))
    #     return x