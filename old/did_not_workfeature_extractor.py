import tensorflow as tf
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
from seaborn import color_palette
import cv2


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

    def __init__(self,n_classes,model_size,max_output_size,iou_threshold,confidence_threshold,data_format=None):
        self.n_classes = n_classes
        self.model_size = model_size
        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold
        self.data_format = data_format

        mp = tf.keras.layers.MaxPool2D
        config = CfgParser(path=os.path.join(os.getcwd(),'yolov3_tiny.cfg')).get_config()
        self.layers = []
        self.yolos = []
        with tf.variable_scope('Feature_Extractor'):
            self.img_in = tf.keras.Input(shape=(None,None,3)) # TODO: shape, dtype set correctly
            
            for index,x in enumerate(config):
                index = index-1
                print(index,x['type'])
                
                if len(self.layers)==0:
                    inp = self.img_in #TODO
                else:
                    inp = self.layers[index-1]

                

                if x['type'] == 'convolutional':
                    if x['activation'] == 'leaky':
                        activ = tf.nn.leaky_relu
                    else:
                        activ = None
                    
                    try:
                        batch_normalize = int(x["batch_normalize"])
                    except:
                        batch_normalize = 0

                    self.layers.append(self.conv_layer(inp,
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
                    self.layers.append(mp(pool_size=size,strides=stride)(inp))
                    
                elif x['type'] == 'upsample':
                    self.layers.append(self.upsample(inp))
                elif x['type'] == 'route':
                    x['layers'] = x['layers'].split(',')
                    start = int(x['layers'][0])
                    try:
                        stop = int(x['layers'[1]])
                    except: 
                        stop = 0
                    if stop == 0:
                        route = tf.keras.layers.concatenate([self.layers[index+start],self.layers[index-1]],axis=3)
                        self.layers.append(route) #AXIS TODO
                    else:
                        route = tf.keras.layers.concatenate([self.layers[index+start],self.layers[stop]],axis=3)
                        self.layers.append(route)
                    
                elif x['type'] == 'shortcut':
                    index2 = int(x['from'])
                    self.layers.append(self.layers[index-1]+self.layers[index2])
                elif x['type'] == 'yolo':
                    mask = x["mask"].split(",")
                    mask = [int(x) for x in mask]

                    anchors = x["anchors"].split(",")
                    anchors = [int(a) for a in anchors]
                    anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
                    anchors = [anchors[i] for i in mask]
                    self.layers.append(self.layers[index-1])
                    self.yolos.append(self.yolo(inp,anchors))
                elif x['type'] == 'net':
                    print('TODO, overall config grabber not yet implemented')

    def __call__(self,training=True):
        with tf.variable_scope('Feature_Extractor',reuse=True):
            inputs = tf.concat(self.yolos,axis=1)
            inputs = self.build_boxes(inputs)

            box_dicts = self.non_max_suppression(input)
            return box_dicts





             

    


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
        detection = Detection_Layer(anchors,n_classes=self.n_classes,img_size=self.model_size,data_format=self.data_format)(x)
        return detection


    def build_boxes(self,inputs):
        center_x, center_y, width, height, confidence, classes = \
            tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

        top_left_x = center_x - width / 2
        top_left_y = center_y - height / 2
        bottom_right_x = center_x + width / 2
        bottom_right_y = center_y + height / 2

        boxes = tf.concat([top_left_x, top_left_y,
                        bottom_right_x, bottom_right_y,
                        confidence, classes], axis=-1)

        return boxes
    
    def non_max_suppression(self,inputs):
        batch = tf.unstack(inputs)
        boxes_dicts = []
        for boxes in batch:
            boxes = tf.boolean_mask(boxes, boxes[:, 4] > self.confidence_threshold)
            classes = tf.argmax(boxes[:, 5:], axis=-1)
            classes = tf.expand_dims(tf.to_float(classes), axis=-1)
            boxes = tf.concat([boxes[:, :5], classes], axis=-1)

            boxes_dict = dict()
            for cls in range(self.n_classes):
                mask = tf.equal(boxes[:, 5], cls)
                mask_shape = mask.get_shape()
                if mask_shape.ndims != 0:
                    class_boxes = tf.boolean_mask(boxes, mask)
                    boxes_coords, boxes_conf_scores, _ = tf.split(class_boxes,
                                                                [4, 1, -1],
                                                                axis=-1)
                    boxes_conf_scores = tf.reshape(boxes_conf_scores, [-1])
                    indices = tf.image.non_max_suppression(boxes_coords,
                                                        boxes_conf_scores,
                                                        self.max_output_size,
                                                        self.iou_threshold)
                    class_boxes = tf.gather(class_boxes, indices)
                    boxes_dict[cls] = class_boxes[:, :5]

            boxes_dicts.append(boxes_dict)

        return boxes_dicts


    def load_images(self,img_names):
        """Loads images in a 4D array.

        Args:
            img_names: A list of images names.
            model_size: The input size of the model.
            data_format: A format for the array returned
                ('channels_first' or 'channels_last').

        Returns:
            A 4D NumPy array.
        """
        imgs = []

        for img_name in img_names:
            img = Image.open(img_name)
            img = img.resize(size=self.model_size)
            img = np.array(img, dtype=np.float32)
            img = np.expand_dims(img, axis=0)
            imgs.append(img)

        imgs = np.concatenate(imgs)

        return imgs


    def load_class_names(self,file_name):
        """Returns a list of class names read from `file_name`."""
        with open(file_name, 'r') as f:
            class_names = f.read().splitlines()
        return class_names


    def draw_boxes(self,img_names, boxes_dicts, class_names):
        """Draws detected boxes.

        Args:
            img_names: A list of input images names.
            boxes_dict: A class-to-boxes dictionary.
            class_names: A class names list.
            model_size: The input size of the model.

        Returns:
            None.
        """
        colors = ((np.array(color_palette("hls", 80)) * 255)).astype(np.uint8)
        for num, img_name, boxes_dict in zip(range(len(img_names)), img_names,
                                            boxes_dicts):
            img = Image.open(img_name)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font='../input/futur.ttf',
                                    size=(img.size[0] + img.size[1]) // 100)
            resize_factor = \
                (img.size[0] / self.model_size[0], img.size[1] / self.model_size[1])
            for cls in range(len(class_names)):
                boxes = boxes_dict[cls]
                if np.size(boxes) != 0:
                    color = colors[cls]
                    for box in boxes:
                        xy, confidence = box[:4], box[4]
                        xy = [xy[i] * resize_factor[i % 2] for i in range(4)]
                        x0, y0 = xy[0], xy[1]
                        thickness = (img.size[0] + img.size[1]) // 200
                        for t in np.linspace(0, 1, thickness):
                            xy[0], xy[1] = xy[0] + t, xy[1] + t
                            xy[2], xy[3] = xy[2] - t, xy[3] - t
                            draw.rectangle(xy, outline=tuple(color))
                        text = '{} {:.1f}%'.format(class_names[cls],
                                                confidence * 100)
                        text_size = draw.textsize(text, font=font)
                        draw.rectangle(
                            [x0, y0 - text_size[1], x0 + text_size[0], y0],
                            fill=tuple(color))
                        draw.text((x0, y0 - text_size[1]), text, fill='black',
                                font=font)

            display(img)

class Detection_Layer(tf.keras.layers.Layer):
    def __init__(self,anchors,n_classes,img_size,data_format,trainable=True,name=None,activity_regularizer=None,**kwargs):
        super(Detection_Layer,self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=tf.keras.regularizers.get(activity_regularizer),
        **kwargs)
        self.anchors = anchors
        self.n_classes = n_classes
        self.img_size = img_size
        self.data_format = data_format

    def build(self, input_shape):
        super(Detection_Layer,self).build(input_shape)

    def __call__(self,inputs,*args,**kwargs):
        n_anchors = len(self.anchors)
        inputs = tf.keras.layers.Conv2D(filters=n_anchors*(5+self.n_classes),kernel_size=1,strides=1,use_bias=True,data_format=self.data_format)(inputs)
        shape = inputs.get_shape().as_list()
        grid_shape = shape[2:4] if self.data_format == 'channels_first' else shape[1:3]
        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 2, 3, 1])
        inputs = tf.reshape(inputs, [-1, n_anchors * grid_shape[0] * grid_shape[1],5 + self.n_classes])
        strides = (self.img_size[0] // grid_shape[0], self.img_size[1] // grid_shape[1])

        box_centers, box_shapes, confidence, classes = \
            tf.split(inputs, [2, 2, 1, self.n_classes], axis=-1)

        x = tf.range(grid_shape[0], dtype=tf.float32)
        y = tf.range(grid_shape[1], dtype=tf.float32)
        x_offset, y_offset = tf.meshgrid(x, y)
        x_offset = tf.reshape(x_offset, (-1, 1))
        y_offset = tf.reshape(y_offset, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.tile(x_y_offset, [1, n_anchors])
        x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
        box_centers = tf.nn.sigmoid(box_centers)
        box_centers = (box_centers + x_y_offset) * strides

        anchors = tf.tile(self.anchors, [grid_shape[0] * grid_shape[1], 1])
        box_shapes = tf.exp(box_shapes) * tf.to_float(anchors)

        confidence = tf.nn.sigmoid(confidence)

        classes = tf.nn.sigmoid(classes)

        inputs = tf.concat([box_centers, box_shapes,
                            confidence, classes], axis=-1)

        return inputs


        



