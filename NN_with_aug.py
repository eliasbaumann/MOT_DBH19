import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
from autoaugment import CIFAR10Policy
from utils import sparse_cost_sensitive_loss,onehot
from PIL import Image
from model import feature_extractor
from utils import sparse_cost_sensitive_loss,onehot
import itertools

def cv_resize_pad(img,desired_size):
    osize = img.shape[:2]
    ratio = float(desired_size)/max(osize)
    nsize = tuple([int(x*ratio) for x in osize])
    
    img = cv2.resize(img, (nsize[1],nsize[0]))
    dw = desired_size - nsize[1]
    dh = desired_size - nsize[0]

    top,bottom = dh//2,dh-(dh//2)
    left,right = dw//2,dw-(dw//2)

    color = [0,0,0]
    return cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=color)

def load_images(data_dir,resize=None,seq_len=None):
    data_desc = pd.read_csv(data_dir+"labelsImgPath.csv",sep=",")
    img_list = []
    shapes = []
    for i in data_desc['filename']:
        img_raw = cv2.imread(data_dir+i,cv2.IMREAD_COLOR)
        
        img_list.append(img_raw)
        shapes.append((img_raw.shape[0],img_raw.shape[1]))


    shapes = np.array(shapes)
    if(resize==None):
        resize = np.int8((np.mean(shapes[:,0])+np.mean(shapes[:,1]))/2.0)
    res_imgs = []
    for img in img_list:
        nimg = cv_resize_pad(img,resize)
        # More augmentation 
        res_imgs.append(nimg)

    images = np.array(res_imgs)
    
    g_images,labels = generate_seqs(images,data_desc)
    g_images = tf.keras.preprocessing.sequence.pad_sequences(g_images,maxlen=seq_len,dtype='float32')
    return g_images,labels

def generate_seqs(images,data_desc,onehot_lab=True):
    idx = []
    runn_idx = 0
    img_seqs = []
    labels = []
    label = None
    tid = 0 
    for _,row in data_desc.iterrows():
        if(tid != row['trackid']):
            if(len(idx)!=0):
                idx = list(map(lambda x: x+runn_idx,idx))
                img_seqs.append(np.array(images[idx]))
                labels.append(label)
                runn_idx = runn_idx + len(idx)
            
            tid = row['trackid']
            idx = [row['framenr']-2] #TODO
        else:
            idx.append(row['framenr']-2)
        label = row['class']
    if(onehot_lab):
        labels = onehot(labels,label_dict={'boat':1,'nature':0})
    return img_seqs,labels

def make_img_from_tensor(np_ary):
    img_list=[]
    for i in range(0,20):
        img_list.append(Image.fromarray(np_ary[i,:,:],'RGB'))
    return img_list

def augment(np_ary):
    transformed_list = []
    img_list = make_img_from_tensor(np_ary)
    policy = CIFAR10Policy()
    policy.draw()
    for img in img_list:
        transformed_list.append(np.array(policy(img)))
    return transformed_list 

def run(data_dir,batchsize=50, n_epochs=50):

    tf.reset_default_graph()
    train_dir = data_dir +'train/'
    test_dir = data_dir +'test/'
    train_imgs,train_labels = load_images(train_dir)
    test_imgs,test_labels = load_images(test_dir,resize=train_imgs.shape[2],seq_len=train_imgs.shape[1])
    n_samples = train_imgs.shape[0]

    #train_dataset = tf.data.Dataset.from_tensor_slices((train_imgs,train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_imgs,test_labels))
    train_img = tf.data.Dataset.from_tensor_slices(train_imgs)
    train_labels = tf.data.Dataset.from_tensor_slices(train_labels)

    train_dataset = tf.data.Dataset.zip((train_img, train_labels))
    train_dataset = train_dataset.shuffle(buffer_size=100,reshuffle_each_iteration=True).batch(batchsize).repeat()
    test_dataset = test_dataset.shuffle(buffer_size=100,reshuffle_each_iteration=True).batch(batchsize)
    
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
    x,y = iterator.get_next()
    train_iterator = iterator.make_initializer(train_dataset)
    test_iterator = iterator.make_initializer(test_dataset)
    # apply random augmentations
    ft_extr = feature_extractor()
    logits = ft_extr.create_3dconv_model(x)

    
    #loss = weighted_ce(next_element[1],model,.1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
    prediction = tf.nn.softmax(logits)
    #cnf_matrix = tf.math.confusion_matrix(predictions=tf.to_float(tf.argmax(prediction,1)),labels=tf.to_float(tf.argmax(y, 1)),num_classes=2)
    equality = tf.equal(tf.to_float(tf.argmax(prediction,1)), tf.to_float(tf.argmax(y, 1)))    
    accuracy = tf.reduce_mean(tf.to_float(equality))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(loss)
      

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_iterator)
        saver = tf.train.Saver()

        for epoch in range(n_epochs):
            ep_loss = []
            ep_cnf = []
            for _ in range(int(n_samples/batchsize)):
                _,b_loss = sess.run([optimizer,loss],feed_dict={'is_training:0':True})  #cnf_mat cnf_matrix
                ep_loss.append(b_loss)
                #ep_cnf.append(cnf_mat)

            print(np.mean(ep_loss))
            print(np.mean(ep_cnf,axis=0))
            if(n_epochs%10==0):
                save_path = saver.save(sess,data_dir+str(epoch)+"_checkpoint.ckpt")
    
        print('predicting..')
        save_path = saver.save(sess,data_dir+"final_checkpoint.ckpt")

        with tf.SessionL() as sess:
            sess.run(test_iterator)
            result_set = []
            try:
                while True:
                    pred = sess.run(prediction,feed_dict={'is_training:0':False})
                    result_set.append(pred)
            except:
                pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, default="/fast/AG_Kainmueller/jrumber/Hackathon/data/") 

    args = parser.parse_args()
    data_dir = args.__dict__['dir']
    tf.reset_default_graph()
    run(data_dir,batchsize=50, n_epochs=200)
    print("done")
