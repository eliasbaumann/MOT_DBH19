import numpy as np
import tensorflow as tf
import pandas as pd
import cv2

from mpi4py import MPI

from model import feature_extractor
from utils import sparse_cost_sensitive_loss,onehot,weighted_ce

import xgboost as xgb

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.imagenet_utils import preprocess_input

# from feature_extractor import Feature_extractor


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

        img_raw = cv2.imread(data_dir+i,0) #cv2.IMREAD_COLOR
        # img_raw = cv2.Laplacian(img_raw,cv2.CV_64F)
        img_raw = cv2.normalize(img_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        img_list.append(img_raw)
        shapes.append((img_raw.shape[0],img_raw.shape[1]))


    shapes = np.array(shapes)
    if(resize==None):
        resize = np.int8((np.mean(shapes[:,0])+np.mean(shapes[:,1]))/2.0)
    res_imgs = []
    for img in img_list:
        nimg = cv_resize_pad(img,resize)
        
        # nimg = np.expand_dims(nimg,-1)
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

def fourier_transform(data,only_real=True):
    fft_data = np.fft.rfftn(data,axes=(1,2,3))
    fft_data = (fft_data - np.mean(fft_data))/np.std(fft_data)
    if(only_real):
        fft_data = np.real(fft_data)
    return fft_data

def surf_extract(img):
    #surf = cv2.xfeatures2d.SURF_create(400)
    #kp, des = surf.detectAndCompute(img, None)
    return None

def run_xgb(train,trainlab,test,testlab):
    train_set = xgb.DMatrix(data=train,label=trainlab)
    test_set = xgb.DMatrix(data=test,label=testlab)
    param = {'max_depth': 5, 'eta': .1, 'objective': 'binary:logistic','nthread':6,'eval_metric':'aucpr','predictor':'gpu_predictor','colsample_bytree':.6}
    num_round = 200
    evallist = [(test_set, 'eval'), (train_set, 'train')]
    bst = xgb.train(param, train_set, num_round,evallist,early_stopping_rounds=50)
    preds = bst.predict(test_set)
    return preds


def apply_pretrained_model(train_imgs,test_imgs):
    resnet = MobileNetV2(weights='imagenet',pooling = max, include_top = False,input_shape=train_imgs.shape[2:])
    
    reshaped_imgs = np.reshape(train_imgs,(-1,)+train_imgs.shape[2:])
    reshaped_t_imgs = np.reshape(test_imgs,(-1,)+test_imgs.shape[2:])

    train_imgs_pp = preprocess_input(reshaped_imgs)
    test_imgs_pp = preprocess_input(reshaped_t_imgs)
    
    train_imgs_feat = resnet.predict(train_imgs_pp)
    test_imgs_feat = resnet.predict(test_imgs_pp)
    
    train_imgs_feat = np.reshape(train_imgs_feat,(train_imgs.shape[:2]+train_imgs_feat.shape[1:]))
    test_imgs_feat = np.reshape(test_imgs_feat,(test_imgs.shape[:2]+test_imgs_feat.shape[1:]))
    return train_imgs_feat,test_imgs_feat

def train_with_tf(train_imgs,train_labels,test_imgs,test_labels,n_epochs=10,batchsize=50):

    # additional_train_imgs = np.repeat(train_imgs[:122,:],4,axis=0)
    # additional_train_labels = np.repeat(train_labels[:122,:],4,axis=0)

    # train_imgs = np.concatenate((train_imgs,additional_train_imgs),axis=0)
    # train_labels = np.concatenate((train_labels,additional_train_labels),axis=0)
    
    # Fourier transform
    # fft_train = fourier_transform(train_imgs)

    # Random cnn features

    # train_imgs,test_imgs = apply_pretrained_model(train_imgs,test_imgs)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_imgs,train_labels))
    valid_dataset = tf.data.Dataset.from_tensor_slices((train_imgs,train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_imgs,test_labels))
    
    n_samples = train_imgs.shape[0]
    
    train_dataset = train_dataset.shuffle(buffer_size=100,reshuffle_each_iteration=True).batch(batchsize).repeat().prefetch(50)
    valid_dataset = valid_dataset.batch(batchsize)
    test_dataset = test_dataset.batch(batchsize)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
    next_element = iterator.get_next()

    train_iterator = iterator.make_initializer(train_dataset)
    valid_iterator = iterator.make_initializer(valid_dataset)
    test_iterator = iterator.make_initializer(test_dataset)

    ft_extr = feature_extractor()

    # abandoned as well:
    #model = ft_extr.create_lstm_model(next_element[0])
    #model,pre_fc = ft_extr.small_model(next_element[0])
    model,pre_fc = ft_extr.create_3dconv_model(next_element[0])

    # abandoned approaches
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=next_element[1]))
    #loss = sparse_cost_sensitive_loss(model,next_element[1],[[1.,1.],[1.,1.]]) #TODO label to onehot
    loss = weighted_ce(next_element[1],model,.1)

    prediction = tf.nn.softmax(model)
    
    cnf_matrix = tf.math.confusion_matrix(predictions=tf.to_float(tf.argmax(prediction,1)),labels=tf.to_float(tf.argmax(next_element[1], 1)),num_classes=2)
    equality = tf.equal(tf.to_float(tf.argmax(prediction,1)), tf.to_float(tf.argmax(next_element[1], 1)))    
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
            for _ in range(int(n_samples/batchsize+.5)):
                _,b_loss,cnf_mat = sess.run([optimizer,loss,cnf_matrix],feed_dict={'is_training:0':True})
                ep_loss.append(b_loss)
                ep_cnf.append(cnf_mat)

            print(np.mean(ep_loss))
            print(np.mean(ep_cnf,axis=0))
            if(n_epochs%10==0):
                save_path = saver.save(sess,data_dir+str(epoch)+"_checkpoint.ckpt")

        encoded_train=[]
        sess.run(valid_iterator)
        try:
            while True:
                enc_pred = sess.run(pre_fc,feed_dict={'is_training:0':False})
                encoded_train.append(enc_pred)
        except:
            pass
        
            
        
        encoded_train = np.vstack(np.array(encoded_train))

        print('predicting..')
        save_path = saver.save(sess,data_dir+"final_checkpoint.ckpt")
        sess.run(test_iterator)
        result_set = []
        encoded_test = []
        try:
            while True:
                pred,pre_feats = sess.run([prediction,pre_fc],feed_dict={'is_training:0':False})
                result_set.append(pred)
                encoded_test.append(pre_feats)
        except:
            pass
        # range
        # sess.run(optimizer,cost)
    # training is false and true sometimes
        
    # save_path = saver.save(sess, "path.ckpt")
    # saver.restore(sess, "path.ckpt")
    encoded_test = np.vstack(np.array(encoded_test))
    result_set = np.vstack(np.array(result_set))

    return encoded_train,encoded_test,result_set

def covert_to_max

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, default="C:/Users/Elias/Desktop/DB19/Hackathon/MultiFrame_propolsTrackingClassification/") 

    args = parser.parse_args()
    data_dir = args.__dict__['dir']
    train_dir = data_dir +'train/'
    test_dir = data_dir +'test/'
    train_imgs,train_labels = load_images(train_dir,resize=4,seq_len=5)
    test_imgs,test_labels = load_images(test_dir,resize=train_imgs.shape[2],seq_len=train_imgs.shape[1])
    
   

    encoded_train,encoded_test,result_set = train_with_tf(train_imgs,train_labels,test_imgs,test_labels,50,50)

    # results2 = run_xgb(encoded_train,train_labels,encoded_test,test_labels)
    print(train_imgs.shape)
    results2 = run_xgb(np.reshape(train_imgs,(train_imgs.shape[0],-1)),train_labels,np.reshape(test_imgs,(test_imgs.shape[0],-1)),test_labels)
    results2 = np.array([int(i+.5) for i in results2])

    from time import gmtime, strftime

    np.savetxt(data_dir+"xgb_preds.csv",results2,delimiter=",")
    
    # np.savetxt(data_dir+"preds.csv", result_set, delimiter=",")
    print("done")
    
    
    
    
 

           

        
        
    
        
    #for i in img_list:

    

        
        
    #cv2.cvtColor(csv_dir)

    # with tf.Session() as sess:
    #     asf = Feature_extractor()
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank() # debatable wether i want to think about mpi now..., maybe if it works, put in the effort to make it run on mpi