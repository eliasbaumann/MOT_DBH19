import numpy as np
import tensorflow as tf
import pandas as pd
import cv2

from mpi4py import MPI

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
        img_raw = cv2.imread(data_dir+i,cv2.IMREAD_COLOR)
        
        img_list.append(img_raw)
        shapes.append((img_raw.shape[0],img_raw.shape[1]))


    shapes = np.array(shapes)
    if(resize==None):
        resize = np.int8((np.mean(shapes[:,0])+np.mean(shapes[:,1]))/2.0)
    res_imgs = []
    for img in img_list:
        nimg = cv_resize_pad(img,resize)
        res_imgs.append(nimg)

    images = np.array(res_imgs)
    g_images,labels = generate_seqs(images,data_desc)
    g_images = tf.keras.preprocessing.sequence.pad_sequences(g_images,maxlen=seq_len)
    return g_images,labels

def generate_seqs(images,data_desc):
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
    return img_seqs,labels


    




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, default="C:/Users/Elias/Desktop/DB19/Hackathon/MultiFrame_propolsTrackingClassification/") 

    args = parser.parse_args()
    data_dir = args.__dict__['dir']
    train_dir = data_dir +'train/'
    test_dir = data_dir +'test/'
    train_imgs,train_labels = load_images(train_dir)
    test_imgs,test_labels = load_images(test_dir,resize=train_imgs.shape[2],seq_len=train_imgs.shape[1])

    train_dataset = tf.data.Dataset.from_tensor_slices((train_imgs,train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_imgs,test_labels))
    
    print("done")
    
    
    
    
 

           

        
        
    
        
    #for i in img_list:

    

        
        
    #cv2.cvtColor(csv_dir)

    # with tf.Session() as sess:
    #     asf = Feature_extractor()
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank() # debatable wether i want to think about mpi now..., maybe if it works, put in the effort to make it run on mpi