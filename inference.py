import sys
import cv2
import numpy as np
import os
import keras.backend as K
from keras.backend import tf as ktf
import tensorflow as tf
from keras import layers
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, ZeroPadding2D, MaxPooling2D, Concatenate, \
    Reshape, Lambda,Dropout, Activation
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.utils import plot_model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint

from keras.layers.merge import Concatenate, Add
from keras.optimizers import SGD

import keras_segmentation
from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.data_utils.data_loader import get_image_arr , get_segmentation_arr
import random

from keras_segmentation import train 
from keras_segmentation import predict

import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import glob
import itertools
from tqdm import tqdm
from PIL import Image

n_classes =3

input_path = "/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/input/"
final_prediction_path = "/mnt/disk3/rohit2/bhomik_work/flixstock/Solution/Fused_net_prediction_2/" 

list_of_images = os.listdir(input_path)
total_num_of_images = len(list_of_images)

full_path_raw_images = [os.path.join(input_path,name) for i,name in enumerate(list_of_images)]

def fused_model():

    # input_tensor = Input(shape=(473, 473, 3))

    tnet = keras_segmentation.models.pspnet.pspnet_50(n_classes=3, input_height=473, input_width=473)

    input_tensor = tnet.input

    input = Lambda(lambda i: i[:, :, :, 0:3])(input_tensor)

    x = Reshape((tnet.output_height,tnet.output_width , n_classes))(tnet.output)

    # x = Concatenate(axis=3)([input, tnet.get_layer('interp_5').output])
    x = Concatenate(axis=3)([input, x])

    # x = tf.image.resize_images(x,(320,320))
    x = Lambda(lambda image: ktf.image.resize_images(image, (320,320)))(x)
    


    # 1. Conv+ReLu
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv1_1')(x)   # First convolutional layers
    x = BatchNormalization()(x)
    
    # 2. Conv+ReLu
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv1_2')(x)
    x = BatchNormalization()(x)

    orig_1 = x
    
    # 3. MaxPool
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # 4. Conv+ReLu
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu', name='conv2_1')(x)
    x = BatchNormalization()(x)

     # 5. Conv+ReLu
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu', name='conv2_2')(x)
    x = BatchNormalization()(x)

    orig_2 = x

    #6. MaxPool
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    #7. Conv+ReLu
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_1')(x)
    x = BatchNormalization()(x)

    #8. Conv+ReLu
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_2')(x)
    x = BatchNormalization()(x)

    #9. Conv+ReLu
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_3')(x)
    x = BatchNormalization()(x)

    orig_3 = x

    #10. MaxPool
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    #11. Conv+ReLu
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_1')(x)
    x = BatchNormalization()(x)

    #12. Conv+ReLu
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_2')(x)
    x = BatchNormalization()(x)

    #13. Conv+ReLu
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_3')(x)
    x = BatchNormalization()(x)

    orig_4 = x
    
    #14. MaxPool
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    #15. Conv+ReLu
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_1')(x)
    x = BatchNormalization()(x)

    #16. Conv+ReLU
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_2')(x)
    x = BatchNormalization()(x)

    #17. Conv+ReLU
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv5_3')(x)
    x = BatchNormalization()(x)

    orig_5 = x

    
    # Decoder
    x = Conv2D(512, (5,5), activation='relu', padding = 'same', name='deconv5',kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(256, (5,5), activation='relu', padding = 'same', name='deconv4',kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(128, (5,5), activation='relu', padding = 'same', name='deconv3',kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2,2))(x)
    x = Conv2D(64, (5,5), activation='relu', padding = 'same', name='deconv2',kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D((2,2))(x)
    x = Conv2D(64, (5,5), activation='relu', padding = 'same', name='deconv1',kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)

            
    x = Conv2D(1, (5, 5), activation='relu', padding='same', name='Raw_Alpha_Pred', kernel_initializer='he_normal',
               bias_initializer='zeros')(x)

    # x = tf.image.resize_images(x,(473,473))
    
    fused_model = Model(inputs=tnet.input, outputs=x)
    
    # fused_model = Model(inputs=tnet.input, outputs=x)

    fused_model.summary()
    
    return fused_model




# def infer():



if __name__ == '__main__':
    model_combined = fused_model()
    model_combined.summary()
    print ('Architecture of Fused Model done')
    # model_combined.load_weights("/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/fused_best_model.hdf5")
    model_combined.load_weights("/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/pspnet_training_2_checkpoint/pretrained_tnet.hdf5", 
        by_name=True)
    model_combined.load_weights("/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/mnet_checkpoint_loc/best_model.hdf5", 
        by_name=True)


    input_w = 473
    input_h = 473

    output_w = 320
    output_h = 320


    for j in range(total_num_of_images):

        sample_img = cv2.imread(full_path_raw_images[j],1)

        #Read color image
        # sample_img = cv2.imread("/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/input/GVRRC519@7656571@f10.jpg",1)
        height, width, channels = sample_img.shape
        # prediction = np.zeros( (height,width ))
        
        #resize it to 473 x 473
        sample_img_resized = cv2.resize(sample_img, (input_w,input_h), interpolation=cv2.INTER_NEAREST)
        sample_img_resized = sample_img_resized.astype(np.float32)

        # bgr to rgb conversion
        sample_img_resized = sample_img_resized[ : , : , ::-1 ]
        rgb_array = np.array([sample_img_resized])

        pred = model_combined.predict(rgb_array)[0]   # pred is of 320 x 320

        #if flattened
        pred = pred.reshape(( output_h ,  output_w ) )

        seg_img = np.zeros( (output_w,output_h) )
        seg_img[:,:] = pred[:,:]

        # print (np.unique(pred))
        print ('max and min : ', np.max(pred), np.min(pred))
        norm_image = cv2.normalize(seg_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        prediction = cv2.resize(norm_image, (width,height), interpolation=cv2.INTER_NEAREST )

        save_path = final_prediction_path + full_path_raw_images[j].rsplit("/",1)[-1].split(".")[0]+'.png'
        print (save_path)
        cv2.imwrite(save_path, prediction)
