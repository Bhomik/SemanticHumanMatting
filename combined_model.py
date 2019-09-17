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
from keras.optimizers import Adam

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


random.seed(0)
IMAGE_ORDERING = 'channels_last'
# from custom_layers.unpooling_layer import Unpooling

#DIM 
#Encoder has 14 convolutional layers and 5 MaxPool layers.
#Decoder has 6 convolutional layers and 5 unpooling layers/

n_classes = 3
IMAGE_ORDERING = 'channels_last'
input_path = "/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/input/"
mask_path = "/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/mask/"
# remapped_mask_path = "/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/GT_labels_mask/"

list_of_images = os.listdir(input_path)
total_num_of_images = len(list_of_images)

full_path_raw_images = [os.path.join(input_path,name) for i,name in enumerate(list_of_images)]
full_path_mask_images = [os.path.join(mask_path, (name.rsplit(".",1)[0]+'.png')) for i,name in enumerate(list_of_images)]
# full_path_remapped_mask_images = [os.path.join(remapped_mask_path, (name.rsplit(".",1)[0]+'.png')) for i,name in enumerate(list_of_images)]

img_width = 473
img_height = 473
num_channels = 3

batch_size = 2
n_classes =3 
input_height = 473 
input_width = 473
output_height = 320 
output_width = 320
do_augment=False





def alpha_loss(y_true, y_pred):
    diff = np.subtract(y_true, y_pred)
    squared_diff = np.square(diff)
    loss = np.sum(squared_diff)
    return loss

def compositional_loss(y_true, y_pred):
    y_true_R = y_true[:,:,0]
    y_true_G = y_true[:,:,1]
    y_true_B = y_true[:,:,2]

    y_pred_R = y_pred[:,:,0]
    y_pred_G = y_pred[:,:,1]
    y_pred_B = y_pred[:,:,2]

    diff_R = np.subtract(y_true_R, y_pred_R)
    diff_G = np.subtract(y_true_G, y_pred_G)
    diff_B = np.subtract(y_true_B, y_pred_B)

    squared_diff_R = np.square(diff_R)
    squared_diff_G = np.square(diff_G)
    squared_diff_B = np.square(diff_B)

    composite_loss = np.sum(squared_diff_R) + np.sum(squared_diff_G) + np.sum(squared_diff_B)

    return composite_loss


# calculation of loss function 
def total_loss(y_true, y_pred):
    gamma = 0.5
    alpha_prediction_loss = alpha_loss(alpha_matte, alpha_predicted)
    composite_loss = compositional_loss(raw_image, composition_image)
    loss_p = gamma*alpha_loss + (1-gamma)*(compositional_loss)
    total_loss = loss_p + factor*cross_entropy_loss
    return total_loss


class_colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(5000)  ]


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

    o = (Reshape((   320*320 , -1    )))(x)

    o = (Activation('softmax'))(o)

    # x = tf.image.resize_images(x,(473,473))
    
    fused_model = Model(inputs=tnet.input, outputs=o)
    
    # fused_model = Model(inputs=tnet.input, outputs=x)

    fused_model.summary()
    
    return fused_model

# def build_encoder_decoder(tnet):

#     # Encoder
#     # input_tensor = Input(tensor=inp, shape=(320, 320, 6))
#     # # input_tensor = inp
#     # print('shape in mnet: ' + str(K.int_shape(input_tensor)))

#     input_tensor = tnet.input

#     input = Lambda(lambda i: i[:, :, :, 0:3])(input_tensor)

#     x = Reshape((tnet.output_height,tnet.output_width , n_classes))(tnet.output)

#     # x = Concatenate(axis=3)([input, tnet.get_layer('interp_5').output])
#     x = Concatenate(axis=3)([input, x])

#     # x = tf.image.resize_images(x,(320,320))
#     x = Lambda(lambda image: ktf.image.resize_images(image, (320,320)))(x)

#     # 1. Conv+ReLu
#     x = ZeroPadding2D((1, 1))(x)
#     x = Conv2D(64, (3, 3), activation='relu', name='conv1_1')(x)   # First convolutional layers
#     x = BatchNormalization()(x)
    
#     # 2. Conv+ReLu
#     x = ZeroPadding2D((1, 1))(x)
#     x = Conv2D(64, (3, 3), activation='relu', name='conv1_2')(x)
#     x = BatchNormalization()(x)

#     orig_1 = x
    
#     # 3. MaxPool
#     x = MaxPooling2D((2, 2), strides=(2, 2))(x)

#     # 4. Conv+ReLu
#     x = ZeroPadding2D((1, 1))(x)
#     x = Conv2D(128, (3, 3), activation='relu', name='conv2_1')(x)
#     x = BatchNormalization()(x)

#      # 5. Conv+ReLu
#     x = ZeroPadding2D((1, 1))(x)
#     x = Conv2D(128, (3, 3), activation='relu', name='conv2_2')(x)
#     x = BatchNormalization()(x)

#     orig_2 = x

#     #6. MaxPool
#     x = MaxPooling2D((2, 2), strides=(2, 2))(x)

#     #7. Conv+ReLu
#     x = ZeroPadding2D((1, 1))(x)
#     x = Conv2D(256, (3, 3), activation='relu', name='conv3_1')(x)
#     x = BatchNormalization()(x)

#     #8. Conv+ReLu
#     x = ZeroPadding2D((1, 1))(x)
#     x = Conv2D(256, (3, 3), activation='relu', name='conv3_2')(x)
#     x = BatchNormalization()(x)

#     #9. Conv+ReLu
#     x = ZeroPadding2D((1, 1))(x)
#     x = Conv2D(256, (3, 3), activation='relu', name='conv3_3')(x)
#     x = BatchNormalization()(x)

#     orig_3 = x

#     #10. MaxPool
#     x = MaxPooling2D((2, 2), strides=(2, 2))(x)

#     #11. Conv+ReLu
#     x = ZeroPadding2D((1, 1))(x)
#     x = Conv2D(512, (3, 3), activation='relu', name='conv4_1')(x)
#     x = BatchNormalization()(x)

#     #12. Conv+ReLu
#     x = ZeroPadding2D((1, 1))(x)
#     x = Conv2D(512, (3, 3), activation='relu', name='conv4_2')(x)
#     x = BatchNormalization()(x)

#     #13. Conv+ReLu
#     x = ZeroPadding2D((1, 1))(x)
#     x = Conv2D(512, (3, 3), activation='relu', name='conv4_3')(x)
#     x = BatchNormalization()(x)

#     orig_4 = x
    
#     #14. MaxPool
#     x = MaxPooling2D((2, 2), strides=(2, 2))(x)

#     #15. Conv+ReLu
#     x = ZeroPadding2D((1, 1))(x)
#     x = Conv2D(512, (3, 3), activation='relu', name='conv5_1')(x)
#     x = BatchNormalization()(x)

#     #16. Conv+ReLU
#     x = ZeroPadding2D((1, 1))(x)
#     x = Conv2D(512, (3, 3), activation='relu', name='conv5_2')(x)
#     x = BatchNormalization()(x)

#     #17. Conv+ReLU
#     x = ZeroPadding2D((1, 1))(x)
#     x = Conv2D(512, (3, 3), activation='relu', name='conv5_3')(x)
#     x = BatchNormalization()(x)

#     orig_5 = x

    
#     # Decoder
#     x = Conv2D(512, (5,5), activation='relu', padding = 'same', name='deconv5',kernel_initializer='he_normal',
#                bias_initializer='zeros')(x)
#     x = BatchNormalization()(x)

#     x = UpSampling2D((2,2))(x)
#     x = Conv2D(256, (5,5), activation='relu', padding = 'same', name='deconv4',kernel_initializer='he_normal',
#                bias_initializer='zeros')(x)
#     x = BatchNormalization()(x)

#     x = UpSampling2D((2,2))(x)
#     x = Conv2D(128, (5,5), activation='relu', padding = 'same', name='deconv3',kernel_initializer='he_normal',
#                bias_initializer='zeros')(x)
#     x = BatchNormalization()(x)

#     x = UpSampling2D((2,2))(x)
#     x = Conv2D(64, (5,5), activation='relu', padding = 'same', name='deconv2',kernel_initializer='he_normal',
#                bias_initializer='zeros')(x)
#     x = BatchNormalization()(x)
    
#     x = UpSampling2D((2,2))(x)
#     x = Conv2D(64, (5,5), activation='relu', padding = 'same', name='deconv1',kernel_initializer='he_normal',
#                bias_initializer='zeros')(x)
#     x = BatchNormalization()(x)

            
#     x = Conv2D(1, (5, 5), activation='relu', padding='same', name='Raw_Alpha_Pred', kernel_initializer='he_normal',
#                bias_initializer='zeros')(x)

#     model = Model(inputs=input_tensor, outputs=x)
#     return model


# def build_TNet():
#     tnet = keras_segmentation.models.pspnet.pspnet_50(n_classes=3, input_height=473, input_width=473)
#     # tnet = keras_segmentation.models.pspnet.pspnet_50( n_classes=3)
#     # pspnet.summary()
    
#     return tnet
# def get_image_arr( path , width , height , imgNorm="sub_mean" , odering='channels_first' ):


#     if type( path ) is np.ndarray:
#         img = path
#     else:
#         img = cv2.imread(path, 1)

#     if imgNorm == "sub_and_divide":
#         img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
#     elif imgNorm == "sub_mean":
#         img = cv2.resize(img, ( width , height ))
#         img = img.astype(np.float32)
#         img[:,:,0] -= 103.939
#         img[:,:,1] -= 116.779
#         img[:,:,2] -= 123.68
#         img = img[ : , : , ::-1 ]
#     elif imgNorm == "divide":
#         img = cv2.resize(img, ( width , height ))
#         img = img.astype(np.float32)
#         img = img/255.0

#     if odering == 'channels_first':
#         img = np.rollaxis(img, 2, 0)
#     return img



def get_segmentation_arr( path , nClasses ,  width , height , no_reshape=False ):

    seg_labels = np.zeros((  height , width  , nClasses ))
        
    if type( path ) is np.ndarray:
        img = path
    else:
        img = cv2.imread(path, 1)

    img = cv2.resize(img, ( width , height ) , interpolation=cv2.INTER_NEAREST )
    img = img[:, : , 0]

    for c in range(nClasses):
        seg_labels[: , : , c ] = (img == c ).astype(int)


    
    if no_reshape:
        return seg_labels

    seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
    return seg_labels

def custom_semantic_segmentation_generator( images , segs ,  batch_size,  n_classes , 
            input_height , input_width , output_height , output_width  , 
            do_augment=False ):
    
    print ('inside semantic')

    img_seg_pairs = []

    for im in range(total_num_of_images):
        img_seg_pairs.append((images[im,:,:,:] , segs[im,:,:,:]) )

    zipped = itertools.cycle( img_seg_pairs  )

    while True:
        X = []
        Y = []
        for _ in range( batch_size) :
            im , seg = next(zipped) 
            if do_augment:
                img , seg[:,:,0] = augment_seg( img , seg[:,:,0] )

            X.append(im)
            seg = np.reshape(seg, ( output_width*output_height , 1 ))
            Y.append( seg  )
            # Y.append(seg)

        yield np.array(X) , np.array(Y)

if __name__ == '__main__':

    model_combined = fused_model()
    model_combined.summary()
    print ('Architecture of Fused Model done')

    #Pretrained Tnet weights
    model_combined.load_weights("/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/pspnet_training_2_checkpoint/pretrained_tnet.hdf5", 
        by_name=True)

    # Pretrained Mnet weights
    model_combined.load_weights("/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/mnet_checkpoint_loc/best_model.hdf5", 
        by_name=True)

    print ('loaded weights in fused model')

    model_combined.compile(loss=alpha_loss,
              optimizer = Adam(lr=0.001),
              metrics=['acc'])

    filepath = '/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/fused_best_model.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor="acc", verbose=1, save_best_only=True, mode="max")
    callbacks_list = [checkpoint]


    training_images = np.zeros((total_num_of_images,) + (img_width,img_height,num_channels))
    training_gt = np.zeros((total_num_of_images,) + (output_width,output_height,1))

    for j in range(total_num_of_images):

        raw_img = cv2.imread(full_path_raw_images[j],1)  # 3 channel bgr images
        raw_img_resized = cv2.resize(raw_img, (img_width,img_height), interpolation=cv2.INTER_NEAREST)
        raw_img_resized = raw_img_resized.astype(np.float32)
        raw_img_resized = raw_img_resized*1.0/255.0
        raw_img_resized = raw_img_resized[ : , : , ::-1 ]  # BGR to RGB conversion

        # raw_img_resized[:,:,0] -= 103.939
        # raw_img_resized[:,:,1] -= 116.779
        # raw_img_resized[:,:,2] -= 123.68

        mask_img = cv2.imread(full_path_mask_images[j],-1)    
        mask_img_resized = cv2.resize(mask_img, (output_width,output_height), interpolation=cv2.INTER_NEAREST)
        mask_img_resized = mask_img_resized*1.0/255.0
        mask_img_resized = mask_img_resized.reshape((output_width,output_height,1))

        #order is B G R Bs Us Fs
        training_images[j,:,:,0] = raw_img_resized[:,:,0]
        training_images[j,:,:,1] = raw_img_resized[:,:,1]
        training_images[j,:,:,2] = raw_img_resized[:,:,2]

        training_gt[j,:,:,:] = mask_img_resized


    train_gen = custom_semantic_segmentation_generator( training_images , training_gt, batch_size, n_classes, input_height,
                                    input_width , output_height , output_width, do_augment=False)

    history = model_combined.fit_generator(train_gen, steps_per_epoch=252, epochs=5, verbose=1, callbacks=callbacks_list)


    acc = history.history['acc']

    loss = history.history['loss']

    fig = plt.figure()
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train_loss'], loc='upper left')

    plt.show()
    fig.savefig('fused_loss.png', format = 'png')

    fig = plt.figure()
    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train_accuracy'], loc='upper left')

    plt.show()
    fig.savefig('fused_accuracy.png', format = 'png')








