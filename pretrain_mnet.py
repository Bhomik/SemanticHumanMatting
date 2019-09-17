import cv2
import sys
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

from keras.layers.merge import Concatenate, Add
from keras.optimizers import SGD

import keras_segmentation
from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.data_utils.data_loader import get_image_arr , get_segmentation_arr
import random

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from keras_segmentation import train 
from keras_segmentation import predict

import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import glob
import itertools
from tqdm import tqdm
from PIL import Image

# input_height = 473 
# input_width = 473

img_width = 320
img_height = 320
num_channels = 6

batch_size = 4
n_classes = 3 
input_height = 320 
input_width = 320
output_height = 320 
output_width = 320
do_augment=False

random.seed(0)
class_colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(5000)  ]

IMAGE_ORDERING = 'channels_last'


input_path = "/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/input/"
trimap_location_path = "/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/3channel_trimap/15/"
mask_path = "/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/mask/"
# remapped_mask_path = "/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/GT_labels_mask/"

list_of_images = os.listdir(input_path)
list_of_trimaps = os.listdir(trimap_location_path)

total_num_of_images = len(list_of_images)

full_path_raw_images = [os.path.join(input_path,name) for i,name in enumerate(list_of_images)]
full_path_trimap_images = [os.path.join(trimap_location_path, (name.rsplit(".",1)[0]+'.png') ) for i,name in enumerate(list_of_images)]
full_path_mask_images = [os.path.join(mask_path, (name.rsplit(".",1)[0]+'.png')) for i,name in enumerate(list_of_images)]
# full_path_remapped_mask_images = [os.path.join(remapped_mask_path, (name.rsplit(".",1)[0]+'.png')) for i,name in enumerate(list_of_images)]



def buildmnet():

    # Encoder

    input_tensor = Input(shape=(input_height, input_width, 6))
    # input_tensor = Input(tensor=inp, shape=(320, 320, 6))

    print('shape in mnet: ' + str(K.int_shape(input_tensor)))

    # 1. Conv+ReLu
    x = ZeroPadding2D((1, 1))(input_tensor)
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

 #    o = (Reshape((   output_height*output_width , -1    )))(o)

	# o = (Activation('softmax'))(o)

    model = Model(inputs=input_tensor, outputs=x)
    return model


def demo_sample():
	mnet = buildmnet()
	mnet.summary()
	mnet.load_weights("/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/mnet_checkpoint_loc/best_model.hdf5")

	sample_img = cv2.imread("/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/input/GVRRC519@7656571@f10.jpg",1)
	sample_img_resized = cv2.resize(sample_img, (320,320), interpolation=cv2.INTER_NEAREST)

	# w,h = sample_img.shape
	print (sample_img.shape)
	plt.imshow(sample_img)
	plt.show()

	sample_trimap_img = cv2.imread("/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/3channel_trimap/15/GVRRC519@7656571@f10.png",1)
	sample_trimap_img_resized = cv2.resize(sample_trimap_img, (320,320), interpolation=cv2.INTER_NEAREST)

	sample_img_resized = sample_img_resized.astype(np.float32)
	# sample_img_resized[:,:,0] -= 103.939
	# sample_img_resized[:,:,1] -= 116.779
	# sample_img_resized[:,:,2] -= 123.68
	sample_img_resized = sample_img_resized[ : , : , ::-1 ]

	rgb_array = np.array([sample_img_resized])
	trimap_array = np.array([sample_trimap_img_resized])
	print (rgb_array.shape)
	print (trimap_array.shape)

	final_array = np.concatenate((rgb_array,trimap_array),axis=3)
	print (final_array.shape)

	pred = mnet.predict(final_array)[0]
	pred = pred
	seg_img = np.zeros( (320,320) )
	seg_img[:,:] = pred[:,:,0]
	# seg_img = cv2.resize(seg_img, (w,h), interpolation=cv2.INTER_NEAREST)
	print (np.unique(pred))
	print ('max and min : ', np.max(pred), np.min(pred))
	norm_image = cv2.normalize(seg_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype = cv2.CV_32F)
	print (np.unique(norm_image))
	print ('max and min : ', np.max(norm_image), np.min(norm_image))
	plt.imshow(seg_img)
	plt.show()

# demo_sample()

# sys.exit()


def alpha_loss(y_true, y_pred):
    diff = np.subtract(y_true, y_pred)
    squared_diff = np.square(diff)
    loss = np.sum(squared_diff)
    print ('here inside alpha loss')
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
			Y.append(seg)
		yield np.array(X) , np.array(Y)



X = np.zeros((total_num_of_images,) + (img_width,img_height,num_channels))
Y = np.zeros((total_num_of_images,) + (output_width,output_height,1))


for j in range(total_num_of_images):
	# print (full_path_raw_images[j])
	# print (full_path_trimap_images[j])
	# print (full_path_mask_images[j])
	raw_img = cv2.imread(full_path_raw_images[j],1)
	raw_img_resized = cv2.resize(raw_img, (img_width,img_height), interpolation=cv2.INTER_NEAREST)
	raw_img_resized = raw_img_resized.astype(np.float32)
	# raw_img_resized[:,:,0] -= 103.939
	# raw_img_resized[:,:,1] -= 116.779
	# raw_img_resized[:,:,2] -= 123.68
	raw_img_resized = raw_img_resized[ : , : , ::-1 ]
	# raw_img_resized = raw_img_resized*1.0/255.0

	trimap_img = cv2.imread(full_path_trimap_images[j],1)
	trimap_img_resized = cv2.resize(trimap_img, (img_width,img_height), interpolation=cv2.INTER_NEAREST)


	mask_img = cv2.imread(full_path_mask_images[j],-1)
	mask_img_resized = cv2.resize(mask_img, (output_width,output_height), interpolation=cv2.INTER_NEAREST)
	mask_img_resized = mask_img_resized*1.0/255.0
	mask_img_resized = mask_img_resized.reshape((output_width,output_height,1))

	#order is B G R Bs Us Fs
	X[j,:,:,0] = raw_img_resized[:,:,0]
	X[j,:,:,1] = raw_img_resized[:,:,1]
	X[j,:,:,2] = raw_img_resized[:,:,2]

	X[j,:,:,3] = trimap_img_resized[:,:,0]
	X[j,:,:,4] = trimap_img_resized[:,:,1]
	X[j,:,:,5] = trimap_img_resized[:,:,2]

	Y[j,:,:,:] = mask_img_resized


print ('finished with reading input+trimap and corresponding GT')

train_gen = custom_semantic_segmentation_generator( X , Y, batch_size, n_classes, input_height,
									input_width , output_height , output_width, do_augment=False)

mnet = buildmnet()
mnet.summary()

mnet.compile(loss=alpha_loss,
              optimizer = 'adadelta',
              metrics=['acc'])

filepath = '/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/mnet_checkpoint_loc/best_model.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor="acc", verbose=1, save_best_only=True, mode="max")
callbacks_list = [checkpoint]
history = mnet.fit_generator(
      train_gen,
      steps_per_epoch=126,  
      epochs=30,
      verbose=1,
      callbacks=callbacks_list)

acc = history.history['acc']

loss = history.history['loss']

fig = plt.figure()
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train_loss'], loc='upper left')

plt.show()
fig.savefig('/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/mnet_checkpoint_loc/loss.png', format = 'png')

fig = plt.figure()
plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('Epoch')
plt.legend(['Train_accuracy'], loc='upper left')

plt.show()
fig.savefig('/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/mnet_checkpoint_loc/accuracy.png', format = 'png')
