import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import random
import re
import sys
from PIL import Image
import argparse
from keras.utils import np_utils
# import pspnet



import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping

import keras_segmentation
from keras_segmentation.models.model_utils import transfer_weights
# from model_utils import transfer_weights
from keras_segmentation import train 
from keras_segmentation import predict

plt.switch_backend('TkAgg')

keras.backend.set_image_data_format('channels_last')

parser = argparse.ArgumentParser()


parser.add_argument("command", type = str)
# parser.add_argument("--checkpoints_path", type = str  )
# parser.add_argument("--input_path", type = str , default = "")
# parser.add_argument("--output_path", type = str , default = "")
args = parser.parse_args()
command = sys.argv[1]


# Download this pretrained model (PSPNET-50 trained on ADE-20K dataset) and place it in .keras/datasets/
# https://www.dropbox.com/s/0uxn14y26jcui4v/pspnet50_ade20k.h5?dl=1:

#Load this model and transfer its weights to new model
pre_trained_psp_model = keras_segmentation.pretrained.pspnet_50_ADE_20K()
new_model = keras_segmentation.models.pspnet.pspnet_50( n_classes=3)
new_model.summary()
transfer_weights( pre_trained_psp_model , new_model) # transfer weights from pre-trained model to your model



input_path = "/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/input/"
trimap_path = "/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/trimap_labels/15/"
checkpoints_path = "/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/pspnet_training_3_checkpoint/"
output_path = "/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/predictions/"


if command == "train_tnet":
  
  # For pretraining on a given dataset, provide 

  # train images, 
  # annotations path (images with labels 0 (background) ,1 (uncertain) ,2 (foreground) )
  # checkpoints path - where model is saved after each epoch

  new_model.train( 
      train_images =  input_path,
      train_annotations = trimap_path,
      checkpoints_path = checkpoints_path, 
      epochs=30,
      # val_images="/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/val_frames" , 
      # val_annotations="/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/val_trimap_gt",
      n_classes=3 
  )

else:
    print("Invalid command " , command )

new_model.save_weights("/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/pspnet_training_2_checkpoint/pretrained_tnet.hdf5")
new_model.load_weights("/mnt/disk3/rohit2/bhomik_work/flixstock/shm_data/pspnet_training_2_checkpoint/pretrained_tnet.hdf5")

  #Predict the output of TNET
if ".jpg" in input_path or ".png" in input_path or ".jpeg" in input_path:
  print ("Predicting on an image")
  new_model.predict( inp=input_path, out_fname=output_path  , checkpoints_path=checkpoints_path  )
else:
  print ("Predicting on a folder")
  new_model.predict_multiple( inp_dir=input_path , out_dir=output_path  , checkpoints_path=checkpoints_path   )