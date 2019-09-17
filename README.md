# SemanticHumanMatting
This code provides the implementation of Semantic Human Matting paper (by Alibaba)

Instructions/Points while implementation :

The paper takes two networks -
1.) Semantic Segmentation Network (E.g. Pspnet-50, Unet etc.)
2.) Encoder Decode Network for Image Matting problem

inference.py contains the code for loading the model and its weights from a given location (model location) and runs on a given set of RGB images. It stores the result in a folder provided in the code itself. Please change the location of input path and output path in inference.py to run the code.

Semantic Segmentation Network

1.) Pretrain the PSPNET-50 on your dataset or you can also pretrain it on a big dataset of yours usage: 

I have taken already available PSPNET-50 pretrained on ADE-20K dataset from this location - 
https://www.dropbox.com/s/0uxn14y26jcui4v/pspnet50_ade20k.h5?dl=1:
Please download it and place it in .keras/datasets/pspnet50_ade20k.h5

The architecture of pspnet-50 has been imported from https://github.com/qubvel/segmentation_models
Transfer the weights from .keras/datasets/pspnet50_ade20k.h5 to the model architecture defined by you.

Now, in order to pretrain this model on the dataset (for e.g. 501 images for image matting problem),
run the command-

for training  --> python pretrain_tnet.py train_tnet
train_images is the location of raw images
train_annotations is the location of trimaps (by dilation) created by me from the masks.
epochs can be set by the user. I have set to 30

The model weights are located in this file -  
pretrained_tnet.hdf5, the model cannot be shared as its too large, however I can share on google drive.

For just prediction and no training, run the command
python pretrain_tnet.py x where x can be anything apart from "train_tnet"

The network takes 473 x 473 x 3 image as input and returns a 3 channel trimap of shape (223729x3) which is reshaped to 473x473x3 and concatenated to raw image.

The 3 channel image with 3 channel trimap is then sent to Matting network (encoder-decoder architecture as mentioned in paper) with encoder similar to VGG except dense layers.


MATTING NETWORK 

6 channel input is implemented.
Batch normalization has been applied after each conv layer as mentioned in paper.
Conv6 and deconv6 have been removed
encoder has 13 conv layers and 4 max pool layers
decoder has 6 conv layers and 4 unpool layers

For pretraining this network, python command is --> python pretrain_mnet.py

custom image datagenerator has been implemented for loading 6 channel input data and mask image has been rescaled from (0-255) to (0-1). The order of 6 channels are Red, Green, Blue, Background, Uncertain, Foreground.

custom loss function called alpha loss is implemented to calculate the difference between Ground Truth Mask and Predicted Mask.





