# SemanticHumanMatting
This code provides the implementation of Semantic Human Matting paper (by Alibaba)

Instructions/Points while implementation :

The paper takes two networks -
1.) Semantic Segmentation Network (E.g. Pspnet-50, Unet etc.)
2.) Encoder Decode Network for Image Matting problem

inference.py contains the code for loading the model and its weights from a given location (model location) and runs on a given set of RGB images. It stores the result in a folder provided in the code itself. Please change the location of input path and output path in inference.py to run the code.

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
pretrained_tnet.hdf5


