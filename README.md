# SemanticHumanMatting
This code provides the implementation of Semantic Human Matting paper (by Alibaba)

Instructions/Points while implementation :

The paper takes two networks -
1.) Semantic Segmentation Network (E.g. Pspnet-50, Unet etc.)
2.) Encoder Decode Network for Image Matting problem

inference.py contains the code for loading the model and its weights from a given location (model location) and runs on a given set of RGB images. It stores the result in a folder provided in the code itself. Please change the location of input path and output path in inference.py to run the code.

