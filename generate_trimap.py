import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

flixstock_data_dir = '/home/hitech/flixstock/shm_data/'

#Number for pixels for dilation purpose for creation of trimaps
pixels_for_trimap = [15,21]

images_dir = os.path.join(flixstock_data_dir, 'input')
mask_dir = os.path.join(flixstock_data_dir, 'mask')
# trimap_dir = os.path.join(flixstock_data_dir, 'trimap_labels')
trimap_dir = os.path.join(flixstock_data_dir, '3channel_trimap')

# new_GT_dir = os.path.join(flixstock_data_dir, 'GT_labels_mask')


if not (os.path.exists(trimap_dir)):
	print ('Creating Trimap Folder')
	os.mkdir(trimap_dir)
	if ((os.path.exists(trimap_dir))):
		print ('Trimap Folder created')
	for pix in pixels_for_trimap:
		os.mkdir(os.path.join(trimap_dir,str(pix)))

aplha_matte_list = os.listdir(mask_dir)
full_path_alpha_matte_GT = [os.path.join(mask_dir, name) for i,name in enumerate(aplha_matte_list)]


# def create_gt(img):
# 	width, height = img.shape
# 	gt = np.ones((width,height),dtype=np.uint8)
# 	# print (np.unique(img))
# 	# cv2.namedWindow("image",0)
# 	# cv2.imshow("image",img)
# 	# cv2.waitKey(0)
# 	gt[img==0]=0
# 	gt[img==255]=2
# 	# print (np.unique(gt))
# 	# plt.imshow(img)
# 	# plt.show()
# 	return gt

def create_trimap(img,pix,struc="None"):
	image = img/255
	size =  2*pix+1;
	kernel = np.ones((size,size), np.uint8)
	dilated = cv2.dilate(image, kernel, iterations = 1)*255
	eroded = cv2.erode(image, kernel, iterations=1)*255
	res = dilated.copy()
	# print (image.shape)
	width,height = image.shape
	foreground = np.zeros((width,height,1),dtype=np.uint8)
	background = np.zeros((width,height,1),dtype=np.uint8)
	uncertain = np.zeros((width,height,1),dtype=np.uint8)
	
	foreground[((dilated==255)&(eroded==255))]= 255
	background[((dilated==0)&(eroded==0))] = 255
	uncertain[((dilated==255)&(eroded==0))] = 255

	# n_fg_pix = np.sum(foreground == 1)
	# # print('Number of fg pixels:', n_fg_pix)

	# n_bg_pix = np.sum(background == 1)
	# # print('Number of fg pixels:', n_bg_pix)

	# n_uncertain_pix = np.sum(uncertain == 1)
	# # print('Number of fg pixels:', n_uncertain_pix)
	# assert (n_fg_pix+n_bg_pix+n_uncertain_pix) == width*height

	trimap = cv2.merge((background,uncertain,foreground))
	# print ('trimap shape is', trimap.shape, type(trimap))
	# print (np.unique(trimap[:,]))
	# cv2.namedWindow("trimap",0)
	# cv2.imshow("trimap",trimap[:,:,0])
	# cv2.waitKey(0)
	return trimap




for pix in pixels_for_trimap:
	for i,image in enumerate(full_path_alpha_matte_GT):
		img = cv2.imread(image,-1) # Reading Gray
		print (img.shape)
		if (len(img.shape)!=2):
			print("Error: Mask image not correct")
			sys.exit()
		trimap = create_trimap(img,pix)
		# cv2.namedWindow("image",0)
		# cv2.imshow("image",f)
		# cv2.waitKey(10000)
		trimap_path = trimap_dir + "/" + str(pix) + "/" + image.split('/')[-1]
		# cv2.imwrite(trimap_path,trimap)





