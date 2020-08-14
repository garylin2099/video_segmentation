import os
from math import *
import cv2
from skimage.transform import resize
from skimage.io import imread, imshow, concatenate_images,imsave
from tqdm import tqdm_notebook, tnrange
import numpy as np

# IMG_MASK_PATH = "./smooth_test_img/davis/"
IMG_MASK_PATH = "./smooth_test_img/leaf/"
IM_WIDTH = 480
IM_HEIGHT = 480

def saveSplit(ids,chunk_num):
    ### Adjust Split Folder###
    # save_path = "./smooth_all/davis/"
    save_path = "./smooth_all/leaf/"

    ### Adjust photo ###
    for _, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        print(id_)
        # img = imread(TRAIN_PATH + '/' + id_ + '.jpg')
        # mask = imread(TRAIN_PATH + '/' + id_ + '.png')
        img_mask = imread(IMG_MASK_PATH + '/' + id_)

        # im_split = split2(img)
        # mask_split = split2(mask)
        im_mask_split = split2(img_mask)

        # for i in range(len(im_split)):
        #     if i>=int(0.5*len(im_split)):
        #         name = id_ + str(i).zfill(3) #here INC
        #         imsave(savetrain + name + ".jpg", im_split[i])
        #         imsave(savetrain + name + ".png", mask_split[i])
        #     else:
        #         name = id_ + str(i).zfill(3) #here INC
        #         imsave(savetest + name + ".jpg", im_split[i])
        #         imsave(savetest + name + ".png", mask_split[i])

        for i in range(len(im_mask_split)):
            name = str(i).zfill(2) + "_" + id_ #here INC
            imsave(save_path + name, im_mask_split[i])



def split2(im):
    # print(im.shape)
    ret = []

    for i in np.arange(0, im.shape[0] - IM_HEIGHT + 1, IM_HEIGHT):
        for j in np.arange(0, im.shape[1] - IM_WIDTH + 1, IM_WIDTH):
            s = im[i:i+IM_HEIGHT, j:j+IM_WIDTH]
            ret.append(s)

    return ret


# leaf_ids = set([f_name[:-4] for f_name in os.listdir(IMG_MASK_PATH)])
leaf_ids = set([f_name for f_name in os.listdir(IMG_MASK_PATH)])
saveSplit(leaf_ids,chunk_num=2)

print("split over")