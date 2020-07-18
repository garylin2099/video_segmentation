import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images,imsave
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import img_to_array, load_img

from constants import *

def prepare_data(ids, im_width, im_height, test_size, seed=42, path=None):
    X = np.zeros((len(ids) * 5, im_height, im_width, 3), dtype=np.float32)
    y = np.zeros((len(ids) * 5, im_height, im_width, N_CLASSES), dtype=np.float32) #### adjust classes
    # X = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)
    # y = np.zeros((len(ids), im_height, im_width, N_CLASSES), dtype=np.float32) #### adjust classes
    i = 0
    for _, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images
        x_img = cv2.imread(path + '/' + id_ + '.jpg')
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        x_img = img_to_array(x_img)
        x_img = resize(x_img, (im_height, im_width, 3), mode = 'constant', preserve_range = True)
        # Load masks
        mask = cv2.imread(path + '/' + id_ + '.png')
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = img_to_array(mask)
        mask = resize(mask, (im_height, im_width, 3), mode = 'constant', preserve_range = True)
        ground_truth = np.full((im_height, im_width, N_CLASSES), 0) ### adjust classes
        for typep in TYPES_TO_COLORS:
            if typep == 'other':
                type_indices = np.argwhere(mask[:,:,:] < TYPES_TO_CHANNEL[typep]) # an array containing all the indices that match the pixels        
            elif typep == 'nasturtium' or typep == 'borage' or typep == 'bok_choy':
                type_indices = np.argwhere(mask[:,:,TYPES_TO_CHANNEL[typep]] > 230) # an array containing all the indices that match the pixels     
            elif typep == 'plant1' or typep == 'plant2':
                type_indices = np.argwhere((mask[:,:,TYPES_TO_CHANNEL[typep][0]] > 230) & (mask[:,:,TYPES_TO_CHANNEL[typep][1]] > 230))
            else: 
                type_indices = np.argwhere((mask[:,:,TYPES_TO_CHANNEL[typep][0]] > 230) & (mask[:,:,TYPES_TO_CHANNEL[typep][1]] > 100))
            for type_index in type_indices:
                ground_truth[type_index[0], type_index[1], :] = BINARY_ENCODINGS[typep]
        
        if (_/len(ids))*100 % 10 == 0:
            print(_)
            print((_/len(ids))*100) 
        # Save images
        X[i] = x_img
        y[i] = ground_truth

        # Apply Vertical Flip Augmentation
        X[i+1] = np.flip(x_img, 0)
        y[i+1] = np.flip(ground_truth, 0)

        # Apply Horizontal Flip Augmentation
        X[i+2] = np.flip(x_img, 1)
        y[i+2] = np.flip(ground_truth, 1)

        # Apply 90-degrees Rotation Augmentation
        X[i+3] = np.rot90(x_img)
        y[i+3] = np.rot90(ground_truth)

        # Apply 180-degrees Rotation Augmentation
        X[i+4] = np.rot90(np.rot90(x_img))
        y[i+4] = np.rot90(np.rot90(ground_truth))

        i = i + 5
        # i = i + 1

    # return train_test_split(X, y, test_size=test_size, random_state=seed)
    return X, y

# def saveSplit(ids,chunk_num):
#     ### Adjust Split Folder###
#     saveSp = "./split2/"
#     saveMask = "./split2/"
#     ### Adjust photo ###
#     for _, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
#         img = imread(TRAIN_PATH + '/' + id_ + '.jpg')
#         mask = imread(TRAIN_PATH + '/' + id_ + '.png')

#         im_split = split(img, chunk_num)
#         mask_split = split(mask, chunk_num)

#         for i in range(len(im_split)):
#             name = id_ + str(i).zfill(3) #here INC
#             ti = cv2.resize(im_split[i], (IM_WIDTH, IM_HEIGHT)) #here SIZE
#             tm = cv2.resize(mask_split[i], (IM_WIDTH, IM_HEIGHT)) #here SIZE
#             imsave(saveSp + name + ".jpg", ti)
#             imsave(saveMask + name + ".png", tm)
    

# def split(im, chunk_num=1):
#     inc = int(3024/chunk_num)

#     yran = int(3024 / inc) 
#     xran = int(4032 / inc) 

#     offset = int((4032 - xran*inc)/2)
        
#     ret = []
#     sy = 0
#     sx = offset
    
#     for y in range(yran):
#         for x in range(xran):
#             s = im[sy:sy+inc,sx:sx+inc]
#             ret.append(s)
#             sx += inc
#         sx = offset
#         sy += inc
    
#     return ret