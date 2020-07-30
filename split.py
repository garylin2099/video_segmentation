import tensorflow as tf
import segmentation_models as sm
import torch
from data_utils import *
from eval_utils import *
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from segmentation_models import get_preprocessing
from segmentation_models.losses import JaccardLoss, CategoricalCELoss
from segmentation_models.metrics import IOUScore
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from math import *

#Intermediate   Maturation1 Maturation2 Pruned  Flowering
# STAGE = 'Intermediate'
# STAGE = 'Maturation1'
# STAGE = 'Maturation2'
# STAGE = 'Pruned'
# STAGE2 = 'Flowering'

IMG_MASK_PATH = "./overhead_0202_to_0216/"

def saveSplit(ids,chunk_num):
    ### Adjust Split Folder###
    save_path = "./video_all_new/"

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

    # for i in np.arange(0, im.shape[0] - IM_HEIGHT, IM_HEIGHT):
    #     for j in np.arange(0, im.shape[1] - IM_WIDTH, IM_WIDTH):
    #         s = im[i:i+IM_HEIGHT, j:j+IM_WIDTH]
    #         ret.append(s)
    for j in np.arange(0, im.shape[1] - IM_WIDTH, IM_WIDTH):
        for i in np.arange(0, im.shape[0] - IM_HEIGHT, IM_HEIGHT):
            s = im[i:i+IM_HEIGHT, j:j+IM_WIDTH]
            ret.append(s)

    return ret
        
            

def split(im, chunk_num=1):
    inc = int(3024/chunk_num)

    yran = int(3024 / inc) 
    xran = int(4032 / inc) 

    offset = int((4032 - xran*inc)/2)
        
    ret = []
    sy = 0
    sx = offset
    
    for y in range(yran):
        for x in range(xran):
            s = im[sy:sy+inc,sx:sx+inc]
            ret.append(s)
            sx += inc
        sx = offset
        sy += inc
    
    return ret


# leaf_ids = set([f_name[:-4] for f_name in os.listdir(IMG_MASK_PATH)])
leaf_ids = set([f_name for f_name in os.listdir(IMG_MASK_PATH)])
saveSplit(leaf_ids,chunk_num=2)

print("split over")