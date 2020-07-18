import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from skimage.io import imread, imshow, concatenate_images,imsave
from tqdm import tqdm_notebook, tnrange
from constants import *
from statistics import *

def plot_iou_curve(results, title):
  plt.figure(figsize=(8, 8))
  plt.title(title)
  train_iou = np.array(results.history["iou_score"])
  val_iou = np.array(results.history["val_iou_score"])
  plt.plot(train_iou, label='train_iou')
  plt.plot(val_iou, label='val_iou')
  plt.plot(np.argmax(val_iou), np.max(val_iou), marker="x", color="b", label="best iou")
  plt.xlabel("Epochs")
  plt.ylabel("IOU Score")
  plt.legend()
  plt.show()
  plt.savefig('./results/plot_iou_curve.png')

def plot_loss_curve(results, title):
  plt.figure(figsize=(8, 8))
  plt.title(title)
  exp_train_loss = np.exp(results.history["loss"])
  exp_val_loss = np.exp(results.history["val_loss"])
  plt.plot(exp_train_loss, label="train_loss")
  plt.plot(exp_val_loss, label="val_loss")
  plt.plot(np.argmin(exp_val_loss), np.min(exp_val_loss), marker="x", color="r", label="lowest loss")
  plt.xlabel("Epochs")
  plt.ylabel("IOU Score")
  plt.legend()
  plt.show()
  plt.savefig('./results/plot_loss_curve.png')

def generate_full_label_map(test_image, model):
    base_map = np.full((test_image.shape[0], test_image.shape[1]), 0)
    for i in np.arange(0, test_image.shape[0] - IM_HEIGHT, 512):
        for j in np.arange(0, test_image.shape[1] - IM_WIDTH, 512):
            temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
            temp[0] = test_image[i:i+IM_HEIGHT, j:j+IM_WIDTH]
            prediction = np.argmax(model.predict(temp)[0], axis=-1)
            for x in np.arange(prediction.shape[0]):
                for y in np.arange(prediction.shape[1]):
                    base_map[i+x][j+y] = prediction[x][y]
        if j<test_image.shape[1] - IM_WIDTH:
            j = test_image.shape[1] - IM_WIDTH
            temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
            temp[0] = test_image[i:i+IM_HEIGHT, j:j+IM_WIDTH]
            prediction = np.argmax(model.predict(temp)[0], axis=-1)
            for x in np.arange(prediction.shape[0]):
                for y in np.arange(prediction.shape[1]):
                    base_map[i+x][j+y] = prediction[x][y]

    if i<test_image.shape[0] - IM_HEIGHT:
        i = test_image.shape[0] - IM_HEIGHT
        for j in np.arange(0, test_image.shape[1] - IM_WIDTH+1, 512):
            temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
            temp[0] = test_image[i:i+IM_HEIGHT, j:j+IM_WIDTH]
            prediction = np.argmax(model.predict(temp)[0], axis=-1)
            for x in np.arange(prediction.shape[0]):
                for y in np.arange(prediction.shape[1]):
                    base_map[i+x][j+y] = prediction[x][y]
        if j<test_image.shape[1] - IM_WIDTH:
            j = test_image.shape[1] - IM_WIDTH
            temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
            temp[0] = test_image[i:i+IM_HEIGHT, j:j+IM_WIDTH]
            prediction = np.argmax(model.predict(temp)[0], axis=-1)
            for x in np.arange(prediction.shape[0]):
                for y in np.arange(prediction.shape[1]):
                    base_map[i+x][j+y] = prediction[x][y]

    # if i<test_image.shape[0] - IM_HEIGHT:
    #     i = test_image.shape[0] - IM_HEIGHT
    #     j = test_image.shape[1] - IM_WIDTH
    #     temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
    #     temp[0] = test_image[i:i+IM_HEIGHT, j:j+IM_WIDTH]
    #     prediction = np.argmax(model.predict(temp)[0], axis=-1)
    #     for x in np.arange(prediction.shape[0]):
    #         for y in np.arange(prediction.shape[1]):
    #             base_map[i+x][j+y] = prediction[x][y]
    #     print(i,j)

    return base_map

# def generate_full_label_map(test_image, model):
#     base_map = np.full((test_image.shape[0], test_image.shape[1]), 0)
#     for i in np.arange(0, test_image.shape[0] - IM_HEIGHT, 256):
#         for j in np.arange(0, test_image.shape[1] - IM_WIDTH, 256):
#             temp = np.zeros((1, 512, 512,3))
#             temp[0] = resize(test_image[i:i+IM_HEIGHT, j:j+IM_WIDTH], (512, 512, 3), mode = 'constant', preserve_range = True)
#             prediction = np.argmax(model.predict(temp)[0], axis=-1)
#             for x in np.arange(prediction.shape[0]):
#                 for y in np.arange(prediction.shape[1]):
#                     base_map[i+x][j+y] = prediction[x][y]
#     return base_map


# def colors_to_labels(original_mask):
#     ground_truth_label_map = np.full((original_mask.shape[0],original_mask.shape[1]), 0)
#     for j in range(len(COLORS)):
#         pred_indices = np.argwhere((original_mask[:, :, 1] == COLORS[j][1]) & (original_mask[:, :, 2] == COLORS[j][2]))
#         for pred_index in pred_indices:
#             ground_truth_label_map[pred_index[0]][pred_index[1]] = j
#     return ground_truth_label_map

def colors_to_labels(original_mask):
    ground_truth_label_map = np.full((original_mask.shape[0],original_mask.shape[1]), 0)
    count = 0
    for typep in TYPES_TO_COLORS:
        if typep == 'other':
            type_indices = np.argwhere(original_mask[:,:,:] < TYPES_TO_CHANNEL[typep]) # an array containing all the indices that match the pixels        
        elif typep == 'nasturtium' or typep == 'borage' or typep == 'bok_choy':
            type_indices = np.argwhere(original_mask[:,:,TYPES_TO_CHANNEL[typep]] > 230) # an array containing all the indices that match the pixels     
        elif typep == 'plant1' or typep == 'plant2':
            type_indices = np.argwhere((original_mask[:,:,TYPES_TO_CHANNEL[typep][0]] > 230) & (original_mask[:,:,TYPES_TO_CHANNEL[typep][1]] > 230))
        else: 
            type_indices = np.argwhere((original_mask[:,:,TYPES_TO_CHANNEL[typep][0]] > 230) & (original_mask[:,:,TYPES_TO_CHANNEL[typep][1]] > 100))
        for type_index in type_indices:
            ground_truth_label_map[type_index[0], type_index[1]] = count
        count += 1
    return ground_truth_label_map

def labels_to_colors(label_map):
    predicted_mask = np.full((label_map.shape[0], label_map.shape[1], 3), 0)
    for j in range(len(COLORS)):
        pred_indices = np.argwhere(label_map == j)
        for pred_index in pred_indices:
            predicted_mask[pred_index[0], pred_index[1], :] = COLORS[j]
    return predicted_mask

def iou_score(target, prediction, label):
    target_tf = (np.array(target) == label)
    pred_tf = (np.array(prediction) == label)
    intersection = np.logical_and(target_tf, pred_tf)
    union = np.logical_or(target_tf, pred_tf)
    if np.count_nonzero(union)>0:
      iou_score = np.count_nonzero(intersection) / np.count_nonzero(union)
    else:
      iou_score = 1.
    return iou_score

def prepare_img_and_label_map(test_id, model):
    test_image = cv2.cvtColor(cv2.imread('{}/{}.jpg'.format(TEST_PATH, test_id)), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(TEST_PATH + '/' + test_id + '.png')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    label_map = generate_full_label_map(test_image, model)
    unet_mask = labels_to_colors(label_map)
    return test_image, mask, unet_mask

def show_test_truth_prediction(test_image, mask, unet_mask,test_id,num):
    plt.figure(figsize=(8, 24))
    _, axes = plt.subplots(3, 1)
    axes[0].set_title('Original Image')
    axes[0].imshow(test_image)
    axes[1].set_title('Ground Truth')
    axes[1].imshow(mask)
    axes[2].set_title('Unet Predicted Mask')
    axes[2].imshow(unet_mask)
    plt.tight_layout()
    plt.show()
    plt.savefig('./results/mask'+test_id+num+'.png')

    imsave('./results/maskonly'+test_id+num+'.png', unet_mask)

# def show_test_truth_prediction(test_image, mask, unet_mask,test_id,num):
#     plt.figure(figsize=(8, 24))
#     _, axes = plt.subplots(2, 1)
#     axes[0].set_title('Original Image')
#     axes[0].imshow(test_image)
#     axes[1].set_title('Unet Predicted Mask')
#     axes[1].imshow(unet_mask)
#     plt.show()
#     plt.tight_layout()
#     plt.savefig('./results/mask'+test_id+num+'.png')

#     imsave('./results/maskonly'+test_id+num+'.png', unet_mask)

def predict_mask(test_id, model):
    test_image, mask, unet_mask = prepare_img_and_label_map(test_id, model)
    show_test_truth_prediction(test_image, mask, unet_mask)

def categorical_iou_eval(test_ids, model):
    unet_iou = {}

    unet_iou['index'] = []

    for category in TYPES:
        unet_iou[category] = []
    i = 0
    for _, id_ in tqdm_notebook(enumerate(test_ids), total=len(test_ids)):
        test_image, mask, unet_mask = prepare_img_and_label_map(id_, model)
        print('measuring IoU loss with test image {}'.format(TEST_PATH + id_ + '.jpg'))

        truth_label_map = colors_to_labels(mask)
        label_map = generate_full_label_map(test_image, model)

        for j in range(len(COLORS)):
            unet_iou[TYPES[j]].append(iou_score(truth_label_map, label_map, j))

        unet_iou['index'].append(id_)

        show_test_truth_prediction(test_image, mask, unet_mask, id_,'0')

    unet_iou['index'].append('mean')
    for j in range(len(COLORS)):
      meanval = mean(unet_iou[TYPES[j]])
      unet_iou[TYPES[j]].append(meanval)


    unet_iou_table = pd.DataFrame(unet_iou)
    unet_iou_table.to_csv(IOU_EVAL_FILE)
    print('Complete Evaluation of Categorical IoU Score on Test Images and Saved to file {}'.format(IOU_EVAL_FILE))


def eval_premask(test_ids, model):
    unet_iou = {}

    unet_iou['index'] = []

    for category in TYPES:
        unet_iou[category] = []

    for _, id_ in tqdm_notebook(enumerate(test_ids), total=len(test_ids)):
      original_image = cv2.cvtColor(cv2.imread('{}/{}.jpg'.format(TEST_PATH, id_)), cv2.COLOR_BGR2RGB)
      # test_image = load_img('{}/{}.jpg'.format(TEST_PATH, id_), grayscale=False, color_mode='rgb')
      test_image = img_to_array(original_image)
      # test_image = resize(test_image, (3024, 4032, 3), mode = 'constant', preserve_range = True)
      # test_image = test_image[:,400:4032]
      test_image = resize(test_image, (IM_HEIGHT, IM_WIDTH, 3), mode = 'constant', preserve_range = True)
      temp = np.zeros((1, IM_HEIGHT, IM_WIDTH, 3))
      temp[0] = test_image[:, :]
      prediction = np.argmax(model.predict(temp)[0], axis=-1)
      unet_mask = labels_to_colors(prediction)

      mask = cv2.imread(TEST_PATH + '/' + id_ + '.png')
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
      truth_label_map = colors_to_labels(mask)

      for j in range(len(COLORS)):
            unet_iou[TYPES[j]].append(iou_score(truth_label_map, prediction, j))

      unet_iou['index'].append(id_)
      
      show_test_truth_prediction(original_image, mask, unet_mask, id_,'1')


    unet_iou['index'].append('mean')
    for j in range(len(COLORS)):
      meanval = mean(unet_iou[TYPES[j]])
      unet_iou[TYPES[j]].append(meanval)

    unet_iou_table = pd.DataFrame(unet_iou)
    unet_iou_table.to_csv(IOU_EVAL_FILE)
    print('Complete Evaluation of Categorical IoU Score on Test Images and Saved to file {}'.format(IOU_EVAL_FILE))

    
 