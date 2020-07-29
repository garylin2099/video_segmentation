import argparse, glob, os, cv2, sys, pickle, random
import numpy as np

from constants import *
import matplotlib.pyplot as plt
import pandas as pd

class DataLoader():
    def __init__(self, im_size, nbr_frames, data_path):
        self.im_size = im_size
        # self.dataset_size = [1024, 2048]
        self.nbr_frames = nbr_frames
        # self.L = glob.glob(os.path.join(cfg.cityscapes_dir, 'gtFine', 'train', "*", "*labelTrainIds.png"))
        self.L = glob.glob(os.path.join(data_path, "*.png"))
        # random.shuffle(self.L)
        self.idx = 0
        self.data_path = data_path
    
    def get_next_sequence(self):
        # H, W = self.dataset_size
        h, w = self.im_size

        # shuffle at each new epoch
        if (self.idx + 1) % len(self.L) == 0:
            random.shuffle(self.L)

        im_path = self.L[self.idx % len(self.L)] # the gt path, later split into image path
        self.idx += 1

        parts = im_path.split('/')[-1].split('_')
        # city, seq, frame = parts[0], parts[1], parts[2]
        seq, frame = parts[0], parts[2]

        images = []
        # gt = cv2.imread(im_path, 0)[i0:i1, j0:j1]
        gt = cv2.imread(im_path, 0)
        self.gt_intensity_to_class(gt)
        # print(np.unique(gt))
        
        for dt in range(-self.nbr_frames + 1, 1):
            t = int(frame) + dt
            
            # frame_path = os.path.join(cfg.cityscapes_video_dir, 'leftImg8bit_sequence', 'train', 
            #         city, ("%s_%s_%06d_leftImg8bit.png" % (city, seq, t)))
            frame_path = os.path.join(self.data_path, ("%s_02_%02d_2020_cal.jpg" % (seq, t)))
            # images.append(cv2.imread(frame_path, 1).astype(np.float32)[i0:i1,j0:j1][np.newaxis,...])
            im = cv2.imread(frame_path, 1)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # a = im.astype(np.float32)[np.newaxis,...]
            # print("image shape is", a.shape)
            images.append(im.astype(np.float32)[np.newaxis,...]) # shape: (1, 512, 512, 3)
        return images, gt
    
    def gt_intensity_to_class(self, gt):
        # print(np.unique(gt))
        for i in range(len(TYPE_INTENSITY)):
            gt[gt == TYPE_INTENSITY[i]] = i
        gt[gt == 187] = 4 # fix some individual masks for having intensity 178 as 187
        # print(np.unique(gt))

def plot_loss_curve(results_train, results_val, title):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    plt.plot(range(len(results_train)), results_train, label="train_loss")
    plt.plot(range(len(results_val)), results_val, label="val_loss")
    plt.plot(np.argmin(results_val), np.min(results_val), marker="x", color="r", label="lowest loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    # plt.show()
    plt.savefig('./results/%s.png' % title)

def labels_to_colors(label_map):
    predicted_mask = np.full((label_map.shape[0], label_map.shape[1], 3), 0)
    for j in range(len(COLORS)):
        pred_indices = np.argwhere(label_map == j)
        for pred_index in pred_indices:
            predicted_mask[pred_index[0], pred_index[1], :] = COLORS[j]
    predicted_mask = cv2.cvtColor(predicted_mask.astype("uint8"),\
         cv2.COLOR_RGB2BGR) # convert RGB value to BGR and unsigned int8 for imwrite
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

def colors_to_labels(gt_path):
    gt = cv2.imread(gt_path, 0) # grayscale image with 7 different intensities
    # intensity to plant type
    for i in range(len(TYPE_INTENSITY)):
        gt[gt == TYPE_INTENSITY[i]] = i
    gt[gt == 187] = 4 # fix some individual masks for having intensity 178 as 187
    return gt # shape 512x512, each entry 0-7

def gt_to_one_hot_map(gt_path):
    gt = colors_to_labels(gt_path)
    one_hot_map = np.zeros((1, gt.shape[0], gt.shape[1], N_CLASSES))
    for i in range(N_CLASSES):
        one_hot_map[0,:,:,i] = (gt == i)*1
    return one_hot_map

def categorical_iou_eval_each_im(gt_path, pred_label_map, iou):
    id_ = gt_path.split('/')[-1][:-4]
    print('measuring IoU {}'.format(id_))

    truth_label_map = colors_to_labels(gt_path)

    plants_each_im = np.zeros(N_CLASSES)
    for j in range(len(COLORS)):
        plants_each_im[j] = iou_score(truth_label_map, pred_label_map, j)
        iou[TYPES[j]].append(plants_each_im[j])
    iou['index'].append(id_)
    iou['im_avrg'].append(np.mean(plants_each_im))
    print('IoU for this image is {:.3f}'.format(np.mean(plants_each_im)))


def iou_mean(iou):
    iou['index'].append('mean')
    plant_total = np.zeros(N_CLASSES)
    for j in range(len(COLORS)):
        meanval = np.mean(iou[TYPES[j]])
        iou[TYPES[j]].append(meanval)
        plant_total[j] = meanval
    iou_total = np.mean(plant_total)
    iou['im_avrg'].append(iou_total)
    print('average iou on test set is {:.8f}'.format(iou_total))
    iou_table = pd.DataFrame(iou)
    iou_table.to_csv(IOU_EVAL_FILE)
    print('Complete Evaluation of Categorical IoU Score on Test Images and Saved to file {}'.format(IOU_EVAL_FILE))

# def show_test_truth_prediction(test_image, mask, unet_mask,test_id,num):
#     plt.figure(figsize=(8, 24))
#     _, axes = plt.subplots(3, 1)
#     axes[0].set_title('Original Image')
#     axes[0].imshow(test_image)
#     axes[1].set_title('Ground Truth')
#     axes[1].imshow(mask)
#     axes[2].set_title('Unet Predicted Mask')
#     axes[2].imshow(unet_mask)
#     plt.tight_layout()
#     plt.show()
#     plt.savefig('./results/mask'+test_id+num+'.png')

#     imsave('./results/maskonly'+test_id+num+'.png', unet_mask)