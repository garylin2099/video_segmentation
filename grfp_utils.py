import argparse, glob, os, cv2, sys, pickle, random
import numpy as np

from constants import *
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, im_size, nbr_frames):
        self.im_size = im_size
        # self.dataset_size = [1024, 2048]
        self.nbr_frames = nbr_frames
        # self.L = glob.glob(os.path.join(cfg.cityscapes_dir, 'gtFine', 'train', "*", "*labelTrainIds.png"))
        self.L = glob.glob(os.path.join(VD_TRAIN_PATH, "*.png"))
        # random.shuffle(self.L)
        self.idx = 0
    
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
            frame_path = os.path.join(VD_TRAIN_PATH, ("%s_02_%02d_2020_cal.jpg" % (seq, t)))
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

def plot_loss_curve(results, title):
    plt.figure(figsize=(8, 8))
    plt.title(title)
    # print(results)
    plt.plot(range(len(results)), results, label="train_loss")
    # plt.plot(exp_val_loss, label="val_loss")
    # plt.plot(np.min(results), marker="x", color="r", label="lowest loss")
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