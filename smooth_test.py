import argparse, glob, os, cv2, sys, pickle
import numpy as np
import tensorflow as tf
from models.stgru import STGRU
from models.flownet2 import Flownet2
from tensorflow.python.framework import ops

from constants import *
import segmentation_models as sm
from grfp_utils import *
from collections import OrderedDict

bilinear_warping_module = tf.load_op_library('./misc/bilinear_warping.so')
@ops.RegisterGradient("BilinearWarping")
def _BilinearWarping(op, grad):
  return bilinear_warping_module.bilinear_warping_grad(grad, op.inputs[0], op.inputs[1])

def evaluate(args):
    im_size = [480, 480]
    
    if args.flow == 'flownet2':
        with tf.variable_scope('flow'):
            flow_network = Flownet2(bilinear_warping_module)
            flow_img0 = tf.placeholder(tf.float32)
            flow_img1 = tf.placeholder(tf.float32)
            flow_tensor = flow_network(flow_img0, flow_img1, flip=True)
    
    im_input = tf.placeholder(tf.float32)
    flow_input = flow_tensor
    warped_output = bilinear_warping_module.bilinear_warping(im_input, flow_input)

    # saver = tf.train.Saver([k for k in tf.global_variables() if not k.name.startswith('flow/')])
    # saver = tf.train.Saver([k for k in tf.trainable_variables() if not k.name.startswith('flow/')])
    if args.flow in ['flownet1', 'flownet2']:
        saver_fn = tf.train.Saver([k for k in tf.global_variables() if k.name.startswith('flow/')])

    with tf.Session() as sess:

        if args.flow == 'flownet1':
            saver_fn.restore(sess, './checkpoints/flownet1')
        elif args.flow == 'flownet2':
            saver_fn.restore(sess, './checkpoints/flownet2')

        # weight_print = sess.run(weight_check)
        # for k, v in weight_print.items():
        #     print("%s, max %.5f, min abs %.5f, mean abs %.5f, median abs %.5f" \
        #          % (k, np.max(v), np.min(np.abs(v)), np.mean(np.abs(v)), np.median(np.abs(v))))
        #     # print(v)

        # initialization to store iou
        # iou_dict = OrderedDict()
        # iou_dict['index'] = []
        # for category in TYPES:
        #     iou_dict[category] = []
        # iou_dict['im_avrg'] = []

        # mask_dict = OrderedDict()

        eval_path = './smooth_all/%s/' % args.eval_set
        # L = glob.glob(os.path.join(eval_path, "*1.jpg"))
        if args.eval_set == 'leaf':
            L = glob.glob(os.path.join(eval_path, "*05_2020_cal.jpg"))
        elif args.eval_set == 'davis':
            L = glob.glob(os.path.join(eval_path, "*1.jpg"))

        for (progress_counter, gt_path) in enumerate(L):
            parts = gt_path.split('/')[-1].split('_')

            if args.eval_set == 'leaf':
                seq, frame = parts[0], parts[2]
            elif args.eval_set == 'davis':
                seq, frame = parts[0], parts[1][:-4]

            # print("Processing sequence %d/%d" % (progress_counter+1, len(L)))
            for dt in range(-args.frames + 1, 1):
                first_frame = dt == -args.frames + 1
                t = int(frame) + dt
                
                # frame_path = os.path.join(eval_path, ("%s_02_%02d_2020_cal.jpg" % (seq, t)))
                if args.eval_set == 'leaf':
                    frame_path = os.path.join(eval_path, ("%s_02_%02d_2020_cal.jpg" % (seq, t)))
                elif args.eval_set == 'davis':
                    frame_path = os.path.join(eval_path, ("%s_%05d.jpg" % (seq, t)))

                # im = cv2.imread(frame_path, 1).astype(np.float32)[np.newaxis,...]
                im = cv2.imread(frame_path, 1)
                # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)[np.newaxis,...]
                im = im.astype(np.float32)[np.newaxis,...]

                # Compute optical flow
                if not first_frame:
                    flow = sess.run(flow_tensor, feed_dict={flow_img0: im, flow_img1: last_im})
                    # warped_img = bilinear_warping_module.bilinear_warping(last_im, flow)
                    warped_img = sess.run(warped_output, feed_dict={im_input:last_im, flow_input: flow})
                    print("frame t-1")
                    print(np.sum(im != 0))
                    print("warped frame")
                    print(np.sum(warped_img != 0))
                    print(np.sum(np.abs(warped_img) > 0.01))
                    print(np.sum(np.abs(warped_img) > 0.1))
                    # print(warped_img.shape) # 1, 480, 480, 3
                    if args.eval_set == 'leaf':
                        cv2.imwrite('./smooth_all/leaf_warped/%s_02_%02d_warped.jpg' % (seq, t), warped_img[0])
                    elif args.eval_set == 'davis':
                        cv2.imwrite('./smooth_all/davis_warped/%s_%05d_warped.jpg' % (seq, t), warped_img[0])
                last_im = im
        
        # iou_mean(iou_dict, '%s_inf%d_fix%d_gt%d_%s.csv' % (args.ckpt[14:], args.frames, args.original_static, args.first_gt, args.eval_set))

        # iou_dict_overall = OrderedDict()
        # iou_dict_overall['index'] = []
        # for category in TYPES:
        #     iou_dict_overall[category] = []
        # iou_dict_overall['im_avrg'] = []
        # iou_filename = 'overall_%s_inf%d_gt%d_%s.csv' % (args.ckpt[14:], args.frames, args.first_gt, args.eval_set)
        # iou_overall(iou_dict_overall, mask_dict, iou_filename)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evluate GRFP on the CityScapes validation set.')

    parser.add_argument('--flow', help='Which optical flow method to use.', required=True)
    parser.add_argument('--frames', type=int, help='Number of frames to use.', default=2, required=False)
    parser.add_argument('--eval_set', help='what data to test smoothness on.', required=True)
    
    args = parser.parse_args()

    assert args.flow in ['flownet1', 'flownet2', 'farneback'], "Unknown flow method %s." % args.flow
    assert args.frames >= 1 and args.frames <= 20, "The number of frames must be between 1 and 20."
    
    evaluate(args)