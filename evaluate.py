import argparse, glob, os, cv2, sys, pickle
import numpy as np
import tensorflow as tf
# import config as cfg
from models.stgru import STGRU
from models.lrr import LRR
from models.dilation import dilation10network
from models.flownet2 import Flownet2
# from models.flownet1 import Flownet1
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
    data_split = 'val'
    # nbr_classes = 19
    nbr_classes = 7
    # im_size = [1024, 2048]
    im_size = [512, 512]
    # image_mean = [72.39,82.91,73.16] # the mean is automatically subtracted in some modules e.g. flownet2, so be careful
    
    if args.flow == 'flownet2':
        with tf.variable_scope('flow'):
            flow_network = Flownet2(bilinear_warping_module)
            flow_img0 = tf.placeholder(tf.float32)
            flow_img1 = tf.placeholder(tf.float32)
            flow_tensor = flow_network(flow_img0, flow_img1, flip=True)

    RNN = STGRU([nbr_classes, im_size[0], im_size[1]], [7, 7], bilinear_warping_module)
    
    input_images_tensor, input_flow, \
        input_segmentation, prev_h, new_h, \
        prediction = RNN.get_one_step_predictor()
    
    # h_input = tf.placeholder(tf.float32)
    # softmax_result = RNN.softmax_last_dim(h_input)
    weight_check = RNN.print_weights()

    im0 = tf.placeholder(tf.float32)
    last_im0 = tf.placeholder(tf.float32)
    flow0 = tf.placeholder(tf.float32)
    h0 = tf.placeholder(tf.float32)
    x0 = tf.placeholder(tf.float32)
    h_check0, r0, h_prev_warped0, h_prev_reset0, h_tilde0, z0, xh_x0, hh_r0, xz_x0, hz_r0, h_prev_check0 =\
         RNN.print_GRU_cell(im0, last_im0, flow0, h0, x0)

    # if args.static == 'lrr':
    #     static_input = tf.placeholder(tf.float32)
    #     static_network = LRR()
    #     static_output = static_network(static_input)
    # elif args.static == 'unet':
    #     static_network = sm.Unet(backbone_name=BACKBONE, encoder_weights=None, activation=ACTIVATION_FN, classes=N_CLASSES)

    saver = tf.train.Saver([k for k in tf.global_variables() if not k.name.startswith('flow/')])
    # saver = tf.train.Saver([k for k in tf.trainable_variables() if not k.name.startswith('flow/')])
    if args.flow in ['flownet1', 'flownet2']:
        saver_fn = tf.train.Saver([k for k in tf.global_variables() if k.name.startswith('flow/')])
    
    if args.static == 'unet':
        static_network = sm.Unet(backbone_name=BACKBONE, encoder_weights=None, activation=ACTIVATION_FN, classes=N_CLASSES)

    with tf.Session() as sess:
        if args.ckpt != '': # load checkpoints models saved in training
            saver.restore(sess, './checkpoints/%s' % (args.ckpt))
        
        if args.original_static == 1:
            print("use original network")
            static_network.load_weights(TEST_MODEL) # use original unet, this should overwrite weights restored by checkpoint

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
        iou_dict = OrderedDict()
        iou_dict['index'] = []
        for category in TYPES:
            iou_dict[category] = []
        iou_dict['im_avrg'] = []

        if args.eval_set == 'train':
            eval_path = VD_TRAIN_PATH
            L = glob.glob(os.path.join(VD_TRAIN_PATH, "*.png"))
        elif args.eval_set == 'val':
            eval_path = VD_VALIDATION_PATH
            L = glob.glob(os.path.join(VD_VALIDATION_PATH, "*.png"))
        else:
            eval_path = VD_TEST_PATH
            # L = glob.glob(os.path.join(VD_TEST_PATH, "*02_1[56]_2020_cal.png"))
            L = glob.glob(os.path.join(VD_TEST_PATH, "*.png"))
        
        # L = glob.glob(os.path.join(cfg.cityscapes_dir, 'gtFine', data_split, "*", "*labelIds.png"))
        # L = glob.glob(os.path.join(VD_TRAIN_PATH, "*.png"))
        # L = glob.glob(os.path.join(VD_VALIDATION_PATH, "*.png"))
        # L = glob.glob(os.path.join(VD_TEST_PATH, "*02_16_2020_cal.png"))
        for (progress_counter, gt_path) in enumerate(L):
            parts = gt_path.split('/')[-1].split('_')
            # city, seq, frame = parts[0], parts[1], parts[2]
            seq, frame = parts[0], parts[2]

            # print("Processing sequence %d/%d" % (progress_counter+1, len(L)))
            for dt in range(-args.frames + 1, 1):
            # for dt in range(-args.frames + 1, 1, 2):
                first_frame = dt == -args.frames + 1
                t = int(frame) + dt
                
                # frame_path = os.path.join(cfg.cityscapes_video_dir, 'leftImg8bit_sequence', data_split, 
                #         city, ("%s_%s_%06d_leftImg8bit.png" % (city, seq, t)))
                # im = cv2.imread(frame_path, 1).astype(np.float32)[np.newaxis,...]
                
                # frame_path = os.path.join(VD_TRAIN_PATH, ("%s_02_%02d_2020_cal.jpg" % (seq, t)))
                # frame_path = os.path.join(VD_VALIDATION_PATH, ("%s_02_%02d_2020_cal.jpg" % (seq, t)))
                # frame_path = os.path.join(VD_TEST_PATH, ("%s_02_%02d_2020_cal.jpg" % (seq, t)))
                frame_path = os.path.join(eval_path, ("%s_02_%02d_2020_cal.jpg" % (seq, t)))

                # im = cv2.imread(frame_path, 1).astype(np.float32)[np.newaxis,...]
                im = cv2.imread(frame_path, 1)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)[np.newaxis,...]

                # Compute optical flow
                if not first_frame:
                    if args.flow == 'flownet2':
                        flow = sess.run(flow_tensor, feed_dict={flow_img0: im, flow_img1: last_im})

                # Static segmentation
                if args.static == 'lrr':
                    x = sess.run(static_output, feed_dict={static_input: im})
                elif args.static == 'unet':
                    x = static_network.predict(im)
                
                if first_frame:
                    first_gt = os.path.join(eval_path, ("%s_02_%02d_2020_cal.png" % (seq, t)))
                    if args.frames != 1 and args.first_gt == 1 and os.path.isfile(first_gt): # if first frame gt exists and we use grfp
                        # leverage first gt mask in subsequent segmentation
                        h = gt_to_one_hot_map(first_gt).astype('float32') # h is 1x512x512x7 0-1 map
                        # print(h.shape)
                        h = h * 10
                        # h = h * 0 # let mask be random
                        # y_h = sess.run(softmax_result, feed_dict={h_input: h})
                        # print(y_h[0,50,50,:])
                        print("use the ground truth mask of the first frame")
                    else: # if first frame gt doesnt exist or we only use single-frame approach
                        # the hidden state is simple the static segmentation for the first frame
                        h = x # x shape (1, 512, 512, 7)
                        pred = np.argmax(h, axis=3)
                else:
                    inputs = {
                        input_images_tensor: np.stack([last_im, im]),
                        input_flow: flow,
                        input_segmentation: x,
                        prev_h: h
                    }
                    # GRFP

                    #### check weights to diagnose ###
                    # input0 = {
                    #     im0: im, last_im0: last_im, flow0: flow, h0: h, x0: x
                    # }
                    # h_check, r, h_prev_warped, h_prev_reset, h_tilde, z, xh_x, hh_r, xz_x, hz_r, h_prev_check = \
                    #     sess.run([h_check0, r0, h_prev_warped0, h_prev_reset0, h_tilde0, z0, xh_x0, hh_r0, xz_x0, hz_r0, h_prev_check0],\
                    #          feed_dict = input0)
                    # print("h, max %.5f, min %.5f, mean %.5f" % (np.max(h_check), np.min(h_check), np.mean(h_check)))
                    # print("r, max %.5f, min %.5f, mean %.5f" % (np.max(r), np.min(r), np.mean(r)))
                    # print("h_prev_warped, max %.5f, min %.5f, mean %.5f" % (np.max(h_prev_warped), np.min(h_prev_warped), np.mean(h_prev_warped)))
                    # print("h_prev_reset, max %.5f, min %.5f, mean %.5f" % (np.max(h_prev_reset), np.min(h_prev_reset), np.mean(h_prev_reset)))
                    # print("h_tilde, max %.5f, min %.5f, mean %.5f" % (np.max(h_tilde), np.min(h_tilde), np.mean(h_tilde)))
                    # print("z, max %.5f, min %.5f, mean %.5f" % (np.max(z), np.min(z), np.mean(z)))
                    # print("h_prev_input")
                    # print(np.sum(h_prev_check != 0))
                    # print(np.sum(np.abs(h_prev_check) > 0.01))
                    # print(np.sum(np.abs(h_prev_check) > 0.1))
                    # print("h_prev_warped")
                    # print(np.sum(h_prev_warped != 0))
                    # print(np.sum(np.abs(h_prev_warped) > 0.01))
                    # print(np.sum(np.abs(h_prev_warped) > 0.1))
                    # print("r")
                    # print(np.sum(r != 0))
                    # print(np.sum(np.abs(r) > 0.01))
                    # print(np.sum(np.abs(r) > 0.1))
                    # print("h_reset")
                    # print(np.sum(h_prev_reset != 0))
                    # print(np.sum(np.abs(h_prev_reset) > 0.01))
                    # print(np.sum(np.abs(h_prev_reset) > 0.1))
                    # print("next")
                    # print("xh_x")
                    # print(np.sum(xh_x != 0))
                    # print(np.sum(np.abs(xh_x) > 0.01))
                    # print(np.sum(np.abs(xh_x) > 0.1))
                    # print("hh_r")
                    # print(np.sum(hh_r != 0))
                    # print(np.sum(np.abs(hh_r) > 0.01))
                    # print(np.sum(np.abs(hh_r) > 0.1))
                    # print("xz_x")
                    # print(np.sum(xz_x != 0))
                    # print(np.sum(np.abs(xz_x) > 0.01))
                    # print(np.sum(np.abs(xz_x) > 0.1))
                    # print("hz_r")
                    # print(np.sum(hz_r != 0))
                    # print(np.sum(np.abs(hz_r) > 0.01))
                    # print(np.sum(np.abs(hz_r) > 0.1))
                    # print("next")
                    ################

                    h, pred = sess.run([new_h, prediction], feed_dict=inputs)

                last_im = im

            # save it
            S = pred[0] # S is a 2D predicted map, each entry 0-7, 512x512
            S_new = S.copy()
            # for (idx, train_idx) in cs_id2trainid.iteritems():
            #     S_new[S == train_idx] = idx
            
            categorical_iou_eval_each_im(gt_path, S, iou_dict)

            # output_path = '%s_%s_%s.png' % (city, seq, frame)
            if args.original_static == 1:
                # output_path = '%s_02_%s_pred_%df_fix.png' % (seq, frame, args.frames)
                output_path = '%s_02_%s_pred_%s_inf%d_fix_gt%d.png' % (seq, frame, args.ckpt[14:], args.frames, args.first_gt)
            else:
                output_path = '%s_02_%s_pred_%s_inf%d.png' % (seq, frame, args.ckpt[14:], args.frames)
            # cv2.imwrite(os.path.join(cfg.cityscapes_dir, 'results', output_path), S_new)
            S_color = labels_to_colors(S)
            cv2.imwrite(os.path.join('./pred_mask_%s' % args.eval_set, output_path), S_color)
        
        iou_mean(iou_dict, '%s_inf%d_fix%d_gt%d_%s.csv' % (args.ckpt[14:], args.frames, args.original_static, args.first_gt, args.eval_set))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evluate GRFP on the CityScapes validation set.')

    parser.add_argument('--static', help='Which static network to use.', required=True)
    parser.add_argument('--flow', help='Which optical flow method to use.', required=True)
    parser.add_argument('--frames', type=int, help='Number of frames to use.', default=5, required=False)
    parser.add_argument('--ckpt', help='Which checkpoint file to load from. Specify relative to the ./checkpoints/ directory.', default='', required=False)
    parser.add_argument('--original_static', type=int, help='whether to use original weights for static nn', default=0, required=False)
    parser.add_argument('--eval_set', help='evaluate on train or val or test', default='test', required=False)
    parser.add_argument('--first_gt', type=int, help='whether to use the first frame gt', default=1, required=False)

    args = parser.parse_args()

    assert args.flow in ['flownet1', 'flownet2', 'farneback'], "Unknown flow method %s." % args.flow
    assert args.static in ['dilation', 'dilation_grfp', 'lrr', 'lrr_grfp', 'unet'], "Unknown static method %s." % args.static
    assert args.frames >= 1 and args.frames <= 20, "The number of frames must be between 1 and 20."
    
    evaluate(args)