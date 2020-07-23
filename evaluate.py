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

    if args.static == 'lrr':
        static_input = tf.placeholder(tf.float32)
        static_network = LRR()
        static_output = static_network(static_input)
    elif args.static == 'unet':
        static_network = sm.Unet(backbone_name=BACKBONE, encoder_weights=None, activation=ACTIVATION_FN, classes=N_CLASSES)

    saver = tf.train.Saver([k for k in tf.global_variables() if not k.name.startswith('flow/')])
    if args.flow in ['flownet1', 'flownet2']:
        saver_fn = tf.train.Saver([k for k in tf.global_variables() if k.name.startswith('flow/')])

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

        # L = glob.glob(os.path.join(cfg.cityscapes_dir, 'gtFine', data_split, "*", "*labelIds.png"))
        # L = glob.glob(os.path.join(VD_VALIDATION_PATH, "*.png"))
        L = glob.glob(os.path.join(VD_TRAIN_PATH, "*.png"))
        for (progress_counter, im_path) in enumerate(L):
            parts = im_path.split('/')[-1].split('_')
            # city, seq, frame = parts[0], parts[1], parts[2]
            seq, frame = parts[0], parts[2]

            # print("Processing sequence %d/%d" % (progress_counter+1, len(L)))
            for dt in range(-args.frames + 1, 1):
                first_frame = dt == -args.frames + 1
                t = int(frame) + dt
                
                # frame_path = os.path.join(cfg.cityscapes_video_dir, 'leftImg8bit_sequence', data_split, 
                #         city, ("%s_%s_%06d_leftImg8bit.png" % (city, seq, t)))
                # im = cv2.imread(frame_path, 1).astype(np.float32)[np.newaxis,...]
                # frame_path = os.path.join(VD_VALIDATION_PATH, ("%s_02_%02d_2020_cal.jpg" % (seq, t)))
                frame_path = os.path.join(VD_TRAIN_PATH, ("%s_02_%02d_2020_cal.jpg" % (seq, t)))
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
                    # the hidden state is simple the static segmentation for the first frame
                    h = x
                    pred = np.argmax(h, axis=3)
                    # print("first image")
                    # print(np.unique(pred))
                else:
                    inputs = {
                        input_images_tensor: np.stack([last_im, im]),
                        input_flow: flow,
                        input_segmentation: x,
                        prev_h: h
                    }
                    # GRFP
                    h, pred = sess.run([new_h, prediction], feed_dict=inputs)
                    # print("intermediate")
                    # print(np.unique(pred))

                last_im = im

            # save it
            S = pred[0]
            S_new = S.copy()
            # for (idx, train_idx) in cs_id2trainid.iteritems():
            #     S_new[S == train_idx] = idx
            
            # print("final")
            # print(np.unique(S_new))

            # output_path = '%s_%s_%s.png' % (city, seq, frame)
            if args.original_static == 1:
                # print("write, fix")
                output_path = '%s_02_%s_pred_%df_fix.png' % (seq, frame, args.frames)
            else:
                # print("write, end to end")
                output_path = '%s_02_%s_pred_%df.png' % (seq, frame, args.frames)
            # cv2.imwrite(os.path.join(cfg.cityscapes_dir, 'results', output_path), S_new)
            S_color = labels_to_colors(S)
            cv2.imwrite(os.path.join('./pred_mask_train', output_path), S_color)


        # # Evaluate using the official CityScapes code
        # evalPixelLevelSemanticLabeling.main([])



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evluate GRFP on the CityScapes validation set.')

    parser.add_argument('--static', help='Which static network to use.', required=True)
    parser.add_argument('--flow', help='Which optical flow method to use.', required=True)
    parser.add_argument('--frames', type=int, help='Number of frames to use.', default=5, required=False)
    parser.add_argument('--ckpt', help='Which checkpoint file to load from. Specify relative to the ./checkpoints/ directory.', default='', required=False)
    parser.add_argument('--original_static', type=int, help='whether to use original weights for static nn', default=0, required=False)

    args = parser.parse_args()

    assert args.flow in ['flownet1', 'flownet2', 'farneback'], "Unknown flow method %s." % args.flow
    assert args.static in ['dilation', 'dilation_grfp', 'lrr', 'lrr_grfp', 'unet'], "Unknown static method %s." % args.static
    assert args.frames >= 1 and args.frames <= 20, "The number of frames must be between 1 and 20."
    
    evaluate(args)