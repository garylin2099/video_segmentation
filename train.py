import argparse, glob, os, cv2, sys, pickle, random
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
# import matplotlib.pyplot as plt
import segmentation_models as sm
from grfp_utils import *

bilinear_warping_module = tf.load_op_library('./misc/bilinear_warping.so')
@ops.RegisterGradient("BilinearWarping")
def _BilinearWarping(op, grad):
  return bilinear_warping_module.bilinear_warping_grad(grad, op.inputs[0], op.inputs[1])

def train(args):
    # nbr_classes = 19
    nbr_classes = 7

    # learning rates for the GRU and the static segmentation networks, respectively
    # learning_rate = 2e-5 # original paper
    learning_rate = 0.001
    # static_learning_rate = 2e-12 # original paper
    # static_learning_rate_lrr = 1 # second stage
    
    # The total number of iterations and when the static network should start being refined
    nbr_iterations = 11200
    # t0_dilation_net = 5000
    # t0_dilation_net = 0

    im_size = [512, 512]
    # image_mean = [72.39,82.91,73.16] # the mean is automatically subtracted in some modules e.g. flownet2, so be careful

    assert args.static in ['dilation', 'lrr', 'unet'], "Only dilation and LRR are supported for now."

    if args.flow == 'flownet2':
        with tf.variable_scope('flow'):
            flow_network = Flownet2(bilinear_warping_module)
            flow_img0 = tf.placeholder(tf.float32)
            flow_img1 = tf.placeholder(tf.float32)
            flow_tensor = flow_network(flow_img0, flow_img1, flip=True)

    RNN = STGRU([nbr_classes, im_size[0], im_size[1]], [7, 7], bilinear_warping_module)

    gru_opt, gru_loss, gru_prediction, gru_learning_rate, \
        gru_input_images_tensor, gru_input_flow_tensor, \
        gru_input_segmentation_tensor, gru_targets = RNN.get_optimizer(args.frames)
    unary_grad_op = tf.gradients(gru_loss, gru_input_segmentation_tensor)

    gru_loss_val, gru_input_images_tensor_val, gru_input_flow_tensor_val, \
        gru_input_segmentation_tensor_val, gru_targets_val = RNN.get_validation_loss(args.frames)

    # if args.static == 'lrr':
    #     static_input = tf.placeholder(tf.float32)
    #     static_network = LRR()
    #     static_output = static_network(static_input)

    #     # static_learning_rate = tf.placeholder(tf.float32) # variable learning rate

    #     unary_opt, unary_dLdy = static_network.get_optimizer(static_input, static_output, static_learning_rate)
    # elif args.static == 'unet':
    #     static_network = sm.Unet(backbone_name=BACKBONE, encoder_weights=None, activation=ACTIVATION_FN, classes=N_CLASSES)

    # random.seed(5)
    # np.random.seed(5)
    # tf.compat.v1.random.set_random_seed(5)

    data_loader = DataLoader(im_size, args.frames, VD_TRAIN_PATH) # arg.frames is how many frames to use
    data_loader_val = DataLoader(im_size, args.frames, VD_VALIDATION_PATH)

    loss_history = np.zeros(nbr_iterations)
    loss_history_smoothed = np.zeros(nbr_iterations)
    loss_history_val = np.zeros(nbr_iterations)
    loss_history_smoothed_val = np.zeros(nbr_iterations)

    vars_trainable = [k for k in tf.trainable_variables() if not k.name.startswith('flow/')]
    # vars_static = [k for k in vars_trainable if not k in RNN.weights.values()]
    # loader_static = tf.train.Saver(vars_static)
    saver = tf.train.Saver(vars_trainable)
    
    if args.flow in ['flownet1', 'flownet2']:
        saver_fn = tf.train.Saver([k for k in tf.trainable_variables() if k.name.startswith('flow/')])
    
    if args.static == 'unet':
        static_network = sm.Unet(backbone_name=BACKBONE, encoder_weights=None, activation=ACTIVATION_FN, classes=N_CLASSES)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # if args.static == 'lrr':
        #     loader_static.restore(sess, './checkpoints/lrr_pretrained')
        # elif args.static == 'dilation':
        #     assert False, "Pretrained dilation model will soon be released."
        #     saver.restore(sess, './checkpoints/dilation_grfp')
        if args.static == 'unet':
            static_network.load_weights(TEST_MODEL)
            print("use pre-trained unet at %s" % TEST_MODEL)

        use_ckpt = 0
        if args.ckpt is not None and args.ckpt != '':
            # saver.restore(sess, './checkpoints/round1/%s' % (args.ckpt))
            use_ckpt = 1

        if args.flow == 'flownet1':
            saver_fn.restore(sess, './checkpoints/flownet1')
        elif args.flow == 'flownet2':
            saver_fn.restore(sess, './checkpoints/flownet2')
        
        min_val_loss = 1e6

        for training_it in range(nbr_iterations):

            images, ground_truth = data_loader.get_next_sequence()

            # Optical flow
            optflow = []
            for frame in range(1, args.frames):
                im, last_im = images[frame], images[frame-1]
                if args.flow == 'flownet2':
                    flow = sess.run(flow_tensor, feed_dict={flow_img0: im, flow_img1: last_im})
                optflow.append(flow)

            # Static segmentation
            static_segm = []
            for frame in range(args.frames):
                im = images[frame]
                if args.static == 'unet':
                    # print("static seg, input shape", im.shape)
                    x = static_network.predict(im)
                    # print(x.shape)
                elif args.static == 'lrr':
                    x = sess.run(static_output, feed_dict={static_input: im})
                    # print("output tensor shape", x.shape)
                static_segm.append(x)

            # GRFP
            rnn_input = {
                gru_learning_rate: learning_rate,
                # gru_learning_rate: learning_rate * (1-(training_it+1)/nbr_iterations),
                gru_input_images_tensor: np.stack(images),
                gru_input_flow_tensor: np.stack(optflow),
                gru_input_segmentation_tensor: np.stack(static_segm),
                gru_targets: ground_truth,
            }

            _, loss, pred, unary_grads = sess.run([gru_opt, gru_loss, 
               gru_prediction, unary_grad_op], feed_dict=rnn_input)
            loss_history[training_it] = loss

            ### validate as training goes ###
            images_val, ground_truth_val = data_loader_val.get_next_sequence()
            # Optical flow
            optflow = []
            for frame in range(1, args.frames):
                im, last_im = images_val[frame], images_val[frame-1]
                flow = sess.run(flow_tensor, feed_dict={flow_img0: im, flow_img1: last_im})
                optflow.append(flow)
            # Static segmentation
            static_segm = []
            for frame in range(args.frames):
                im = images_val[frame]
                x = static_network.predict(im)
                static_segm.append(x)
            # gru
            rnn_input_val = {
                gru_input_images_tensor_val: np.stack(images_val),
                gru_input_flow_tensor_val: np.stack(optflow),
                gru_input_segmentation_tensor_val: np.stack(static_segm),
                gru_targets_val: ground_truth_val,
            }
            loss_val = sess.run(gru_loss_val, feed_dict=rnn_input_val)
            loss_history_val[training_it] = loss_val
            ### validation part ends ###
            
            if training_it < 300:
                loss_history_smoothed[training_it] = np.mean(loss_history[0:training_it+1])
                loss_history_smoothed_val[training_it] = np.mean(loss_history_val[0:training_it+1])
            else:
                loss_history_smoothed[training_it] = 0.997*loss_history_smoothed[training_it-1] + 0.003*loss
                loss_history_smoothed_val[training_it] = 0.997*loss_history_smoothed_val[training_it-1] + 0.003*loss_val

            # Refine the static network?
            # The reason that a two-stage training routine is used
            # is because there is not enough GPU memory (with a 12 GB Titan X)
            # to do it in one pass.
            # if training_it+1 > t0_dilation_net:
            #     for k in range(len(images)-3, len(images)):
            #         g = unary_grads[0][k]
            #         im = images[k]
            #         _ = sess.run([unary_opt], feed_dict={
            #           static_input: im,
            #           unary_dLdy: g
            #           ,static_learning_rate: static_learning_rate_lrr * (1-(training_it+1)/nbr_iterations)**2
            #         })

            if (training_it+1) % 20 == 0:
                print("Iteration %d/%d: Training Loss %.3f" % (training_it+1, nbr_iterations, loss_history_smoothed[training_it]))
                print("Iteration %d/%d: Validation Loss %.3f" % (training_it+1, nbr_iterations, loss_history_smoothed_val[training_it]))

            if training_it+1 >= 112 and (training_it+1) % 112 == 0:
                print("epoch %d result" % ((training_it+1) / 112))
                print(np.mean(loss_history[(training_it+1-112): (training_it+1)]))
                print(np.mean(loss_history_val[(training_it+1-28): (training_it+1)])) # 28 images in validation set

                val_loss_last_epoch = np.mean(loss_history_val[(training_it+1-28): (training_it+1)])
                if val_loss_last_epoch < min_val_loss:
                    saver.save(sess, './checkpoints/%s_%s_tr%d_best_lr-3' % (args.static, args.flow, args.frames))
                    print("validation loss improved from %.4f to %.4f, save checkpoint, training epoch is %d" %\
                         (min_val_loss, val_loss_last_epoch, (training_it+1) / 112))
                    min_val_loss = val_loss_last_epoch
                else:
                    print("validation loss didn't improve from %.4f" % min_val_loss)

            if training_it > 0 and (training_it+1) % 5600 == 0:
                saver.save(sess, './checkpoints/%s_%s_tr%d_it%d_lr-3' % (args.static, args.flow, args.frames, training_it+1))

        # loss_hist_file = np.asarray(loss_history)
        # np.savetxt("./loss_hist/loss_hist_%s_%s_it%d.csv" % (args.static, args.flow, nbr_iterations + use_ckpt * 6000), loss_hist_file, delimiter=",")

        plot_loss_curve(loss_history_smoothed, loss_history_smoothed_val, \
            "%s_%s_loss_curve_f%d_lr-3" % (args.static, args.flow, args.frames))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tran GRFP on the CityScapes training set.')

    parser.add_argument('--static', help='Which static network to use.', required=True)
    parser.add_argument('--flow', help='Which optical flow method to use.', required=True)
    parser.add_argument('--frames', type=int, help='Number of frames to use.', default=5, required=False)
    parser.add_argument('--ckpt', help='checkpoint weights to use.', required=False)

    args = parser.parse_args()

    assert args.flow in ['flownet1', 'flownet2', 'farneback'], "Unknown flow method %s." % args.flow
    assert args.static in ['dilation', 'dilation_grfp', 'lrr', 'lrr_grfp', 'unet'], "Unknown static method %s." % args.static
    assert args.frames >= 1 and args.frames <= 20, "The number of frames must be between 1 and 20."
    
    train(args)