import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


latest_ckp = './checkpoints/unet_flownet2_tr2_best_lr-1'
print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')