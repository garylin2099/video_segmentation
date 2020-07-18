#Intermediate   Maturation1 Maturation2 Pruned  Flowering
# STAGE = 'Intermediate'
# STAGE = 'Maturation1'
# STAGE = 'Maturation2'
# STAGE = 'Pruned'
# STAGE = 'Flowering'
# STAGE = 'all'
STAGE = 'train0-5_val6_no_augment'

VD_TRAIN_PATH = './video/train/'
VD_VALIDATION_PATH = './video/val'
SF_TRAIN_PATH = './single_frame/train/'
SF_VALIDATION_PATH = './single_frame/val'
MODEL_PATH = './models'

IM_WIDTH = 512
IM_HEIGHT = 512
N_CLASSES = 7
BATCH_SIZE = 32
N_EPOCHS = 100

RANDOM_SEED = 42

BACKBONE = 'resnet18'
ARCHITECTURE = 'unet'
ACTIVATION_FN = 'relu'

LOSS_FN = 'weighted_ce'

LOSS_WEIGHTS = [1.5, 2, 2.5, 1.5]

BASELINE_FILE = '{}/baseline_model.h5'.format(MODEL_PATH)
LAST_SAVED_MODEL = '{}/unet_resnet18_weighted_jaccard.h5'.format(MODEL_PATH)
CHECKPOINT_FILE = ('{}/{}_{}_{}_'+STAGE+'.h5').format(MODEL_PATH, ARCHITECTURE, BACKBONE, LOSS_FN)


TYPES_TO_COLORS = {
    'other': (0,0,0), # all < 5
    'nasturtium': (0, 0, 254), #b > 230
    'borage': (251, 1, 6), #r > 230
    'bok_choy':(33, 254, 6), # g > 230
    'plant1': (0, 255, 255), #g and b > 230
    'plant2': (251, 2, 254), #r and b > 230
    'plant3': (252, 127, 8) #r > 230 and g >100
}
TYPES_TO_CHANNEL= {
    'other': (5,5,5),
    'nasturtium': 2,
    'borage': 0, 
    'bok_choy':1,
    'plant1': (1,2),
    'plant2': (0,2),
    'plant3': (0,1)
}
BINARY_ENCODINGS = {
    'other': [1,0,0,0,0,0,0],
    'nasturtium': [0,1,0,0,0,0,0],
    'borage': [0,0,1,0,0,0,0],
    'bok_choy': [0,0,0,1,0,0,0],
    'plant1': [0,0,0,0,1,0,0],
    'plant2': [0,0,0,0,0,1,0], 
    'plant3': [0,0,0,0,0,0,1]
}


COLORS = [(0, 0, 0), (0, 0, 254), (251, 1, 6), (33, 254, 6), (0, 255, 255), (251, 2, 254), (252, 127, 8)] 
TYPES = ['other','nasturtium','borage','bok_choy', 'plant1', 'plant2', 'plant3']


IOU_TEST_RATIO = 1.0

#################################################################
# #Intermediate   Maturation1 Maturation2 Pruned  Flowering
# STAGE_TEST = 'Intermediate'
# STAGE_TEST = 'Maturation1'
# STAGE_TEST = 'Maturation2'
# STAGE_TEST = 'Pruned'
# STAGE_TEST = 'Flowering'
# STAGE_TESTm = 'all'
# TEST_PATH = './Overhead/'+ STAGE_TEST
# IOU_EVAL_FILE = 'unet_iou_eval'+STAGE_TEST+STAGE_TESTm+'.csv'

# TEST_MODEL =  ('{}/{}_{}_{}_'+STAGE_TESTm+'.h5').format(MODEL_PATH, ARCHITECTURE, BACKBONE, LOSS_FN)
# TEST_MODEL = LAST_SAVED_MODEL
# TEST_MODEL = CHECKPOINT_FILE

######################################################################
#Intermediate   Maturation1 Maturation2 Pruned  Flowering
# STAGE_TEST = 'Intermediate'
# STAGE_TEST = 'Maturation1'
# STAGE_TEST = 'Maturation2'
# STAGE_TEST = 'Pruned'
# STAGE_TEST = 'Flowering'
STAGE_TEST = 'train0-5_val6_no_augment'

TEST_PATH = './single_frame/test/'
IOU_EVAL_FILE = 'unet_iou_eval'+STAGE_TEST+'trainonly.csv'

TEST_MODEL =  ('{}/{}_{}_{}_'+STAGE_TEST+'.h5').format(MODEL_PATH, ARCHITECTURE, BACKBONE, LOSS_FN)
# TEST_MODEL = LAST_SAVED_MODEL
# TEST_MODEL = CHECKPOINT_FILE
