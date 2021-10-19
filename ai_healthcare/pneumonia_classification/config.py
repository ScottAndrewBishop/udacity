# Imports
# ==================================================

from keras.layers import (
    Dense, 
    Dropout, 
    Flatten,
    MaxPooling2D
)



# I/O Constants
# ==================================================

# File Extensions
# --------------------------------------------------
MODEL_EXT   = ".json"
WEIGHT_EXT  = ".best.hdf5".format("xray_class")
HISTORY_EXT = ".pickle"


# Inputs
# --------------------------------------------------
ALL_XRAY_PATH  = "/data/Data_Entry_2017.csv"
SAMPLE_PATH    = "samples/sample_labels.csv"

TEST_DICOMS = [
    "dicom/test1.dcm",
    "dicom/test2.dcm",
    "dicom/test3.dcm",
    "dicom/test4.dcm",
    "dicom/test5.dcm",
    "dicom/test6.dcm"
]


# Outputs
# --------------------------------------------------
WEIGHT_PATH  = "weights"
MODEL_PATH   = "models"
HISTORY_PATH = "histories"



# Data Constants
# ==================================================

DV             = "Class"
UID            = "Patient ID"
IMG_IDX_COL    = "Image Index"
IMG_PATH_COL   = "Path"
LABEL_COL      = "Finding Labels"
TARGET_DISEASE = "Pneumonia"



# Filter Constants
# ==================================================

PNEUMONIA_MEAN_AGE = 45
PNEUMONIA_STD_AGE  = 18

YOUNGEST_AGE = PNEUMONIA_MEAN_AGE - (2 * PNEUMONIA_STD_AGE)
OLDEST_AGE   = PNEUMONIA_MEAN_AGE + (2 * PNEUMONIA_STD_AGE)

OTSU_THRESH = 0.05


# Split Constants
# ==================================================

VAL_DATA_RATIO = 0.2



# Augmentation Constants
# ==================================================

H_FLIP     = True
V_FLIP     = False
H_SHIFT    = 0.1
W_SHIFT    = 0.1 
SHEAR      = 0.1
ZOOM       = 0.1
FILL_MODE  = "constant"
CVAL       = 0.0
BRIGHTNESS = [0.85, 1.15]
ROTATION   = 10
IMG_SIZE   = (224, 224)
CLASS_MODE = "binary"

# https://en.wikipedia.org/wiki/Grayscale
Y = [0.2126, 0.7152, 0.0722]

# https://pytorch.org/vision/stable/models.html
IM = [0.485, 0.456, 0.406]
IS = [0.229, 0.224, 0.225]

IMAGENET_MU    = (Y[0] * IM[0]) + (Y[1] * IM[1]) + (Y[2] * IM[2])
IMAGENET_SIGMA = (Y[0] * IS[0]) + (Y[1] * IS[1]) + (Y[2] * IS[2])



# Model Constants
# ==================================================

INPUT_IMG_SIZE = (1, IMG_SIZE[0], IMG_SIZE[0], 3)

EPOCHS           = 50
EARLY_STOP       = 20
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 512

ACTIVATION        = "relu"
OUTPUT_ACTIVATION = "sigmoid"

LOSS        = "binary_crossentropy"
METRICS     = ["accuracy"]
LEARN_RATE  = 0.001
LR_FACTOR   = 0.1
LR_PATIENCE = 10

CALLBACK_MONITOR = "val_loss"
CALLBACK_MODE    = "min"


# ResNet50
# --------------------------------------------------
# config.RESNET50_OUTPUT_LAYER = ""


# VGG16
# --------------------------------------------------
VGG16_OUTPUT_LAYER  = "block5_conv3"
VGG16_ARCHITECTURES = {
    "model_1" : [
        # Add MaxPool 1x1 for CAM
        # -------------------------
        MaxPooling2D(pool_size = (1, 1)),
        
        # Add VGG16's MaxPool Following Final Conv2D
        # -------------------------
        MaxPooling2D(pool_size = (2, 2)),
        
        # Tune VGG16's Architecture
        # -------------------------
        MaxPooling2D(pool_size = (7, 7)),
        Flatten(),
        Dropout(0.5),
        Dense(256, activation = ACTIVATION),
        Dense(1, activation = OUTPUT_ACTIVATION)
    ],
    "model_2" : [
        # MaxPool 1x1 for CAM
        # -------------------------
        MaxPooling2D(pool_size = (1, 1)),
        
        # VGG16 MaxPool Following Final Conv2D
        # -------------------------
        MaxPooling2D(pool_size = (2, 2)),
        
        # Tune VGG16's Architecture
        # -------------------------
        Flatten(),
        Dropout(0.5),
        Dense(1024, activation = ACTIVATION),
        Dropout(0.5),
        Dense(512, activation = ACTIVATION),
        Dropout(0.5),
        Dense(256, activation = ACTIVATION),
        Dense(1, activation = OUTPUT_ACTIVATION)
    ],
    "model_3" : [
        # Add MaxPool 1x1 for CAM
        # -------------------------
        MaxPooling2D(pool_size = (1, 1)),
        
        # Add VGG16's MaxPool Following Final Conv2D
        # -------------------------
        MaxPooling2D(pool_size = (2, 2)),
        
        # Tune VGG16's Architecture
        # -------------------------
        MaxPooling2D(pool_size = (7, 7)),
        Flatten(),
        Dropout(0.5),
        Dense(1, activation = OUTPUT_ACTIVATION)
    ],
}


# Xception
# --------------------------------------------------
# config.XCEPTION_OUTPUT_LAYER = ""



# Evaluation Constants
# ==================================================

BEST_MODEL = "model_3"

BEST_MODEL_PATH    = f"{MODEL_PATH}/vgg16/{BEST_MODEL}{MODEL_EXT}"
BEST_WEIGHT_PATH   = f"{WEIGHT_PATH}/vgg16/{BEST_MODEL}{WEIGHT_EXT}"
BEST_MODEL_HISTORY = f"{HISTORY_PATH}/vgg16/{BEST_MODEL}{HISTORY_EXT}"



# Inferencing Constants
# ==================================================

VALID_PATIENT_POSITIONS = ["AP", "PA"]
VALID_MODALITY          = ["DX"]
VALID_BODY_PART         = ["CHEST"]

BEST_THRESH = 0.49