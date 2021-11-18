# Import
# ==================================================

# Python Modules
# -------------------------
import os
import numpy as np



# I/O 
# ==================================================

# Directories
# -------------------------
DATA_DIR         = "data"
IMAGE_DIR        = "images"
HAAR_CASCADE_DIR = "haar_cascade"
MODEL_DIR        = "models"


# Images
# -------------------------
TRAIN_DATA_FILEPATH = os.path.join("..", "..", DATA_DIR, "training")
TEST_DATA_FILEPATH  = os.path.join("..", "..", DATA_DIR, "test")
TEST_IMAGE_PATH     = os.path.join(IMAGE_DIR, "obamas.jpg")
SUNGLASS_FILEPATH   = os.path.join(IMAGE_DIR, "sunglasses.png")


# Keypoints
# -------------------------
TRAIN_KEYPOINTS_PATH = os.path.join("..", "..", DATA_DIR, "training_frames_keypoints.csv")
TEST_KEYPOINTS_PATH  = os.path.join("..", "..", DATA_DIR, "test_frames_keypoints.csv")


# Haar Cascades
# -------------------------
HAAR_FACE_FILENAME   = "haar_cascade_frontalface_default.xml"
HAAR_EYE_FILENAME    = "haar_cascade_eye.xml"
HAAR_SMILE_FILENAME  = "haar_cascade_smile.xml"


# Models
# -------------------------
MODEL_VER  = 2
MODEL_NAME = f"keypoints_model_{MODEL_VER}.pt"



# Transform 
# ==================================================

NORM_KP_MEAN = 100.0
NORM_KP_STD  = 50.0

RESCALE_SIZE = 250
CROP_SIZE    = 224



# Training 
# ==================================================

EPOCHS = 20

DATA_LOADER_BATCH_SIZE  = 32
DATA_LOADER_SHUFFLE     = True
DATA_LOADER_NUM_WORKERS = 0

NUM_KEYPOINTS = 68



# Optimizer
# ==================================================

OPTIM_LR      = 0.001
OPTIM_BETAS   = (0.9, 0.999)
OPTIM_EPSILON = 1e-8



# Plot
# ==================================================

MAX_ERROR_PLOTS = 8



# Haar Cascade
# ==================================================

HAAR_PADDING = 50


# Face
# -------------------------
FACE_SCALE_FACTOR  = 1.2
FACE_MIN_NEIGHBORS = 2


# Eyes
# -------------------------
EYE_SCALE_FACTOR  = 1.3
EYE_MIN_NEIGHBORS = 1


# Smiles
# -------------------------
SMILE_SCALE_FACTOR  = 1.2
SMILE_MIN_NEIGHBORS = 40



# Filter
# ==================================================

FILTER_IMAGE_IDX = 800

# Brow
# -------------------------
RIGHT_BROW_EDGE_KP = 17
LEFT_BROW_EDGE_KP  = 26


# Nose
# -------------------------
TOP_NOSE_KP        = 27
BOTTOM_NOSE_KP     = 34



# Udactiy 
# ==================================================

DELAY     = 4 * 60
MIN_DELAY = 2 * 60

INTERVAL     = 4 * 60
MIN_INTERVAL = 2 * 60

KEEP_ALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL      = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS  = {
    "Metadata-Flavor" : "Google"
}