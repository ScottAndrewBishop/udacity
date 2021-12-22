# Imports
# ==================================================

# Python Modules
# -------------------------
import os
import sys



# Global Constants
# ==================================================

RANDOM_SEED = 19920917



# I/O Constants
# ==================================================

# Coco
# -------------------------
COCO_API_ROOT = f"/opt"
COCO_API_DIR  = f"{COCO_API_ROOT}/cocoapi"

COCO_API_PATH = f"{COCO_API_DIR}/PythonAPI"
COCO_ANN_PATH = f"{COCO_API_DIR}/annotations"
COCO_IMG_PATH = f"{COCO_API_DIR}/images"

COCO_TRAIN_DATA = f"train2014"
COCO_VAL_DATA   = f"val2014"
COCO_TEST_DATA  = f"test2014"

ANN_CAPTIONS_TRAIN_FILE = f"{COCO_ANN_PATH}/captions_{COCO_TRAIN_DATA}.json"

ANN_CAPTIONS_VAL_FILE = f"{COCO_ANN_PATH}/captions_{COCO_VAL_DATA}.json"
ANN_INSTANCE_VAL_FILE = f"{COCO_ANN_PATH}/instances_{COCO_VAL_DATA}.json"

ANN_INFO_TEST_FILE = f"{COCO_ANN_PATH}/image_info_{COCO_TEST_DATA}.json"


# Logs
# -------------------------
LOG_DIR        = f"logs"
TRAIN_LOG_FILE = f"{LOG_DIR}/train_log.txt"


# Model
# -------------------------
MODEL_DIR = "models"


# Vocab
# -------------------------
VOCAB_DIR  = f"vocab"
VOCAB_FILE = f"{VOCAB_DIR}/vocab.pkl"



# NLP Constants
# ==================================================

START_WORD = "<start>"
END_WORD   = "<end>"
UNK_WORD   = "<unk>"

LOAD_VOCAB_FILE = True
VOCAB_THRESH    = 5



# COCO Constants
# ==================================================

COCO_UID_COL     = "id"
COCO_IMG_ID_COL  = "image_id"
COCO_IMG_COL     = "images"
COCO_URL_COL     = "coco_url"
COCO_CAPTION_COL = "caption"
COCO_FN_COL      = "file_name"



# Transform Constants
# ==================================================

IMAGENET_MU    = (0.485, 0.456, 0.406)      
IMAGENET_SIGMA = (0.229, 0.224, 0.225)
    


# Hyperparameter Constants
# ==================================================

INPUT_SIZE  = 224
EMBED_SIZE  = 256
HIDDEN_SIZE = 512
NUM_LAYERS  = 2



# Train Constants
# ==================================================

LOAD_WEIGHTS = False
BEST_EPOCH   = 1

TRAIN_BATCH_SIZE = 32
TRAIN_EPOCHS     = 3
TRAIN_SAVE_FREQ  = 1
TRAIN_PRINT_FREQ = 100



# Inference Constants
# ==================================================

TEST_BATCH_SIZE = 1