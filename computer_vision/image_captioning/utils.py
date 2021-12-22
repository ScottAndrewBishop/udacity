# Imports
# ==================================================

# Python Modules
# -------------------------
import os
import random
import json
import numpy             as np
import matplotlib.pyplot as plt

from tqdm import tqdm


# Torch Modules
# -------------------------
import torch
import torch.utils.data as data


# Third Party Modules
# -------------------------
from pycocotools.coco import COCO


# Custom Modules
# -------------------------
import config

from Dataset    import CocoDataset
from Vocabulary import Vocabulary



# Data Methods
# ==================================================

def get_loader(transform, mode, batch_size, ann_file, vocab_from_file):
    """
    """
    
    # Preprocessing Checks
    # --------------------------------------------------
    assert mode in ["train", "test"], "[ERROR] mode must be one of 'train' or 'test'"
    
    if vocab_from_file == False: 
        assert mode == "train", "[ERROR] to generate vocab from captions file, must be in training mode"

    if mode == "train":
        if vocab_from_file == True: 
            assert os.path.exists(config.VOCAB_FILE), "[ERROR] vocab_file does not exist"
            
        img_folder       = f"{config.COCO_IMG_PATH}/{config.COCO_TRAIN_DATA}"
        annotations_file = ann_file
        
    if mode == "test":
        assert batch_size == 1,                   "[ERROR] please change batch_size to 1 if testing your model"
        assert os.path.exists(config.VOCAB_FILE), "[ERROR] must first generate vocab.pkl from training data"
        assert vocab_from_file == True,           "[ERROR] change vocab_from_file to True"
        
        img_folder       = f"{config.COCO_IMG_PATH}/{config.COCO_TEST_DATA}"
        annotations_file = config.ANN_INFO_TEST_FILE

        
    # Load COCO Data
    # --------------------------------------------------
    dataset = CocoDataset(
        transform        = transform,
        mode             = mode,
        batch_size       = batch_size,
        ann_file         = annotations_file,
        vocab_from_file  = vocab_from_file,
        img_folder       = img_folder
    )

    
    # Train Mode Settings
    # --------------------------------------------------
    if mode == 'train':
        indices         = dataset.get_train_indices()
        initial_sampler = data.sampler.SubsetRandomSampler(indices = indices)
        
        data_loader = data.DataLoader(
            dataset       = dataset, 
            num_workers   = 0,
            batch_sampler = data.sampler.BatchSampler(
                sampler    = initial_sampler,
                batch_size = dataset.batch_size,
                drop_last  = False)
        )
        
    else:
        data_loader = data.DataLoader(
            dataset     = dataset,
            batch_size  = dataset.batch_size,
            shuffle     = True,
            num_workers = 0
        )

    return data_loader



# Inference Methods
# ==================================================

def clean_sentence(data_loader, output):
    """
    """
    
    sentence = []
    
    for word in [data_loader.dataset.vocab.idx2word.get(x) for x in output]:
        if   word == config.END_WORD:   break
        elif word == config.START_WORD: continue
        else:                           sentence.append(word)
    
    return " ".join(sentence).replace(" .", ".")


def get_prediction(encoder, decoder, device, data_loader):
    """
    """
    
    orig_image, image = next(iter(data_loader))
    
    plt.figure(figsize = (20, 10))
    plt.imshow(np.squeeze(orig_image))
    plt.title("Sample Image")
    plt.axis("off")
    plt.show()
    
    image    = image.to(device)
    features = encoder(image).unsqueeze(1)
    output   = decoder.sample(features)    
    sentence = clean_sentence(data_loader, output)
    
    print(sentence)