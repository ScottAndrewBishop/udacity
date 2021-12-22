# Imports
# ==================================================

# Python Modules
# -------------------------
import os
import nltk
import json
import numpy as np

from PIL  import Image
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

from Vocabulary import Vocabulary



# Dataset Classes
# ==================================================

class CocoDataset(data.Dataset):
    """
    """

    def __init__(self, transform, mode, batch_size, ann_file, vocab_from_file, img_folder):
        """
        """
        
        self.transform  = transform
        self.mode       = mode
        self.batch_size = batch_size
        self.img_folder = img_folder
        self.vocab      = Vocabulary(
            ann_file, 
            vocab_from_file
        )
        
        if self.mode == "train":
            self.coco = COCO(config.ANN_CAPTIONS_TRAIN_FILE)
            self.ids  = list(self.coco.anns.keys())
            
            all_tokens = [
                nltk.tokenize.word_tokenize(
                    str(self.coco.anns[self.ids[index]][config.COCO_CAPTION_COL]).lower()
                ) for index in tqdm(np.arange(len(self.ids)))
            ]
            
            self.caption_lengths = [len(token) for token in all_tokens]
            
        else:
            test_info  = json.loads(open(ann_file).read())
            self.paths = [item[config.COCO_FN_COL] for item in test_info[config.COCO_IMG_COL]]

        
    def get_train_indices(self):
        """
        """
        
        sel_length  = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices     = list(np.random.choice(all_indices, size=self.batch_size))
        
        return indices
    
    
    def __getitem__(self, index):
        """
        """
        
        if self.mode == 'train':
            ann_id  = self.ids[index]
            caption = self.coco.anns[ann_id][config.COCO_CAPTION_COL]
            img_id  = self.coco.anns[ann_id][config.COCO_IMG_ID_COL]
            path    = self.coco.loadImgs(img_id)[0][config.COCO_FN_COL]

            image = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            image = self.transform(image)

            tokens  = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            
            caption = torch.Tensor(caption).long()

            return (
                image, 
                caption
            )


        else:
            path = self.paths[index]

            PIL_image  = Image.open(os.path.join(self.img_folder, path)).convert("RGB")
            orig_image = np.array(PIL_image)
            image      = self.transform(PIL_image)

            return (
                orig_image, 
                image
            )

    
    def __len__(self):
        """
        """
        
        if self.mode == "train": return len(self.ids)
        else:                    return len(self.paths)