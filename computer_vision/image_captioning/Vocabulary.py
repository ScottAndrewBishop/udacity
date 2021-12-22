# Imports
# ==================================================

# Python Modules
# -------------------------
import nltk
import pickle
import os.path

from collections import Counter


# Third Paty Modules
# -------------------------
from pycocotools.coco import COCO


# Custom Modules
# -------------------------
import config



# Vocab Classes
# ==================================================

class Vocabulary(object):
    """
    """

    def __init__(self, ann_file, vocab_from_file):
        """
        """
        
        self.vocab_from_file  = vocab_from_file
        self.vocab_threshold  = config.VOCAB_THRESH
        self.vocab_file       = config.VOCAB_FILE
        self.start_word       = config.START_WORD
        self.end_word         = config.END_WORD
        self.unk_word         = config.UNK_WORD
        self.annotations_file = ann_file
        
        self.get_vocab()

        
    def get_vocab(self):
        """
        """
        
        if self.vocab_from_file & os.path.exists(self.vocab_file):
            with open(self.vocab_file, "rb") as f:
                vocab         = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            
        else:
            self.build_vocab()
            with open(self.vocab_file, "wb") as f:
                pickle.dump(self, f)
        
        
    def build_vocab(self):
        """
        """
        
        self.init_vocab()
        
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        
        self.add_captions()

        
    def init_vocab(self):
        """
        """
        
        self.word2idx = {}
        self.idx2word = {}
        self.idx      = 0

        
    def add_word(self, word):
        """
        """
        
        if not word in self.word2idx:
            self.word2idx[word]      = self.idx
            self.idx2word[self.idx]  = word
            self.idx                += 1

            
    def add_captions(self):
        """
        """
        
        coco    = COCO(self.annotations_file)
        counter = Counter()
        ids     = coco.anns.keys()
        
        for i, id in enumerate(ids):
            caption = str(coco.anns[id][config.COCO_CAPTION_COL])
            tokens  = nltk.tokenize.word_tokenize(caption.lower())
            
            counter.update(tokens)

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

            
    def __call__(self, word):
        """
        """
        
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        
        return self.word2idx[word]

    
    def __len__(self):
        """
        """
        
        return len(self.word2idx)