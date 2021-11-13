# Imports 
# ==================================================

# Python Modules
# -------------------------
import random
import collections
import numpy as np


# Torch Modules
# -------------------------
import torch
import torch.nn            as nn
import torch.nn.functional as F


# Custom Modules
# -------------------------
import config



# Class
# ==================================================

class ReplayBuffer():
    def __init__(self, action_size, batch_size, buffer_size, cuda):
        self.seed = random.seed(config.RANDOM_SEED)
        self.cuda = cuda
        
        self.action_size = action_size
        self.batch_size  = batch_size
        self.memory      = collections.deque(maxlen = buffer_size)
        
        self.exp = collections.namedtuple(
            typename    = "exp",
            field_names = [
                "state",
                "action",
                "reward",
                "next_state",
                "done"
            ]
        )
        
        
    def add(self, state, action, reward, next_state, done):
        self.memory.append(
            self.exp(state, action, reward, next_state, done)
        )
        
        
    def sample(self):
        exp = random.sample(
            population = self.memory,
            k          = self.batch_size
        )
        
        return(
            torch.FloatTensor(np.vstack([e.state      for e in exp if e is not None])                 ).to(self.cuda), 
            torch.LongTensor( np.vstack([e.action     for e in exp if e is not None])                 ).to(self.cuda), 
            torch.FloatTensor(np.vstack([e.reward     for e in exp if e is not None])                 ).to(self.cuda), 
            torch.FloatTensor(np.vstack([e.next_state for e in exp if e is not None])                 ).to(self.cuda),
            torch.FloatTensor(np.vstack([e.done       for e in exp if e is not None]).astype(np.uint8)).to(self.cuda)
        )
    
    
    def __len__(self):
        return len(self.memory)