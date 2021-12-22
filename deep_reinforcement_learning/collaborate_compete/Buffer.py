# Imports
# ==================================================

# Python Modules
# --------------------------------------------------
import random
import numpy as np

from collections import (
    deque, 
    namedtuple
)


# Torch Modules
# --------------------------------------------------
import torch


# Custom Modules
# --------------------------------------------------
import config
import utils



# Buffer Classes
# ==================================================

class ReplayBuffer:
    """
    """
    
    def __init__(self, action_size, buffer_size, batch_size, device):
        """
        """
        
        self.seed = np.random.seed(config.RANDOM_SEED)
        
        random.seed(config.RANDOM_SEED)
        
        self.device      = device
        self.action_size = action_size
        self.memory      = deque(maxlen = buffer_size) 
        self.batch_size  = batch_size
        self.experience  = namedtuple(
            "Experience", 
            field_names = [
                "state", 
                "action", 
                "reward", 
                "next_state", 
                "done"
            ]
        )
        

    def add(self, state, action, reward, next_state, done):
        """
        """
        
        exp = self.experience(state, action, reward, next_state, done)
        
        self.memory.append(exp)

        
    def sample(self):
        """
        """
        
        exp = random.sample(self.memory, k = self.batch_size)
       
        return (
            torch.from_numpy(np.vstack([e.state      for e in exp if e is not None])                 ).float().to(self.device),
            torch.from_numpy(np.vstack([e.action     for e in exp if e is not None])                 ).float().to(self.device),
            torch.from_numpy(np.vstack([e.reward     for e in exp if e is not None])                 ).float().to(self.device),
            torch.from_numpy(np.vstack([e.next_state for e in exp if e is not None])                 ).float().to(self.device),
            torch.from_numpy(np.vstack([e.done       for e in exp if e is not None]).astype(np.uint8)).float().to(self.device)
        )

    
    def __len__(self):
        return len(self.memory)