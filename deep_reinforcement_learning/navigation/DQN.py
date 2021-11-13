# Imports 
# ==================================================

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

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        
        self.seed = torch.manual_seed(config.RANDOM_SEED)
        
        self.fc_1 = nn.Linear(state_size, 64)
        self.fc_2 = nn.Linear(64,         64)
        self.fc_3 = nn.Linear(64,         action_size)
        
        
    def forward(self, state):
        # First layer
        x = self.fc_1(state)
        x = F.relu(x)
        
        # Second layer
        x = self.fc_2(x)
        x = F.relu(x)
        
        # Third layer (output layer)
        x = self.fc_3(x)
        
        return x
    
    
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        
        self.seed = torch.manual_seed(config.RANDOM_SEED)
        
        # First hidden layer
        self.fc_1 = nn.Linear(state_size, 64)
        
        # Advantage
        self.fc_adv = nn.Linear(64, action_size)
        
        # Value
        self.fc_val = nn.Linear(64, 1)
        
        
    def forward(self, state):
        # Hidden layer
        x = self.fc_1(state)
        x = F.relu(x)
        
        # Value
        value = self.fc_val(x)
        value = F.relu(value)
        
        # Advantage
        advantage = self.fc_adv(x)
        advantage = F.relu(advantage)
        
        # Output layer
        return value + (advantage - torch.mean(advantage, dim = 1, keepdim = True))