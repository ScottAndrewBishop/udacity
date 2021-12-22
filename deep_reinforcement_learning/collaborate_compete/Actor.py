# Imports
# ==================================================

# Torch Modules
# --------------------------------------------------
import torch
import torch.nn            as nn
import torch.nn.functional as F


# Custom Modules
# --------------------------------------------------
import config
import utils



# Actor Classes
# ==================================================

class Actor(nn.Module):
    """
    """
    
    def __init__(self, state_size, action_size):
        """
        """
        
        super(Actor, self).__init__()
        
        self.seed = torch.manual_seed(config.RANDOM_SEED)
        self.leak = config.LEAKINESS
        self.bn   = nn.BatchNorm1d(state_size)

        self.fc1 = nn.Linear(state_size, config.FC1)
        self.fc2 = nn.Linear(config.FC1, config.FC2)
        self.fc3 = nn.Linear(config.FC2, action_size)
        
        self.reset_parameters()

        
    def reset_parameters(self):
        """
        """
        
        nn.init.kaiming_normal_(self.fc1.weight.data, a = self.leak, mode = "fan_in")
        nn.init.kaiming_normal_(self.fc2.weight.data, a = self.leak, mode = "fan_in")
        
        nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

        
    def forward(self, state):
        """
        """
        
        x = F.leaky_relu(self.fc1(self.bn(state)), negative_slope = self.leak)
        x = F.leaky_relu(self.fc2(x),              negative_slope = self.leak)
        
        return torch.tanh(self.fc3(x))