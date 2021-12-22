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



# Critic Classes
# ==================================================

class Critic(nn.Module):
    """
    """
    
    def __init__(self, state_size, action_size):
        """
        """
        
        super(Critic, self).__init__()
        
        self.seed = torch.manual_seed(config.RANDOM_SEED)
        self.leak = config.LEAKINESS
        self.bn   = nn.BatchNorm1d(state_size)
        
        self.fc1 = nn.Linear(              state_size, config.FC1)
        self.fc2 = nn.Linear(config.FC1 + action_size, config.FC2)
        self.fc3 = nn.Linear(              config.FC2, config.FC3)
        self.fc4 = nn.Linear(              config.FC3,          1)
        
        self.reset_parameters()

        
    def reset_parameters(self):
        """
        """
        
        nn.init.kaiming_normal_(self.fc1.weight.data, a = self.leak, mode = "fan_in")
        nn.init.kaiming_normal_(self.fc2.weight.data, a = self.leak, mode = "fan_in")
        
        nn.init.uniform_(self.fc3.weight.data, -3e-3, 3e-3)

        
    def forward(self, state, action):
        """
        """
        
        x = F.leaky_relu(self.fc1(self.bn(state)), negative_slope = self.leak)
        
        x = torch.cat((x, action), dim = 1)
        
        x = F.leaky_relu(self.fc2(x), negative_slope = self.leak)
        x = F.leaky_relu(self.fc3(x), negative_slope = self.leak)
        
        return self.fc4(x)