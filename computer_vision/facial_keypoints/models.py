# Imports
# ==================================================

# Torch Modules
# -------------------------
import torch
import torch.nn            as nn
import torch.nn.init       as I
import torch.nn.functional as F


# Custom Modules
# -------------------------
import config



# Classes
# ==================================================

# Inspired by NaimishNet - https://arxiv.org/pdf/1710.00977.pdf
class Net(nn.Module):
    """
    """

    def __init__(self):
        """
        """
        
        super(Net, self).__init__()

        # Input (224 x 224 x 1)
        # --------------------------------------------------
        
        
        # Convolution Layers
        # --------------------------------------------------
        self.conv1 = nn.Conv2d(1, 32, 5) # (220 x 220 x 32)
        self.pool1 = nn.MaxPool2d(2, 2)  # (110 x 110 x 32)
        self.drop1 = nn.Dropout(0.1)     # (110 x 110 x 32)
        
        self.conv2 = nn.Conv2d(32, 64, 4) # (107 x 107 x 64)
        self.pool2 = nn.MaxPool2d(2, 2)   # ( 53 x  53 x 64)
        self.drop2 = nn.Dropout(0.2)      # ( 53 x  53 x 64)
        
        self.conv3 = nn.Conv2d(64, 128, 3) # (51 x 51 x 128)
        self.pool3 = nn.MaxPool2d(2, 2)    # (25 x 25 x 128)
        self.drop3 = nn.Dropout(0.3)       # (25 x 25 x 128)
        
        self.conv4 = nn.Conv2d(128, 256, 2) # (24 x 24 x 256)
        self.pool4 = nn.MaxPool2d(2, 2)     # (12 x 12 x 256)
        self.drop4 = nn.Dropout(0.4)        # (12 x 12 x 256)
        
        self.conv5 = nn.Conv2d(256, 256, 1) # (12 x 12 x 256)
        self.pool5 = nn.MaxPool2d(2, 2)     # ( 6 x  6 x 256)
        self.drop5 = nn.Dropout(0.5)        # ( 6 x  6 x 256)
        
        
        # Flatten (6 * 6 * 256 x 1)
        # --------------------------------------------------
        
        
        # Fully Connected Layers
        # --------------------------------------------------
        self.fc1   = nn.Linear(9216, 1024) # (1024)
        self.drop6 = nn.Dropout(0.5)       # (1024)
        
        self.fc2   = nn.Linear(1024, 1024) # (1024)
        self.drop7 = nn.Dropout(0.5)       # (1024)
        
        self.fc3 = nn.Linear(1024, config.NUM_KEYPOINTS * 2) # (136)
        
        I.xavier_uniform_(self.fc1.weight.data)
        I.xavier_uniform_(self.fc2.weight.data)
        I.xavier_uniform_(self.fc3.weight.data)

        
    def forward(self, x):
        """
        """
        
        # Convolutions
        # --------------------------------------------------
        x = self.drop1(self.pool1(F.elu(self.conv1(x))))
        x = self.drop2(self.pool2(F.elu(self.conv2(x))))
        x = self.drop3(self.pool3(F.elu(self.conv3(x))))
        x = self.drop4(self.pool4(F.elu(self.conv4(x))))
        x = self.drop5(self.pool5(F.elu(self.conv5(x))))
        
        
        # Flatten
        # --------------------------------------------------
        x = x.view(x.size(0), -1)
        
        
        # Fully Connected Layers
        # --------------------------------------------------
        x = self.drop6(F.elu(self.fc1(x)))
        x = self.drop7(F.elu(self.fc2(x)))
        x = self.fc3(x)
        
        return x