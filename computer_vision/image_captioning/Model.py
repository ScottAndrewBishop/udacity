# Imports
# ==================================================

# Torch Modules
# -------------------------
import torch
import torch.nn           as nn
import torchvision.models as models


# Custome Modules
# -------------------------
import config



# Model Classes
# ==================================================

# Encoders 
# -------------------------
class EncoderCNN(nn.Module):
    """
    """
    
    def __init__(self):
        """
        """
        
        super(EncoderCNN, self).__init__()
        
        
        # Load Pre-Trained ResNet Model
        # --------------------------------------------------
        resnet = models.resnet50(pretrained = True)
        
        
        # Remove Final FC Layer
        # --------------------------------------------------
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules     = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        
        # Define Final FC Layer (Input to RNN Decoder)
        # --------------------------------------------------
        self.embed = nn.Linear(resnet.fc.in_features, config.EMBED_SIZE)
        self.norm  = nn.BatchNorm1d(config.EMBED_SIZE)

        
    def forward(self, images):
        """
        """
        
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.norm(features)
        
        return features
    

# Decoders 
# -------------------------
class DecoderRNN(nn.Module):
    """
    """
    
    def __init__(self, vocab_size):
        """
        """
        
        super(DecoderRNN, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, config.EMBED_SIZE)
        self.lstm  = nn.LSTM(config.EMBED_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, batch_first = True)
        self.fc    = nn.Linear(config.HIDDEN_SIZE, vocab_size)
    
    
    def forward(self, features, captions):
        """
        """
        
        embeddings = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        
        out, _ = self.lstm(embeddings)
        out    = self.fc(out)
        
        return out

    
    def sample(self, inputs, states = None, max_len = 20):
        """
        """
        
        words = []
        
        for i in range(max_len):                                    
            out, states = self.lstm(inputs, states)        
            out         = self.fc(out.squeeze(1))
            pred        = out.max(1)[1]
            
            words.append(pred.item())
            
            inputs = self.embed(pred)
            inputs = inputs.unsqueeze(1)                   

        return words