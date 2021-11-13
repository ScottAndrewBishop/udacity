# Global Constants
# --------------------------------------------------
RANDOM_SEED = 42


# Enviornment Constants
# --------------------------------------------------
UNITY_APP = "Banana"

ENV_TRAIN_MODE = True


# Agent Constants
# --------------------------------------------------                         
AGENT_BATCH_SIZE   = 64          
AGENT_BUFFER_SIZE  = 10000            
AGENT_GAMMA        = 0.99     
AGENT_LR           = 0.0001             
AGENT_TAU          = 0.001   
AGENT_UPDATE_EVERY = 4            
AGENT_USE_DUEL     = False        
AGENT_USE_DOUBLE   = True 


# Training Constants
# --------------------------------------------------
N_EPISODES = 2000
MAX_T      = 1000
EPS_START  = 1.0
EPS_END    = 0.01
EPS_DECAY  = 0.995

CHECKPOINT_DIR  = "checkpoints"
CHECKPOINT_NAME = "checkpoint.pth"

SCORES_DIR  = "scores"
SCORES_NAME = "scores.npy"

SCORE_WINDOW_SIZE = 100


# Plot Constants
# --------------------------------------------------
PLOT_DIR  = "images"
PLOT_NAME = "results.jpg"