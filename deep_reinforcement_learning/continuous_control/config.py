# Global Constants
# ==================================================

RANDOM_SEED = 42



# I/O Constants
# ==================================================

CHECKPOINT_DIR    = "checkpoints"
CHECKPOINT_ACTOR  = f"{CHECKPOINT_DIR}/actor.pth"
CHECKPOINT_CRITIC = f"{CHECKPOINT_DIR}/critic.pth"

SCORES_DIR  = "scores"
SCORES_NAME = f"{SCORES_DIR}/scores.npy"

PLOT_DIR  = "images"
PLOT_NAME = f"{PLOT_DIR}/results.jpg"



# Enviornment Constants
# ==================================================

UNITY_APP      = "Reacher"
ENV_TRAIN_MODE = True


# Agent Constants
# ==================================================

FC1         = 256, 
FC2         = 128, 
FC3         = 128,
LEAKINESS   = 1e-2,
ACTOR_LR    = 1e-4,
CRITIC_LR   = 3e-4,
BUFFER_SIZE = 1000000,
BATCH_SIZE  = 1024,
GAMMA       = 0.99,
TAU         = 1e-3,
DECAY       = 1e-4,
MU          = 0.0, 
THETA       = 0.15, 
SIGMA       = 0.2


# Training Constants
# ==================================================
 
N_EPISODES  = 2000
MAX_T       = 1000 
EPOCHS      = 1000
WINDOW_SIZE = 100
GOAL        = 30.0