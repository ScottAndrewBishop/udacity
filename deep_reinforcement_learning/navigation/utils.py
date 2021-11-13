# Imports 
# ==================================================

# Python Modules
# -------------------------
import random
import collections
import numpy             as np
import matplotlib.pyplot as plt


# Torch Modules
# -------------------------
import torch
import torch.nn            as nn
import torch.nn.functional as F


# Custom Modules
# -------------------------
import config



# Train Methods 
# ==================================================

def train(agent, env, n_episodes, max_t, eps_start, eps_end, eps_decay):
    """
    """
    
    scores_window = collections.deque(maxlen = config.SCORE_WINDOW_SIZE)  
    eps           = eps_start                  
    scores        = []  
    
    brain_name = env.brain_names[0]
    
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode = config.ENV_TRAIN_MODE)[brain_name]
        state    = env_info.vector_observations[0]
        score    = 0
        
        for t in range(max_t):
            action     = agent.act(state, eps)
            env_info   = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward     = env_info.rewards[0]
            done       = env_info.local_done[0]
            
            agent.step(state, action, reward, next_state, done)
            
            score += reward
            state  = next_state
            
            if done:
                break 
                
        scores_window.append(score)       
        scores.append(score)    
        
        eps = max(eps_end, eps_decay * eps)
        
        if i_episode % 100 == 0:
            print(
                f"==================================================\n"
                f"Episode        {i_episode}\n"
                f"Average Score: {np.round(np.mean(scores_window), 2)}\n"
            )
            
        if np.mean(scores_window) >= 13:                
            print(
                f"==================================================\n"
                f"Environment Solved! {i_episode - 100} Episodes\n"
                f"Average Score:      {np.round(np.mean(scores_window), 2)}\n"
            )
                
            torch.save(agent.local_qnet.state_dict(), f"{config.CHECKPOINT_DIR}/{config.CHECKPOINT_NAME}")
                
            break
            
    env.close()
    np.save(f"{config.SCORES_DIR}/{config.SCORES_NAME}", scores)
    
    return scores



# Plot Methods 
# ==================================================

def plot_learning():
    """
    """
    
    scores = np.load(f"{config.SCORES_DIR}/{config.SCORES_NAME}")
    fig    = plt.figure(figsize = (20, 8))
    ax     = fig.add_subplot(111)

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.grid(alpha = 0.25)
    
    plt.savefig(f"{PLOT_DIR}/{PLOT_NAME}")
    plt.show()