# Imports 
# ==================================================

# Python Modules
# -------------------------
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns

from collections import (
    deque, 
    namedtuple
)


# Torch Modules
# -------------------------
import torch


# Custom Modules
# -------------------------
import config



# Train Methods 
# ==================================================

def ddpg(agent, env, brain_name, n_agents):
    """
    """
    
    # Initialize Scores
    # --------------------------------------------------
    scores_window = deque(maxlen = config.WINDOW_SIZE) 
    max_scores    = [] 
    moving_avgs   = []
    
    
    # Train Agent
    # --------------------------------------------------
    for i_episode in range(1, config.EPOCHS + 1):
        env_info       = env.reset(train_mode = config.ENV_TRAIN_MODE)[brain_name]
        states         = env_info.vector_observations
        episode_scores = np.zeros(n_agents) 
        
        agent.reset()

        for t in range(config.MAX_T):
            actions     = agent.act(states)
            env_info    = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards     = env_info.rewards
            dones       = env_info.local_done
            
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(
                    states      = states,
                    actions     = actions,
                    rewards     = rewards,
                    next_states = next_states, 
                    dones       = dones,
                    t
                )
            
            episode_scores += np.array(rewards)
            states          = next_states
            
            if np.any(dones):
                break

        best_score = np.max(episode_scores)
        
        max_scores.append(best_score)
        scores_window.append(best_score)
        
        average_score = np.mean(scores_window)
        
        
        # Log Message
        # --------------------------------------------------
        print(
            f"--------------------------------------------------\n"
            f"[INFO] episode:       {i_episode}\n"
            f"       average score: {round(average_score, 2)}\n"
            f"       best score:    {round(best_score,    2)}"
        )

        if average_score >= config.GOAL:
            print(
                f"--------------------------------------------------\n"
                f"[INFO] environment solved in {i_episode - config.WINDOW_SIZE} episodes\n"
                f"       average score is {round(average_score, 2)}"
            )
            
            torch.save(agent.actor_local.state_dict(),  config.CHECKPOINT_ACTOR)
            torch.save(agent.critic_local.state_dict(), config.CHECKPOINT_ACTOR)
            
            break

            
    np.save(config.SCORES_NAME, scores)
    
    
    
# Plot Methods 
# ==================================================

def plot_learning():
    """
    """
    
    scores = np.load(config.SCORES_NAME)
    fig    = plt.figure(figsize = (20, 8))
    ax     = fig.add_subplot(111)

    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.grid(alpha = 0.25)
    
    plt.savefig(config.PLOT_NAME)
    plt.show()