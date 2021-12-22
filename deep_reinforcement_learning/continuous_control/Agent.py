# Imports
# ==================================================

# Python Modules
# --------------------------------------------------
import random
import copy
import numpy as np


# Torch Modules
# --------------------------------------------------
import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim


# Custom Modules
# --------------------------------------------------
import config
import utils
import Actor
import Critic
import Buffer



# Agent Classes
# ==================================================

class Agent():
    """
    """
    
    def __init__(self, state_size, action_size, n_agents):
        """
        """
        
        self.seed = np.random.seed(config.RANDOM_SEED)
        
        random.seed(config.RANDOM_SEED)
        
        if torch.cuda.is_available():
            self.device = "cuda:0"
            print(f"[INFO] training on CUDA")
            
        else:
            self.device = "cpu"
            print(f"[INFO] training on CPU")
            
        self.state_size  = state_size
        self.action_size = action_size
        self.n_agents    = n_agents  

        
        # Actor
        # --------------------------------------------------
        self.actor_local  = Actor(self.state_size, self.action_size, self.fc1, self.fc2, self.leakiness).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size, self.fc1, self.fc2, self.leakiness).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = config.ACTOR_LR)

        
        # Critic
        # --------------------------------------------------
        self.critic_local  = Critic(self.state_size, self.action_size, self.fc1, self.fc2, self.fc3, self.leakiness).to(self.device)
        self.critic_target = Critic(self.state_size, self.action_size, self.fc1, self.fc2, self.fc3, self.leakiness).to(self.device)
        
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = config.CRITIC_LR)

        
        # Noise
        # --------------------------------------------------
        self.noise = OUNoise(self.action_size)

        
        # Replay Buffer
        # --------------------------------------------------
        self.memory    = Buffer.ReplayBuffer(self.action_size, config.BUFFER_SIZE, config.BATCH_SIZE, self.device)
        self.timesteps = 0  

        
    def step(self, states, actions, rewards, next_states, dones):
        """
        """
        
        self.timesteps += 1
        
        for i in range(self.n_agents):
            self.memory.add(
                states[i], 
                actions[i], 
                rewards[i], 
                next_states[i], 
                dones[i]
            )

            
        if (len(self.memory) > config.BATCH_SIZE) and (self.timesteps % self.n_agents == 0):
            for _ in range(10):
                exp = self.memory.sample()
                
                self.learn(exp)

                
    def act(self, states):
        """
        """
        
        states = torch.from_numpy(states).float().to(self.device)
        
        self.actor_local.eval()
        
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
            
        self.actor_local.train()
        
        actions += [self.noise.sample() for _ in range(self.n_agents)]
            
        return np.clip(actions, -1, 1)

    
    def reset(self):
        """
        """
        
        self.noise.reset()

        
    def learn(self, experiences):
        """
        """
        
        states, actions, rewards, next_states, dones = experiences

        
        # Critic Update
        # --------------------------------------------------
        actions_next = self.actor_target(next_states)
        
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets      = rewards + (config.GAMMA * Q_targets_next * (1 - dones))
        Q_expected     = self.critic_local(states, actions)
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        
        # Actor Update
        # --------------------------------------------------
        actions_pred =  self.actor_local(states)
        actor_loss   = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        
        # Target Update
        # --------------------------------------------------
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local,  self.actor_target)

        
    def soft_update(self, local_model, target_model):
        """
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(config.TAU * local_param.data + (1.0 - config.TAU) * target_param.data)

            

class OUNoise:
    """
    """
    
    def __init__(self, size):
        """
        """
        
        self.seed = np.random.seed(config.RANDOM_SEED)
        self.mu   = config.MU * np.ones(size)
        
        random.seed(config.RANDOM_SEED)
        
        self.reset()
        
        
    def reset(self):
        """
        """
        
        self.state = copy.copy(self.mu)
        
        
    def sample(self):
        """
        """
        
        x          = self.state
        dx         = config.THETA * (self.mu - x) + config.SIGMA * np.array([np.random.random() for i in range(len(x))])
        self.state = x + dx
        
        return self.state