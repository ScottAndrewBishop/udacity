# Imports 
# ==================================================

# Python Modules
# -------------------------
import random
import numpy as np


# Torch Modules
# -------------------------
import torch
import torch.nn            as nn
import torch.optim         as optim
import torch.nn.functional as F


# Custom Modules
# -------------------------
import config
import DQN
import Buffer



# Class
# ==================================================

class Agent():
    def __init__(self, action_size, state_size, batch_size, buffer_size, gamma, lr, tau, update_every, use_duel, use_double):
        self.seed = random.seed(config.RANDOM_SEED)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("[INFO] using CUDA")
            
        else:
            self.device = torch.device("cpu")
            print("[INFO] using CPU")
        
        
        self.action_size  = action_size
        self.batch_size   = batch_size
        self.buffer_size  = buffer_size
        self.gamma        = gamma
        self.lr           = lr
        self.state_size   = state_size
        self.tau          = tau
        self.update_every = update_every
        self.use_duel     = use_duel
        self.use_double   = use_double
        
        
        # Dueling DQN
        # --------------------------------------------------
        if self.use_duel:
            self.local_qnet  = DQN.DuelingDQN(self.state_size, self.action_size).to(self.device)
            self.target_qnet = DQN.DuelingDQN(self.state_size, self.action_size).to(self.device)
        
        
        # DQN
        # --------------------------------------------------
        else:
            self.local_qnet  = DQN.DQN(self.state_size, self.action_size).to(self.device)
            self.target_qnet = DQN.DQN(self.state_size, self.action_size).to(self.device)
        
        
        self.optimizer = optim.Adam(
            self.local_qnet.parameters(), 
            lr = self.lr
        )
        
        self.memory = Buffer.ReplayBuffer(self.action_size, self.batch_size, self.buffer_size, self.device)
        self.t      = 0
        
        
    def act(self, state, eps):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.local_qnet.eval()
        
        with torch.no_grad():
            action_values = self.local_qnet(state)
            
        self.local_qnet.train()
        
        if random.random() > eps: return np.argmax(action_values.cpu().data.numpy()).astype('int')
        else:                     return random.choice(np.arange(self.action_size))
        
        
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        self.t = (self.t + 1) % self.update_every
        
        if self.t == 0 and len(self.memory) > self.batch_size:
            self.learn(self.memory.sample(), self.gamma)
            
            
    def learn(self, exp, gamma):
        states, actions, rewards, next_states, dones = exp
        
        
        # Double Q-Network
        # --------------------------------------------------
        if self.use_double:
            local_qnet_actions = torch.LongTensor(self.local_qnet(next_states).detach().max(1)[1].unsqueeze(1))
            local_qnet_next    = self.target_qnet(next_states).gather(1, local_qnet_actions)
            q_targets          = rewards + (gamma * local_qnet_next * (1 - dones))
            q_expected         = self.local_qnet(states).gather(1, actions)
        
        
        # Default Q-Network
        # --------------------------------------------------
        else:
            q_targets_next = self.target_qnet(next_states).detach().max(1)[0].unsqueeze(1)    
            q_targets      = rewards + (gamma * q_targets_next * (1 - dones))
            q_expected     = self.local_qnet(states).gather(1, actions)
        
        
        loss = F.mse_loss(q_expected, q_targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        for target_param, local_param in zip(self.target_qnet.parameters(), self.local_qnet.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)