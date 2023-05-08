from collections import namedtuple, deque
import matplotlib.pyplot as plt
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import random

from model_dqn import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 0
BATCH_SIZE = 64# minibatch size
GAMMA = 0.995# discount factor
ACTION_SIZE = 4
UPDATE_EVERY = 4
TAU = 0.001# for soft update of target parameters
LR = 5e-4# learning rate
CAPACITY_MEMORY = 100000# replay buffer size

#PriorizedExperienceReplay hyperparameters
BETA=0.04
ALPHA=0.1


class AgentDQN():
    def __init__(self, state_size, action_size, seed=SEED):
        
        """
        Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.gamma = GAMMA
        self.seed = random.seed(seed)
        self.action_size = action_size
        self.target_net = DQN(state_size, action_size, seed=seed).to(device)
        self.policy_net = DQN(state_size, action_size, seed=seed).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = PriorizedExpReplay(BATCH_SIZE, CAPACITY_MEMORY, seed=self.seed)
        self.target_net.eval()
        
        self.t_step=0
        
        
    def step(self,state, action, reward, next_state, done):
        """
        Add new experience to replay buffer
        If enough sample are collecteted, learn from a subset priorized.
        """
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step+1)%UPDATE_EVERY
        if self.t_step==0:
            if len(self.memory.memory_replay)>self.memory.batch_size:
                experiences = self.memory.weighted_sample()
                self.learn(experiences)
        
        
    def act(self, state, eps):
        """
        Choose an action according to the current state and the epsilon-greedy policy.
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        return
        ======
        action (int) : action
        """
        
        state = torch.from_numpy(state).float().to(device)
        
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        
        if np.random.rand()<eps:
            action = np.random.choice(np.arange(self.action_size))
        else:
            action = np.int32(np.argmax(action_values.cpu().numpy()))                  
        return action
        
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): 
                tuple of namedtuples (states, actions, rewards, next_states, dones, exp_index, adjust_IS_normalized)
        """
        states, actions, rewards, next_states, dones, exp_index, adjust_IS_normalized = experiences #, adjust_IS_normalized
        
        actions = actions.type(torch.LongTensor).to(device)
        next_actions_target_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        
        Q_observed = rewards+self.gamma*next_actions_target_values*(1-dones)
        Q_expected = torch.gather(self.policy_net(states), 1, actions)
        
        #update errors to get more probabilities to choose experiences with large errors in next batchs 
        diff_errors = Q_observed-Q_expected
        self.memory.update_errors(diff_errors,exp_index)
        
        #compute loss with correction of the bias from non uniform sampling
        loss = ((diff_errors**2)*torch.tensor(adjust_IS_normalized).to(device)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_target_net(TAU)
        
        return diff_errors
        
        
        
    def update_target_net(self, tau):
        target_weights = self.target_net.state_dict()
        policy_weights = self.policy_net.state_dict()
        for key in self.target_net.state_dict():
            target_weights[key] = target_weights[key]+(tau)*(policy_weights[key]-target_weights[key])
        self.target_net.load_state_dict(target_weights)
        
        
        
        
        
        
class PriorizedExpReplay():
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, batch_size, capacity_memory, seed, epsilon=0.01):
        """Initialize a ReplayBuffer object.

        Params
        ======
            
            batch_size (int): size of each training batch
            buffer_size (int): maximum size of buffer
            seed (int): random seed
            epsilon (float): 
        """
        self.batch_size = batch_size
        self.capacity_memory = capacity_memory
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=['state','action','reward', 'next_state', 'done'])
        self.memory_replay = deque(maxlen=capacity_memory)
        self.errors = deque(maxlen=capacity_memory)
        self.max_error = 1
        self.epsilon = epsilon
        self.b = BETA
        self.a = ALPHA
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        exp = self.experience(state, action, reward, next_state, done)
        self.memory_replay.append(exp)
        
        #assign maximal error for new experiences
        self.errors.append(self.max_error)
        
    def weighted_sample(self):
        """Select sample a batch of experiences from memory with Importance Sampling"""
        assert len(self.memory_replay)==len(self.errors)
        weights = (np.abs(self.errors)+self.epsilon)**self.a
        proba = weights/weights.sum()
        current_len_memory = len(self.memory_replay)
        sample_size = min(current_len_memory, self.batch_size)
        experiences_index = np.random.choice(np.arange(current_len_memory), p=proba, size=sample_size, replace=True)
        experiences = [self.memory_replay[i] for i in experiences_index]
        states, actions, rewards, next_states, dones = zip(*experiences)
        list_info = []
        for columns in (states, actions, rewards, next_states, dones):
            list_info.append(torch.from_numpy(np.vstack(columns)+0).float().to(device))
        states, actions, rewards, next_states, dones, = list_info
        
        adjust_IS = (len(self.memory_replay)*proba)**(-self.b)
        adjust_IS_normalized = adjust_IS/np.max(adjust_IS)
        
        
        
        return  states, actions, rewards, next_states, dones, experiences_index, adjust_IS_normalized
    
    def update_errors(self,errors, experiences_index):
        """
        Update errors to the experiences where
        
        Params:
        =======
            errors (tensor): error compute for a batch of experiences
            experiences_index (np.array(int)): array of indexes corresponding to each experiences"""
        [self.update_single_error(error,i) for error,i in zip(errors.squeeze(1),experiences_index)]
        self.max_error = max(self.errors)
        #self.max_error = np.quantile(self.errors, 0.9)#avoid outliers in TD_errors
        
    def update_single_error(self,error, index):
        self.errors[index] = error.detach().cpu().numpy()
       