import torch 
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    
    def __init__(self, state_size, action_size, seed, size_1=64, size_2=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            size_1 (int): Number of nodes in first hidden layer
            size_2 (int): Number of nodes in second hidden layer
        """
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.lin1 = nn.Linear(self.state_size, size_1)
        self.lin2 = nn.Linear(size_1, size_2)
        self.lin3 = nn.Linear(size_2, self.action_size)
        
    def forward(self, input_state):
        x = self.lin1(input_state)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return x