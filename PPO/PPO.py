import torch.nn as nn 

#niżej struktura zapropomowana przez co-pilota (bardzo nie chciał się do tego przyznać)

# Define the neural network architecture
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, std=0.0):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.ones(1, action_dim) * std)
        self.apply(self.init_weights)