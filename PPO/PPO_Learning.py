import torch.nn as nn 
import torch.optim as optim
from torch import zeros,Tensor
from SA_ENV import SA_env
from PPO.PPO_Model import PPO_NN
import gymnasium as gym
#niżej struktura zapropomowana przez co-pilota (bardzo nie chciał się do tego przyznać)

# # Define the neural network architecture
# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim, std=0.0):
#         super(ActorCritic, self).__init__()
#         self.actor = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, action_dim),
#         )
#         self.critic = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1),
#         )
#         self.log_std = nn.Parameter(torch.ones(1, action_dim) * std)
#         self.apply(self.init_weights)

class PPO:
    def __init__(self):
        #params
        self.num_envs = 4
        self.num_steps = 128
        self.total_timesteps = 25000
        self.batch_size = int(self.num_envs * self.num_steps)

        self.envs = gym.vector.SyncVectorEnv(
            [ SA_env for i in range(self.num_envs)]
        )
        
        self.agent = PPO_NN(self.envs)

        self.optimizer = optim.Adam(self.agent.parameters(), lr = 0.001, eps=1e-5)
        
        self.obs = zeros((self.num_steps, self.num_envs) + self.envs.single_observation_space.shape)
        self.actions = zeros((self.num_steps, self.num_envs) + self.envs.single_action_space.shape)
        self.logprobs = zeros((self.num_steps, self.num_envs))
        self.rewards = zeros((self.num_steps, self.num_envs))
        self.dones = zeros((self.num_steps, self.num_envs))
        self.values = zeros((self.num_steps, self.num_envs))

        # TRY NOT TO MODIFY: start the game (i hope i wont need to)
        self.global_step = 0
        self.next_obs = Tensor(self.envs.reset())
        self.next_done = zeros(self.num_envs)
        self.num_updates = self.total_timesteps // self.batch_size
        print("next obs shape",self.next_obs.shape)
        print("agent.getValue(next obs)",self.agent.getValue(self.next_obs))
        print("agent.getValue(next obs) shape",self.agent.getValue(self.next_obs).shape)

    def show_agent(self):
        print(self.agent)
