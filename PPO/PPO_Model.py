import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical

def layer_init(layer,std = np.sqrt(2), bias_const = 0.0):
    nn.init.orthogonal_(layer.weight,std)
    nn.init.constant_(layer.bias,bias_const)
    return layer

class PPO_NN(nn.Module):

    def __init__(self, envs, obs_num,actions_num):
        super(PPO_NN,self).__init__()
        self.layer_size = 64
        if envs != None:
            obs_space = np.array(envs.single_observation_space.shape).prod()
            act_space =  envs.single_action_space.n
        else:
            obs_space = obs_num
            act_space = actions_num

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_space, self.layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.layer_size, self.layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.layer_size, 1),std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_space, self.layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.layer_size, self.layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.layer_size,act_space),std=0.01),
        )
        

    def get_value(self,x):
        return self.critic(x)
    
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def get_action(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        return probs.sample()
    
PPO_MODELS = [PPO_NN]