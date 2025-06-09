import torch.nn as nn
import torch
import numpy as np
from torch.distributions.categorical import Categorical

def layer_init(layer,std = np.sqrt(2), bias_const = 0.0):
    nn.init.orthogonal_(layer.weight,std)
    nn.init.constant_(layer.bias,bias_const)
    return layer

class PPO_NN_v2(nn.Module):

    def __init__(self, envs, obs_num = 0,actions_num = 0,layer_size = 256):
        super(PPO_NN_v2,self).__init__()
        self.layer_size = layer_size
        if envs != None:
            obs_space = np.array(envs.single_observation_space.shape).prod()
            act_space =  envs.single_action_space.n
        else:
            assert(obs_num != 0 and actions_num != 0)
            obs_space = obs_num
            act_space = actions_num

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_space, self.layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.layer_size, self.layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.layer_size, 1),std=0.5),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_space, self.layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.layer_size, self.layer_size)),
            nn.Tanh(),
            layer_init(nn.Linear(self.layer_size,act_space),std=0.03),
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
    
    def get_deterministic_action(self, x):
        return torch.argmax(self.actor(x), dim=-1)

class PPO_NN(nn.Module):

    def __init__(self, envs, obs_num = 0,actions_num = 0,layer_size = 128):
        super(PPO_NN,self).__init__()
        self.layer_size = layer_size
        if envs != None:
            obs_space = np.array(envs.single_observation_space.shape).prod()
            act_space =  envs.single_action_space.n
        else:
            assert(obs_num != 0 and actions_num != 0)
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
    
PPO_MODELS = [PPO_NN,PPO_NN_v2]