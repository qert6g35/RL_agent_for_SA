import torch.nn as nn
import torch.nn.functional as F
# The TUTORIAL neural network architecture

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size)
        )

    def forward(self, x):
        return x + self.block(x)

class DuelingDQN_NN(nn.Module):
    def __init__(self, obs_size, action_space):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU()
        )

        self.hidden = [
            ResidualBlock(128),
            ResidualBlock(128)
        ]
        
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )

    def forward(self, x):
        features = self.feature(x)
        for leayer in self.hidden:
            features = leayer(features)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DQN_NN_V1(nn.Module):
    # just a deeper NN with some extra tweeking
    def __init__(self, n_observations, n_actions):
        super(DQN_NN_V1, self).__init__()
        self.layers_standard_width = 128
        self.input = nn.Linear(n_observations, self.layers_standard_width)
        self.inner = [
            nn.Linear(self.layers_standard_width, self.layers_standard_width),
            nn.Linear(self.layers_standard_width, self.layers_standard_width)
            ]
        self.output = nn.Linear(self.layers_standard_width, n_actions)

        nn.init.xavier_uniform_(self.input.weight)
        self.input.bias.data.fill_(0.01)
        
        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.fill_(0.01)

        for layer in self.inner:
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)


        # self.input = nn.LeakyReLU(n_observations, self.layers_standard_width)
        # self.inner = [
        #     nn.LeakyReLU(self.layers_standard_width, self.layers_standard_width),
        #     nn.RReLU(self.layers_standard_width, self.layers_standard_width)
        #     ]
        # self.output = nn.ReLU(self.layers_standard_width, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.leaky_relu(self.input(x))
        x = F.rrelu(self.inner[0](x))
        x = F.relu(self.inner[1](x))
        return self.output(x)
        # x = self.input(x)
        # for layer in self.inner:
        #     x = layer(x)
        # return self.output(x)


class DQN_NN_Tutorial(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN_NN_Tutorial, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)