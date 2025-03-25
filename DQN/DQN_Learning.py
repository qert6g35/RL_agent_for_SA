import DQN_Models as models
import DQN_objs as objs
import torch.nn as nn
from numpy import amax
import torch
from datetime import datetime


class DQN:
    
    def __init__(self, n_observations, n_actions,load_model_path=None):
        self.memory = objs.ReplayMemory(10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0
        self.policy_net = models.DQN_NN(n_observations, n_actions)
        if load_model_path is not None:
            self.policy_net.load_state_dict(torch.load(load_model_path))
        self.target_net = models.DQN_NN(n_observations, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)

    def learnNetwork(self, state_batch, action_batch, reward_batch, nextState_batch, done):
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            nextState_batch)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in nextState_batch
                                                    if s is not None])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(state_batch.size(0))
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        # if not done:
        #     with nn.no_grad():
        #         target = reward + self.gamma * amax(self.target_net.predict(nextState)[0])
        # else:
        #     target = reward
        # target_f = self.policy_net(state)
        # target_f[0][action] = target
        # # Train the model
        # self.optimizer.zero_grad()
        # self.loss = self.loss(target_f, self.policy_net(state))
        # self.loss.backward()
        

    def saveModel(self):
        torch.save(self.policy_net.state_dict(), "DQN_policy_model_"+datetime.today().strftime('%Y_%m_%d_%H_%M'))