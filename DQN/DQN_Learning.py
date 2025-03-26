import DQN.DQN_Models as models
import DQN.DQN_objs as objs
import torch.nn as nn
from numpy import amax
import DQN.DQN_SA as DQN_SA
import torch
from datetime import datetime
from itertools import count
import random


class DQN:
    
    def __init__(self, env:DQN_SA.SA_env = DQN_SA.SA_env() ,load_model_path=None):
        self.memory = objs.ReplayMemory(10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.0001
        self.policy_net = models.DQN_NN(env.action_space, env.observation_space)
        if load_model_path is not None:
            self.policy_net.load_state_dict(torch.load(load_model_path))
        self.target_net = models.DQN_NN(env.action_space, env.observation_space)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.env = env


    def learnNetwork(self, state_batch, action_batch, reward_batch, nextState_batch): #works for any size of batch
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
        
    def run(self, episodes):
        for i_episode in range(episodes):
            # Initialize the environment and get its state
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward])
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.learnNetwork()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    plot_durations()
                    break

    def saveModel(self):
        torch.save(self.policy_net.state_dict(), "DQN_policy_model_"+datetime.today().strftime('%Y_%m_%d_%H_%M'))

    def select_action(self, state):
        if(self.epsilon > self.epsilon_min):
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
        if random.random() < self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([self.env.actions[random.randrange(self.env.action_space)]], dtype=torch.long)