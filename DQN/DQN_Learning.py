import DQN.DQN_Models as models
import DQN.DQN_objs as objs
import torch.nn as nn
from numpy import amax
import DQN.DQN_SA as DQN_SA
import torch
from datetime import datetime
from itertools import count
import random
import math
import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class DQN:
    
    def __init__(self, env:DQN_SA.SA_env = DQN_SA.SA_env() ,load_model_path=None):
        self.memory = objs.ReplayMemory(10000)
        self.gamma = 0.95    # discount rate
        self.tau = 0.02    # target network replacment factor
        self.batch_size = int(env.max_steps/2)

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.015

        self.policy_net = models.DQN_NN(env.observation_space,env.action_space)
        if load_model_path is not None:
            self.policy_net.load_state_dict(torch.load(load_model_path))
        self.target_net = models.DQN_NN(env.observation_space,env.action_space)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.env = env


    def learnNetwork(self, memory_sample): #works for any size of batch
        batch = objs.Transition(*zip(*memory_sample))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        #print(batch.state.size)
        state_batch = torch.cat(batch.state)
        #print(batch.reward.size)
        reward_batch = torch.cat(batch.reward)
        #print(batch.state.size)
        action_batch = torch.cat(batch.action)


        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        #loss = criterion(torch.tensor(self.policy_net(self.env.observation())),torch.tensor([ 0.0,  0.0,  0.0,  69.0,   0.0,   0.0,0.0,  0.0,   0.0,  0.0,  0.0,  0.0,0.0,   0.0,   0.0]))


        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
    def run(self, episodes):
        start_learning_date_sample = datetime.today().strftime('%Y_%m_%d_%H_%M')
        print(f'Started learning {start_learning_date_sample}')
        for i_episode in range(episodes):
            print(" ")
            print(f'Learning episode {i_episode}/{episodes}')
            # Initialize the environment and get its state
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, done = self.env.step(action.item())
                reward = torch.tensor([reward])

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                learning_batch = self.sampleMemoryBatch()
                if learning_batch != None:
                    self.learnNetwork(learning_batch)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    run_history = self.env.getFullParametersHistory()
                    self.plot_data_non_blocking([a[-2] for a in run_history],[a[-1] for a in run_history])
                    break

    def saveModel(self,verssioning):
        torch.save(self.policy_net.state_dict(), "DQN_policy_model_"+verssioning)

    def sampleMemoryBatch(self):
        if self.batch_size <= len(self.memory):
            return self.memory.sample(batch_size=self.batch_size)
        return None


    def select_action(self, state)->torch.Tensor:
        if(type(state)!= torch.Tensor):
            state = torch.tensor(state,dtype=torch.float)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # else:
        #     self.epsilon = self.epsilon_min
        # if random.random() < self.epsilon:
        #     with torch.no_grad():
        #         return torch.tensor([self.env.actions[torch.argmax(self.policy_net(state))]], dtype=torch.float)
        # else:
        #     return torch.tensor([self.env.actions[random.randrange(start=0,stop=self.env.action_space)]], dtype=torch.float)

        if random.random() < self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randrange(start=0,stop=self.env.action_space)]], dtype=torch.long)
            
        
    def plot_data_non_blocking(self,y0,y1):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

        # Left plot
        axes[0].plot(y0, linestyle='-', color='b')
        axes[0].set_title("Reward Plot")
        axes[0].set_xlabel("X-axis")
        axes[0].set_ylabel("Y-axis")

        # Right plot
        axes[1].plot(y1, linestyle='-', color='r')
        axes[1].set_title("Temperature Plot")
        axes[1].set_xlabel("X-axis")
        plt.pause(0.001)  # Small pause to ensure the plot is updated
        if is_ipython:
            display.display(plt.gcf())

















    def FORCE_learnNetwork(self, state_batch, action_batch, reward_batch, nextState_batch): #works for any size of batch

        # # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        what_to_force_to_learn = torch.zeros(self.env.action_space)
        what_to_force_to_learn[3] = 6.9
        loss = criterion(
            torch.Tensor(self.policy_net(torch.Tensor(self.env.observation()))),
            torch.Tensor(what_to_force_to_learn))


        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()