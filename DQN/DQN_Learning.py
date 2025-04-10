import DQN.DQN_Models as models
import DQN.DQN_objs as objs
import torch.nn as nn
from numpy import amax
import SA_ENV as SA_ENV
import torch
from datetime import datetime
from itertools import count
import random
import math
import matplotlib
matplotlib.use('TkAgg')  # Force TkAgg backend
import matplotlib.pyplot as plt
import numpy as np
import time

#matplotlib.use('Qt5Agg')

class DQN:
    
    def __init__(self ,load_model_path=None):
        memory_samples_capacity = 200000
        max_steps_for_sa = 10000
        # for memory capacity we give how much transitions we want to store so there goes 
        self.memory = objs.ReplayMemory(int(memory_samples_capacity/max_steps_for_sa))
        self.gamma = 0.95    # discount rate
        self.tau = 0.05    # target network replacment factor
        self.env = SA_ENV.SA_env(max_steps=max_steps_for_sa)
        self.batch_size = int(self.env.max_steps*1.5)
        self.fig  = None
        self.axes = None
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        obs_space = int(np.array(self.env.observation_space.shape).prod())
        action_sapce = int(self.env.action_space.n)
        print(obs_space,action_sapce)
        self.policy_net = models.DuelingDQN_NN(obs_space,action_sapce)
        if load_model_path is not None:
            self.policy_net.load_state_dict(torch.load(load_model_path))
        self.target_net = models.DuelingDQN_NN(obs_space,action_sapce)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001)
        


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

        next_state_values = torch.zeros(len(memory_sample))
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
        # gradien clipping jeżeli dalej będzie mi eksplodowało do dużych temp lub spadało instant do min to można tu coś pogrzebać nawet oba na raz
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        #torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
    def run(self, episodes, file_name_to_save_model:str = None):
        start_learning_date_sample = datetime.today().strftime('%Y_%m_%d_%H_%M')
        print(f'Started learning {start_learning_date_sample}')
        if file_name_to_save_model is not None:
            start_learning_date_sample = file_name_to_save_model
        for i_episode in range(episodes):
            print(" ")
            print(f'Learning episode {i_episode}/{episodes}')
            print("episilon for episode:", self.epsilon)
            # Initialize the environment and get its state
            state,_ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            for t in count():
                if self.epsilon == self.epsilon_min and i_episode%3 == 0 and t == 2500:
                    self.epsilon = 1.0

                action = self.select_action(state)
                observation, reward, done,_,_= self.env.step(action.item())
                reward = torch.tensor([reward])

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                # Store the transition in memory
                #! debugging
                # if state is None or action is None or next_state is None or reward is None :
                #     print(state)
                #     print(action)
                #     print(next_state, " for that next state observation:",)
                #     print(reward)
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if t%10 == 0:
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
                    self.memory.finalizeTrace()
                    self.saveModel(verssioning=start_learning_date_sample,episode=i_episode)
                    run_history = self.env.getFullParametersHistory()

                    #self.plot_data_non_blocking(max_temperature=self.env.starting_temp,Temperature_normalized=[a[4] for a in run_history],Temperature=[a[-2] for a in run_history],Reward=[a[-1] for a in run_history],current_values=[a[0] for a in run_history],best_values=[a[1] for a in run_history])#epsilon_hist)
                    #time.sleep(1)
                    break
            

    def saveModel(self,verssioning,episode):
        eps_indicator = 1+int(episode/100)
        torch.save(self.policy_net.state_dict(), "DQN_policy_model_"+verssioning+"_eps"+str(eps_indicator)+"00")

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
            return torch.tensor([[random.randrange(start=0,stop=self.env.action_space.n)]], dtype=torch.long)
            
        


    def plot_data_non_blocking(self,Reward,max_temperature, Temperature,Temperature_normalized, current_values, best_values):
        if self.fig is None or self.axes is None:
            plt.ion()  # Only call plt.ion() once when first creating the plot
            self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 7))
        else:
            if plt.fignum_exists(self.fig.number):  # Check if the previous plot is still open
                plt.close(self.fig)
            self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 7))

        # Clear axes before replotting
        for row in self.axes:
            for ax in row:
                ax.cla()

        self.axes[0][0].plot(Reward, linestyle='-', color='b')
        self.axes[0][0].set_title("Reward Plot")

        self.axes[0][1].plot(Temperature, linestyle='-', color='r')
        self.axes[0][1].set_title(f"Temperature Plot, maxT: {max_temperature}")

        self.axes[1][0].plot(current_values, linestyle='-', color='b', label="Current")
        self.axes[1][0].plot(best_values, linestyle='-', color='r', label="Best")
        self.axes[1][0].set_title("SA Values")
        self.axes[1][0].legend()

        self.axes[1][1].plot(Temperature_normalized, linestyle='-', color='b')
        self.axes[1][1].set_title("Normalized Temperature Plot")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)  # Allow the GUI to update — doesn't block
        plt.show(block=False)


















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