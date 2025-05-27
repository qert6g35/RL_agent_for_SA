from re import S
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
import os
from torch.utils.tensorboard import SummaryWriter
import time
from collections import deque
from statistics import mean, stdev

#matplotlib.use('Qt5Agg')

class DQN:
    
    def __init__(self ,load_model_path=None,save_model_offset = 0):
        self.episodes = 2000

        self.model_offset = save_model_offset
        memory_samples_capacity = 500000
        max_steps_for_sa = 10000
        # for memory capacity we give how much transitions we want to store so there goes 
        self.memory = objs.ReplayMemory(int(memory_samples_capacity/max_steps_for_sa))
        self.gamma = 0.95    # discount rate
        self.tau = 0.05    # target network replacment factor
        self.env = SA_ENV.SA_env(max_steps=max_steps_for_sa)
        self.batch_size = int(self.env.max_steps*2)
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

        self.starting_lr = 0.002
        self.lr_anneling = "cosine"
        self.lr_cycle = 500 * self.env.max_steps
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.starting_lr)
        self.number_of_env_reward_collected = 5# z ilu ostatnich środowisk zbieramy nagrodę
        self.episode_rewards = deque([0.0 for _ in range (self.number_of_env_reward_collected)], self.number_of_env_reward_collected) # pojemnik na reward
        


    def learnNetwork(self): #works for any size of batch
        memory_sample = self.memory.sample(batch_size=self.batch_size)
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
        return loss
        
    def run(self, file_name_to_save_model:str = None):
        start_learning_date_sample = datetime.today().strftime('%Y_%m_%d_%H_%M')
        start_time = time.time()
        print(f'Started learning {start_learning_date_sample}')
        writer = SummaryWriter(f"runs/DQN_NN_"+start_learning_date_sample)
        if file_name_to_save_model is not None:
            start_learning_date_sample = file_name_to_save_model
        learning_step = 0
        for i_episode in range(1,1+self.episodes):
            print(" ")
            print(f'Learning episode {i_episode}/{self.episodes}')
            print("episilon for episode:", self.epsilon)
            print("learning rate for episode:",self.optimizer.param_groups[0]["lr"])
            # Initialize the environment and get its state
            state,_ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            for t in count():
                if self.epsilon == self.epsilon_min and i_episode%3 == 0 and t == 2500:
                    self.epsilon = 1.0

                action = self.select_action(state)
                observation, reward, done,_,info= self.env.step(action.item())
                reward = torch.tensor([reward], dtype=torch.float32)

                if done:
                    next_state = None
                    self.episode_rewards.append(info["tr"])
                    writer.add_scalar("charts/avr_reward_from_last_"+str(self.number_of_env_reward_collected), mean(self.episode_rewards), i_episode)
                    writer.add_scalar("charts/last_episode_total_reward", info["tr"],i_episode)
                    print("For episode:",i_episode,"we got last env reward:",info["tr"]," mean of last 5:",mean(self.episode_rewards))
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
                if t%50 == 0 or done:
                     # dodje powolne zmniejszanie lr
                    if self.lr_anneling == "linear":
                        updated_lr = self.starting_lr * (1.0 - ((i_episode * self.env.max_steps + t) - 1.0)/(self.episodes * self.env.max_steps))
                    elif self.lr_anneling == "cosine":
                        progress = ((i_episode * self.env.max_steps + t - 1.0) % self.lr_cycle) / self.lr_cycle
                        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                        updated_lr = self.starting_lr * cosine_decay 
                    else:
                        updated_lr = self.starting_lr  
                    self.optimizer.param_groups[0]["lr"] = max(updated_lr,0.000001)

                    if self.batch_size <= len(self.memory):
                        loss = self.learnNetwork()
                    else:
                        loss = None
                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′

                                # TRY NOT TO MODIFY: record rewards for plotting purposes
                    if t%200 == 0:
                        writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], learning_step)
                        if loss != None:
                            writer.add_scalar("losses/value_loss", loss.item(), learning_step)
                        else:
                            writer.add_scalar("losses/value_loss", 0, learning_step)

                        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                        # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                        # writer.add_scalar("losses/explained_variance", explained_var, global_step)
                        if t%1000 == 0:
                            print("LSPS:", int(learning_step / (time.time() - start_time)))
                        writer.add_scalar("charts/LSPS", int(learning_step / (time.time() - start_time)), learning_step)

                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                    self.target_net.load_state_dict(target_net_state_dict)
                    learning_step+=1

                if done:
                    self.memory.finalizeTrace()
                    self.saveModel(verssioning=start_learning_date_sample,eps=i_episode)
                    #run_history = self.env.getFullParametersHistory()

                    #self.plot_data_non_blocking(max_temperature=self.env.starting_temp,Temperature_normalized=[a[4] for a in run_history],Temperature=[a[-2] for a in run_history],Reward=[a[-1] for a in run_history],current_values=[a[0] for a in run_history],best_values=[a[1] for a in run_history])#epsilon_hist)
                    #time.sleep(1)
                    break
            

    def saveModel(self,verssioning,eps):
        episode = eps + self.model_offset
        torch.save(self.policy_net.state_dict(), "DQN_NN_"+verssioning+"_eps"+str(episode))
        if os.path.exists("DQN_NN_"+verssioning+"_eps"+str(episode-1)) and (eps%200 != 0 or eps == 1):
            os.remove("DQN_NN_"+verssioning+"_eps"+str(episode-1))

    def save_model(self,updates):
        if(updates%10 == 0):
            save_update = updates - updates%10
            check_point =  int((self.total_timesteps // self.batch_size) / 20) - int((self.total_timesteps // self.batch_size) / 20)%10
            torch.save(self.agent.state_dict(), self.save_agent_path+"_updates"+str(save_update + self.vers_offset))
            if int(save_update%check_point) != 0:
                self.delete_model(save_update + self.vers_offset - 10)

    # def sampleMemoryBatch(self):
    #     if self.batch_size <= len(self.memory):
    #         return self.memory.sample(batch_size=self.batch_size)
    #     return None


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
            self.policy_net.eval()
            with torch.no_grad():
                action = self.policy_net(state).max(1).indices.view(1, 1)
            self.policy_net.train()
        else:
            action = torch.tensor([[random.randrange(start=0,stop=self.env.action_space.n)]], dtype=torch.long)
        return action
            
        


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