from ast import mod
from cProfile import label
from sympy import total_degree
from DQN import DQN_Models
from PPO import PPO_Model
import SA
import math
import Problem
from itertools import count
import torch
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from collections import deque
import matplotlib.pyplot as plt
from enum import Enum

class TemperatureChangeStrategy(Enum):
    MultiplyCurrent = 1 # mnożymy obecną wartość przez stały mnożnik
    AddStarting = 2 # do obecnej temperatury dodajemy % temperatury startowej

#SA enviroment devined for DQN
class SA_env(gym.Env):

    def __init__(self,
                 preset_problem = None,
                 set_up_learning_on_init = False,
                 use_observation_divs = False,
                 use_time_temp_info = True,
                 use_new_lower_actions = True,
                 steps_per_temp = 10,
                 temperature_change_type:TemperatureChangeStrategy = TemperatureChangeStrategy.AddStarting
                 ):
        # elements that shouldn't change when SA is changed
        self.max_temp_accepting_chance = 0.85
        self.min_temp_accepting_chance = 0.001
        if use_new_lower_actions:
            self.actions = [float(f) * 0.01 for f in range(95,106,1)]#[0.85, 0.88, 0.91, 0.9400000000000001, 0.97, 1.0, 1.03, 1.06, 1.09, 1.12, 1.1500000000000001]
        else:
            self.actions = [float(f) * 0.01 for f in range(80,121,4)]#[0.8, 0.84, 0.88, 0.92, 0.96, 1.0, 1.04, 1.08, 1.12, 1.16, 1.2]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.max_steps = 10000
        self.reward_lowerd_steps = 0.03 * self.max_steps
        self.total_reward = 0
        self.total_improvment = 0
        self.total_range_punhishment = 0
        self.total_hot_walk = 0
        self.total_cold_walk = 0
        self.total_too_fast_changes_fresh = 0
        self.total_too_fast_changes_long = 0
        self.total_good_trends = 0
        self.total_delta_current = 0
        self.total_no_improvment = 0
        self.total_cold_slow_changes = 0
        self.done = False
        self.use_observation_divs =use_observation_divs
        self.use_time_temp_info = use_time_temp_info
        self.stesp_of_stagnation = 0
        self.stesp_of_noice = 0
        self.stesp_in_hot = 0
        self.stesp_in_cold = 0
        self.steps_per_temp = steps_per_temp
        self.temp_change_strategy = temperature_change_type
        self.steps_without_correction = 0
        self.last_best_value = 0
        #print("there will be no reward for first steps:",self.no_reward_steps)
        
        self.run_history = []
        self.norm_reward_scale = 10.0
        self.temp_history_size = 40
        self.temp_short_size = 8
        self.last_temps = deque([0.5 for _ in range(self.temp_history_size)],maxlen=int(self.temp_history_size))

        l = [0, 0, 0, 0, 0]
        h = [1, 1, 1, 1, 1]
        if use_observation_divs:
            l += [-1,-1]
            h += [1,1]
        if use_time_temp_info:
            l += [0,0,0,-1,-1]
            h += [1,1,1,1,1]
        low = np.array(l, dtype=np.float32)
        high = np.array(h, dtype=np.float32) 
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        #extra elements
        self.render_mode = None
        self.window = None
        self.clock = None
        
        if set_up_learning_on_init:
            if preset_problem != None:
                self.SA = SA.SA(preset_problem)
            else:
                self.SA = SA.SA()
            self.max_steps = self.estimate_sa_steps()
            self.SA_steps = int(self.max_steps / self.steps_per_temp)
            self.reward_lowerd_steps = 0.03 * self.SA_steps
            # elements that should change when SA is 
            deltaEnergy = self.SA.problem.EstimateDeltaEnergy()
            if deltaEnergy <= 0:
                deltaEnergy = self.SA.problem.EstimateDeltaEnergy()
                if deltaEnergy <= 0:
                    print("Used upperbound for delta energy!!")
                    deltaEnergy = self.SA.problem.getUpperBound()/10
            self.starting_temp = (deltaEnergy)/-math.log(self.max_temp_accepting_chance)
            self.min_temp = (deltaEnergy)/-math.log(self.min_temp_accepting_chance)
            #print("we have starting temp:",self.starting_temp)
            #print("we have min temp:",self.min_temp)
            self.current_temp = self.starting_temp
            self.run_history.append(self.observation() + [self.current_temp,0])
            self.last_temps = deque([0.5 for _ in range(self.temp_history_size)],maxlen=int(self.temp_history_size))
        else:
            self.SA = None
            self.starting_temp = 0.0
            self.min_temp = 0.0
            self.current_temp = self.starting_temp
        pass

    def reset(self,seed=None, options=None,preset_problem = None,initial_solution = None,reset_sa=True):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        if self.SA == None:
            self.SA = SA.SA(preset_problem=preset_problem,initial_solution=initial_solution)
        elif reset_sa:
            self.SA.reset(preset_problem=preset_problem,initial_solution=initial_solution)
        deltaEnergy = self.SA.problem.EstimateDeltaEnergy()
        if deltaEnergy <= 0:
            deltaEnergy = self.SA.problem.EstimateDeltaEnergy()
            if deltaEnergy <= 0:
                print("Used upperbound for delta energy!!")
                deltaEnergy = self.SA.problem.getUpperBound()/10
        self.starting_temp = (deltaEnergy)/-math.log(self.max_temp_accepting_chance)
        self.min_temp = (deltaEnergy)/-math.log(self.min_temp_accepting_chance)
        self.stesp_of_stagnation
        self.max_steps = self.estimate_sa_steps()
        self.SA_steps = int(self.max_steps / self.steps_per_temp)
        self.reward_lowerd_steps = 0.03 * self.SA_steps
        
        self.last_temps = deque([0.5 for _ in range(self.temp_history_size)],maxlen=int(self.temp_history_size))
        #print("we have starting temp:",self.starting_temp)
        #print("we have min temp:",self.min_temp)
        self.current_temp = self.starting_temp
        self.stesp_of_noice = 0
        self.stesp_of_stagnation = 0 
        self.stesp_in_hot = 0
        self.stesp_in_cold = 0
        self.steps_without_correction = 0
        self.last_best_value = self.SA.best_solution_value

        # # #! zaawansowane plotowanie na potrzeby oprzedstawienia temperatury i przebiegu poprzedniej instacjni 
        # if(len(self.run_history)>10):
        #     fig, axs = plt.subplots(3, 3, figsize=(8, 15))
        #     axs[0][0].plot([x[0] for x in self.run_history], color='blue',label = "current")
        #     axs[0][0].plot([x[1] for x in self.run_history], color='green',label = "current")
     
        #     axs[1][0].plot([x[-2] for x in self.run_history], color='green',label = "temp") 

        #     axs[2][0].plot([x[0] - x[1] for x in self.run_history], color='red')

        #     axs[0][1].plot([x[-1] for x in self.run_history], color='red',label = "reward")
        #     #axs[0][1].set_ylim(-1, 1)
            
        #     axs[1][1].plot([x[-3] for x in self.run_history], color='green',label = "good_trends ")
        #     #axs[1][1].set_ylim(-1, 1)

        #     axs[2][1].plot([x[-4] for x in self.run_history], color='blue',label = "cold punishment")
        #     #axs[2][1].set_ylim(-1, 1)

        #     axs[0][2].plot([x[5] for x in self.run_history], color='red',label = "mean")
        #     #axs[0][2].set_ylim(-1, 1)

        #     axs[1][2].plot([x[-5] for x in self.run_history], color='red',label = "too fast changes")
        #     #axs[1][2].set_ylim(-1, 1)

        #     axs[2][2].plot([x[-6] for x in self.run_history], color='red',label = "hot punishment")
        #     #axs[2][2].set_ylim(-1, 1)

            
        #     fig.legend()
        #     # Dostosowanie wyglądu

        #     plt.tight_layout()
        #     plt.show()

        obs = self.observation()
        self.run_history = [obs + [0.5,0]]
        self.done = False
        self.total_reward = 0
        self.total_improvment = 0
        self.total_no_improvment = 0
        self.total_range_punhishment = 0
        self.total_hot_walk = 0
        self.total_cold_walk = 0
        self.total_too_fast_changes_fresh = 0
        self.total_too_fast_changes_long = 0
        self.total_good_trends = 0
        self.total_delta_current = 0
        self.total_cold_slow_changes = 0
        return self.observation(), self.info() #!!! we pas none as info
    
    def makeTempChangeStep(self,action_number):
        was_temp_lower_than_min = False
        if (self.temp_change_strategy == TemperatureChangeStrategy.AddStarting):
            self.current_temp += self.starting_temp * (self.actions[action_number] - 1 )#= 0.9 * self.current_temp * self.actions[action_number] + self.current_temp * 0.1
        elif (self.temp_change_strategy == TemperatureChangeStrategy.MultiplyCurrent):
            self.current_temp = 0.9 * self.current_temp * self.actions[action_number] + self.current_temp * 0.1
        else:
            assert(False)
        if self.current_temp < self.min_temp:
            was_temp_lower_than_min = True
            self.current_temp = self.min_temp

        if self.current_temp > self.starting_temp*10:
            self.current_temp = self.starting_temp*10

        teperature_factor = (self.current_temp - self.min_temp) /(self.starting_temp - self.min_temp)
        self.last_temps.append(min(teperature_factor,2.0)/2.0)
        
        self.SA.step(self.current_temp,steps_per_temperature=self.steps_per_temp)

        if self.last_best_value != self.SA.best_solution_value:
            self.steps_without_correction = 0
            self.last_best_value = self.SA.best_solution_value
        else:
            self.steps_without_correction += 1

        return was_temp_lower_than_min,teperature_factor

    def step(self,action_number):
        was_temp_lower_than_min,teperature_factor = self.makeTempChangeStep(action_number)

        new_observation = self.observation()
            #self.run_history.append( new_observation + self.run_history[-1][-6:])

        improvement = abs( self.run_history[-1][1] - new_observation[1] )
        reward = 0.0
        
        if improvement > 0:
            reward = self.norm_reward_scale * improvement
            reward = min(math.log1p(reward * self.SA.steps_done)*2,10)

            if self.SA.steps_done > self.max_steps * 0.2:
                reward = max(2,reward)  # bonus za poprawę po jakimś czasie #math.log(reward * self.SA.steps_done + 1)*10  #reward * (math.pow(self.SA.steps_done + 1,2)/2) #(math.log(self.SA.steps_done + 1)/2)

        improvment_reward = reward

        #! spłaszczenie nagrody w początkowym stadium przeszukiwania
        if self.SA.steps_done < self.reward_lowerd_steps:
            reward = reward * (self.SA.steps_done/self.reward_lowerd_steps)  
        
        #! kary za przekroczenie granic temperaturowych
        range_punhishment = 0
        if was_temp_lower_than_min:
            range_punhishment -= 0.25
        elif teperature_factor >= 1.5:
            punishment = 0.25 * (int(teperature_factor)-1)
            if punishment > self.norm_reward_scale:
                punishment = self.norm_reward_scale
            range_punhishment -= punishment # silna kara za każdą krotność przekroczenia temperatur
        else:
            range_punhishment += 0.00025 # śladowa nagroda za pozostawanie w dobrym zakreśie

        reward += range_punhishment

        cold_steps_punishment, hot_steps_punishment = self.stepsInWrongRangePunishment(new_observation)
       # print(cold_steps_punishment,hot_steps_punishment)
        reward += cold_steps_punishment 
        reward += hot_steps_punishment

        # ! pozostałość po każe za kroki bez poprawy 
        reward = reward - new_observation[3] * 0.02 # ten wsp już jest znormalizowany więc kara rośnie aż do 2 (ale dowolna poprawa max value zresetuje tą karę)
        self.total_no_improvment -= new_observation[3] * 0.02 

    
        #!! kara za zbyt gwałtowne zmiany
        too_fast_changes_short = 0
        too_fast_changes_long = 0
        if self.use_time_temp_info:
            if new_observation[-4]>0.01:
                too_fast_changes_short -= 0.08
            if new_observation[-3]>0.0175: #[temp_mean, temp_std_fresh,temp_std_full, temp_trend_fresh,temp_trend_full]
                too_fast_changes_long -= 0.08
        reward += too_fast_changes_short
        reward += too_fast_changes_long

        #? nagroda za zgodne trendy
        good_trends = 0
        if self.use_time_temp_info:
            if (new_observation[-1]>0 and new_observation[-2]>0) or (new_observation[-1]<0 and new_observation[-2]<0): #[temp_mean, temp_std_fresh,temp_std_full, temp_trend_fresh,temp_trend_full]
                good_trends += 0.012 #! osłabiamy istotność zgodnych trendów agent nad wyrost uczy się tej taktyki (zamist standardowego /2 dla wszystkich jest /2.99)
        reward += good_trends
        delta_current_reward = 0

        #! kara za zmianę gówngo trendu tak żeby agent znie zmienial go za czensto
        #good_trends = 0
        #if self.use_time_temp_info:
        #    if (new_observation[-1]>0 and new_observation[-2]>0) or (new_observation[-1]<0 and new_observation[-2]<0): #[temp_mean, temp_std_fresh,temp_std_full, temp_trend_fresh,temp_trend_full]
        #reward += good_trends
        #delta_current_reward = 0
        
        #! nowe w G2
        #? drobna nagroda za utrzymywanie małych (bliskich 0) wartości trendu w okolicach wychładzania
        cold_seraching = 0
        if self.use_time_temp_info:
            if new_observation[-5]< 0.035 and abs(new_observation[-1]) < 0.005: #[temp_mean, temp_std_fresh,temp_std_full, temp_trend_fresh,temp_trend_full]
                cold_seraching += 0.012
        reward += cold_seraching
        self.total_cold_slow_changes += cold_seraching

        #! drobna nagroda za poprawę currentalue v 
        #! Została zmniejszona za to że agent ją zbyt często grindował
        delta_current = self.run_history[-1][0] - new_observation[0]
        if delta_current > 0:
            #print("adding mini_reward for good exploration direction:",min(2.0* delta_current ** 0.3,0.5))
            #print("how far is new_current to new_best",(new_observation[0] - new_observation[1]))
            delta_current_reward = 0.0035 * (1 - max(min((new_observation[0] - new_observation[1])*5.0,0.9),0.1))
        reward += delta_current_reward


        #! TO MOŻE NAM POMÓC Z OGARNIĘĆIEM WYBUCHAJĄCYCH WARTOŚCI PRZY STEROWANIU
        # normalizacja nagrody
        #reward_pre_norm = reward
        reward = max(min(reward,self.norm_reward_scale),-self.norm_reward_scale)/self.norm_reward_scale
        #if(reward_pre_norm/self.norm_reward_scale - reward != 0):
        #    print("for temp:",teperature_factor," reward:",reward,"pre minmax reward",reward_pre_norm/self.norm_reward_scale)
        if self.SA.steps_done < self.max_steps:
            is_terminated = False
        else:
            is_terminated = True
            self.done = True
        #[self.current_temp,reward])#
        self.run_history.append( new_observation +[self.current_temp,reward])# [-new_observation[3] * 3/self.norm_reward_scale,hot_steps_punishment/self.norm_reward_scale,too_fast_changes/self.norm_reward_scale,cold_steps_punishment/self.norm_reward_scale,good_trends/self.norm_reward_scale,min(teperature_factor,2.0)/2.0,reward])
        self.total_reward += reward
        self.total_improvment += improvment_reward
        self.total_range_punhishment += range_punhishment
        self.total_hot_walk += hot_steps_punishment
        self.total_cold_walk += cold_steps_punishment
        self.total_too_fast_changes_fresh += too_fast_changes_short
        self.total_too_fast_changes_long += too_fast_changes_long
        self.total_good_trends += good_trends
        self.total_delta_current += delta_current_reward
        return new_observation, reward , is_terminated, False, self.info()

    def stepsInWrongRangePunishment(self,new_observation):
        cold_walk_punishment = 0
        hot_walk_punishment = 0
        if new_observation[-5] < 0.04:
            cold_walk_punishment -= 0.015
            self.stesp_in_cold += 1
            self.stesp_in_hot = 0
        elif self.stesp_in_cold > 0:
            self.stesp_in_cold -= 2 

        if new_observation[-5] > 0.4:
            hot_walk_punishment -= 0.015
            self.stesp_in_hot += 1
            self.stesp_in_cold = 0
        elif self.stesp_in_hot > 0:
            self.stesp_in_hot -= 2

        #! nowy element w G2. zerujemy karę za przebywanie w zimnie jak udało nam się faktycznie coś odnaleść
        if self.steps_without_correction <= 1:
            self.stesp_in_cold = 0
        
        if(self.stesp_in_cold > 5):#! nowy element w G2, opuźniamy karę za chłodzenie
            cold_walk_punishment -= (self.stesp_in_cold - 5)/self.SA_steps * 0.2
        hot_walk_punishment -= self.stesp_in_hot/self.SA_steps * 0.15
       # print(self.stesp_in_cold,cold_walk_punishment,self.stesp_in_hot,hot_walk_punishment)
        return cold_walk_punishment,hot_walk_punishment
        

    def getStagnationAndNoicePunishments(self,observation):
        stagnation_punishment = 0
        too_nocey_punnishment = 0 
        if self.use_time_temp_info:
            #! kara za stagnacje temperatury, nie za duża ale taka statystyczna i powolna
            if self.SA.steps_done >= self.temp_history_size and observation[-3] < 0.0122137:#obs + [temp_mean, temp_std_fresh,temp_std_full, temp_trend_fresh,temp_trend_full]
                if self.stesp_of_stagnation <= 0:
                    self.stesp_of_stagnation = self.temp_history_size-1
                self.stesp_of_stagnation += 1
                stagnation_punishment -= min(1.0 * self.stesp_of_stagnation / (len(self.last_temps)*25),0.5)
            else:
                self.stesp_of_stagnation = max(0,int(self.stesp_of_noice/2) -1)
            #! kara za zbyt duży szum ! (czy ta kara ma sens ?) 
            if self.SA.steps_done >= self.temp_history_size and observation[-3] > 0.1 :
                if self.stesp_of_noice <= 0:
                    self.stesp_of_noice = 4
                self.stesp_of_noice += 1
                too_nocey_punnishment -= min(1.0 * self.stesp_of_noice / (len(self.last_temps)*5),2)
            else:
                self.stesp_of_noice = max(0,int(self.stesp_of_noice/2) -1)
        else:
            #! kara za stagnacje temperatury, nie za duża ale taka statystyczna i powolna
            if self.SA.steps_done >= self.temp_history_size and np.std(self.last_temps) < 0.0122137:
                if self.stesp_of_stagnation <= 0:
                    self.stesp_of_stagnation = len(self.last_temps)-1
                self.stesp_of_stagnation += 1
                stagnation_punishment -= min(1.0 * self.stesp_of_stagnation / (len(self.last_temps)*25),0.5)
            else:
                self.stesp_of_stagnation = max(0,int(self.stesp_of_noice/2) -1)
            #! kara za zbyt duży szum ! (czy ta kara ma sens ?)
            if self.SA.steps_done >= self.temp_history_size and np.std(list(self.last_temps)[:-self.temp_short_size]) > 0.1 :
                if self.stesp_of_noice <= 0:
                    self.stesp_of_noice = len(self.last_temps)-1
                self.stesp_of_noice += 1
                too_nocey_punnishment -= min(1.0 * self.stesp_of_noice / (len(self.last_temps)*5),2)
            else:
                self.stesp_of_noice = max(0,int(self.stesp_of_noice/2) -1)
        return stagnation_punishment ,too_nocey_punnishment 
    
    def getFullParametersHistory(self):
        return self.run_history

    def info(self): # nie sądze by to było potzebne więc zostawiam pusty set 
        if self.done:
            return {
                "total":self.total_reward,
                "improvment":self.total_improvment,
                "no_improvment":self.total_no_improvment,
                "range":self.total_range_punhishment,
                "hot":self.total_hot_walk,
                "cold":self.total_cold_walk ,
                "noice_short":self.total_too_fast_changes_fresh,
                "noice_long":self.total_too_fast_changes_long,
                "trends":self.total_good_trends,
                "deltaC":self.total_delta_current,
                "slow_clod_changes":self.total_cold_slow_changes
                }#{"current_solution":self.SA.current_solution,"best_solution":self.SA.best_solution,"current_temperature":self.current_temp}
        return {}

    def observation(self):
        normalize_factor = 1.0 / self.SA.problem.getUpperBound()

        # obs = [
        #     self.SA.current_solution_value, 
        #     self.SA.best_solution_value,
        #     self.SA.steps_done/self.max_steps, # (tutaj mamy ile już zrobiliśmy w %) zamienić kroki+max_kroki na % ile zostało 
        #     self.getStepsWithoutCorrection(), # dodać ilość korków od ostatniej poprawy 
        #     self.starting_temp,
        #     self.min_temp,
        #     self.current_temp
        #     ]
        
        obs = [
            self.SA.current_solution_value * normalize_factor, 
            self.SA.best_solution_value * normalize_factor,
            self.SA.steps_done/self.max_steps, # (tutaj mamy ile już zrobiliśmy w %) zamienić kroki+max_kroki na % ile zostało 
            self.steps_without_correction/self.SA_steps, # dodać ilość korków od ostatniej poprawy (dr)
            (self.current_temp - self.min_temp)/(self.starting_temp - self.min_temp), # dodatkowa normalizacja tempreatury. z racji na to że zakres temperatury tez jest dobierany zaleznie od zadania 
            ]
        
        if self.use_observation_divs: # dodnie informacji o różnicach LOL bez sensu za mało info a wprowadza tylko ekstra szum
            if not self.run_history:
                pre_csv = 0
                pre_temp = (self.current_temp - self.min_temp)/(self.starting_temp - self.min_temp)
            else:
                pre_csv = self.run_history[-1][0]
                pre_temp = self.run_history[-1][4]

            obs.append(pre_csv - self.SA.current_solution_value * normalize_factor) #! dodatkowa różnica curent value od ostatniego kroku (jest dodatnia jak mamy poprawę)
            obs.append((self.current_temp - self.min_temp)/(self.starting_temp - self.min_temp) - pre_temp) #! dodatkowa różnica curent TEMPERATURE od ostatniego kroku
        
        if self.use_time_temp_info:
            temp_mean = np.mean(self.last_temps)
            temp_std_full = np.std(self.last_temps)
            temp_std_fresh = np.std(list(self.last_temps)[-self.temp_short_size:])
            temp_trend_full = np.polyfit(range(len(self.last_temps)), self.last_temps, 1)[0]
            temp_trend_fresh = np.polyfit(range(self.temp_short_size), list(self.last_temps)[-self.temp_short_size:], 1)[0]
            obs = obs + [temp_mean, temp_std_fresh,temp_std_full, temp_trend_fresh,temp_trend_full]

        return obs
    
    def estimate_sa_steps(self,n = 0):
        if n == 0:
            n = self.SA.problem.dim
        if n <= 100:
            alpha = 15.0
            min_steps = 15000
        elif n <= 200:
            alpha = 11.0
            min_steps = self.estimate_sa_steps(100)
        elif n <= 500:
            alpha = 8
            min_steps = self.estimate_sa_steps(200)
        return min(max(int(alpha * (n ** 1.59)),min_steps),1e5)
   
    def runTest(self,model,generate_plot_data = False):
        obs = self.observation()
        self.run_history = []
        for t in count():
            #getting new temperature
            with torch.no_grad():
                if type(model) in PPO_Model.PPO_MODELS:
                    actionNR = model.get_action(torch.tensor(obs, dtype=torch.float32))
                elif type(model) in DQN_Models.DQN_MODELS:
                    actionNR = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                    actionNR = actionNR.max(1).indices.view(1, 1).item()

                #based on choosed action
                #perform SA step
                self.makeTempChangeStep(action_number=actionNR)
                
            #collecting data
            self.run_history.append( obs + [self.current_temp,0])
            obs = self.observation()

            if t%int(self.max_steps/(10*self.steps_per_temp)) == 0 :
                print("test proges",t,"/",self.max_steps)

            if self.SA.steps_done >= self.max_steps:
                break

        unnormalize_factor =  self.SA.problem.getUpperBound() 
        if generate_plot_data:
            transposed_run_history = list(map(list, zip(*self.run_history)))
            return [x * unnormalize_factor for x in transposed_run_history[1]],[x * unnormalize_factor for x in transposed_run_history[0]],transposed_run_history[-2]  #best_values,current_values,temperature_values
        else:
            return [x[1] * unnormalize_factor for x in self.run_history] 


    def render(self):
        return super().render()
    
    def close(self):
        return super().close()