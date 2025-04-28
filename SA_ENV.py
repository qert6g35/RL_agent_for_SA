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

#SA enviroment devined for DQN
class SA_env(gym.Env):

    def __init__(self,preset_problem = None,set_up_learning_on_init = False,use_observation_divs = True):
        # elements that shouldn't change when SA is changed
        self.max_temp_accepting_chance = 0.85
        self.min_temp_accepting_chance = 0.001
        self.actions = [float(f) * 0.01 for f in range(80,121,4)]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.max_steps = 10000
        self.reward_lowerd_steps = 0.03 * self.max_steps
        self.total_reward = 0
        self.done = False
        self.use_observation_divs =use_observation_divs
        self.stesp_of_stagnation = 0
        #print("there will be no reward for first steps:",self.no_reward_steps)
        
        self.run_history = []
        self.norm_reward_scale = 10.0
        self.last_temps = deque([],maxlen=int(max(10,self.max_steps*0.003)))

        low = np.array([0, 0, 0, 0, 0,-1,-1], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 100,1,1], dtype=np.float32) #! uwaga możliwe że trzeba będzie określić maksymalną temperaturę dla środowiska
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
            self.reward_lowerd_steps = 0.03 * self.max_steps
            # elements that should change when SA is 
            deltaEnergy = self.SA.problem.EstimateDeltaEnergy(50)
            if deltaEnergy <= 0:
                deltaEnergy = self.SA.problem.EstimateDeltaEnergy(100)
                if deltaEnergy <= 0:
                    print("Used upperbound for delta energy!!")
                    deltaEnergy = self.SA.problem.getUpperBound()/10
            self.starting_temp = (deltaEnergy)/-math.log(self.max_temp_accepting_chance)
            self.min_temp = (deltaEnergy)/-math.log(self.min_temp_accepting_chance)
            #print("we have starting temp:",self.starting_temp)
            #print("we have min temp:",self.min_temp)
            self.current_temp = self.starting_temp
            self.run_history.append(self.observation() + [self.current_temp,0])
            self.last_temps = deque([],maxlen=int(max(10,self.max_steps*0.003)))
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
        deltaEnergy = self.SA.problem.EstimateDeltaEnergy(50)
        if deltaEnergy <= 0:
            deltaEnergy = self.SA.problem.EstimateDeltaEnergy(100)
            if deltaEnergy <= 0:
                print("Used upperbound for delta energy!!")
                deltaEnergy = self.SA.problem.getUpperBound()/10
        self.starting_temp = (deltaEnergy)/-math.log(self.max_temp_accepting_chance)
        self.min_temp = (deltaEnergy)/-math.log(self.min_temp_accepting_chance)
        self.stesp_of_stagnation
        self.max_steps = self.estimate_sa_steps()
        self.reward_lowerd_steps = 0.05 * self.max_steps
        self.last_temps = deque([],maxlen=int(max(10,self.max_steps*0.003)))
        #print("we have starting temp:",self.starting_temp)
        #print("we have min temp:",self.min_temp)
        self.current_temp = self.starting_temp

        #! zaawansowane plotowanie na potrzeby oprzedstawienia temperatury i przebiegu poprzedniej instacjni 
        # if(len(self.run_history)>10):
        #     fig, axs = plt.subplots(3, 3, figsize=(8, 15))
        #     axs[0][0].plot([x[0] for x in self.run_history], color='blue',label = "current")
        #     axs[0][0].plot([x[1] for x in self.run_history], color='green',label = "current")
     
        #     axs[1][0].plot([x[-2] for x in self.run_history], color='green',label = "temp") 

        #     axs[2][0].plot([x[0] - x[1] for x in self.run_history], color='red',label = "curent - best")

        #     axs[0][1].plot([x[-1] for x in self.run_history], color='red',label = "reward")
        #     axs[0][1].set_ylim(-1, 1)
            
        #     axs[1][1].plot([x[-3] for x in self.run_history], color='red',label = "delta_current")
        #     axs[1][1].set_ylim(-1, 1)

        #     axs[2][1].plot([x[-4] for x in self.run_history], color='red',label = "stagnation reward")
        #     axs[2][1].set_ylim(-1, 1)

        #     axs[1][2].plot([x[-5] for x in self.run_history], color='red',label = "(main) improvment reward")
        #     axs[1][2].set_ylim(-1, 1)
            
        #     fig.legend()
        #     # Dostosowanie wyglądu

        #     plt.tight_layout()
        #     plt.show()

        obs = self.observation()
        self.run_history = [obs + [self.current_temp,0]]
        self.done = False
        self.total_reward = 0
        return self.observation(), self.info() #!!! we pas none as info

    def step(self,action_number):
        was_temp_lower_than_min = False
        self.current_temp = 0.9 * self.current_temp * self.actions[action_number] + self.current_temp * 0.1
        if self.current_temp < self.min_temp:
            was_temp_lower_than_min = True
            self.current_temp = self.min_temp

        if self.current_temp > self.starting_temp*10:
            self.current_temp = self.starting_temp*10

        self.SA.step(self.current_temp)
        new_observation = self.observation()

        improvement = abs( self.run_history[-1][1] - new_observation[1] )
        reward = 0.0
        
        if improvement > 0:
            reward = self.norm_reward_scale * improvement
            reward = math.log1p(reward * self.SA.steps_done) * 10

            if self.SA.steps_done > self.max_steps * 0.02:
                reward = max(2.0,reward)  # bonus za poprawę po jakimś czasie #math.log(reward * self.SA.steps_done + 1)*10  #reward * (math.pow(self.SA.steps_done + 1,2)/2) #(math.log(self.SA.steps_done + 1)/2)

        improvment_reward = reward

        #! spłaszczenie nagrody w początkowym stadium przeszukiwania
        if self.SA.steps_done < self.reward_lowerd_steps:
            reward = reward * (self.SA.steps_done/self.reward_lowerd_steps)  
        
        teperature_factor = (self.current_temp - self.min_temp) /(self.starting_temp - self.min_temp)
        self.last_temps.append(teperature_factor)

        #! dodatkowe duże kary za przekroczenie granic temperaturowych
        range_punhishment = 0
        if was_temp_lower_than_min:
            range_punhishment -= 1.0
        elif self.current_temp > self.starting_temp * 3:
            punishment = 0.5 * (int(teperature_factor)-1)
            if punishment > self.norm_reward_scale:
                punishment = self.norm_reward_scale
            range_punhishment -= punishment # silna kara za każdą krotność przekroczenia temperatur

        steps_without_solution_correction = self.run_history[-1][3] * self.max_steps

        # znaczne uproszczenie tego jak wygląda poprzednia funkcja okreslająca wa
        if steps_without_solution_correction > self.max_steps * 0.02:
            if teperature_factor < 0.2:
                range_punhishment -= 1.0 + 1.5 * (0.2 - teperature_factor)  #! lekkie wychłodzenie = mała kara
            elif teperature_factor > 1.5:
                range_punhishment -= 1.0 + (teperature_factor - 1.5) * 1.5  #! grzanie bez efektu = większa kara
            else:
                # w sensownym zakresie i próbujesz – mikro nagroda
                range_punhishment += 0.5
        reward += range_punhishment
        #! kara za stagnacje temperatury, dodatkowo kara zwiększa się aż do 5.0
        stagnation_punishment = 0
        if len(self.last_temps) >= self.last_temps.maxlen-1 and np.std(self.last_temps) < 0.0192137:
            if self.stesp_of_stagnation == 0:
                self.stesp_of_stagnation = len(self.last_temps)-1
            self.stesp_of_stagnation += 1
            stagnation_punishment -= min(1.0 * self.stesp_of_stagnation / (len(self.last_temps)*6),2.5)
        else:
            self.stesp_of_stagnation = max(0,int(math.sqrt(self.stesp_of_stagnation)) -1)
            
        reward += stagnation_punishment

        # if (teperature_factor <= 2):
        #     if (teperature_factor < 0.09):
        #         reward -= (steps_without_solution_correction/self.max_steps*(8))*(1/(0.09)-1) # jak trzymamy temp poniżej 0.09% / 200% możliwej to karamy max 10 razy mocniej by nie wyjebać kary zbyt dużej
        #     elif teperature_factor < 1:
        #         reward -= (steps_without_solution_correction/self.max_steps*(8))*(1/(teperature_factor)-1) # odpowieni współczyniik kary jak mamy temperaturą niższą niż startowa a utkneliśmy w minimum
        #     elif steps_without_solution_correction > self.max_steps*0.02:
        #         reward += max(15 - 30*(steps_without_solution_correction-self.max_steps*0.02)/self.max_steps,-1)  # NAGRADZAMY GO JAK PRUBUJE SZYKAĆ NOWYCH ROZWIĄZAŃ GDY DAWNO CZEGOS NIE ZNALEŹLIŚMY !!!!!!!
        # else:
        #     reward -= (steps_without_solution_correction/self.max_steps*10) # tej sytuacji nie chcemy dodatkowo obciążać bo i tak mamy karę za zbyt dużą temperaturę 
        delta_current_reward = 0
        #! drobna nagroda za poprawę currentalue v 
        #! Uwaga nagroda ta jest zwiększona aż do 2.0 ale za to skaluje się z odległością między current a best czyli tym bardziej go nagdzadzamy im bardziej current zbliża się do obecnego best value 
        delta_current = self.run_history[-1][0] - new_observation[0]
        if delta_current > 0:
            #print("adding mini_reward for good exploration direction:",min(2.0* delta_current ** 0.3,0.5))
            #print("how far is new_current to new_best",(new_observation[0] - new_observation[1]))
            delta_current_reward = min(3.0* delta_current ** 0.2,1) * (1 - max(min((new_observation[0] - new_observation[1])*5.0,0.9),0.1))
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

        self.run_history.append( new_observation + [self.current_temp,reward])#[improvment_reward/self.norm_reward_scale,stagnation_punishment/self.norm_reward_scale,delta_current_reward/self.norm_reward_scale,self.current_temp,reward])
        self.total_reward += reward
        return new_observation, reward , is_terminated, False, self.info()

    
    def getFullParametersHistory(self):
        return self.run_history

    def info(self): # nie sądze by to było potzebne więc zostawiam pusty set 
        if self.done:
            return {"tr":self.total_reward}#{"current_solution":self.SA.current_solution,"best_solution":self.SA.best_solution,"current_temperature":self.current_temp}
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
            self.getStepsWithoutCorrection()/self.max_steps, # dodać ilość korków od ostatniej poprawy (dr)
            (self.current_temp - self.min_temp)/(self.starting_temp - self.min_temp), # dodatkowa normalizacja tempreatury. z racji na to że zakres temperatury tez jest dobierany zaleznie od zadania 
            ]
        
        if self.use_observation_divs:
            if not self.run_history:
                pre_csv = 0
                pre_temp = 0#(self.current_temp - self.min_temp)/(self.starting_temp - self.min_temp)
            else:
                pre_csv = 0#self.run_history[-1][0]
                pre_temp = 0#self.run_history[-1][4]

            obs.append(pre_csv - self.SA.current_solution_value * normalize_factor) #! dodatkowa różnica curent value od ostatniego kroku (jest dodatnia jak mamy poprawę)
            obs.append((self.current_temp - self.min_temp)/(self.starting_temp - self.min_temp) - pre_temp) #! dodatkowa różnica curent TEMPERATURE od ostatniego kroku
            
        return obs
    
    def getStepsWithoutCorrection(self,new_best_value:float = None):
        if len(self.run_history) == 0:
            return 0.0
        value_to_compare_to = self.run_history[-1][1]
        if new_best_value is not None:
            value_to_compare_to = new_best_value
        steps_without_solution_correction = 0
        hist_len = len(self.run_history)
        for i in count():
            if self.run_history[-1-i][1] != value_to_compare_to or i+1 == hist_len:
                steps_without_solution_correction = i
                break
        #print("for new value:",value_to_compare_to,"and history:",self.run_history[:]," we got swsc:",steps_without_solution_correction)
        #a = input("TEST:")
        return steps_without_solution_correction
    
    def estimate_sa_steps(self):
        n = self.SA.problem.dim
        if n < 100:
            alpha = 8.0
        elif n < 200:
            alpha = 6.0
        elif n < 500:
            alpha = 4.5
        elif n < 1000:
            alpha = 3.0
        else:
            alpha = 2.0
        return int(alpha * (n ** 1.5))
    
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
                self.current_temp = self.current_temp * self.actions[actionNR]
                if self.current_temp < self.min_temp:
                    self.current_temp = self.min_temp
                if self.current_temp > self.starting_temp * 10:
                    self.current_temp = self.starting_temp * 10
            
            #perform SA step
            self.SA.step(self.current_temp)

            #collecting data
            self.run_history.append( obs + [self.current_temp,0])
            obs = self.observation()

            if self.SA.steps_done == self.max_steps:
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