from ast import mod
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

#SA enviroment devined for DQN
class SA_env(gym.Env):

    def __init__(self,preset_problem = None,max_steps = 5000,set_up_learning_on_init = False):
        # elements that shouldn't change when SA is changed
        self.max_temp_accepting_chance = 0.85
        self.min_temp_accepting_chance = 0.001
        self.actions = [float(f) * 0.01 for f in range(80,121,4)]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.no_reward_steps = max(int(max_steps * 0.005),5)
        self.total_reward = 0
        self.done = False
        #print("there will be no reward for first steps:",self.no_reward_steps)
        self.max_steps = max_steps
        self.run_history = []
        self.norm_reward_scale = 10.0
        self.last_temps = deque([],maxlen=10)

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
        #print("we have starting temp:",self.starting_temp)
        #print("we have min temp:",self.min_temp)
        self.current_temp = self.starting_temp
        obs = self.observation()
        self.run_history = [obs + [self.current_temp,0]]
        self.done = False
        self.total_reward = 0
        return self.observation(), self.info() #!!! we pas none as info

    def step(self,action_number):
        was_temp_lower_than_min = False
        self.current_temp = self.current_temp * self.actions[action_number]
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
            reward = math.log1p(reward * self.SA.steps_done) * 5

            if self.SA.steps_done > self.max_steps * 0.02:
                reward += 2.0  # bonus za poprawę po jakimś czasie #math.log(reward * self.SA.steps_done + 1)*10  #reward * (math.pow(self.SA.steps_done + 1,2)/2) #(math.log(self.SA.steps_done + 1)/2)

        #* do X kroków nie oferujemy nagrody za poprawienie best value
        if self.SA.steps_done < self.no_reward_steps:
            reward = 0.0    
        
        teperature_factor = (self.current_temp - self.min_temp) /(self.starting_temp - self.min_temp)
        self.last_temps.append(teperature_factor)

        #* kara za przekroczenie granic temperaturowych
        if was_temp_lower_than_min:
            reward -= 2.0
        elif self.current_temp > self.starting_temp * 3:
            punishment = 0.5 * (int(teperature_factor)-1)
            if punishment > self.norm_reward_scale:
                punishment = self.norm_reward_scale
            reward -= punishment # silna kara za każdą krotność przekroczenia temperatur

        steps_without_solution_correction = self.run_history[-1][3] * self.max_steps

        # znaczne uproszczenie tego jak wygląda poprzednia funkcja okreslająca wa
        if steps_without_solution_correction > self.max_steps * 0.02:
            if teperature_factor < 0.2:
                reward -= 1.0 + 2.0 * (0.2 - teperature_factor)  # lekkie wychłodzenie = mała kara
            elif teperature_factor > 2.0:
                reward -= 1.0 + (teperature_factor - 2.0) * 1.5  # grzanie bez efektu = też kara
            else:
                # w sensownym zakresie i próbujesz – mikro nagroda
                reward += 0.5
                if len(self.last_temps) >= self.last_temps.maxlen-1 and np.std(self.last_temps) < 0.02:
                    reward -= 1.5

        # if (teperature_factor <= 2):
        #     if (teperature_factor < 0.09):
        #         reward -= (steps_without_solution_correction/self.max_steps*(8))*(1/(0.09)-1) # jak trzymamy temp poniżej 0.09% / 200% możliwej to karamy max 10 razy mocniej by nie wyjebać kary zbyt dużej
        #     elif teperature_factor < 1:
        #         reward -= (steps_without_solution_correction/self.max_steps*(8))*(1/(teperature_factor)-1) # odpowieni współczyniik kary jak mamy temperaturą niższą niż startowa a utkneliśmy w minimum
        #     elif steps_without_solution_correction > self.max_steps*0.02:
        #         reward += max(15 - 30*(steps_without_solution_correction-self.max_steps*0.02)/self.max_steps,-1)  # NAGRADZAMY GO JAK PRUBUJE SZYKAĆ NOWYCH ROZWIĄZAŃ GDY DAWNO CZEGOS NIE ZNALEŹLIŚMY !!!!!!!
        # else:
        #     reward -= (steps_without_solution_correction/self.max_steps*10) # tej sytuacji nie chcemy dodatkowo obciążać bo i tak mamy karę za zbyt dużą temperaturę 

        # drobna nagroda za poprawę current value
        delta_current = self.run_history[-1][0] - new_observation[0]
        if delta_current > 0:
            reward += 0.1* delta_current ** 0.5

        #! TO MOŻE NAM POMÓC Z OGARNIĘĆIEM WYBUCHAJĄCYCH WARTOŚCI PRZY STEROWANIU
        # normalizacja nagrody
        reward = max(min(reward,self.norm_reward_scale),-self.norm_reward_scale)/self.norm_reward_scale
        #print("reward:",reward)
        if self.SA.steps_done < self.max_steps:
            is_terminated = False
        else:
            is_terminated = True
            self.done = True

        

        self.run_history.append( new_observation + [self.current_temp,reward])
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
        if not self.run_history:
            pre_csv = 0
            pre_temp = (self.current_temp - self.min_temp)/(self.starting_temp - self.min_temp)
        else:
            pre_csv = self.run_history[-1][0]
            pre_temp = self.run_history[-1][4]
        obs = [
            self.SA.current_solution_value * normalize_factor, 
            self.SA.best_solution_value * normalize_factor,
            self.SA.steps_done/self.max_steps, # (tutaj mamy ile już zrobiliśmy w %) zamienić kroki+max_kroki na % ile zostało 
            self.getStepsWithoutCorrection()/self.max_steps, # dodać ilość korków od ostatniej poprawy (dr)
            (self.current_temp - self.min_temp)/(self.starting_temp - self.min_temp), # dodatkowa normalizacja tempreatury. z racji na to że zakres temperatury tez jest dobierany zaleznie od zadania 
            pre_csv - self.SA.current_solution_value * normalize_factor, #! dodatkowa różnica curent value od ostatniego kroku (jest dodatnia jak mamy poprawę)
            (self.current_temp - self.min_temp)/(self.starting_temp - self.min_temp) - pre_temp, #! dodatkowa różnica curent TEMPERATURE od ostatniego kroku
            ]

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
            return [x * unnormalize_factor for x in transposed_run_history[1]],[x * unnormalize_factor for x in transposed_run_history[0]],transposed_run_history[-3]  #best_values,current_values,temperature_values
        else:
            return [x[1] * unnormalize_factor for x in self.run_history] 


    def render(self):
        return super().render()
    
    def close(self):
        return super().close()