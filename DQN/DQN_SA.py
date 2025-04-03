import SA
import math
import Problem
from itertools import count

#SA enviroment devined for DQN
class SA_env:

    def __init__(self,preset_problem = None,max_steps = 5000):
        if preset_problem != None:
            self.SA = SA.SA(preset_problem)
        else:
            self.SA = SA.SA()
        self.max_temp_accepting_chance = 0.85
        # elements that should change when SA is changed
        self.starting_temp = (self.SA.problem.getUpperBound()/4)/-math.log(self.max_temp_accepting_chance)
        print("we have starting temp:",self.starting_temp)
        self.current_temp = self.starting_temp
        

        # elements that shouldn't change when SA is changed
        self.min_temp = 0.1
        self.actions = [float(f) * 0.01 for f in range(80,121,4)]
        self.no_reward_steps = int(max_steps * 0.01)
        print("there will be no reward for first steps:",self.no_reward_steps)
        self.max_steps = max_steps
        self.action_space = len(self.actions)
        self.run_history = []
        self.observation_space = len(self.observation())
        self.run_history.append([0 for _ in range(self.observation_space + 2)])
        self.run_history[0][-2] = self.current_temp
        pass

    def reset(self,preset_problem = None):
        if preset_problem != None:
            self.SA = SA.SA(preset_problem)
        else:
            self.SA = SA.SA()
        self.starting_temp = (self.SA.problem.getUpperBound()/4)/-math.log(self.max_temp_accepting_chance)
        print("we have starting temp:",self.starting_temp)
        self.current_temp = self.starting_temp
        self.run_history = [[0 for _ in range(self.observation_space + 2)]]
        self.run_history[0][-2] = self.current_temp
        return self.observation()

    def step(self,action_number):
        was_temp_lower_than_min = False
        self.current_temp = self.current_temp * self.actions[action_number]
        if self.current_temp < self.min_temp:
            was_temp_lower_than_min = True
            self.current_temp = self.min_temp

        self.SA.step(self.current_temp)
        new_observation = self.observation()

        
        # obliczamy nagrodę
        # TODO nagrody których dodanie trzeba wykonać / przemyśleć
        #! różne źródła nagród powinniśmy ze sobą nawzajem ważyć
        #? KROKI BEZ POPRAWY WARTOŚCI FUNKCJI CELU NIE POWINNY BYĆ NAGRODZONE
        # kara za zdropowanie temperatury do 0 zbyt szybko i nie podnoszenie jej przez dłuższy czas
        # kara za zakończenie ze zbyt wysoką temperaturę
        # nagroda za zakończenie z niską temperaturą
        # kara za zbyt gwałtowną zmianę temperatury (machanie góra dół lub wybieranie tylko gwałtowniejszych zmian)
        #*  nagrody które już 
        # kara za każde x kroków bez poprawy best_solution 
        # nagroda za popawę najleprzej wartości
        # kara za temperaturę przekraczającą <temp_min,starting_temp*2>

        # nagroda za poprawę best value
        reward = self.run_history[-1][1] - new_observation[1]
        if reward < 0:
            reward = -reward
        #! uwaga tutaj najbardziej newraligncze miejsce dycutujące o tym jak wygląda nagroda
        reward = math.log(reward * self.SA.steps_done + 1)  #reward * (math.pow(self.SA.steps_done + 1,2)/2) #(math.log(self.SA.steps_done + 1)/2)
        #* do X kroków nie oferujemy nagrody za poprawienie best value
        if self.SA.steps_done < self.no_reward_steps:
            reward = 0.0    
        
        #* kara za przekroczenie granic temperaturowych
        if was_temp_lower_than_min:
            reward -= 10
        elif self.current_temp > self.starting_temp:
            reward -= 2 * int(self.current_temp / self.starting_temp)

        #* kara za każde x kroków bez poprawy best_solution 
        steps_without_solution_correction = self.run_history[-1][3]

        reward -= int(steps_without_solution_correction/self.max_steps*25)

        if self.SA.steps_done < self.max_steps:
            is_terminated = False
        else:
            is_terminated = True

        

        self.run_history.append( new_observation + [self.current_temp,reward])

        return new_observation, reward , is_terminated

    
    def getFullParametersHistory(self):
        return self.run_history

    def observation(self,normalize = True,norm_reward_scale = 100.0):
        normalize_factor = norm_reward_scale / self.SA.problem.getUpperBound()

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
            self.SA.current_solution_value, 
            self.SA.best_solution_value,
            self.SA.steps_done/self.max_steps, # (tutaj mamy ile już zrobiliśmy w %) zamienić kroki+max_kroki na % ile zostało 
            self.getStepsWithoutCorrection(), # dodać ilość korków od ostatniej poprawy 
            (self.current_temp - self.min_temp)/(self.starting_temp - self.min_temp), # dodatkowa normalizacja tempreatury. z racji na to że zakres temperatury tez jest dobierany zaleznie od zadania 
            ]

        if normalize:
            obs[0],obs[1] = obs[0] * normalize_factor, obs[1] * normalize_factor

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
    
    