import SA
import math
import Problem

class SA_env:

    def __init__(self,preset_problem = None,max_steps = 5000,strating_temp = 1000):
        if preset_problem != None:
            self.SA = SA.SA(preset_problem)
        else:
            self.SA = SA.SA(problem=Problem.VRP())

        self.actions = [f * 0.01 for f in range(80,121,2)]
        self.starting_temp = strating_temp
        self.current_temp = strating_temp
        self.max_steps = max_steps
        self.observation_history = []
        self.action_space = len(self.actions)
        self.observation_space = len(self.observation())
        pass

    def step(self,action_number):
        pre_observation = self.observation()
        self.current_temp = self.current_temp * self.actions[action_number]
        self.SA.step(self.current_temp)
        post_observation = self.observation()
        # obliczamy nagrodę (trzeba dobrze przemyśleć poniższe pomysły na nagrody/kary)
        # różne źródła nagród powinniśmy ze sobą nawzajem ważyć
        # !!! KROKI BEZ POPRAWY WARTOŚCI FUNKCJI CELU NIE POWINNY BYĆ NAGRODZONE !!! 
        # kara za każdy krok bez poprawy best_solution 
        # kara za zakończenie ze zbyt wysoką temperaturę
        # nagroda za poprawę best_solution
        # nagroda za zakończenie z niską temperaturą
        # kara za zbyt gwałtowną zmianę temperatury (machanie góra dół lub wybieranie tylko gwałtowniejszych zmian)
        # kara za temperaturę przekraczającą startową temperaturę
        reward = pre_observation[1] - post_observation[1]
        if reward < 0:
            reward = -reward

        reward = math.log(reward * int(self.SA.steps_done / 10) + 1)
        
        if self.SA.steps_done < self.max_steps:
            is_terminated = False
        else:
            is_terminated = True

        return self.observation(), reward , is_terminated, False#is_terminated_but_have_next_state

    def reset(self,preset_problem = None):
        if preset_problem != None:
            self.SA = SA.SA(preset_problem)
        else:
            self.SA = SA.SA()
        self.current_temp = self.starting_temp
        self.observation_history = []
        return self.observation()

    def observation(self):
        # WARTO DODAĆ NORMALIZACJĘ DANYCH !!!
        return [self.SA.current_solution_value, self.SA.best_solution_value,self.SA.steps_done,self.current_temp]