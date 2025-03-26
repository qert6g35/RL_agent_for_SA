import SA
import math

class SA_env():

    def __init__(self,preset_problem = None,max_steps = 5000):
        if preset_problem != None:
            self.SA = SA.SA(preset_problem)
        else:
            self.SA = SA.SA()

        self.actions = [f * 0.01 for f in range(80,121,2)]
        self.action_space = len(self.actions)
        
        self.observation_space = len(self.observation())

        self.max_steps = max_steps
        pass

    def step(self,action_number):
        pre_observation = self.observation()
        self.SA.step(self.actions[action_number])
        post_observation = self.observation()

        reward = pre_observation[1] - post_observation[1]
        if reward < 0:
            reward = -reward

        reward = math.log(reward * int(self.SA.steps_done / 10) + 1)
        
        if self.SA.steps_done < self.max_steps:
            is_terminated = False
        else:
            is_terminated = True

        return self.observation(), reward , is_terminated, False#is_terminated_but_have_next_state

    def resert(self):
        pass

    def observation(self):
        # WARTO DODAĆ NORMALIZACJĘ DANYCH !!!
        return [self.SA.current_solution_value, self.SA.best_solution_value,self.SA.steps_done]