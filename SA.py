import Problem
import random
import math
from TempSheduler import TempSheduler
    

class SA:
    def __init__(self, preset_problem:Problem.Problem = None, initial_solution = None,skip_initialization = False):
        if skip_initialization == True:
            #params
            self.steps_done = 0
            self.problem = None
            #setting up initial solution
            self.current_solution = []
            self.current_solution_value = 0
            # setting first best solution
            self.best_solution = self.current_solution
            self.best_solution_value = self.current_solution_value
        else:
            self.reset(preset_problem,initial_solution)

    def reset(self,preset_problem:Problem.Problem = None, initial_solution = None):
        #params
        self.steps_done = 0
        if preset_problem is not None:
            self.problem = preset_problem
        else:
            self.problem = Problem.TSP()
        #setting up initial solution
        if initial_solution != None:
            self.current_solution = initial_solution
        else:
            self.current_solution = self.problem.get_initial_solution()
        self.current_solution_value = self.problem.objective_function(self.current_solution)
        # setting first best solution
        self.best_solution = self.current_solution
        self.best_solution_value = self.current_solution_value

    def acceptNewSolution(self,new_solution,new_solution_value = None):
        if new_solution_value == None:
            new_solution_value = self.problem.objective_function(new_solution)
        self.current_solution = new_solution
        self.current_solution_value = new_solution_value
        #check if new solution is better than best
        if self.problem.isFirstBetter(new_solution_value,self.best_solution_value):
            self.best_solution = new_solution
            self.best_solution_value = new_solution_value

    def step(self,temp_value:float,steps_per_temperature = 1):
        for _ in range(steps_per_temperature):
            #get random neighbor
            new_solution = self.problem.get_random_neighbor(self.current_solution)
            new_solution_value = self.problem.objective_function(new_solution)

            #check if new solution is better than current
            if self.problem.isFirstBetter(new_solution_value,self.current_solution_value):
                self.acceptNewSolution(new_solution,new_solution_value)
            else:
                if random.random() < math.exp(-(new_solution_value - self.current_solution_value)/temp_value):
                    self.acceptNewSolution(new_solution,new_solution_value)

            self.steps_done += 1

    def run(self, max_steps:int, temp_shadeuling_model:TempSheduler,generate_plot_data = False):
        best_values = []
        current_values = []
        temperature_values = []
        for _ in range(max_steps):
            #getting new temperature
            temp = temp_shadeuling_model.getTemp(self.steps_done)
            
            #perform SA step
            self.step(temp)

            #collecting data
            best_values.append(self.best_solution_value)
            if generate_plot_data:
                current_values.append(self.current_solution_value)
            if generate_plot_data:
                temperature_values.append(temp)

        if generate_plot_data:
            return best_values,current_values,temperature_values
        return best_values

