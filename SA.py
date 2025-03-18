from Problem import Problem
import random
import math

class SA:
    def __init__(self, problem:Problem, initial_solution = None):
        self.problem = problem
        if initial_solution != None:
            self.current_solution = initial_solution
        else:
            self.current_solution = problem.initial_solution()
        self.best_solution = self.current_solution
        self.best_solution_value = problem.objective_function(self.current_solution)

    def step(self,temp_value:float):
        new_solution = self.problem.get_random_neighbor(self.current_solution)
        new_solution_value = self.problem.objective_function(new_solution)
        if new_solution_value < self.best_solution_value:
            self.best_solution = new_solution
            self.best_solution_value = new_solution_value
        if new_solution_value < self.problem.objective_function(self.current_solution):
            self.current_solution = new_solution
        else:
            if random.random() < math.exp(-(new_solution_value - self.problem.objective_function(self.current_solution))/temp_value):
                self.current_solution = new_solution