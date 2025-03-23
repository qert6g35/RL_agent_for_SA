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
        self.current_solution_value = problem.objective_function(self.current_solution)
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

    def step(self,temp_value:float):

        
        #get random neighbor
        new_solution = self.problem.get_random_neighbor(self.current_solution)
        new_solution_value = self.problem.objective_function(new_solution)


        #check if new solution is better than current
        if self.problem.isFirstBetter(new_solution_value,self.current_solution_value):
            self.acceptNewSolution(new_solution,new_solution_value)
        else:
            if random.random() < math.exp(-(new_solution_value - self.current_solution_value)/temp_value):
                self.acceptNewSolution(new_solution,new_solution_value)