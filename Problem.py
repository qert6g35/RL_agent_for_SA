from abc import ABC, abstractmethod
from typing import Any
import random
import os
import tsplib95

class Problem(ABC):
    """Abstract base class for an optimization problem."""

    @abstractmethod
    def objective_function(self, x: Any) -> float:
        """Evaluate the objective function at a given solution."""
        pass

    def isFirstBetter(x,y):
        """Defines if x is better than y"""
        pass

    @abstractmethod
    def get_initial_solution(self) -> Any:
        """Return an initial solution for the optimization problem."""
        pass

    @abstractmethod
    def get_random_neighbor(self, x: Any) -> Any:
        """Return a neighbor of a given solution."""
        pass

class VRP(Problem):
    #its VRP with returns
    def __init__(self,distances = None):
        if distances is not None:
            self.distances = distances
        else:
            problem = tsplib95.load(self.choose_random_file('VRP_examle'))
            self.distances = [[problem.get_weight(i, j) for j in problem.get_nodes()] for i in problem.get_nodes()]
        super().__init__()

    def objective_function(self, x: Any) -> float:
        return sum([self.distances[x[i-1]][x[i]] for i in range(len(x))])
    
    def get_initial_solution(self):
        initial_solution = [f for f in range(len(self.distances))]
        random.shuffle(initial_solution)
        return initial_solution
    
    def get_random_neighbor(self, x: Any):
        swap_place_a = random.randint(0,len(x))
        swap_place_b = random.randint(0,len(x))
        if swap_place_a > swap_place_b:
            swap_place_a, swap_place_b = swap_place_b, swap_place_a
        return x[:swap_place_a] + x[swap_place_a:swap_place_b][::-1] + x[swap_place_b:]
    
    def isFirstBetter(self,x,y):
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return x < y
        elif isinstance(x, list) and isinstance(y, list):
            return self.objective_function(x) < self.objective_function(y)  # Example comparison for lists
        else:
            raise TypeError("Unsupported types for comparison")
    
    def choose_random_file(self,folder_path):
        """Returns the full path of a randomly chosen file from the given folder."""
        if not os.path.isdir(folder_path):
            raise ValueError(f"Invalid directory: {folder_path}")
        
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        if not files:
            raise FileNotFoundError("No files found in the given folder.")
        
        return os.path.join(folder_path, random.choice(files))

