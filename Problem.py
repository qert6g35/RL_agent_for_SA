from abc import ABC, abstractmethod
from typing import Any
import random

class Problem(ABC):
    """Abstract base class for an optimization problem."""

    @abstractmethod
    def objective_function(self, x: Any) -> float:
        """Evaluate the objective function at a given solution."""
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
    def __init__(self,distances = None):
        if distances is not None:
            self.distances = distances
        else:
            self.distances = [
                [0, 1, 2, 3],
                [1, 0, 4, 5],
                [2, 4, 0, 6],
                [3, 5, 6, 0]
            ]
        super().__init__()

    def isFirstBetter(x,y):
        return x < y

    def objective_function(self, x: Any) -> float:
        return sum([self.distances[x[i-1]][x[i]] for i in range(len(x))])
    
    def get_initial_solution(self):
        return [f for f in range(len(self.distances))]
    
    def get_random_neighbor(self, x: Any):
        swap_place = random.randint(1,len(x)-1)
        return x[swap_place:] + x[:swap_place]
    
print(VRP().objective_function([0,1,2,3])) #1+4+6+3 = 14