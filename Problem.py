from abc import ABC, abstractmethod
from typing import Any
import random
import os
import tsplib95
import networkx as nx
import numpy as np

class Problem(ABC):
    """Abstract base class for an optimization problem."""

    @abstractmethod
    def objective_function(self, x: Any) -> float:
        """Evaluate the objective function at a given solution."""
        pass

    def isFirstBetter(self,x,y):
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
    
    @abstractmethod
    def getUpperBound(self) -> float:
        '''Should calculate (or at least aproximate) upperbound for given problem'''
        pass

    @abstractmethod
    def EstimateDeltaEnergy(self,n):
        '''Shold get n random inputs and output np.mean(x_value - x_neighbour_value,...)'''

class TSP(Problem):
    #its TSP with returns
    def __init__(self):
        # if distances is not None:
        #     self.distances = distances
        # else:
        problem = tsplib95.load(self.choose_random_file('TSP_examle'))
        while problem.dimension > 2000:
            problem = tsplib95.load(self.choose_random_file('TSP_examle'))
        print("choosed problem with dimention:",problem.dimension)
        self.graph = problem.get_graph(normalize=True)
        self.upperBound = None
        super().__init__()

    def objective_function(self, x: Any) -> float:
        return sum([self.graph.edges[x[0],x[1]]["weight"] for i in range(len(x))])
    
    def get_initial_solution(self):
        initial_solution = [f for f in range(len(self.graph.nodes))]
        random.shuffle(initial_solution)
        return initial_solution
    
    def get_random_neighbor(self, x: Any):
        swap_place_a = random.randint(0,len(x)-1)
        swap_place_b = random.randint(0,len(x)-1)
        while swap_place_a == swap_place_b:
            swap_place_b = random.randint(0,len(x)-1)
        # smart neighbour devinition 
        if swap_place_a > swap_place_b:
            swap_place_a, swap_place_b = swap_place_b, swap_place_a
        return x[:swap_place_a] + x[swap_place_a:swap_place_b][::-1] + x[swap_place_b:]
        
        # dump neighbour devinition
        #x[swap_place_a],x[swap_place_b] = x[swap_place_b],x[swap_place_a]
        return x

    
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

    def evaluate_tsp_files(self,folder_path = 'TSP_examle'):
        if not os.path.isdir(folder_path):
            raise ValueError(f"Invalid directory: {folder_path}")
        
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        
        if not files:
            raise FileNotFoundError("No files found in the given folder.")
        

        available_problems = 0
        available_weak_problems = 0
        all_problems = 0
        for file in files:
            all_problems+=1
            path = os.path.join(folder_path, file)
            problem = tsplib95.load(path)
            if problem.dimension < 2000:
                available_weak_problems+=1
                G = problem.get_graph()

                # Check connectivity
                if isinstance(G, nx.DiGraph):
                    is_connected = nx.is_strongly_connected(G)  # For directed graphs
                else:
                    is_connected = nx.is_connected(G)  # For undirected graphs
                if is_connected :
                    available_problems += 1
        
        print("there are:",all_problems," in folder and ",available_problems," are available, and", available_weak_problems - available_problems)
            



    def getUpperBound(self):
        if self.upperBound is None:
            upperBound = 0
            for i in range(len(self.graph.nodes)):
                maxValue = 0
                for j in range(len(self.graph.nodes)):
                    if i != j and self.graph.edges[i,j]["weight"] > maxValue:
                        maxValue = self.graph.edges[i,j]["weight"]
                upperBound += maxValue
            self.upperBound = upperBound
            print("upperbound is:",self.upperBound)
            # upperBoundSolution = [0]
            # not_visited = [f for f in range(1,len(self.graph.nodes))]
            # while len(not_visited) > 0:
            #     farthest_to_last = -1
            #     dist_to_farthest = 0
            #     for id in not_visited:
            #         if self.graph.edges[upperBoundSolution[-1],id]["weight"] > dist_to_farthest:
            #             farthest_to_last = id
            #             dist_to_farthest = self.graph.edges[upperBoundSolution[-1],id]["weight"]
            #     upperBoundSolution.append(not_visited.pop(not_visited.index(farthest_to_last)))
            # self.upperBound = self.objective_function(upperBoundSolution)    
        return self.upperBound
    
    def EstimateDeltaEnergy(self,n):
        deltas = []
        for _ in range(n):
            x = self.get_initial_solution()
            x_value = self.objective_function(x)
            deltas.append(abs(x_value - self.objective_function(self.get_random_neighbor(x))))
        return np.mean(deltas)

