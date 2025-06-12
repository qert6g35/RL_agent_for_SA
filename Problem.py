from abc import ABC, abstractmethod
from typing import Any
import random
import os
from torch import randint
import tsplib95
import networkx as nx
import numpy as np
import random as r
import math

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
        pass

    @abstractmethod
    def getDimention(self):
        pass

class TSP(Problem):
    #its TSP with returns
    def __init__(self,generation_dim = 0,range = [50,300],generate_harder_instance = False,use_lib_instances = False):
        # if distances is not None:
        #     self.distances = distances
        # else:
        if len(range) == 2:
            generation_dim = r.randint(range[0],range[1])
        if use_lib_instances:
            problem = tsplib95.load(self.choose_random_file('TSP_examle'))
            problems_with_optimal_tour = ["kroA100", "kroA150", "kroA200", "kroB100", "kroB150", "kroB200"]
            while problem.name not in problems_with_optimal_tour:
                problem = tsplib95.load(self.choose_random_file('TSP_examle'))
            print("choosed problem",problem.name)
            self.name = problem.name
            self.dim = problem.dimension
            self.graph = problem.get_graph(normalize=True)
        else:
            self.name = "g_"+str(generation_dim)
            self.dim = generation_dim
            if generate_harder_instance:
                self.graph = self.generate_tsp_graph_random_distances()
            else:
                self.graph = self.generate_tsp_graph()
        
        self.upperBound = None
        self.use_smart_neighbour = False
        super().__init__()


    def objective_function(self, x: Any) -> float:
        return sum([self.graph.edges[x[-1],x[0]]["weight"]] + [self.graph.edges[x[i],x[i+1]]["weight"] for i in range(len(x)-1)])
    
    def get_initial_solution(self):
        initial_solution = [f for f in range(self.dim)]
        random.shuffle(initial_solution)
        return initial_solution
    
    def get_random_neighbor(self, x: Any):
        if self.use_smart_neighbour:
            swap_place_a = random.randint(0,len(x))
            swap_place_b = random.randint(0,len(x))
            while swap_place_a == swap_place_b or swap_place_a+1 == swap_place_b or swap_place_a-1 == swap_place_b:
                swap_place_b = random.randint(0,len(x))
            if swap_place_a > swap_place_b:
                swap_place_a, swap_place_b = swap_place_b, swap_place_a
            return x[:swap_place_a] + x[swap_place_a:swap_place_b][::-1] + x[swap_place_b:]
        
        swap_place_a = random.randint(0,len(x)-1)
        swap_place_b = swap_place_a
        while swap_place_a == swap_place_b:
            swap_place_b = random.randint(0,len(x)-1)

        x[swap_place_a],x[swap_place_b] = x[swap_place_b],x[swap_place_a]
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
        dimentions = []
        for file in files:
            all_problems+=1
            path = os.path.join(folder_path, file)
            problem = tsplib95.load(path)
            if problem.dimension in range(50,300):
                dimentions.append(problem.name)
            # if problem.dimension < 2000:
            #     available_weak_problems+=1
            #     G = problem.get_graph()

            #     # Check connectivity
            #     if isinstance(G, nx.DiGraph):
            #         is_connected = nx.is_strongly_connected(G)  # For directed graphs
            #     else:
            #         is_connected = nx.is_connected(G)  # For undirected graphs
            #     if is_connected :
            #         available_problems += 1
        dimentions.sort()
        print(dimentions)
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
        return self.upperBound
    
    def EstimateDeltaEnergy(self):
        n = self.dim
        deltas = []
        for _ in range(1000):
            x = self.get_initial_solution()
            x_value = self.objective_function(x)
            deltas.append(abs(x_value - self.objective_function(self.get_random_neighbor(x))))
        return np.mean(deltas), np.max(deltas), np.min(deltas)

    def getDimention(self):
        return self.dim
    
    def generate_tsp_graph_random_distances(self):
        """
        Generuje instancję problemu TSP jako pełny graf,
        gdzie odległość między każdą parą miast jest losowa.
        Pozycje miast nie są używane.
        """
        print(f"\nGenerowanie problemu TSP (losowe odległości) o wymiarze: {self.dim}\n")
        
        G = nx.Graph()

        # Dodajemy wierzchołki (miasta) do grafu
        for i in range(self.dim):
            G.add_node(i)

        # Określamy zakres dla losowych odległości, np. od 1 do 100
        min_dist = self.dim
        max_dist = self.dim * 200

        # Dodajemy krawędzie między każdą parą miast z losową wagą.
        # Pętla for j in range(i + 1, ...) zapewnia, że każdą parę miast rozważamy tylko raz.
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                # Losujemy wagę (odległość) dla tej krawędzi
                weight = random.randint(min_dist, max_dist)
                G.add_edge(i, j, weight=weight)

        print(f"Wygenerowano problem (losowe odległości) o wymiarze: {self.dim}")
        return G

    def generate_tsp_graph(self):
        print("")
        print("Generateing TSP problem with dim:",self.dim)
        print("")
        """Generuje instancję problemu TSP jako pełny graf z wagami euklidesowymi."""
        G = nx.Graph()
        coord_range = self.dim * 10

        # Generujemy losowe współrzędne dla każdego miasta
        positions = {
            i: (random.uniform(0, coord_range), random.uniform(0, coord_range))
            for i in range(self.dim)
        }

        # Dodajemy wierzchołki do grafu z atrybutami pozycji
        for node, pos in positions.items():
            G.add_node(node, pos=pos)

        # Dodajemy krawędzie z wagą jako odległość euklidesowa
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                weight = math.hypot(x2 - x1, y2 - y1)
                G.add_edge(i, j, weight=weight)
        print("generated problem with dim:",self.dim)
        return G

