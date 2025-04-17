from abc import ABC, abstractmethod
import math
class TempSheduler(ABC):
    """Abstract base class for an time sheduler module."""
    
    ''' Returns temperature for given step

        step: int - number of current step for which we want to get temperature
        
        return:float - temperature for given step
    '''
    @abstractmethod 
    def getTemp(self,step)->float:
        pass

class LinearTempSheduler(TempSheduler):
    ''' Linear temperature sheduler
    
        start_temp:float - starting temperature
        end_temp:float - ending temperature
        steps:int - number of steps to reach end temperature
    '''

    def __init__(self,start_temp:float,end_temp:float,end_steps:int):
        self.reset(start_temp = start_temp,end_steps=end_steps,end_temp=end_temp)

    def getTemp(self,step):# tutaj podajemy tylko jeden agrument, który to jekt krok aglorytmu
        if step >= self.end_steps:
            return self.end_temp
        return self.start_temp - self.temp_diff * step
    
    def reset(self,start_temp:float,end_temp:float,end_steps:int):
        self.start_temp = start_temp
        self.end_steps = end_steps
        self.end_temp = end_temp
        self.temp_diff = (start_temp - end_temp) / end_steps
    
class ConstTempSheduler(TempSheduler):
    ''' Linear temperature sheduler
    
        start_temp:float - starting temperature
        end_temp:float - ending temperature
        steps:int - number of steps to reach end temperature
    '''

    def __init__(self,temp:float = None):
        if temp is not None:
            self.const_temp = temp
        else:
            self.const_temp = 100

    def getTemp(self, *args):
        return self.const_temp
    
class GeometricTempSheduler(TempSheduler):
    '''
    Geometric temperature scheduler:
    T(step) = start_temp * (decay_rate)^step
    decay_rate obliczane tak, by T(end_steps) = end_temp
    '''
    def __init__(self, start_temp: float, end_temp: float, end_steps: int):
        self.reset(start_temp, end_temp, end_steps)

    def getTemp(self, step):
        temp = self.start_temp * (self.decay_rate ** step)
        return max(temp, self.end_temp)

    def reset(self, start_temp: float, end_temp: float, end_steps: int):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.end_steps = end_steps
        # decay_rate obliczony tak, że temp schodzi do end_temp po end_steps krokach
        self.decay_rate = (end_temp / start_temp) ** (1 / end_steps)


class ReciprocalTempSheduler(TempSheduler):
    '''
    Reciprocal temperature scheduler:
    T(step) = start_temp / (1 + decay_factor * step)
    decay_factor obliczane na podstawie end_steps
    '''
    def __init__(self, start_temp: float, end_temp: float, end_steps: int):
        self.reset(start_temp, end_temp, end_steps)

    def getTemp(self, step):
        temp = self.start_temp / (1 + self.decay_factor * step)
        return max(temp, self.end_temp)

    def reset(self, start_temp: float, end_temp: float, end_steps: int):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.end_steps = end_steps
        self.decay_factor = (start_temp / end_temp - 1) / end_steps

class LogarithmicTempSheduler(TempSheduler):
    '''
    Logarithmic temperature scheduler:
    T(step) = start_temp / log(1 + decay_factor * step)
    decay_factor obliczane na podstawie end_steps
    '''
    def __init__(self, start_temp: float, end_temp: float, end_steps: int):
        self.reset(start_temp, end_temp, end_steps)

    def getTemp(self, step):
        if step == 0:
            return self.start_temp
        temp = self.start_temp / math.log(1 + self.decay_factor * step)
        return max(temp, self.end_temp)

    def reset(self, start_temp: float, end_temp: float, end_steps: int):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.end_steps = end_steps
        self.decay_factor = (math.exp(start_temp / end_temp) - 1) / end_steps