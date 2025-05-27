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
        
    def __call__(self, *args, **kwds):
        return self.getTemp(args)
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

class LinearScheduler_FirstKind(TempSheduler):
    def __init__(self, start_temp: float = 1.1, end_temp: float = 0.1, total_steps: int = 1):
        self.reset(start_temp,end_temp,total_steps)

    def reset(self, start_temp: float, end_temp: float, total_steps: int):
        self.f = start_temp
        self.l = end_temp
        self.T = total_steps

    def getTemp(self, step: int):
        return ((self.l - self.f) / (self.T - 1)) * (step - 1) + self.f
    
    def __call__(self, *args, **kwds):
        return self.getTemp(args[0])

class ConstTempSheduler(TempSheduler):
    def __init__(self, start_temp: float = 1.1, end_temp: float = 0.1, total_steps: int = 1):
        self.reset(start_temp,end_temp,total_steps)

    def reset(self, start_temp: float, end_temp: float, total_steps: int):
        if start_temp is not None:
            self.const_temp = (start_temp + end_temp)/2
        else:
            self.const_temp = 100

    def getTemp(self, step):
        return self.const_temp
    
    def __call__(self, *args, **kwds):
        return self.getTemp(args[0])
    
class GeometricTempSheduler(TempSheduler):
    '''
    Geometric temperature scheduler:
    T(step) = start_temp * (decay_rate)^step
    decay_rate obliczane tak, by T(end_steps) = end_temp
    '''
    def __init__(self, start_temp: float = 1.1, end_temp: float = 0.1, total_steps: int = 1):
        self.reset(start_temp, end_temp, total_steps)

    def getTemp(self, step):
        temp = self.start_temp * (self.decay_rate ** step)
        return max(temp, self.end_temp)

    def reset(self, start_temp: float, end_temp: float, end_steps: int):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.end_steps = end_steps
        # decay_rate obliczony tak, że temp schodzi do end_temp po end_steps krokach
        self.decay_rate = (end_temp / start_temp) ** (1 / end_steps)
    
    def __call__(self, *args, **kwds):
        return self.getTemp(args[0])

class GeometricScheduler_FirstKind(TempSheduler):
    def __init__(self, start_temp: float = 1.1, end_temp: float = 0.1, total_steps: int = 1):
        self.reset(start_temp,end_temp,total_steps)

    def reset(self, start_temp: float, end_temp: float, total_steps: int):
        self.f = start_temp
        self.l = end_temp
        self.T = total_steps

    def getTemp(self, step: int):
        ratio = self.l / self.f
        return self.f * (ratio ** ((step - 1) / (self.T - 1)))
    
    def __call__(self, *args, **kwds):
        return self.getTemp(args[0])
    
class GeometricScheduler_SecondKind(TempSheduler):
    def __init__(self, start_temp: float = 1.1, end_temp: float = 0.1, total_steps: int = 1):
        self.reset(start_temp,end_temp,total_steps)

    def reset(self, start_temp: float, end_temp: float, total_steps: int):
        self.f = start_temp
        self.l = end_temp
        self.T = total_steps
        self.ratio = self.f / self.l

    def getTemp(self, step: int):  
        return self.f + self.l * (1 - self.ratio ** ((step - 1) / (self.T - 1)))
    
    def __call__(self, *args, **kwds):
        return self.getTemp(args[0])

class ReciprocalTempSheduler(TempSheduler):
    '''
    Reciprocal temperature scheduler:
    T(step) = start_temp / (1 + decay_factor * step)
    decay_factor obliczane na podstawie end_steps
    '''
    def __init__(self, start_temp: float = 1.1, end_temp: float = 0.1, total_steps: int = 1):
        self.reset(start_temp, end_temp, total_steps)

    def getTemp(self, step):
        temp = self.start_temp / (1 + self.decay_factor * step)
        return max(temp, self.end_temp)

    def reset(self, start_temp: float, end_temp: float, end_steps: int):
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.end_steps = end_steps
        self.decay_factor = (start_temp / end_temp - 1) / end_steps

    def __call__(self, *args, **kwds):
        return self.getTemp(args[0])

class ReciprocalScheduler_FirstKind:
    def __init__(self, start_temp: float = 1.1, end_temp: float = 0.1, total_steps: int = 1):
        self.reset(start_temp,end_temp,total_steps)

    def reset(self, start_temp: float, end_temp: float, total_steps: int):
        self.f = start_temp
        self.l = end_temp
        self.T = total_steps

    def getTemp(self, step: int):
        numerator = self.l * self.f * (self.T - 1)
        denominator = (self.l * self.T - self.f) + (self.f - self.l) * step
        return numerator / denominator
    
    def __call__(self, *args, **kwds):
        return self.getTemp(args[0])
    
class ReciprocalScheduler_SecondKind:
    def __init__(self, start_temp: float = 1.1, end_temp: float = 0.1, total_steps: int = 1):
        self.reset(start_temp,end_temp,total_steps)

    def reset(self, start_temp: float, end_temp: float, total_steps: int):
        self.f = start_temp
        self.l = end_temp
        self.T = total_steps

    def getTemp(self, step: int):
        numerator = self.l * self.f * (self.T - 1)
        denominator = (self.l * self.T - self.f) + (self.f - self.l) * (self.T - step + 1)
        return self.f + self.l - numerator / denominator
    
    def __call__(self, *args, **kwds):
        return self.getTemp(args[0])

        
class LogarithmicScheduler_FirstKind:
    def __init__(self, start_temp: float = 1.1, end_temp: float = 0.1, total_steps: int = 1):
        self.reset(start_temp,end_temp,total_steps)

    def reset(self, start_temp: float, end_temp: float, total_steps: int):
        self.f = start_temp
        self.l = end_temp
        self.T = total_steps
        self.log_T1 = math.log(self.T + 1)
        self.log_2 = math.log(2)
        self.numerator = self.l * self.f * (self.log_T1 - self.log_2)

    def getTemp(self, step: int):
        log_t1 = math.log(step + 1)
        denominator = self.l * self.log_T1 - self.f * self.log_2 + (self.f - self.l) * log_t1
        return self.numerator / denominator
    
    def __call__(self, *args, **kwds):
        return self.getTemp(args[0])
    
class LogarithmicScheduler_SecondKind:
    def __init__(self, start_temp: float = 1.1, end_temp: float = 0.1, total_steps: int = 1):
        self.reset(start_temp,end_temp,total_steps)

    def reset(self, start_temp: float, end_temp: float, total_steps: int):
        self.f = start_temp
        self.l = end_temp
        self.T = total_steps
        self.log_T1 = math.log(self.T + 1)
        self.log_2 = math.log(2)
        self.numerator = self.l * self.f * (self.log_T1 - self.log_2)

    def getTemp(self, step: int):
        log_t1 = math.log(self.T - step + 2)
        denominator = self.l * self.log_T1 - self.f * self.log_2 + (self.f - self.l) * log_t1
        return self.numerator / denominator
    
    def __call__(self, *args, **kwds):
        return self.getTemp(args[0])