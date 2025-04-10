from abc import ABC, abstractmethod

class TempSheduler(ABC):
    """Abstract base class for an time sheduler module."""
    
    ''' Returns temperature for given step

        step: int - number of current step for which we want to get temperature
        
        return:float - temperature for given step
    '''
    @abstractmethod 
    def getTemp(self,*args)->float:
        pass

class LinearTempSheduler(TempSheduler):
    ''' Linear temperature sheduler
    
        start_temp:float - starting temperature
        end_temp:float - ending temperature
        steps:int - number of steps to reach end temperature
    '''

    def __init__(self,start_temp:float,end_temp:float,end_steps:int):
        self.reset(start_temp = start_temp,end_steps=end_steps,end_temp=end_temp)

    def getTemp(self,*args):# tutaj podajemy tylko jeden agrument, który to jekt krok aglorytmu
        if args[0] >= self.end_steps:
            return self.end_temp
        return self.start_temp - self.temp_diff * args[0]
    
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