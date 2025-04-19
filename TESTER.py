class TempSheduler:
    """Abstract base class for an time sheduler module."""
    
    ''' Returns temperature for given step

        step: int - number of current step for which we want to get temperature
        
        return:float - temperature for given step
    '''
    def __init__(self):
        pass

    def getTemp(self,step)->float:
        print("getTemp for step",step)
        return step
        
    def __call__(self, *args, **kwds):
        return self.getTemp(args[0])
    

t = TempSheduler()

for i in range(10):
    print("t(i) =",t(i))