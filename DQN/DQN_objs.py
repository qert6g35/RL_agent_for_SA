from collections import namedtuple,deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.current_trace = []

    def push(self, *args):
        """Save a transition"""
        self.current_trace.append(Transition(*args))

    def sample(self, batch_size):
        sample_list = self.current_trace.copy()
        elems_to_pick = [i for i in range(len(self.memory))]
        while len(sample_list) < batch_size:
            id = random.randrange(len(elems_to_pick))
            choice = elems_to_pick.pop(id)
            sample_list.extend(self.memory[choice])
        #print(len(sample_list))
        return sample_list
    
    def finalizeTrace(self):
        #print("current trace size:",len(self.current_trace))
        self.memory.append(self.current_trace)
        self.current_trace = []

    def __len__(self):
        """Returns total number of transitions (not traces)"""
        return len(self.current_trace) + sum(len(inner) for inner in self.memory)