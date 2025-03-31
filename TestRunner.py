from DQN.DQN_Learning import DQN
import Problem
import TempSheduler
import matplotlib.pyplot as plt
import tsplib95
import random
import torch
import SA

DQN_eng = DQN()

DQN_eng.run(100)


#Learning episode 42/100
# Traceback (most recent call last):
#   File "c:\Users\piotr\OneDrive\Pulpit\MAGISTERKA\RL_agent_for_SA\TestRunner.py", line 12, in <module>
#     DQN_eng.run(100)
#   File "c:\Users\piotr\OneDrive\Pulpit\MAGISTERKA\RL_agent_for_SA\DQN\DQN_Learning.py", line 85, in run
#     state = self.env.reset()
#             ^^^^^^^^^^^^^^^^
#   File "c:\Users\piotr\OneDrive\Pulpit\MAGISTERKA\RL_agent_for_SA\DQN\DQN_SA.py", line 64, in reset
#     else:

#   File "c:\Users\piotr\OneDrive\Pulpit\MAGISTERKA\RL_agent_for_SA\SA.py", line 14, in __init__
#     self.problem = Problem.VRP()
#                    ^^^^^^^^^^^^^
#   File "c:\Users\piotr\OneDrive\Pulpit\MAGISTERKA\RL_agent_for_SA\Problem.py", line 36, in __init__
#     self.distances = [[problem.get_weight(i, j) for j in problem.get_nodes()] for i in problem.get_nodes()]
#                        ^^^^^^^^^^^^^^^^^^^^^^^^
# MemoryError
# PS C:\Users\piotr\OneDrive\Pulpit\MAGISTERKA\RL_agent_for_SA> 
#

# def CHECK_IF_NETWORK_IS_LEARNING_GIVEN_DATA():
#     DQN_engine = DQN()
#     DQN_engine.epsilon = 0
#     action_selction = DQN_engine.policy_net(torch.Tensor(DQN_engine.env.observation()))
#     print(action_selction)

#     for i in range(1000):
#         DQN_engine.FORCE_learnNetwork(None,None,None,None)
#         if i%100 == 0:
#             action_selction = DQN_engine.policy_net(torch.Tensor(DQN_engine.env.observation()))
#             print(action_selction)

#     DQN_engine.epsilon = 0
#     action_selction = DQN_engine.policy_net(torch.Tensor(DQN_engine.env.observation()))
#     print(action_selction)

def show_SA_with_VRP():
    problem = Problem.VRP()

    print(problem.getUpperBound())

    simulated_anneling_engine = SA.SA(problem)

    max_steps = 200000

    starting_temp = 100000
    ending_temp = 0.001
    end_steps = int(max_steps/2)

    TSM = TempSheduler.ConstTempSheduler(temp=starting_temp)
    #TSM = TempSheduler.LinearTempSheduler(start_temp=starting_temp,end_temp=ending_temp,end_steps=end_steps)

    best_values_list,current_values_list,temp_values_list = simulated_anneling_engine.run(max_steps=max_steps, 
                                temp_shadeuling_model=TSM,
                                collect_best_values=True,
                                collect_current_values=True,
                                collect_temperature_shadule=True)

    # Creating subplots
    plt.figure(figsize=(12, 10))

    # First plot: Best and Current Values
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st plot
    plt.plot(best_values_list, label="Best Values", color="blue")
    plt.plot(current_values_list, label="Current Values", color="orange")
    plt.xlabel("Steps")
    plt.ylabel("Objective Function Value")
    plt.title("Simulated Annealing Progress")
    plt.legend()
    plt.grid()

    # Second plot: Temperature Values
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd plot
    plt.plot(temp_values_list, label="Temperature", color="green")
    plt.xlabel("Steps")
    plt.ylabel("Temperature")
    plt.title("Temperature Schedule")
    plt.legend()
    plt.grid()

    # Show the plots
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()