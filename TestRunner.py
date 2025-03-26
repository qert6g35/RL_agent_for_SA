import DQN.DQN_Learning as DQN
import Problem
import TempSheduler
import matplotlib.pyplot as plt
import tsplib95
import random
import torch

DQN = DQN.DQN()
DQN.epsilon = 0
action_selction = DQN.select_action(DQN.env.observation())
print(action_selction)

for _ in range(1000):
    DQN.learnNetwork(None,None,None,None)

DQN.epsilon = 0
action_selction = DQN.select_action(DQN.env.observation())
print(action_selction)
# problem = tsplib95.load('VRP_example/a280.tsp')
# distance_matrix = [[problem.get_weight(i, j) for j in problem.get_nodes()] for i in problem.get_nodes()]
# print(distance_matrix)
# problem = Problem.VRP(distances=distance_matrix)

# simulated_anneling_engine = SA.SA(problem)


# max_steps = 500000

# starting_temp = 1000
# ending_temp = 0.001
# end_steps = int(max_steps/2)

# TSM = TempSheduler.LinearTempSheduler(start_temp=starting_temp,end_temp=ending_temp,end_steps=end_steps)

# best_values_list,current_values_list,temp_values_list = simulated_anneling_engine.run(max_steps=max_steps, 
#                               temp_shadeuling_model=TSM,
#                               collect_best_values=True,
#                               collect_current_values=True,
#                               collect_temperature_shadule=True)


# # Creating subplots
# plt.figure(figsize=(12, 10))

# # First plot: Best and Current Values
# plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st plot
# plt.plot(best_values_list, label="Best Values", color="blue")
# plt.plot(current_values_list, label="Current Values", color="orange")
# plt.xlabel("Steps")
# plt.ylabel("Objective Function Value")
# plt.title("Simulated Annealing Progress")
# plt.legend()
# plt.grid()

# # Second plot: Temperature Values
# plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd plot
# plt.plot(temp_values_list, label="Temperature", color="green")
# plt.xlabel("Steps")
# plt.ylabel("Temperature")
# plt.title("Temperature Schedule")
# plt.legend()
# plt.grid()

# # Show the plots
# plt.tight_layout()  # Adjust layout to prevent overlap
# plt.show()