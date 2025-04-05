import Problem
import TempSheduler
from DQN.DQN_Models import DQN_NN_V1
import DQN.DQN_SA as DQN_SA
import Problem
import TempSheduler
import matplotlib.pyplot as plt
import SA
import torch

def plot_SA_output(best_values_list,current_values_list,temp_values_list):
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



def TestGivenTempSheduler(TSM):
    problem = Problem.TSP()

    print(problem.getUpperBound())

    simulated_anneling_engine = SA.SA(problem)

    max_steps = 200000

    #starting_temp = 100000
    #ending_temp = 0.001
    #end_steps = int(max_steps/2)

    #TSM = TempSheduler.ConstTempSheduler(temp=starting_temp)
    #TSM = TempSheduler.LinearTempSheduler(start_temp=starting_temp,end_temp=ending_temp,end_steps=end_steps)

    best_values_list,current_values_list,temp_values_list = simulated_anneling_engine.run(max_steps=max_steps, 
                                temp_shadeuling_model=TSM,
                                collect_best_values=True,
                                collect_current_values=True,
                                collect_temperature_shadule=True)
    
    return best_values_list,current_values_list,temp_values_list



DQN_SA_engine = DQN_SA.SA_env()

DQN_nn = DQN_NN_V1(DQN_SA_engine.observation_space,DQN_SA_engine.action_space)
DQN_nn.load_state_dict(torch.load('NN_Models\V1\B\DQN_policy_model_DQN_V1_B_ok_300_500_eps'))
print("start runTest")
best_values_list,current_values_list,temp_values_list = DQN_SA_engine.runTest(model=DQN_nn)
print("finalised test")

plot_SA_output(best_values_list,current_values_list,temp_values_list)


