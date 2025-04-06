import Problem
import TempSheduler
from DQN.DQN_Models import DQN_NN_V1,DuelingDQN_NN
import SA_ENV as SA_ENV
import Problem
import TempSheduler
import matplotlib.pyplot as plt
import SA
import torch

def plot_SA_output(lbest_values_list,lcurrent_values_list,ltemp_values_list,dbest_values_list,dcurrent_values_list,dtemp_values_list):
        # Creating subplots
    plt.figure(figsize=(12, 10))

    # First plot: Best and Current Values
    plt.subplot(2, 2, 1)  # 2 rows, 1 column, 1st plot
    plt.plot(lbest_values_list, label="Linear Best Values", color="blue")
    plt.plot(lcurrent_values_list, label="Linear Current Values", color="orange")
    plt.xlabel("Steps")
    plt.ylabel("Objective Function Value")
    plt.title("Simulated Annealing Progress")
    plt.legend()
    plt.grid()

    # Second plot: Temperature Values
    plt.subplot(2, 2, 2)  # 2 rows, 1 column, 2nd plot
    plt.plot(ltemp_values_list, label="Linear Temperature", color="green")
    plt.xlabel("Steps")
    plt.ylabel("Temperature")
    plt.title("Temperature Schedule")
    plt.legend()
    plt.grid()

        # First plot: Best and Current Values
    plt.subplot(2, 2, 3)  # 2 rows, 1 column, 1st plot
    plt.plot(dbest_values_list, label="DQN Best Values", color="blue")
    plt.plot(dcurrent_values_list, label="DQN Current Values", color="orange")
    plt.xlabel("Steps")
    plt.ylabel("Objective Function Value")
    plt.title("Simulated Annealing Progress")
    plt.legend()
    plt.grid()

    # Second plot: Temperature Values
    plt.subplot(2, 2, 4)  # 2 rows, 1 column, 2nd plot
    plt.plot(dtemp_values_list, label="DQN Temperature", color="green")
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

NUM_TESTS = 2



DQN_SA_engine = SA_ENV.SA_env()
# DQN_model = DuelingDQN_NN()

DQN_nn = DQN_NN_V1(len(DQN_SA_engine.observation()),DQN_SA_engine.action_space.n)
DQN_nn.load_state_dict(torch.load('NN_Models/DQN/V1/B/ekplodujące_zanikające_wartości/DQN_policy_model_DQN_V1_B_ok_300_500_eps'))

LinerSA = SA.SA()

for i in range(NUM_TESTS):
    new_problem = Problem.TSP()
    initial_solution = new_problem.get_initial_solution()

    DQN_SA_engine.reset(preset_problem=new_problem,initial_solution=initial_solution)
    LinerSA.reset(preset_problem=new_problem,initial_solution=initial_solution)

    LinearTS = TempSheduler.LinearTempSheduler(DQN_SA_engine.starting_temp,DQN_SA_engine.min_temp,DQN_SA_engine.max_steps)

    linear_BV,linear_CV,linear_TV = LinerSA.run(max_steps=DQN_SA_engine.max_steps, 
                                temp_shadeuling_model=LinearTS,
                                collect_best_values=True,
                                collect_current_values=True,
                                collect_temperature_shadule=True)
    
    DQN_BV,DQN_CV,DQN_TV = DQN_SA_engine.runTest(model=DQN_nn)

    plot_SA_output(linear_BV,linear_CV,linear_TV,DQN_BV,DQN_CV,DQN_TV)

# DQN_nn = DQN_NN_V1(DQN_SA_engine.observation_space,DQN_SA_engine.action_space)
# DQN_nn.load_state_dict(torch.load('NN_Models/V1/B/DQN_policy_model_DQN_V1_B_ok_300_500_eps'))
# print("start runTest")
# best_values_list,current_values_list,temp_values_list = DQN_SA_engine.runTest(model=DQN_nn)
# print("finalised test")

# plot_SA_output(best_values_list,current_values_list,temp_values_list)


