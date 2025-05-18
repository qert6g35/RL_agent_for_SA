from cProfile import label
from glob import glob
from re import S
from turtle import color, st

import test
import Problem
import TempSheduler
from DQN.DQN_Models import DQN_NN_V1,DuelingDQN_NN
from PPO.PPO_Model import PPO_NN,PPO_NN_v2
import SA_ENV as SA_ENV
import Problem
import TempSheduler
import matplotlib.pyplot as plt
import SA
import torch
from collections import Counter
from typing import List
import os
import csv


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

def compareTempSheduler():
    start = 100
    end = 1
    steps = 1000
    # TS: List[TempSheduler.TempSheduler] = [
    #     TempSheduler.LinearTempSheduler(start_temp=start,end_temp=end,end_steps=steps),
    #     TempSheduler.LinearScheduler_FirstKind(start_temp=start,end_temp=end,total_steps=steps),

    #     TempSheduler.ReciprocalTempSheduler(start_temp=start,end_temp=end,end_steps=steps),
    #     TempSheduler.ReciprocalScheduler_FirstKind(start_temp=start,end_temp=end,total_steps=steps),
    #     TempSheduler.ReciprocalScheduler_SecondKind(start_temp=start,end_temp=end,total_steps=steps),

    #     TempSheduler.GeometricTempSheduler(start_temp=start,end_temp=end,end_steps=steps),
    #     TempSheduler.GeometricScheduler_FirstKind(start_temp=start,end_temp=end,total_steps=steps),
    #     TempSheduler.GeometricScheduler_SecondKind(start_temp=start,end_temp=end,total_steps=steps),

    #     TempSheduler.LogarithmicTempSheduler(start_temp=start,end_temp=end,end_steps=steps),
    #     TempSheduler.LogarithmicScheduler_FirstKind(start_temp=start,end_temp=end,total_steps=steps),
    #     TempSheduler.LogarithmicScheduler_SecondKind(start_temp=start,end_temp=end,total_steps=steps),        
    # ]
    # TS_type_number = [2,3,3,3]

    TS: List[TempSheduler.TempSheduler] = [
        #TempSheduler.LinearTempSheduler(start_temp=start,end_temp=end,end_steps=steps),
        TempSheduler.LinearScheduler_FirstKind(start_temp=start,end_temp=end,total_steps=steps),

        #TempSheduler.ReciprocalTempSheduler(start_temp=start,end_temp=end,end_steps=steps),
        TempSheduler.ReciprocalScheduler_FirstKind(start_temp=start,end_temp=end,total_steps=steps),
        TempSheduler.ReciprocalScheduler_SecondKind(start_temp=start,end_temp=end,total_steps=steps),

        #TempSheduler.GeometricTempSheduler(start_temp=start,end_temp=end,end_steps=steps),
        TempSheduler.GeometricScheduler_FirstKind(start_temp=start,end_temp=end,total_steps=steps),
        TempSheduler.GeometricScheduler_SecondKind(start_temp=start,end_temp=end,total_steps=steps),

        #TempSheduler.LogarithmicTempSheduler(start_temp=start,end_temp=end,end_steps=steps),
        TempSheduler.LogarithmicScheduler_FirstKind(start_temp=start,end_temp=end,total_steps=steps),
        TempSheduler.LogarithmicScheduler_SecondKind(start_temp=start,end_temp=end,total_steps=steps),        
    ]
    TS_type_number = [1,2,2,2]

    data = [ [] for _ in TS]
    steps_lsit = []

    for step in range(1,steps+1):
        steps_lsit.append(step)
        for i in range(len(TS)):
            data[i].append(TS[i].getTemp(step=step))
    
    # Tworzenie subplotów 2x2
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Porównanie różnych harmonogramów temperatury", fontsize=14)

    id_0 = 0
    id_1 = 0
    global_done = 0
    plot_color = ["orange","green","blue"]
    plot_title = ["Linear","Reciprocal","Geometric","Logarithmic"]
    labels = ["first","second"]
    for num_of_data in TS_type_number:
        axs[id_0, id_1].set_title(plot_title.pop(0))
        for id in range(num_of_data):
            axs[id_0, id_1].plot(steps_lsit, data[id + global_done],label = labels[id],color = plot_color[id])

        global_done += num_of_data
        if id_0 == 0:
            id_0 = 1
        else:
            id_0 = 0
            id_1 = 1
            
    for ax in axs.flat:
        ax.set(xlabel='Krok', ylabel='Temperatura')
        ax.grid(True)
    plt.legend()
    plt.show()

def make_ploting_test():
    env7 = SA_ENV.SA_env(use_observation_divs=True,use_time_temp_info=False)
    env10 = SA_ENV.SA_env()
    # DQN_model = DuelingDQN_NN()
    print(env7.observation_space.shape[0])
    print(env10.observation_space.shape[0])
    # DQN_nn_999 = DuelingDQN_NN(len(DQN_SA_engine.observation()),DQN_SA_engine.action_space.n)
    # DQN_nn_999.load_state_dict(torch.load('NN_Models/DQN/DuelingDQN/E/SMART_TSP/DQN_NN_2025_04_17_01_04_eps999'))

    # DQN_nn_2999 = DuelingDQN_NN(len(DQN_SA_engine.observation()),DQN_SA_engine.action_space.n)
    # DQN_nn_2999.load_state_dict(torch.load('NN_Models/DQN/DuelingDQN/E/SMART_TSP/DQN_NN_2025_04_19_08_34_eps2999'))

    # PPO_nn_31206 = PPO_NN( None,len(DQN_SA_engine.observation()),DQN_SA_engine.action_space.n,layer_size=64)
    # PPO_nn_31206.load_state_dict(torch.load('NN_Models/PPO/A/Smart_TSP/1/PPO_2025_04_16_20_35_updates31206'))

    # PPO_nn_115352 = PPO_NN( None,len(DQN_SA_engine.observation()),DQN_SA_engine.action_space.n,layer_size=64)
    # PPO_nn_115352.load_state_dict(torch.load('NN_Models/PPO/A/Smart_TSP/2/PPO_2025_04_17_22_19_updates115352'))

    # PPO_nn_6563 = PPO_NN( None,len(DQN_SA_engine.observation()),DQN_SA_engine.action_space.n)
    # PPO_nn_6563.load_state_dict(torch.load('PPO_2025_04_22_21_52_updates6563'))

    NN_TS = [
        #("PPO_F1_25k",PPO_NN_v2( None,env10.observation_space.shape[0],env10.action_space.n),SA_ENV.SA_env()),
        #("PPO_F2_25k",PPO_NN_v2( None,env10.observation_space.shape[0],env10.action_space.n),SA_ENV.SA_env()),
        #("PPO_F2_50k",PPO_NN_v2( None,env10.observation_space.shape[0],env10.action_space.n),SA_ENV.SA_env()),
        #("PPO_F2_75k",PPO_NN_v2( None,env10.observation_space.shape[0],env10.action_space.n),SA_ENV.SA_env()),
        #("PPO_F2_100k",PPO_NN_v2( None,env10.observation_space.shape[0],env10.action_space.n),SA_ENV.SA_env()),
        # ("PPO_F2_83k",PPO_NN_v2( None,env10.observation_space.shape[0],env10.action_space.n),SA_ENV.SA_env()),
        # ("PPO_F2_166k",PPO_NN_v2( None,env10.observation_space.shape[0],env10.action_space.n),SA_ENV.SA_env()),
        ("PPO_G2_65k",PPO_NN_v2( None,env10.observation_space.shape[0],env10.action_space.n),SA_ENV.SA_env()),
        ("PPO_G2_130k",PPO_NN_v2( None,env10.observation_space.shape[0],env10.action_space.n),SA_ENV.SA_env()),
    ]

    NN_TS[0][1].load_state_dict(torch.load('NN_Models/PPO/G2/PPO_2025_05_13_00_22_updates65050'))
    NN_TS[1][1].load_state_dict(torch.load('NN_Models/PPO/G2/PPO_2025_05_13_00_22_updates130200'))
    # NN_TS[2][1].load_state_dict(torch.load('PPO_2025_05_06_22_21_updates249980'))
    # NN_TS[3][1].load_state_dict(torch.load('PPO_2025_05_06_22_21_updates255480'))
    #NN_TS[4][1].load_state_dict(torch.load('NN_Models/PPO/F/2/PPO_2025_05_05_08_04_updates104160'))


    new_problem = Problem.TSP()
    initial_solution = new_problem.get_initial_solution()

    for nn_tuple in NN_TS:
        nn_tuple[2].reset(preset_problem=new_problem,initial_solution=initial_solution)
        print("starting temp:",nn_tuple[2].starting_temp)
        print("min temp:",nn_tuple[2].min_temp)

    run_Best = {}
    run_Current = {}
    run_Temp = {}

    for tuple in NN_TS:
        print("RUNNING TEST",tuple[0])
        run_Best[tuple[0]],run_Current[tuple[0]],run_Temp[tuple[0]] = tuple[2].runTest(model=tuple[1],generate_plot_data=True)

    for key in run_Best:
        plt.ion()
        # Tworenie figury i trzech subplotów
        fig, axs = plt.subplots(3, 1, figsize=(8, 10))

        # Wykresy
        print(key,"score:",run_Best[key][-1])
        axs[0].plot(run_Best[key], color='blue')
        axs[0].set_title(key)

        axs[1].plot(run_Current[key], color='green')


        axs[2].plot(run_Temp[key], color='red')


        # Dostosowanie wyglądu
        plt.tight_layout()
        plt.show()


    # for tuple in TS:
    #     run_results[tuple[0]] = tuple[2].run(
    #         max_steps=DQN_SA_engine.max_steps, 
    #         temp_shadeuling_model=tuple[1])

def make_compareing_test(NUM_TESTS):
    test_result = {}

    def save_flat_sa_results_to_csv(file_path="sa_results.csv"):
        nonlocal test_result
        file_exists = os.path.isfile(file_path)

        with open(file_path, mode="a", newline='', encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            
            if not file_exists:
                writer.writerow(["dim", "ts_name", "best_value", "steps_to_best_value"])
            
            for problem_dim, method_results in test_result.items():
                for method_name, best_result, steps_to_best in method_results:
                    writer.writerow([problem_dim, method_name, best_result, steps_to_best])

        test_result = {}
        print(f"✔️ Dane dopisane do pliku: {file_path}")

    def collect_run_result(run_result: dict, problem_dimention: int):
        '''
        For a given problem dimension, collect:
        0 - name of the temperature schedule
        1 - final best result
        2 - number of iterations needed to reach that result
        '''
        refactored_result = []
        
        for key in run_result:
            last_value = run_result[key][-1]
            all_values = run_result[key]
            count_best = Counter(all_values)[last_value]
            iterations_to_best = len(all_values) - count_best

            refactored_result.append([
                key,
                last_value,
                iterations_to_best
            ])
        
        if problem_dimention not in test_result:
            test_result[problem_dimention] = []

        test_result[problem_dimention] += (refactored_result)

    DQN_SA_engine = SA_ENV.SA_env(set_up_learning_on_init=True)
    # DQN_model = DuelingDQN_NN()

    NN_TS = [
        ("PPO_G2_130k",PPO_NN_v2( None,len(DQN_SA_engine.observation()),DQN_SA_engine.action_space.n),SA_ENV.SA_env())
    ]

    NN_TS[0][1].load_state_dict(torch.load('NN_Models/PPO/G2/PPO_2025_05_13_00_22_updates130200'))

    TS: List[TempSheduler.TempSheduler] = [
        #TempSheduler.LinearTempSheduler(start_temp=start,end_temp=end,end_steps=steps),
        ("Linear",TempSheduler.LinearScheduler_FirstKind(),SA.SA(skip_initialization=True)),

        #TempSheduler.ReciprocalTempSheduler(start_temp=start,end_temp=end,end_steps=steps),
        ("ReciprocalV1",TempSheduler.ReciprocalScheduler_FirstKind(),SA.SA(skip_initialization=True)),
        ("ReciprocalV2",TempSheduler.ReciprocalScheduler_SecondKind(),SA.SA(skip_initialization=True)),

        #TempSheduler.GeometricTempSheduler(start_temp=start,end_temp=end,end_steps=steps),
        ("GeometricV1",TempSheduler.GeometricScheduler_FirstKind(),SA.SA(skip_initialization=True)),
        ("GeometricV2",TempSheduler.GeometricScheduler_SecondKind(),SA.SA(skip_initialization=True)),

        #TempSheduler.LogarithmicTempSheduler(start_temp=start,end_temp=end,end_steps=steps),
        ("LogarithmicV1",TempSheduler.LogarithmicScheduler_FirstKind(),SA.SA(skip_initialization=True)),
        ("LogarithmicV2",TempSheduler.LogarithmicScheduler_SecondKind(),SA.SA(skip_initialization=True)),     
    ]


    for i in range(NUM_TESTS):
        new_problem = Problem.TSP()
        initial_solution = new_problem.get_initial_solution()

        # DQN_SA_engine.reset(preset_problem=new_problem,initial_solution=initial_solution)
        # LinerSA.reset(preset_problem=new_problem,initial_solution=initial_solution)

        for nn_tuple in NN_TS:
            nn_tuple[2].reset(preset_problem=new_problem,initial_solution=initial_solution)

        t_max = NN_TS[0][2].starting_temp
        t_min = NN_TS[0][2].min_temp
        # LinearTS.reset(DQN_SA_engine.starting_temp,DQN_SA_engine.min_temp,DQN_SA_engine.max_steps)

        for tuple in TS:
            tuple[1].reset(t_max,t_min,DQN_SA_engine.max_steps)
            tuple[2].reset(preset_problem=new_problem,initial_solution=initial_solution)



        run_results = {}

        for tuple in NN_TS:
            run_results[tuple[0]] = tuple[2].runTest(model=tuple[1])
            
        for tuple in TS:
            run_results[tuple[0]] = tuple[2].run(
                max_steps=DQN_SA_engine.max_steps, 
                temp_shadeuling_model=tuple[1])
        
        collect_run_result(run_results,new_problem.getDimention())
        if i % 1 == 0:
            save_flat_sa_results_to_csv()


# h = "y"            
# while(h == "y"):
#     make_ploting_test()
#     h = input("continue?:")



#make_compareing_test(10000)


compareTempSheduler()




# ✅ 1. Jakość uzyskanego rozwiązania (wartość funkcji celu)
# Podstawowe kryterium.

# Sprawdzasz, jakie najlepsze rozwiązanie udało się osiągnąć przy danym schemacie chłodzenia.

# Im niższa wartość funkcji celu (dla problemów minimalizacji), tym lepiej.

# ✅ 2. Stabilność wyników (odchylenie standardowe / wariancja)
# Dobrze, jeśli dany schemat regularnie daje dobre wyniki, nie tylko "raz się udało".

# Powtarzasz eksperymenty wielokrotnie (np. 30 razy) i analizujesz rozrzut wyników.

# Schemat o niższej wariancji jest bardziej niezawodny.

# ✅ 3. Czas obliczeń (czas działania algorytmu)
# Czy dany schemat osiąga dobre wyniki szybko?

# Warto analizować czas do osiągnięcia najlepszego wyniku lub całkowity czas działania.

# W niektórych zastosowaniach czas ma większe znaczenie niż optymalność.

# ✅ 4. Liczba iteracji do najlepszego rozwiązania
# Ile kroków było potrzebne, żeby znaleźć najlepsze rozwiązanie?

# Schemat, który szybciej dochodzi do optimum, może być lepszy w praktycznych zastosowaniach.


