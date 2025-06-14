import matplotlib.pyplot as plt  # Import matplotlib for plotting
import numpy as np
import pandas as pd

from Problem import TSP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob
import os
from cProfile import label
from glob import glob
from pickle import TRUE
from re import S
from turtle import color, st
from numpy import empty
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
import csv
import math
import threading
import os
import psutil
import multiprocessing
import time

# problem = TSP(use_lib_instances=True)

# rozw = [
#     1, 47, 93, 28, 67, 58, 61, 51, 87, 25, 81, 69, 64, 40, 54, 2, 44, 50, 73, 68,
#     85, 82, 95, 13, 76, 33, 37, 5, 52, 78, 96, 39, 30, 48, 100, 41, 71, 14, 3, 43,
#     46, 29, 34, 83, 55, 7, 9, 57, 20, 12, 27, 86, 35, 62, 60, 77, 23, 98, 91, 45,
#     32, 11, 15, 17, 59, 74, 21, 72, 10, 84, 36, 99, 38, 24, 18, 79, 53, 88, 16, 94,
#     22, 70, 66, 26, 65, 4, 97, 56, 80, 31, 89, 42, 8, 92, 75, 19, 90, 49, 6, 63
# ]
# print(problem.objective_function([x - 1 for x in rozw]))


#   READ ME
#
# z racji na to że zapomniaem uwzgldnić to, że SA wykonuje 10 kroków dla PPO_G2
#
# wartości steps to best value, muszą zostać pomnożone przez 10 (drobna utrata danych ale nie większa niż 9), 
#
# dlatego aby nie zaniżać wsp.  znalezienia lepszego wyniku (który chcemy aby by maly) doamy tez do kazdego wyniku 5

my_list = [
    "E d",
    "E s",
    "F d",
    "F s",
    "G d",
    "G s",
    "G2 d",
    "G2 s",
    "G3_1 d",
    "G3_1 s",
    "G4_1 d",
    "G4_1 s",
    "G6_V0_1 d",
    "G6_V0_1 s",
    "G6_V1_3 d",
    "G6_V1_3 s",
]

# Wczytaj dane
folder_path = "./"#fixed_approach"#Tests/FinalTests/"  # lub np. "./dane/"
pattern = os.path.join(folder_path, "sa_results_final_Fix*.csv")

# 3. Wczytywanie i łączenie danych
all_files = glob.glob(pattern)
df_list = []

for file in all_files:
    temp_df = pd.read_csv(file)
    #temp_df['source_file'] = os.path.basename(file)  # opcjonalnie: dodaj nazwę pliku jako kolumnę
    df_list.append(temp_df)

# 4. Łączenie w jeden DataFrame
df = pd.concat(df_list, ignore_index=True)

# Filtruj tylko wiersze z nazwą zawierającą "PPO_G2_130k"
mask = df["ts_name"].isin(my_list)

# Pomnóż wartość w kolumnie 'stbv' przez np. 2 (możesz zmienić na dowolną wartość)
df.loc[mask, "steps_to_best_value"] = df.loc[mask, "steps_to_best_value"] * 10 + 5

# Opcjonalnie: zapisz wynik do nowego pliku
df.to_csv("first_approach.csv", index=False)

# Podgląd zmodyfikowanych danych
print(df[mask])


# def estimate_sa_steps1(n):
#     if n <= 100:
#         alpha = 15.0
#         min_steps = 1500
#     elif n <= 200:
#         alpha = 11.0
#         min_steps = estimate_sa_steps(100)
#         print(min_steps)
#     elif n <= 500:
#         alpha = 8.0
#         min_steps = estimate_sa_steps(200)
#     else:
#         # Zakładamy, że funkcja ma działać do 500, jak w oryginale
#         return 100000  # wartość ograniczona przez 1e5

#     return min(max(int(alpha * (n ** 1.59))/10, min_steps), int(1e5))

# def estimate_sa_steps(n):
#     if n <= 100:
#         alpha = 15.0
#         min_steps = 15000
#     elif n <= 200:
#         alpha = 11.0
#         min_steps = estimate_sa_steps(100)
#     elif n <= 500:
#         alpha = 8.0
#         min_steps = estimate_sa_steps(200)
#     else:
#         # Zakładamy, że funkcja ma działać do 500, jak w oryginale
#         return 100000  # wartość ograniczona przez 1e5

#     return min(max(int(alpha * (n ** 1.59)), min_steps), int(1e5))
# #print(estimate_sa_steps(109))
# # Zakres wartości n do wykresu
# n_values = np.arange(50, 301)
# step_values = [estimate_sa_steps(n)//10 for n in n_values]
# #step_values2 = [estimate_sa_steps2(n) for n in n_values]

# # Rysowanie wykresu
# plt.figure(figsize=(10, 6))
# plt.plot(n_values, step_values)
# #plt.plot(n_values, step_values2, label="estimate_sa_steps(n)",color="green")
# plt.xlabel("n (rozmiar problemu TSP)")
# plt.ylabel("Liczba kroków (max steps)")
# plt.title("Wartości maksymalnej liczby kroków w zależności od rozmiaru problemu TSP")
# plt.grid(True)
# plt.legend()
# plt.show()


def make_ploting_test():
    env7 = SA_ENV.SA_env(use_observation_divs=True,use_time_temp_info=False)
    DQN_SA_engine = SA_ENV.SA_env()
    # DQN_model = DuelingDQN_NN()
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
       ("E deterministyczna",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],11),SA_ENV.SA_env(use_new_lower_actions=True),True),
       ("E stochastyczna",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],11),SA_ENV.SA_env(use_new_lower_actions=True),False),
       ("F deterministyczna",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],11),SA_ENV.SA_env(use_new_lower_actions=True),False),
       ("F stochastyczna",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],11),SA_ENV.SA_env(use_new_lower_actions=True),False),
       ("G deterministyczna",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],11),SA_ENV.SA_env(use_new_lower_actions=True),True),
       ("G stochastyczna",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],11),SA_ENV.SA_env(use_new_lower_actions=True),False),
       ("G2 deterministyczna",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],11),SA_ENV.SA_env(use_new_lower_actions=True),True),
       ("G2 stochastyczna",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],11),SA_ENV.SA_env(use_new_lower_actions=True),False),
        #("G3_1 s",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],11),SA_ENV.SA_env(use_new_lower_actions=True),False),
        #("G3_2 s",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],11),SA_ENV.SA_env(use_new_lower_actions=True),False),
        #("G3_3 s",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],11),SA_ENV.SA_env(use_new_lower_actions=True),False),
        # #("G3_1 s",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],11),SA_ENV.SA_env(use_new_lower_actions=True),False),
        ("H1 deterministyczna",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],11),SA_ENV.SA_env(use_new_lower_actions=True),True),
        # #("G3_2 s",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],11),SA_ENV.SA_env(use_new_lower_actions=True),False),
        # #("G3_3 d",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],11),SA_ENV.SA_env(use_new_lower_actions=True),True),
        ("H1 stochastyczna",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],11),SA_ENV.SA_env(use_new_lower_actions=True),False),
        #("G4_1",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),False),
        #("G4_2",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),False),
        #("G4_3",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),False),
        ("H2 deterministyczna",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),True),
        ("H2 stochastyczna",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),False),
        ("H3 deterministyczna 1",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),True),
        ("H3 stochastyczna 1",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),False),
        ("H3 deterministyczna 2",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),True),
        ("H3 stochastyczna 2",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),False),
        ("H3 deterministyczna 3",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),True),
        ("H3 stochastyczna 3",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),False),
        ("H3 deterministyczna 4",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),True),
        ("H3 stochastyczna 4",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),False),
        ("H3 deterministyczna 5",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),True),
        ("H3 stochastyczna 5",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),False),
        ("H3 deterministyczna",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),True),
        ("H3 stochastyczna",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),False),
        #("H3 deterministyczna5",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),True),
        #("H3 stochastyczna5",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),False),
        #("G6_V1_1",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),False),
        #("G6_V1_2",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),False),
        
        #("G6_V0_1",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),False),
        #("G6_V0_2",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),False),
        #("G6_V0_3",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env(),False),
        #("G6_3",PPO_NN_v2( None,DQN_SA_engine.observation_space.shape[0],DQN_SA_engine.action_space.n),SA_ENV.SA_env()),
    ]

    NN_load_paths = [
        # ["NN_Models/PPO/E_10SPT/PPO_2025_05_03_19_53_updates24392",5,25],
        # ["NN_Models/PPO/E_10SPT/PPO_2025_05_03_19_53_updates24392",5,25],
        # ["NN_Models/PPO/F/2/PPO_2025_05_05_08_04_updates104160",10,50],
        # ["NN_Models/PPO/F/2/PPO_2025_05_05_08_04_updates104160",10,50],
        # ["NN_Models/PPO/G/PPO_2025_05_06_22_21_updates255480",5,25],
        # ["NN_Models/PPO/G/PPO_2025_05_06_22_21_updates255480",5,25],
        # ["NN_Models/PPO/G2/PPO_2025_05_13_00_22_updates130200_2",5,25],
        # ["NN_Models/PPO/G2/PPO_2025_05_13_00_22_updates130200_2",5,25],
        ["NN_Models/PPO/G3/PPO_2025_05_18_23_00_Best1",8,40],
        ["NN_Models/PPO/G3/PPO_2025_05_18_23_00_Best1",8,40],

        ["NN_Models/PPO/G3/PPO_2025_05_18_23_00_Best2",8,40],
        ["NN_Models/PPO/G3/PPO_2025_05_18_23_00_Best2",8,40],
        ["NN_Models/PPO/G3/PPO_2025_05_18_23_00_Best3",8,40],
        ["NN_Models/PPO/G3/PPO_2025_05_18_23_00_Best3",8,40],
        ["NN_Models/PPO/G4/PPO_2025_05_23_21_43_value3.961440821419651_envs_updated60165",8,40],
        ["NN_Models/PPO/G4/PPO_2025_05_23_21_43_value3.961440821419651_envs_updated60165",8,40],
        ["NN_Models/PPO/G4/PPO_2025_05_23_21_43_value3.774389414221403_envs_updated6565",8,40],
        ["NN_Models/PPO/G4/PPO_2025_05_23_21_43_value3.774389414221403_envs_updated6565",8,40],
        ["NN_Models/PPO/G4/PPO_2025_05_23_21_43_value3.609286070321052_envs_updated76410",8,40],
        ["NN_Models/PPO/G4/PPO_2025_05_23_21_43_value3.609286070321052_envs_updated76410",8,40],
        # ["NN_Models/PPO/G6/PPO_2025_05_27_20_44_Best1",8,40],
        # ["NN_Models/PPO/G6/PPO_2025_05_27_20_44_Best1",8,40],
        # ["NN_Models/PPO/G6/PPO_2025_05_27_20_44_Best2",8,40],
        # ["NN_Models/PPO/G6/PPO_2025_05_27_20_44_Best2",8,40],
        # ["NN_Models/PPO/G6/PPO_2025_05_27_20_44_Best3",8,40],
        # ["NN_Models/PPO/G6/PPO_2025_05_27_20_44_Best3",8,40],
        # ["NN_Models/PPO/G6_with_offset_onFiew_first_steps/PPO_2025_05_27_20_44_Best1_PRE_TREND_ADJUSTMENT",8,40],
        # ["NN_Models/PPO/G6_with_offset_onFiew_first_steps/PPO_2025_05_27_20_44_Best1_PRE_TREND_ADJUSTMENT",8,40],
        # ["NN_Models/PPO/G6_with_offset_onFiew_first_steps/PPO_2025_05_27_20_44_Best2_PRE_TREND_ADJUSTMENT",8,40],
        # ["NN_Models/PPO/G6_with_offset_onFiew_first_steps/PPO_2025_05_27_20_44_Best2_PRE_TREND_ADJUSTMENT",8,40],
        ["NN_Models/PPO/G6_with_offset_onFiew_first_steps/PPO_2025_05_27_20_44_Best3_PRE_TREND_ADJUSTMENT",8,40],
        ["NN_Models/PPO/G6_with_offset_onFiew_first_steps/PPO_2025_05_27_20_44_Best3_PRE_TREND_ADJUSTMENT",8,40],
        #["NN_Models/PPO/G6/PPO_2025_05_27_20_44_Best2",8,40],
        #["NN_Models/PPO/G6/PPO_2025_05_27_20_44_Best2",8,40],
        #["NN_Models/PPO/G6/PPO_2025_05_27_20_44_Best3",8,40],
        #["NN_Models/PPO/G6_with_offset_onFiew_first_steps/PPO_2025_05_27_20_44_Best1_V3",8,40],
        #"PPO_2025_05_23_21_43_value3.609286070321052_envs_updated76410",
        #"NN_Models/PPO/G2/PPO_2025_05_13_00_22_updates130200_2",
        #"NN_Models/PPO/G3/PPO_2025_05_18_23_00_Best1",
    ]

    for path_id in range(len(NN_load_paths)):
        print("loading:",NN_load_paths[path_id][0])
        NN_TS[path_id][1].load_state_dict(torch.load(NN_load_paths[path_id][0]))
        NN_TS[0][2].temp_history_size = NN_load_paths[path_id][2]
        NN_TS[0][2].temp_short_size = NN_load_paths[path_id][1]
    #NN_TS[0][1].load_state_dict(torch.load('NN_Models/PPO/G2/PPO_2025_05_13_00_22_updates65050'))
    #NN_TS[1][1].load_state_dict(torch.load('NN_Models/PPO/G2/PPO_2025_05_13_00_22_updates130200'))
    # NN_TS[2][1].load_state_dict(torch.load('PPO_2025_05_06_22_21_updates249980'))
    # NN_TS[3][1].load_state_dict(torch.load('PPO_2025_05_06_22_21_updates255480'))
    #NN_TS[4][1].load_state_dict(torch.load('NN_Models/PPO/F/2/PPO_2025_05_05_08_04_updates104160'))


    new_problem = Problem.TSP(generation_dim=150,generate_harder_instance=True,use_lib_instances=True)
    initial_solution = new_problem.get_initial_solution()

    for nn_tuple in NN_TS:
        nn_tuple[2].reset(preset_problem=new_problem,initial_solution=initial_solution,use_lower_maxsteps=False,use_harder_TSP=True)
        print("starting temp:",nn_tuple[2].starting_temp)
        print("min temp:",nn_tuple[2].min_temp)

    run_Best = {}
    run_Current = {}
    run_Temp = {}
    run_percentage = {}

    for tuple in NN_TS:
        print("RUNNING TEST",tuple[0])
        run_Best[tuple[0]],run_Current[tuple[0]],run_Temp[tuple[0]],run_percentage[tuple[0]] = tuple[2].runTest(model=tuple[1],generate_plot_data=True,use_deterministic_actions=tuple[3])



    for key in run_Best:
        plt.ion()
        # Tworenie figury i trzech subplotów
        fig, axs = plt.subplots(3, 1, figsize=(8, 10))

        # Wykresy
        #print(key,"score:",run_Best[key][-1])
        axs[0].plot(run_Best[key], color='blue', label='Best score')
        axs[0].legend()
        axs[0].set_title("Przykładowe działanie agenta, wersja " + key)

        axs[1].plot(run_Current[key], color='green', label='Current score')
        axs[1].legend()

        #axs[2].plot(run_percentage[key], color='red', label='Temp score')
        axs[2].plot(run_Temp[key], color='black', label='Temp score')
        if key[0] == 'F':
            y_top = 200
        else:
            y_top = 100
        axs[2].set_ylim(-1, y_top + 1)
        axs[2].set_xlim(-50, len(run_Current[key]) + 50)
        axs[2].legend()
        axs[2].plot(0,0,color = 'white')

        # Dostosowanie wyglądu
        plt.tight_layout()
        plt.show()
    input("")

make_ploting_test()