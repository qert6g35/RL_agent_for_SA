import matplotlib.pyplot as plt  # Import matplotlib for plotting
import numpy as np
import pandas as pd


#   READ ME
#
# z racji na to że zapomniaem uwzgldnić to, że SA wykonuje 10 kroków dla PPO_G2
#
# wartości steps to best value, muszą zostać pomnożone przez 10 (drobna utrata danych ale nie większa niż 9), 
#
# dlatego aby nie zaniżać wsp.  znalezienia lepszego wyniku (który chcemy aby by maly) doamy tez do kazdego wyniku 5


filename = "sa_results_stoch_5.csv"
print("running datachange for file: " + filename)
# Wczytaj dane
df = pd.read_csv(filename)

# Filtruj tylko wiersze z nazwą zawierającą "PPO_G2_130k"
mask = df["ts_name"].isin(['E', 'F_2', 'G', 'G2', 'G3_1', 'G3_2', 'G3_3', 'G4_1', 'G4_2', 'G4_3', 'G6_V1_1', 'G6_V1_2', 'G6_V1_3', 'G6_V0_1', 'G6_V0_2', 'G6_V0_3'])

# Pomnóż wartość w kolumnie 'stbv' przez np. 2 (możesz zmienić na dowolną wartość)
df.loc[mask, "steps_to_best_value"] = df.loc[mask, "steps_to_best_value"] * 10 + 5

# Opcjonalnie: zapisz wynik do nowego pliku
df.to_csv(filename, index=False)

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
#         min_steps = 1500
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
# print(estimate_sa_steps(109))
# # Zakres wartości n do wykresu
# n_values = np.arange(50, 301)
# step_values = [estimate_sa_steps(n) for n in n_values]
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