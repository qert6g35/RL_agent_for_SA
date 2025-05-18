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



# Wczytaj dane
df = pd.read_csv("G2_130K_TestsResult_pre_refactor.csv")

# Filtruj tylko wiersze z nazwą zawierającą "PPO_G2_130k"
mask = df["ts_name"].str.lower() == "ppo_g2_130k"

# Pomnóż wartość w kolumnie 'stbv' przez np. 2 (możesz zmienić na dowolną wartość)
df.loc[mask, "stbv"] = df.loc[mask, "stbv"] * 10 + 5

# Opcjonalnie: zapisz wynik do nowego pliku
df.to_csv("G2_130K_TestsResult.csv", index=False)

# Podgląd zmodyfikowanych danych
print(df[mask])