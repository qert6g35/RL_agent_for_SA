import matplotlib.pyplot as plt  # Import matplotlib for plotting
import numpy as np
def estimate_sa_steps(n):
    alpha=1
    min_steps = 1
    if n <= 100:
        alpha = 15.0
        min_steps = 15000
    elif n <= 200:
        alpha = 11.0
        min_steps = estimate_sa_steps(100)
    elif n <= 500:
        alpha = 8
        min_steps = estimate_sa_steps(200)
    return min(max(int(alpha * (n ** 1.58)),min_steps),1e5)

N = []
for n in range(1,500):
    N.append(estimate_sa_steps(n))


# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(range(1, 500), N, label="Estimated SA Steps", color="blue")
plt.xlabel("n (Problem Size)")
plt.ylabel("Estimated Steps")
plt.title("Estimated Simulated Annealing Steps vs Problem Size")
plt.legend()
plt.grid()
plt.show()