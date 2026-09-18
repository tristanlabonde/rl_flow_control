# This file can be used for blowing schemes displaying, either based on grid input files (e.g. single_grid_input.txt or grids_inputX.txt) or based on Fourier coefficients
# Use this script like this :
#     python draw_input_graph.py name_of_grid_input_file.txt
# Or :
#     python draw_input_graph.py 1.5 -2.3 3.0
# The number of Fourier coefficients is not limited

import sys
import numpy as np
import matplotlib.pyplot as plt

ng = [512, 128, 160]
starting_x = 100
cutting_rate = 1
control_width = 64//cutting_rate
control_length = ng[1]//cutting_rate
nb_actions = control_width * control_length

max_blow = 0.8

x = np.arange(starting_x, starting_x + control_width, 1)
y = np.arange(0, control_length, 1)
X, Y = np.meshgrid(x, y)

def W_single_grid_input():
    data = np.loadtxt('single_grid_input.txt')

    W = np.zeros((control_width, control_length))
    for x, y, w in data:
        int_x = int(x)
        int_y = int(y)
        W[int_x-starting_x, int_y] = w

    return W

def draw_coefficients(c):
    nb_coeffs = len(c)
    print(f"Drawing W distribution with {nb_coeffs} coefficients: {c}")

    x_grid = starting_x + np.arange(0, control_width, dtype=np.float32)*cutting_rate
    y_grid = np.arange(0, control_length, dtype=np.float32)*cutting_rate
    k_modes = np.arange(1, nb_coeffs + 1, dtype=np.float32)

    sin_x = np.sin(2*np.pi/(control_width * cutting_rate) * k_modes[:, np.newaxis] * x_grid[np.newaxis, :])
    sin_y = np.sin(2*np.pi/(control_length * cutting_rate) * k_modes[:, np.newaxis] * y_grid[np.newaxis, :])

    W = np.zeros((control_width, control_length), dtype=np.float32)

    for i in range(nb_coeffs):
        W += c[i] * sin_x[i][:, np.newaxis] * sin_y[i][np.newaxis, :]

    W = np.clip(W, -max_blow, max_blow)
    return W

if sys.argv[1][-4:] != ".txt":
    coeffs = [float(arg) for arg in sys.argv[1:]]
    W = draw_coefficients(coeffs)
else:
    W = W_single_grid_input()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(X, Y, W.T, cmap='viridis', edgecolor='none')

ax.set_title("Blowing Velocity W Distribution Simulated by RL Model")
ax.set_xlabel("Position X")
ax.set_ylabel("Position Y")
ax.set_zlabel("Velocity W")
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

plt.show()