import numpy as np
import os
import sys

ng = [512, 128, 160]
starting_x = 100
cutting_rate = 1
control_width = 64//cutting_rate
control_length = ng[1]//cutting_rate
nb_actions = control_width * control_length

max_blow = 0.8

def draw_coefficients(c):
    nb_coeffs = len(c)
    print(f"Writting W distribution with {nb_coeffs} coefficients: {c}")

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

def create_single_grid_input(wall_blowing_amps):
    filename = "./single_grid_input.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for x in range(control_width):
            for y in range(control_length):
                for i in range(cutting_rate):
                    for j in range(cutting_rate):
                        f.write(f"{starting_x + x*cutting_rate + i} {y*cutting_rate + j} {wall_blowing_amps[x, y]}\n")
        f.close()

coeffs = [float(arg) for arg in sys.argv[1:]]
W = draw_coefficients(coeffs)
create_single_grid_input(W)