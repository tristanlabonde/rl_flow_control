import numpy as np

ng = [512, 128, 160]
starting_x = 100
cutting_rate = 1
control_width = 64//cutting_rate
control_length = ng[1]//cutting_rate
nb_actions = control_width * control_length
data_file = "without_control_data/"

#### P_base_pump computation

data = np.loadtxt(data_file + "stats1d.out")
z, u = data[:, 0], data[:, 1]

print(f"z[0] = {z[0]}, z[1] = {z[1]}")
print(f"u[0] = {u[0]}, u[1] = {u[1]}")

Re = 2000.0
mu = 1.0 / Re
S_wall = 30.0 * 3.0

tau_w_1p = mu * (u[0] / z[0])
P_1p = tau_w_1p * S_wall

du_dz_2p = (4.0 * u[0] - 0.5 * u[1]) / (3.0 * z[0])
tau_w_2p = mu * du_dz_2p
P_2p = tau_w_2p * S_wall

print(f"P_base_pump (Ordre 1) = {P_1p:.6f}")
print(f"P_base_pump (Ordre 2) = {P_2p:.6f}")

##########################