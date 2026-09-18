# This file contains all the hyperparameters used or that can be used for the training process

# blowing area features
ng = [512, 128, 160] # dimensions of the velocity field
starting_x = 100
cutting_rate = 1
nb_width_points = 64
control_width = nb_width_points//cutting_rate
control_length = ng[1]//cutting_rate

# for AI model
nb_snapshots = 20000
nb_components = 3 # velocity u, velocity v, velocity w
nb_actions = control_width * control_length
nb_coeffs = 2

# for reward computation
tke_ref = 0
gamma = 0.3
alpha = 0.15
Nu_baseline = 2.962 # calculated and rounded from 2.9619621858466085 obtained from the baseline simulation with no control
P_base_pump = 0.061 # calculated and rounded from 0.060989 obtained from the calculus operated in the tmp.py file in the "P_base_pump computation" section
blow_area_rate = nb_width_points / ng[0]

# for training
nb_epoch = 50
max_blow = 0.8
learning_rate = 1e-6

# for hyperparameters exploration
starting_coeff = 2
coeff_step = 1
starting_blow = 0.8
blow_step = 0.1