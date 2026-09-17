nb_snapshots = 20000
nb_components = 3 # velocity u, velocity v, velocity w
ng = [512, 128, 160] # dimensions of the velocity field
starting_x = 100
cutting_rate = 1
nb_width_points = 64
control_width = nb_width_points//cutting_rate
control_length = ng[1]//cutting_rate
nb_actions = control_width * control_length

starting_coeff = 2
nb_coeffs = 4
tke_ref = 0
gamma = 0.3
delta_t = 275.0
nb_epoch = 50
starting_blow = 0.4
max_blow = 0.8
blowstep = 0.1
alpha = 0.15
Nu_baseline = 2.962 # calculated and rounded from 2.9619621858466085 obtained from the baseline simulation with no control
P_base_pump = 0.668 # calculated and rounded from 0.6681371117 obtained from the calculus : (1/2) * rho * (u_inf**3) * S_wall * Cf
# rho = fluid density = 1.0
# u_inf = fluid x velocity after the boundary layer = 1.0
# S_wall = wall surface defined in geometry.out = 30.0 * 3.0
# Cf = coefficient of fricton for laminar flow
#    = 0.664/sqrt(Re) where Re is Reynold's number, Re = 2000
blow_area_rate = nb_width_points / ng[0]
learning_rate = 1e-6