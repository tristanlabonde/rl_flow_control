nb_snapshots = 20000
nb_components = 3 # velocity u, velocity v, velocity w
ng = [512, 128, 160] # dimensions of the velocity field
starting_x = 100
cutting_rate = 1
control_width = 64//cutting_rate
control_length = ng[1]//cutting_rate
nb_actions = control_width * control_length
nb_coeffs = 1 # number of Fourier series coefficients to predict

tke_ref = 0
gamma = 0.3
nb_epoch = 50
max_blow = 0.8
blowstep = 0.05