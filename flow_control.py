import subprocess
import numpy as np
import torch
import time
import glob
import os
import re
import rl_fc_models as model
from collections import deque

TKE_ref = 0
GAMMA = 0.3

def parse_results(jobid, verbose=True): #results are stored in data/
    # Read the results of the simulation from the output file stats1d.out and return a 4 x height array where row 0 is the height, row 1 is the u-velocity variance, row 2 is the v-velocity variance and row 3 is the w-velocity variance.
    if verbose:
        print("\tParsing results...")

    if jobid == -1:
        return None
    with open('./jobs/stdout.' + jobid, 'r', encoding='utf-8') as f:
        derniere_ligne = deque(f, maxlen=1)[0]
        if derniere_ligne.strip() != "*** Fim ***":
            return None
    
    results_file = "data/stats1d.out"
    
    data = np.loadtxt(results_file)
    res = np.array([data[:, 0], data[:, 4], data[:, 5], data[:, 6]])
    return res

def compute_scaled_tke(variances, verbose=True):
    # Compute the reward/tke (Turbulent Kinetic Energy) from the given variances of the simulation.
    if verbose:
        print("\tComputing scaled tke...")
    if variances is None:
        if verbose:
            print("\033[31mInvalid results detected.\033[0m")
        return -5  # Return a low reward if results are invalid
    
    z = variances[0]
    u_prime2 = variances[1]
    v_prime2 = variances[2]
    w_prime2 = variances[3]

    tke_profile = 0.5 * (u_prime2 + v_prime2 + w_prime2)

    tke = np.trapezoid(tke_profile, z)

    return tke * 1000  # Scale the TKE value for better numerical stability

def compute_scaled_exergy_tke(variances, action_np, gamma=GAMMA, verbose=True):

    scaled_tke = compute_scaled_tke(variances, verbose)
    blowing_cost = np.mean(np.square(action_np))
    reward = scaled_tke - gamma * blowing_cost
    
    return reward

def compute_nusselt(file_temp, z0, z1, verbose=True):
    print(f"z0: {z0}, z1: {z1}")

    nu_list = []

    if verbose:
        print(f"Computing Nusselt number for files: {file_temp}")

    for file in file_temp:
        try:
            t_data = np.fromfile(file, dtype=np.float64)
            t = np.reshape(t_data, (model.ng[0], model.ng[1], model.ng[2]), order='F')
        except FileNotFoundError:
            if verbose:
                print(f"\033[31mFile {file} not found. Skipping this file.\033[0m")
            continue

        t = t[model.starting_x+model.control_width*model.cutting_rate+1:, :, 0:2] # Extract the temperature data for the specified x-range and first two z-planes
        t_mean_z = np.mean(t, axis=(0, 1))

        nu_wall = (t_mean_z[1] - t_mean_z[0]) / (z1 - z0)

        if nu_wall is not np.isnan(nu_wall):
            nu_list.append(nu_wall)

    if len(nu_list) == len(file_temp):
        return np.mean(nu_list)
    else:
        return -5.0

def compute_exergy_nusselt(z, action_np, gamma=GAMMA, nb_files=3, verbose=True):
    file_temp = sorted(glob.glob("data/sca_001_fld_*.bin"))[-nb_files:]

    Nu_mean = compute_nusselt(file_temp, z[0], z[1])

    blowing_cost = np.mean(np.square(action_np))
    reward = Nu_mean - gamma * blowing_cost
    
    return reward

def compute_thermic_capacity(nb_files, verbose=True):

    log_visu_3d = np.loadtxt("data/log_visu_3d.out")
    geometry = np.loadtxt("data/geometry.out")

    Lx = geometry[1, 0]
    Ly = geometry[1, 1]
    Lz = geometry[1, 2]
    Lx_region = ((model.ng[0] - (model.starting_x+model.control_width*model.cutting_rate+1))/model.ng[0]) * Lx
    S = Lx_region * Ly
    nb_parameters = 9 # Number of parameters in the simulation (u, v, w, T, p, etc.)
    delta_t = log_visu_3d[-nb_parameters, -2] - log_visu_3d[-nb_files*nb_parameters, -2]

    print(f"file numbers for delta_t computation : {log_visu_3d[-nb_parameters, -1]} - {log_visu_3d[-nb_files*nb_parameters, -1]}")

    with open("input.nml", 'r', encoding='utf-8') as f:
        content = f.read()
    alphai_pattern = r"alphai(:)\s*=\s*([-+]?\d*\.?\d+)"
    k = 1.0/float((re.search(alphai_pattern, content)).group(1))

    delta_T = 1.0

    return ((k*delta_T)/Lz)*S*delta_t

def compute_thermal_efficiency(z, action_np, nb_files=3, verbose=True):
    file_temp = sorted(glob.glob("data/sca_001_fld_*.bin"))[-nb_files:]

    Nu_mean = compute_nusselt(file_temp, z[0], z[1])
    Cth = compute_thermic_capacity(nb_files, verbose)
    E_out = Cth * Nu_mean

    E_in = np.mean(np.square(action_np)) # a changer pour passer a la véritable énergie d'entrée, cad la puissance de soufflage multipliée par le temps de simulation plus l'énergie de pompage du canal

    return (E_out / E_in) * 100

def parse_input(filename, verbose=True):
    if verbose:
        print(f"Parsing input file {filename}...")
    
    data = np.loadtxt(filename)
    means = np.array([data[:, 0], data[:, 1], data[:, 2], data[:, 3]]) # position z, means of u, v, w on the x-y corresponding planes

    velocity_field = np.zeros((model.nb_components, model.ng[0], model.ng[1], model.ng[2]))
    for z in range(model.ng[2]):
        velocity_field[0, :, :, z] = means[1][z]
    velocity_field[2, :, :, model.ng[2]-1] = means[3][model.ng[2]-1]
    return velocity_field

def compute_tke_ref(filename, verbose=True):
    # Compute the reference TKE from the input velocity field.
    # UNUSED
    if verbose:
        print(f"Computing reference TKE from {filename}...")
        
    data = np.loadtxt(filename)
    res = np.array([data[:, 0], data[:, 4], data[:, 5], data[:, 6]])
    tke_ref = compute_scaled_tke(res, verbose)
    return tke_ref

def create_grids_input(foldername, wall_blowing_amps, verbose=True):
    if verbose:
        print(f"\tCreating input folder {foldername}")
    for t in range(model.nb_snapshots):
        filename = foldername+"/grids_input"+str(t)+".txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            for x in range(model.control_width):
                for y in range(model.control_length):
                    for i in range(model.cutting_rate):
                        for j in range(model.cutting_rate):
                            f.write(f"{model.starting_x + x*model.cutting_rate + i} {y*model.cutting_rate + j} {wall_blowing_amps[t, x, y]}\n")
            f.close()

    simulation_input = "./wall_blowing_input"
    remove_command = ["rm", "-rf", simulation_input]
    lazy_copy_command = ["ln", "-s", foldername, simulation_input]
    subprocess.run(remove_command, check=True)
    subprocess.run(lazy_copy_command, check=True)
    
def create_single_grid_input(foldername, wall_blowing_amps, verbose=True):
    if verbose:
        print(f"\tCreating input folder {foldername}")
    filename = foldername+"/single_grid_input.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for x in range(model.control_width):
            for y in range(model.control_length):
                for i in range(model.cutting_rate):
                    for j in range(model.cutting_rate):
                        f.write(f"{model.starting_x + x*model.cutting_rate + i} {y*model.cutting_rate + j} {wall_blowing_amps[x, y]}\n")
        f.close()

    simulation_input = "./wall_blowing_input"
    remove_command = ["rm", "-rf", simulation_input]
    lazy_copy_command = ["ln", "-s", foldername, simulation_input]
    subprocess.run(remove_command, check=True)
    subprocess.run(lazy_copy_command, check=True)

def launch_simulation(verbose=True):
    # Launch the simulation using the command sbatch --wait srun.sh.
    command = ["sbatch", "--wait", "srun.sh"]
    reset_command = ["rm", "-rf", "data/*"]

    if verbose:
        print("\tCleaning previous results...")
    subprocess.run(reset_command, check=True)
    try:
        if verbose:
            print("\tLaunching simulation...")
        res = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        if verbose:
            print("\tSimulation completed successfully.")
        jobid = ((res.stdout).split())[-1]
        return jobid
    except subprocess.CalledProcessError as e:
        print("\033[31mAn error occurred during simulation.\033[0m")
        print(e.stderr)
        return -1

def criterion_tke_grids(train_foldername, train_preds, verbose=True):
    # Create the input file for the simulation using the grids predicted by the rl agent, launch the simulation, parse the results, and compute the TKE.
    create_grids_input(train_foldername, train_preds, verbose)
    jobid = launch_simulation(verbose)
    variances = parse_results(jobid, verbose)
    return compute_scaled_tke(variances, verbose)


def criterion_tke_single_grid(train_foldername, train_preds, verbose=True):
    # Create the input file for the simulation using the single grid predicted by the rl agent, launch the simulation, parse the results, and compute the TKE.
    create_single_grid_input(train_foldername, train_preds, verbose)
    jobid = launch_simulation(verbose)
    variances = parse_results(jobid, verbose)
    return compute_scaled_exergy_tke(variances, train_preds, verbose=verbose)

def criterion_nusselt_single_grid(train_foldername, train_preds, verbose=True):
    # Create the input file for the simulation using the single grid predicted by the rl agent, launch the simulation, parse the results, and compute the Nusselt number.
    create_single_grid_input(train_foldername, train_preds, verbose)
    jobid = launch_simulation(verbose)
    variances = parse_results(jobid, verbose)
    return compute_exergy_nusselt(variances[0], train_preds, verbose=verbose)

def train(agent, input_velocity_tensor, input_time_tensor, nb_epoch, optimizer, criterion, scheduler, verbose=True):
    # Train the RL agent for a specified number of epochs, using the provided optimizer, criterion, and scheduler.
    print("Training starts...\n")

    exploration_noise = 0.01

    for epoch in range(nb_epoch):
        if verbose:
            print(f"Epoch {epoch + 1}/{nb_epoch}")
        #valid_loss = 0.0

        agent.train()
        optimizer.zero_grad()
        action_mean = agent(input_velocity_tensor, input_time_tensor)
        distribution = torch.distributions.Normal(action_mean, exploration_noise)
        action_sample = distribution.sample()
        log_prob = distribution.log_prob(action_sample).mean()
        action_np = action_sample.cpu().numpy()
        action_np = np.clip(action_np, -1.0, 1.0)
        train_foldername = f"training_inputs/training_input_epoch{epoch + 1}"
        reward = criterion(train_foldername, action_np, verbose)
        loss = -log_prob * reward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss = loss.item()

        #scheduler.step(valid_loss) 
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:>4}/{nb_epoch} - LR actuel : {current_lr:.6f}\n\ttrain_loss : {train_loss:>9.3f} - reward : {reward:>9.6f}\n")

def init_train(filename, nb_epoch, verbose=True):
    # Initialize the training process by parsing the input velocity field, setting up the device (GPU or CPU), creating the model, optimizer, criterion, and scheduler, and then calling the train function.
    train_velocity_field = parse_input(filename, verbose)
    # global TKE_ref
    # TKE_ref = compute_tke_ref(filename, verbose)
    time_steps = torch.linspace(0.0, 1.0, model.nb_snapshots).unsqueeze(1)

    if torch.cuda.is_available():
        print(f"GPU found : {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Mac GPU detected (MPS). Using graphical acceleration.")
        device = torch.device("mps")
    else:
        print("GPU not found, using CPU.")
        device = torch.device("cpu")

    input_velocity_tensor = torch.from_numpy(np.array([train_velocity_field])).float().to(device)
    input_time_tensor = time_steps.to(device)
    agent = model.FlowControlCoeffSingleGrid().to(device)

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-6, weight_decay=1e-4)
    criterion = criterion_tke_single_grid
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5,patience=2)

    start = time.time()
    train(agent, input_velocity_tensor, input_time_tensor, nb_epoch, optimizer, criterion, scheduler)
    end = time.time()
    print(f"Training took {end-start:.3f}s")