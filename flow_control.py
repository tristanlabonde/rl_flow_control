import subprocess
import numpy as np
import torch
import time
import glob
import os
import re
from collections import deque
import rl_fc_models as models
import hyperparameters as hp

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

def compute_scaled_exergy_tke(variances, action_np, gamma=hp.gamma, verbose=True):

    scaled_tke = compute_scaled_tke(variances, verbose)
    blowing_cost = np.mean(np.square(action_np))
    reward = scaled_tke - gamma * blowing_cost
    
    return reward

def compute_nusselt(file_temp, z0, z1, verbose=True):

    nu_list = []

    if verbose:
        print(f"\tComputing Nusselt number for files: {file_temp}")

    for file in file_temp:
        try:
            T_data = np.fromfile(file, dtype=np.float64)
            T = np.reshape(T_data, (hp.ng[0], hp.ng[1], hp.ng[2]), order='F')
        except FileNotFoundError:
            if verbose:
                print(f"\033[31mFile {file} not found. Skipping this file.\033[0m")
            continue

        T = T[hp.starting_x+hp.control_width*hp.cutting_rate+1:, :, 0:2] # Extract the temperature data for the specified x-range and first two z-planes
        # with open('./saved_std/temperature_z1.txt', 'a', encoding='utf-8') as f:
            # f.write("Values of temperature on z1 plane for file " + file + ": \n\n")
            # f.write(str(T[1]))

        T_mean_z = np.mean(T, axis=(0,1))
        nu_wall = (T_mean_z[1] - T_mean_z[0]) / (z1 - z0)

            # f.write("\n\nMean z1, z0: ")
            # f.write(str(T_mean_z[1]) + " " + str(T_mean_z[0]))
            # f.write("\n\n")
        
        if nu_wall is not np.isnan(nu_wall):
            nu_list.append(nu_wall)

    if len(nu_list) == len(file_temp):
        return np.mean(nu_list)
    else:
        return -5.0

def compute_exergy_nusselt(z, action_np, gamma=hp.gamma, nb_files=4, verbose=True):
    file_temp = sorted(glob.glob("data/sca_001_fld_*.bin"))[-nb_files:]

    Nu_mean = compute_nusselt(file_temp, z[0], z[1])

    blowing_cost = np.mean(np.square(action_np))
    reward = Nu_mean - gamma * blowing_cost
    
    return reward

def compute_thermic_capacity(verbose=True):

    geometry = np.loadtxt("data/geometry.out")

    Lx = geometry[1, 0]
    Ly = geometry[1, 1]
    Lz = geometry[1, 2]
    Lx_region = ((hp.ng[0] - (hp.starting_x+hp.control_width*hp.cutting_rate+1))/hp.ng[0]) * Lx
    S = Lx_region * Ly

    with open("input.nml", 'r', encoding='utf-8') as f:
        content = f.read()
    alphai_pattern = r"alphai\(\:\)\s*=\s*([-+]?\d*\.?\d*)"
    k = 1.0/float((re.search(alphai_pattern, content)).group(1))

    delta_T = 1.0

    return ((k*delta_T)/Lz)*S # * hp.t_eval but simplify when computing thermal efficiency

def compute_thermal_efficiency(z, action_np, nb_files=4, verbose=True):
    file_temp = sorted(glob.glob("data/sca_001_fld_*.bin"))[-nb_files:]

    Nu_mean = compute_nusselt(file_temp, z[0], z[1])
    if verbose:
        print(f"\tNusselt mean: {Nu_mean}")

    Cth = compute_thermic_capacity(verbose)
    if verbose:
        print(f"\tThermic capacity: {Cth}")

    P_out = Cth * Nu_mean
    if verbose:
        print(f"\tP_out: {P_out}")

    P_out_absolute = Cth * (Nu_mean - hp.Nu_baseline)

    amp_blow = np.abs(action_np)
    P_blow = 0.5 * hp.blow_area_rate * np.mean(amp_blow**3)
    if verbose:
        print(f"\tP_blow: {P_blow}")
        
    P_in = hp.P_base_pump + P_blow # * hp.t_eval to have E_in but simplify when computing thermal efficiency
    if verbose:
        print(f"\tP_in: {P_in}")

    return (P_out / P_in)

def parse_input(filename, verbose=True):
    if verbose:
        print(f"Parsing input file {filename}...")
    
    data = np.loadtxt(filename)
    means = np.array([data[:, 0], data[:, 1], data[:, 2], data[:, 3]]) # position z, means of u, v, w on the x-y corresponding planes

    velocity_field = np.zeros((hp.nb_components, hp.ng[0], hp.ng[1], hp.ng[2]))
    for z in range(hp.ng[2]):
        velocity_field[0, :, :, z] = means[1][z]
    velocity_field[2, :, :, hp.ng[2]-1] = means[3][hp.ng[2]-1]
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
    for t in range(hp.nb_snapshots):
        filename = foldername+"/grids_input"+str(t)+".txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            for x in range(hp.control_width):
                for y in range(hp.control_length):
                    for i in range(hp.cutting_rate):
                        for j in range(hp.cutting_rate):
                            f.write(f"{hp.starting_x + x*hp.cutting_rate + i} {y*hp.cutting_rate + j} {wall_blowing_amps[t, x, y]}\n")
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
        for x in range(hp.control_width):
            for y in range(hp.control_length):
                for i in range(hp.cutting_rate):
                    for j in range(hp.cutting_rate):
                        f.write(f"{hp.starting_x + x*hp.cutting_rate + i} {y*hp.cutting_rate + j} {wall_blowing_amps[x, y]}\n")
        f.close()

    simulation_input = "./wall_blowing_input"
    remove_command = ["rm", "-rf", simulation_input]
    lazy_copy_command = ["ln", "-s", foldername, simulation_input]
    subprocess.run(remove_command, check=True)
    subprocess.run(lazy_copy_command, check=True)

def launch_simulation(verbose=True):
    # Launch the simulation using the command sbatch --wait srun.sh.
    command = ["sbatch", "--wait", "srun.sh"]

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
    # Create the input file for the simulation using the single grid predicted by the rl agent, launch the simulation, parse the results, and compute exergy based on TKE.
    create_single_grid_input(train_foldername, train_preds, verbose)
    jobid = launch_simulation(verbose)
    variances = parse_results(jobid, verbose)
    return compute_scaled_exergy_tke(variances, train_preds, verbose=verbose)

def criterion_nusselt_single_grid(train_foldername, train_preds, verbose=True):
    # Create the input file for the simulation using the single grid predicted by the rl agent, launch the simulation, parse the results, and compute exergy based on Nusselt number.
    create_single_grid_input(train_foldername, train_preds, verbose)
    jobid = launch_simulation(verbose)
    variances = parse_results(jobid, verbose)
    return compute_exergy_nusselt(variances[0], train_preds, verbose=verbose)

def criterion_thermal_efficiency_single_grid(train_foldername, train_preds, verbose=True):
    # Create the input file for the simulation using the single grid predicted by the rl agent, launch the simulation, parse the results, and compute thermal efficiency.
    create_single_grid_input(train_foldername, train_preds, verbose)
    jobid = launch_simulation(verbose)
    variances = parse_results(jobid, verbose)
    return compute_thermal_efficiency(variances[0], train_preds, verbose=True)

def train(agent, input_velocity_tensor, input_time_tensor, nb_epoch, optimizer, criterion, scheduler, verbose=True):
    # Train the RL agent for a specified number of epochs, using the provided optimizer, criterion, and scheduler.
    print("Training starts...\n")

    exploration_noise = 0.01
    running_reward_mean = None

    for epoch in range(nb_epoch):
        if verbose:
            print(f"Epoch {epoch + 1}/{nb_epoch}")

        agent.train()
        optimizer.zero_grad()
        action_coeffs_mean = agent(input_velocity_tensor, input_time_tensor)
        distribution = torch.distributions.Normal(action_coeffs_mean, exploration_noise)
        coeffs_sample = distribution.sample()
        log_prob = distribution.log_prob(coeffs_sample).sum()
        w_tensor = agent.generate_grid(coeffs_sample[0])
        action_np = w_tensor.detach().cpu().numpy()
        action_np = np.clip(action_np, -hp.max_blow, hp.max_blow)
        train_foldername = f"training_inputs/training_input_epoch{epoch + 1}"
        reward = criterion(train_foldername, action_np, verbose)

        # EMA (moyenne glissante)
        if running_reward_mean is None:
            running_reward_mean = reward
        else:
            running_reward_mean = (1 - hp.alpha) * running_reward_mean + hp.alpha * reward

        advantage = reward - running_reward_mean
        loss = -log_prob * advantage

        # if advantage > 0:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss = loss.item()
        # else:
        #    optimizer.zero_grad()
        #    train_loss = 0.0

        scheduler.step(running_reward_mean)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1:>4}/{nb_epoch} - LR actuel : {current_lr:.2e}\n\ttrain_loss : {train_loss:.3e} - reward : {reward:>9.6f} - advantage : {advantage:>.3e}\n\tthermal_efficiency : {reward*100:>9.3f}%\n")

def init_train(filename, nb_epoch, verbose=True):
    # Initialize the training process by parsing the input velocity field, setting up the device (GPU or CPU), creating the model, optimizer, criterion, and scheduler, and then calling the train function.
    train_velocity_field = parse_input(filename, verbose)
    # hp.tke_ref
    # hp.tke_ref = compute_tke_ref(filename, verbose)
    time_steps = torch.linspace(0.0, 1.0, hp.nb_snapshots).unsqueeze(1)

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
    agent = models.FlowControlCoeffSingleGrid().to(device)

    optimizer = torch.optim.Adam(agent.parameters(), lr=hp.learning_rate, weight_decay=1e-4)
    criterion = criterion_thermal_efficiency_single_grid
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.5,patience=2)

    start = time.time()
    train(agent, input_velocity_tensor, input_time_tensor, nb_epoch, optimizer, criterion, scheduler)
    end = time.time()
    print(f"Training took {end-start:.3f}s")