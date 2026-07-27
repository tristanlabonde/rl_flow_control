import subprocess
import numpy as np
import torch
import time
import os
from rl_fc_model import RL_FlowControl_Agent, nb_snapshots, nb_components, ng, starting_x, control_width, control_length, cutting_rate

def parse_results(verbose=True): #results are stored in data/
    # Read the results of the simulation from the output file stats1d.out and return a 4 x height array where row 0 is the height, row 1 is the u-velocity variance, row 2 is the v-velocity variance and row 3 is the w-velocity variance.
    if verbose:
        print("\tParsing results...")

    filepath = os.path.join("data", "x_bin")
    if not os.path.exists(filepath):
        return None
    
    results_file = "data/stats1d.out"
    
    data = np.loadtxt(results_file)
    res = np.array([data[:, 0], data[:, 4], data[:, 5], data[:, 6]])
    return res

def compute_tke(variances, verbose=True):
    # Compute the reward/tke (Turbulent Kinetic Energy) from the output of the simulation.
    if verbose:
        print("\tComputing reward...")
    if variances is None:
        if verbose:
            print("\033[31mInvalid results detected.\033[0m")
        return float('-inf')  # Return a low reward if results are invalid
    
    y = variances[0]
    u_prime2 = variances[1]
    v_prime2 = variances[2]
    w_prime2 = variances[3]

    tke_profile = 0.5 * (u_prime2 + v_prime2 + w_prime2)

    tke = np.trapezoid(tke_profile, y)
    
    return tke

def parse_input(filename, verbose=True):
    if verbose:
        print(f"Parsing input file {filename}...")
    
    data = np.loadtxt(filename)
    means = np.array([data[:, 0], data[:, 1], data[:, 2], data[:, 3]]) # position z, means of u, v, w on the x-y corresponding planes

    velocity_field = np.zeros((nb_components, ng[0], ng[1], ng[2]))
    for z in range(ng[2]):
        velocity_field[0, :, :, z] = means[1][z]
    velocity_field[2, :, :, ng[2]-1] = means[3][ng[2]-1]
    return velocity_field

def create_input(foldername, wall_blowing_amps, verbose=True):
    if verbose:
        print(f"\tCreating input folder {foldername}")
    for t in range(nb_snapshots):
        filename = foldername+"/input"+str(t)+".txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            for x in range(control_width):
                for y in range(control_length):
                    for i in range(cutting_rate):
                        for j in range(cutting_rate):
                            f.write(f"{starting_x + x*cutting_rate + i} {y*cutting_rate + j} {wall_blowing_amps[t, x*control_width + y]}\n")
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
        subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        if verbose:
            print("\tSimulation completed successfully.")
    except subprocess.CalledProcessError as e:
        print("\033[31mAn error occurred during simulation.\033[0m")
        print(e.stderr)

def criterion_tke(train_foldername, train_preds, verbose=True):
    # Create the input file for the simulation using the prediction done by the rl agent, launch the simulation, parse the results, and compute the TKE.
    create_input(train_foldername, train_preds, verbose)
    launch_simulation(verbose)
    res = parse_results(verbose)
    return compute_tke(res, verbose)

def train(model, input_velocity_tensor, input_time_tensor, nb_epoch, optimizer, criterion, scheduler, verbose=True):
    # Train the RL agent for a specified number of epochs, using the provided optimizer, criterion, and scheduler.
    print("Training starts...\n")
    
    history = {'train_loss': [], 'reward': []}
    exploration_noise = 0.1

    for epoch in range(nb_epoch):
        if verbose:
            print(f"Epoch {epoch + 1}/{nb_epoch}")
        #valid_loss = 0.0

        model.train()
        optimizer.zero_grad()
        action_mean = model(input_velocity_tensor, input_time_tensor)
        distribution = torch.distributions.Normal(action_mean, exploration_noise)
        action_sample = distribution.sample()
        log_prob = distribution.log_prob(action_sample).sum()
        action_np = action_sample.cpu().numpy()
        action_np = np.clip(action_np, -1.0, 1.0)
        train_foldername = f"training_inputs/training_input_epoch{epoch + 1}"
        reward = criterion(train_foldername, action_np, verbose)
        loss = -log_prob * reward
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        #scheduler.step(valid_loss) 
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['reward'].append(reward)

        print(f"Epoch {epoch+1:>4}/{nb_epoch} - LR actuel : {current_lr:.6f}\n\ttrain_loss : {train_loss:>9.3f} - reward : {reward:>9.3f}\n")

    return history

def init_train(filename, nb_epoch, verbose=True):
    # Initialize the training process by parsing the input velocity field, setting up the device (GPU or CPU), creating the model, optimizer, criterion, and scheduler, and then calling the train function.
    train_velocity_field = parse_input(filename, verbose)
    time_steps = torch.linspace(0.0, 1.0, nb_snapshots).unsqueeze(1)

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
    model = RL_FlowControl_Agent().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = criterion_tke
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5,patience=2)

    start = time.time()
    history = train(model, input_velocity_tensor, input_time_tensor, nb_epoch, optimizer, criterion, scheduler)
    end = time.time()
    print(f"Training took {end-start:.3f}s")