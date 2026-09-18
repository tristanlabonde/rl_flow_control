# This file is where the training is initiated and conduct
import numpy as np
import torch
import time
import rl_fc_models as models
import hyperparameters as hp
import criterions
import parser

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
    train_velocity_field = parser.parse_input(filename, verbose)
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
    criterion = criterions.criterion_thermal_efficiency_single_grid
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.5,patience=3)

    start = time.time()
    train(agent, input_velocity_tensor, input_time_tensor, nb_epoch, optimizer, criterion, scheduler)
    end = time.time()
    print(f"Training took {end-start:.3f}s")