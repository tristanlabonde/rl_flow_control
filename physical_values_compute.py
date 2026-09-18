# This file contains functions that computes physical values such as tke and Nusselt number. It is used for reward computation for the AI.
import numpy as np
import hyperparameters as hp
import glob
import re

###### TKE ######

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

def compute_tke_ref(filename, verbose=True):
    # Compute the reference TKE from the input velocity field.
    # UNUSED
    if verbose:
        print(f"Computing reference TKE from {filename}...")
        
    data = np.loadtxt(filename)
    res = np.array([data[:, 0], data[:, 4], data[:, 5], data[:, 6]])
    tke_ref = compute_scaled_tke(res, verbose)
    return tke_ref

def compute_scaled_exergy_tke(variances, action_np, gamma=hp.gamma, verbose=True):

    scaled_tke = compute_scaled_tke(variances, verbose)
    blowing_cost = np.mean(np.square(action_np))
    reward = scaled_tke - gamma * blowing_cost
    
    return reward

###### Nusselt ######

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

    Cth_P = compute_thermic_capacity(verbose)
    if verbose:
        print(f"\tThermic capacity power: {Cth_P}")

    P_out = Cth_P * Nu_mean
    if verbose:
        print(f"\tP_out: {P_out}")

    P_out_absolute = Cth_P * (Nu_mean - hp.Nu_baseline)

    amp_blow = np.abs(action_np)
    P_blow = 0.5 * hp.blow_area_rate * np.mean(amp_blow**3)
    if verbose:
        print(f"\tP_blow: {P_blow}")
        
    P_in = hp.P_base_pump + P_blow # * hp.t_eval to have E_in but simplify when computing thermal efficiency
    if verbose:
        print(f"\tP_in: {P_in}")

    return (P_out / P_in)

