# The functions contained by this file are used for grid_input files creation
import os
import hyperparameters as hp
import subprocess

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