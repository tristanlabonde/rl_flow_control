# This file contains differents criterions and CaNS' simulation launching function
import subprocess
import parser
import physical_values_compute as phy_compute
import input_grid_create as create_input

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
    create_input.create_grids_input(train_foldername, train_preds, verbose)
    jobid = launch_simulation(verbose)
    variances = parser.parse_results(jobid, verbose)
    return phy_compute.compute_scaled_tke(variances, verbose)


def criterion_tke_single_grid(train_foldername, train_preds, verbose=True):
    # Create the input file for the simulation using the single grid predicted by the rl agent, launch the simulation, parse the results, and compute exergy based on TKE.
    create_input.create_single_grid_input(train_foldername, train_preds, verbose)
    jobid = launch_simulation(verbose)
    variances = parser.parse_results(jobid, verbose)
    return phy_compute.compute_scaled_exergy_tke(variances, train_preds, verbose=verbose)

def criterion_nusselt_single_grid(train_foldername, train_preds, verbose=True):
    # Create the input file for the simulation using the single grid predicted by the rl agent, launch the simulation, parse the results, and compute exergy based on Nusselt number.
    create_input.create_single_grid_input(train_foldername, train_preds, verbose)
    jobid = launch_simulation(verbose)
    variances = parser.parse_results(jobid, verbose)
    return phy_compute.compute_exergy_nusselt(variances[0], train_preds, verbose=verbose)

def criterion_thermal_efficiency_single_grid(train_foldername, train_preds, verbose=True):
    # Create the input file for the simulation using the single grid predicted by the rl agent, launch the simulation, parse the results, and compute thermal efficiency.
    create_input.create_single_grid_input(train_foldername, train_preds, verbose)
    jobid = launch_simulation(verbose)
    variances = parser.parse_results(jobid, verbose)
    return phy_compute.compute_thermal_efficiency(variances[0], train_preds, verbose=True)