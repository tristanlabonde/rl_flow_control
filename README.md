# Author : Tristan LABONDE

srun.sh srun_control.sh input_mean_env.txt

## Purpose of the code

The goal of this code is to train an AI model to find the most optimal way to control a fluid with a blowing area.

## Context

I decided to apply a reinforcement learning method with an AI agent and an environment.
I use the CaNS code to simulate the controlled fluid. I use this code as the environment for my AI agent.
I use a Blasius profile, one wall and a fluid with u_inf velocity (x axis) of 1.0.
Blowing area blows in z axis only and so applies a w velocity to the points of the blowing area. Velocities can be positive (blowing) or negative (suction).

## How to use the code

To start to use my code you need to update main.f90 and control.f90 in the CaNS code so copy this files from the root of this folder to CaNS/src/ and do the command make at CaNS/ to recompile the CaNS code.
Then, it is necessary to decompress rl_flow_control_labonde.tar with the command :
    tar xzvf rl_flow_control_labonde.tar
in CaNS/run in order to put all the following files/folder in the run repertory of CaNS' code :
 - files :
    criterions.py hyperparameters.py input_grid_create.py input_mean_env.txt input.nml main.py parser.py physical_values_compute.py rl_fc_models.py srun_control.sh srun.sh trainer_flow_control.py
 - folder :
    jobs jobs_control 

To launch a training session, complete the hyperparameters.py file with the wanted values. The main interest for a user may be the parameters for hyperparameters exploration at the end of the file, you can build a loop that will launch several trainings, exploring a bunch of values for max_blow or nb_coeff. Be aware that a training of 50 epochs takes approximately 5 hours to finish, so be carefull with the number of different values for hyperparameters you want to explore.

When you have done all of that, you just have to do the command to launch the training(s) session(s):
    sbatch srun_control.sh <nb_epoch>(optional)

## How to see results

There is three way to observe the results of a training :
 - observe the evolution of the reward during the training -> open the result_viewer/training_graphs_generator.py file and read the description at the beginning of the file to know how to use it.
 - observe the blowing scheme applied to a simulation -> open the result_viewer/draw_input_graph.py file and read the description at the beginning of the file to know how to use it.
 - observe a simulation with paraview -> open the CaNS/docs/INFO_VISU.md file and learn how to generate the viewfld_DNS.xmf file.