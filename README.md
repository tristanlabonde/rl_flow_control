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

First, to use my code you need to update main.f90 and control.f90 in the CaNS code so copy this files from the root of this folder to CaNS/src/ and do the command make at CaNS/ to

To launch a training session, complete the hyperparameters.py file with the wanted values. The main interest for a user may be the parameters for hyperparameters exploration at the end of the file, you can build a loop that will launch several trainings, exploring a bunch of values for max_blow or nb_coeff. Be aware that a training of 50 epochs takes approximately 5 hours to finish, so be carefull with the number of different values for hyperparameters you want to explore.