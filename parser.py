# Here are defined the AI model's input parser and the results parser
import numpy as np
from collections import deque
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