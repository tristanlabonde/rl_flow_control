# This file is the main of the AI training code.
# Here is set up the input file for the AI model and the number of epochs of training
# There is the loop for hyperparameters values exploration.
import os
import glob
import sys
import trainer_flow_control as trainer
import hyperparameters as hp

def main():
    argc = len(sys.argv)
    assert argc >= 2, "Usage: python main.py [-v/--verbose] <input_filename> <nb_epoch>(optional)"
    verbose = sys.argv[1] == "--verbose" or sys.argv[1] == "-v"
    filename = sys.argv[1 + int(verbose)]
    if argc >= 3 + int(verbose):
        nb_epoch = int(sys.argv[2 + int(verbose)])
    else:
        nb_epoch = hp.nb_epoch

    nb_coeffs = hp.nb_coeffs
    for nb in range(hp.starting_coeff, nb_coeffs + 1, hp.coeff_step):
        if os.path.exists("./jobs"):
            for item in glob.glob("./jobs/*"):
                os.remove(item)
        hp.nb_coeffs = nb
        print(f"Training with nb_coeffs = {nb}")
        trainer.init_train(filename, nb_epoch, verbose)


if __name__ == "__main__":
    main()
