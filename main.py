import subprocess
import sys
import flow_control
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

    max_blow = hp.max_blow
    nb_coeffs = hp.nb_coeffs
    for blow in range(1, int(max_blow/hp.blowstep) + 1):
        blow = blow * hp.blowstep
        for nb in range(1, nb_coeffs + 1):
            reset_command = ["rm", "-rf", "./jobs/*"]
            subprocess.run(reset_command, check=True)
            hp.nb_coeffs = nb
            hp.max_blow = blow
            print(f"Training with max_blow = {blow}, nb_coeffs = {nb}")
            flow_control.init_train(filename, nb_epoch, verbose)
            copy_stdout_command = ["mv", "./jobs/stdout_control.*", "./saved_stdout/stdout_control_blow_" + str(blow) + "_nb_" + str(nb)]
            subprocess.run(copy_stdout_command, check=True)

if __name__ == "__main__":
    main()
