import os
import glob
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
    max_blow_range = int(max_blow/hp.blowstep)
    for blow in range(3, max_blow_range + 1):
        blow = round(blow * hp.blowstep, 2)
        for nb in range(1, nb_coeffs + 1):
            if os.path.exists("./jobs"):
                for item in glob.glob("./jobs/*"):
                    os.remove(item)        
            hp.nb_coeffs = nb
            hp.max_blow = blow
            print(f"Training with max_blow = {blow}, nb_coeffs = {nb}")
            flow_control.init_train(filename, nb_epoch, verbose)


if __name__ == "__main__":
    main()
