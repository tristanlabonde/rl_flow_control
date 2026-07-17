import sys
import flow_control


def main():
    argc = len(sys.argv)
    assert argc >= 2, "Usage: python main.py [-v/--verbose] <input_filename> <nb_epoch>(optional)"
    verbose = sys.argv[1] == "--verbose" or sys.argv[1] == "-v"
    filename = sys.argv[1 + int(verbose)]
    if argc >= 3 + int(verbose):
        nb_epoch = int(sys.argv[2 + int(verbose)])
    else:
        nb_epoch = 50

    flow_control.init_train(filename, nb_epoch, verbose)


if __name__ == "__main__":
    main()
