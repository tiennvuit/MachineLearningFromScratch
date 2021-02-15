import os
import json


def print_info(args: dict):
    """
    print the information of command arguments
    """

    print(str("-"*87).center(80))
    print(str("| {:^40} | {:^40} |".format('Argument', 'Value')).center(80))
    print(str("-"*87).center(80))
    for key, value in args.items():
        print(str("| {:<40} | {:<40} |".format(key, value)).center(80))
    print(str("-"*87).center(80))


def plot_progress(path:str, save=False):
    pass


def load_config(path):
    """
    Load config file to build model
    """

    if not os.path.exists(path):
        print("INVALID config file path !")
        exit(0)
    
    with open(path, 'r') as f:
        contents = f.read()
    args = json.loads(contents)
    return args