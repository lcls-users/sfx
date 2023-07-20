#!/usr/bin/env python

import argparse
import shutil
import sys
import traceback
import yaml
import os

from btx.misc.shortcuts import AttrDict
from scripts.tasks import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to config file.')
    parser.add_argument('-t', '--task', type=str, help='Task to run.')
    parser.add_argument('-n', '--ncores', type=int, help='Number of cores.')
    config_filepath = parser.parse_args().config
    task = parser.parse_args().task
    cores = parser.parse_args().ncores
    with open(config_filepath, "r") as config_file:
        config = AttrDict(yaml.safe_load(config_file))
        #TODO: check required arguments in config dictionary here.

    if int(cores) > 1:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    else:
        comm = None
        rank = 0
        
    if rank == 0:
        # Create output directory.
        try:
            os.makedirs(config.setup.root_dir, exist_ok=True)
        except:
            print(f"Error: cannot create root path.") 
            return -1 

        path_plus_file = config.setup.root_dir + '/' + config_filepath.split('/')[-1]
        if os.path.exists(path_plus_file):
            os.remove(path_plus_file)
        
        # Copy config file to output directory.
        shutil.copy2(config_filepath, config.setup.root_dir)
        # Call 'task' function if it exists.
    if comm:
        comm.Barrier()

    try:
        globals()[task]
    except Exception as e:
        print(f'{task} not found.')
    globals()[task](config)

    return 0, 'Task successfully executed'

if __name__ == '__main__':
    try:
        retval, status_message = main()
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = 'Error: Task failed.'

    print(status_message)
    exit(retval)
