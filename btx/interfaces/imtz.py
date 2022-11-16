from btx.interfaces.ischeduler import *
import argparse
import subprocess
import os
from btx.interfaces.ischeduler import *

def run_dimple(mtz, pdb, outdir, queue='ffbh3q', ncores=16):
    """
    Run dimple to solve the structure: 
    http://ccp4.github.io/dimple/.
    
    Parameters
    ----------
    mtz : str
        input mtz file
    pdb : str
        input PDB file for phasing or rigid body refinement
    outdir : str
        output directory for storing results
    """
    os.makedirs(outdir, exist_ok=True)
    command = f"dimple {mtz} {pdb} {outdir}\n"

    # make path to executable
    if "TMP_EXE" in os.environ:
        tmp_exe = os.environ['TMP_EXE']
    else:
        tmp_exe = os.path.join(outdir, f'dimple.sh')

    js = JobScheduler(tmp_exe,
                      logdir=outdir,
                      ncores=ncores, 
                      jobname=f'dimple', 
                      queue=queue)
    js.write_header()
    js.write_main(command, dependencies=['ccp4'])
    js.shifterize()
    js.clean_up()
    #js.submit()

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mtz', required=True, type=str, help='Input mtz file of merged data')
    parser.add_argument('--pdb', required=False, type=str, help='Input PDB file')
    # dimple arguments
    parser.add_argument('--dimple', action='store_true', help='Run dimple to solve structure')
    parser.add_argument('--outdir', required=False, type=str, help='Directory for output')
    parser.add_argument('--ncores', required=False, type=int, default=16, help='Number of cores')
    parser.add_argument('--queue', required=False, type=str, default='ffbh3q', help='Queue to submit to')

    return parser.parse_args()

if __name__ == '__main__':

    params = parse_input()
    if params.dimple:
        run_dimple(params.mtz, params.pdb, params.outdir, queue=params.queue, ncores=params.ncores)
