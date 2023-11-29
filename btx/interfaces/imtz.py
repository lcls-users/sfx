from btx.interfaces.ischeduler import *
import argparse
import subprocess
import os
from btx.interfaces.ischeduler import *
from btx.misc.xtal import enforce_symmetry

def write_create_xscale(out_hkl, cell, sg_number=1, exe="create-xscale"):
    """
    Write a create-xscale file for converting an hkl file from CrystFEL 
    to XDS format. Original file can be found here:
    https://github.com/biochem-fan/CrystFEL/blob/master/scripts/create-xscale
    
    Parameters
    ----------
    out_hkl : str
        name of output hkl file
    cell : ndarray, shape (6,)
        unit cell parameters
    sg_number : int
        space group number
    exe : str
        name of executable file
    """
    cell = enforce_symmetry(cell, sg_number)
    
    with open(exe, "w") as f:
        f.write("#!/usr/bin/perl -w\n\n")
        f.write("use strict;\n\n")
        f.write(f"open(FH, '{out_hkl}');\n\n")
        f.write('printf("' + "!FORMAT=XDS_ASCII   MERGE=TRUE   FRIEDEL'S_LAW=TRUE" + "\\n" + '");' + "\n")
        f.write('printf("' + f"!SPACE_GROUP_NUMBER={sg_number}" + "\\n" + '");' + "\n")
        f.write('printf("' + f"!UNIT_CELL_CONSTANTS=      {cell[0]:.2f}    {cell[1]:.2f}   {cell[2]:.2f} {cell[3]:.3f} {cell[4]:.3f}    {cell[5]:.3f}" + "\\n" + '");' + "\n")
        f.write('printf("' + f"!NUMBER_OF_ITEMS_IN_EACH_DATA_RECORD=5" + "\\n" + '");' + "\n")
        f.write('printf("' + f"!X-RAY_WAVELENGTH= -1.0" + "\\n" + '");' + "\n")
        f.write('printf("' + f"!ITEM_H=1" + "\\n" + '");' + "\n")
        f.write('printf("' + f"!ITEM_K=2" + "\\n" + '");' + "\n")
        f.write('printf("' + f"!ITEM_L=3" + "\\n" + '");' + "\n")
        f.write('printf("' + f"!ITEM_IOBS=4" + "\\n" + '");' + "\n")
        f.write('printf("' + f"!ITEM_SIGMA(IOBS)=5" + "\\n" + '");' + "\n")
        f.write('printf("' + f"!END_OF_HEADER" + "\\n" + '");' + "\n\n")
        
        f.write("my $line;\n")
        f.write("while ( $line = <FH> ) {\n")
        f.write("\n\tchomp($line);\n\n")
        f.write("\tif ( $line =~ /^\s+([0-9\-]+)\s+([0-9\-]+)\s+([0-9\-]+)\s+([0-9\.\-]+)\s+([\-]+)\s+([0-9\.\-]+)/ ) {\n\n")
        f.write("\t\tmy $h = $1;\n")
        f.write("\t\tmy $k = $2;\n")
        f.write("\t\tmy $l = $3;\n")
        f.write("\t\tmy $int = $4;\n")
        f.write("\t\tmy $sig = $6;\n\n")
        f.write('\t\tprintf("%6i %6i %5i %9.2f %9.2f\\n", $h, $k, $l, $int, $sig);\n\n')
        f.write("\t} else {\n\n")
        f.write('\t\tprintf(STDERR "Unrecognised:' + " '%s'" + "\\n" + '", $line' + ');' + "\n\n")
        
        f.write("\t}\n\n")
        f.write("}\n\n")
        
        f.write('printf("!END_OF_DATA");\n')
        f.write("close(FH);")

def write_xds_inp(hklfile, res_range=None, anomalous=True, inp_path="."):
    """
    Write the input file required for xdsconv, which outputs
    temp.hkl, F2MTZ.INP, and XDSCONV.LP files. These are the 
    inputs for ccp4's f2mtz program.
    
    Parameters
    ----------
    hklfile : str
        hkl file to convert
    res_range : tuple
        (low_res, high_res) cutoffs in Angstroms
    anomalous : bool
        if False, merge Friedel pairs
    inp_path : str
        working directory
    """
    with open(os.path.join(inp_path, "XDSCONV.INP"), "w") as f:
        f.write(f"INPUT_FILE={hklfile}\n")
        if res_range is not None:
            f.write(f"INCLUDE_RESOLUTION_RANGE={res_range[0]} {res_range[1]}\n")
        f.write("OUTPUT_FILE=temp.hkl  CCP4_I\n")
        if anomalous:
            f.write("FRIEDEL'S_LAW=FALSE\n")
        else:
            f.write("FRIEDEL'S_LAW=TRUE\n")

def write_anomalous_f2mtz(cell, sg_number=1, fname="F2MTZ.INP"):
    """
    Write an F2MTZ.INP file specific for anomalous data.
    
    Parameters
    ----------
    cell : ndarray, shape (6,)
        unit cell parameters
    sg_number : int
        space group number
    fname : str
        output file
    """    
    cell = enforce_symmetry(cell, sg_number)
    
    template = ("TITLE XDS to MTZ\n"
                "FILE temp.hkl\n"
                "SYMMETRY   {space_group}\n"
                "CELL {cell}\n"
                "LABOUT  H K L IMEAN SIGIMEAN I(+) SIGI(+) I(-) SIGI(-) ISYM\n"
                "CTYPOUT H H H   J      Q      K     M      K      M      Y\n"
                "END\n")
    
    context = {
        "space_group": sg_number,
        "cell": "    ".join([f"{c:.3f}" for c in cell])
    }
    
    with open(fname, "w") as outfile:
        outfile.write(template.format(**context))

def f2mtz_command(outmtz):
    """
    Return command to convert from an XDS-formatted hkl
    to an mtz file.
    
    Parameters
    ----------
    outmtz : str
        name of output mtz file
    
    Returns
    -------
    command : str
        commands to run ccp4's f2mtz and cad
    """
         
    command = ("f2mtz HKLOUT temp.mtz<F2MTZ.INP\n"
    f"cad HKLIN1 temp.mtz HKLOUT {outmtz}<<EOF\n"
    "LABIN FILE 1 ALL\n"
    "END\n"
    "EOF\n")
    
    return command

def run_dimple(mtz, pdb, outdir, queue='milano', ncores=16, anomalous=False, slurm_account="lcls",
               slurm_reservation=""):
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
    anomalous : bool
        if True, generate an anomalous difference map
    slurm_account : str
        SLURM account to use. Default: "lcls"
    slurm_reservation : str
        SLURM reservation to use, if one. Default: ""
    """
    os.makedirs(outdir, exist_ok=True)
    command = f"dimple {mtz} {pdb} {outdir}"
    if anomalous:
        command += " --anode"

    js = JobScheduler(os.path.join(outdir, "dimple.sh"),
                      logdir=outdir,
                      ncores=ncores, 
                      jobname=f'dimple', 
                      queue=queue,
                      account=slurm_account,
                      reservation=slurm_reservation)
    js.write_header()
    js.write_main(command + "\n", dependencies=['ccp4'])
    js.clean_up()
    js.submit()

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
    parser.add_argument('--anomalous', required=False, type=bool, default=False, help='Anomalous data')

    return parser.parse_args()

if __name__ == '__main__':

    params = parse_input()
    if params.dimple:
        run_dimple(params.mtz, 
                   params.pdb, 
                   params.outdir, 
                   queue=params.queue, 
                   ncores=params.ncores, 
                   anomalous=params.anomalous)
