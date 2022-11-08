from btx.conversion.zmqhelper import ZmqReceiver
from psana.dgramedit import DgramEdit, AlgDef, DetectorDef
from psana.psexp import TransitionId
import numpy as np
import argparse
import sys
import os
from psana import DataSource

"""
This script is based on Mona's zmq_pullall.py script in the xtc1to2 repository:
https://github.com/monarin/xtc1to2/blob/master/examples/zmq_pullall.py
"""

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', help='Experiment name', required=True, type=str)
    parser.add_argument('-r', '--run', help='Run number', required=True, type=int)
    parser.add_argument('-d', '--det_type', help='Detector name, e.g epix10k2M or jungfrau4M',  required=True, type=str)
    parser.add_argument('-o', '--outfile', help='Output cxi file', required=True, type=str)
    parser.add_argument('-b', '--bufsize', help='Size of buffer for dgram editor', required=False, type=int, default=128000000)

    return parser.parse_args()

def setup_dgram_editors(det_type):
    """
    Set up the dgram editors for cxic0415. The detector definition is based on here:
    https://github.com/slac-lcls/lcls2/blob/master/psana/psana/detector/test_detectors.py
    Parameters
    ----------
    det_type : str
        detector name
    Returns
    -------
    cspad : psana.dgrampy.NamesDef instance
        dgram editor
    runinfo : psana.dgrampy.NamesDef
        another dgram editor
    scan : psana.dgrampy.NamesDef
        yet another dgram editor
    """
    # NameId setup
    nodeId = 1 
    if det_type == 'DscCsPa':
        namesId = {"cxicspad": 0, "runinfo": 1, "scan": 2}
    elif det_type == 'epix10k2M':
        namesId = {"epix10k2M": 0, "runinfo": 1, "scan": 2}
    elif det_type == 'Rayonix':
        namesId = {"Rayonix": 0, "runinfo": 1, "scan": 2}
    else:
        sys.exit("Detector name not recongized")

    # set up config, algorithm, and detector
    config = DgramEdit(transition_id=TransitionId.Configure)
    if det_type == 'DscCsPa':
        alg = AlgDef("raw", 1, 2, 3)
        det = DetectorDef("cxicspad", "cspad", "detnum1234")
    elif det_type == 'epix10k2M':
        alg = AlgDef("raw", 0, 0, 2)
        det = DetectorDef("epix10k2M", "epix", "detnum1234")
    elif det_type == 'Rayonix':
        alg = AlgDef("raw", 0, 0, 2)
        det = DetectorDef("Rayonix", "rayonix", "detnum1234")
    else:
        sys.exit("Detector name not recongized")

    runinfo_alg = AlgDef("runinfo", 0, 0, 1)
    runinfo_det = DetectorDef("runinfo", "runinfo", "")

    scan_alg = AlgDef("raw", 2, 0, 0)
    scan_det = DetectorDef("scan", "scan", "detnum1234")

    # Define data formats
    if det_type == 'Rayonix':
        datadef = {
            "raw": (np.float32, 2),
            "photonEnergy": (np.float64, 0),
            }
    else:
        datadef = {
            "raw": (np.float32, 3),
            "photonEnergy": (np.float64, 0),
        }
    
    runinfodef = {
        "expt": (str, 1),
        "runnum": (np.uint32, 0),
    }

    if det_type == 'Rayonix':
        scandef = {
            "iX": (np.int16, 2),
            "iY": (np.int16, 2),
            "ipx": (np.uint32, 0),
            "ipy": (np.uint32, 0),
            "det_shape": (np.uint32, 1),
            "pixel_index_map": (np.int16, 4),
            "clen": (np.float32, 0),
            }
    else:
        scandef = {
            "iX": (np.int16, 3),
            "iY": (np.int16, 3),
            "ipx": (np.uint32, 0),
            "ipy": (np.uint32, 0),
            "det_shape": (np.uint32, 1),
            "pixel_index_map": (np.int16, 4),
            "clen": (np.float32, 0),
        }

    # Create detetors
    if det_type == 'DscCsPa':
        cspad = config.Detector(det, alg, datadef, nodeId=nodeId, namesId=namesId["cxicspad"])
    elif det_type == 'epix10k2M':
        cspad = config.Detector(det, alg, datadef, nodeId=nodeId, namesId=namesId["epix10k2M"])
    elif det_type == 'Rayonix':
        cspad = config.Detector(det, alg, datadef, nodeId=nodeId, namesId=namesId["Rayonix"])
    else:
        sys.exit("Detector name not recongized")

    runinfo = config.Detector(runinfo_det, 
                              runinfo_alg, 
                              runinfodef, 
                              nodeId=nodeId, 
                              namesId=namesId["runinfo"]
                             )
    scan = config.Detector(scan_det,
                           scan_alg,
                           scandef,
                           nodeId=nodeId,
                           namesId=namesId["scan"]
                          )
    
    return config, cspad, runinfo, scan

def main():
    
    params = parse_input()
    
    # Setup socket for zmq connection
    socket = "tcp://127.0.0.1:5558"
    zmq_recv = ZmqReceiver(socket)
        
    config, cspad, runinfo, scan = setup_dgram_editors(params.det_type)
    xtc2file = open(params.outfile, "wb")
    
    while True:
        obj = zmq_recv.recv_zipped_pickle()

        # Begin timestamp is needed (we calculate this from the first L1Accept)
        # to set the correct timestamp for all transitions prior to the first L1.
        if "start" in obj:
            config_timestamp = obj["config_timestamp"]
            config.updatetimestamp(config_timestamp)
            config.save(xtc2file)

            beginrun = DgramEdit(transition_id=TransitionId.BeginRun, config=config, ts=config_timestamp + 1, bufsize=params.bufsize)
            runinfo.runinfo.expt = 'txisfx00121'
            runinfo.runinfo.runnum = int(obj["run"])
            print(runinfo.runinfo.__dict__)
            beginrun.adddata(runinfo.runinfo)
            scan.raw.iX = obj['iX']
            scan.raw.iY = obj['iY']
            scan.raw.ipx = obj['ipx']
            scan.raw.ipy = obj['ipy']
            scan.raw.det_shape = obj['det_shape']
            scan.raw.pixel_index_map = obj['pixel_index_map']
            scan.raw.clen = obj['clen']
            beginrun.adddata(scan.raw)
            beginrun.save(xtc2file)
            
            beginstep = DgramEdit(transition_id=TransitionId.BeginStep, config=config, ts=config_timestamp + 2, bufsize=params.bufsize)
            beginstep.save(xtc2file)
            
            enable = DgramEdit(transition_id=TransitionId.Enable, config=config, ts=config_timestamp + 3, bufsize=params.bufsize)
            enable.save(xtc2file)
            current_timestamp = config_timestamp + 3

        elif "end" in obj:
            disable = DgramEdit(transition_id=TransitionId.Disable, config=config, ts=current_timestamp + 1, bufsize=params.bufsize)
            disable.save(xtc2file)
            endstep = DgramEdit(transition_id=TransitionId.EndStep, config=config, ts=current_timestamp + 2, bufsize=params.bufsize)
            endstep.save(xtc2file)
            endrun = DgramEdit(transition_id=TransitionId.EndRun, config=config, ts=current_timestamp + 3, bufsize=params.bufsize)
            endrun.save(xtc2file)
            break

        else:
            # Create L1Accept
            d0 = DgramEdit(transition_id=TransitionId.L1Accept, config=config, ts=obj["timestamp"], bufsize=params.bufsize)
            cspad.raw.raw = obj["calib"]
            cspad.raw.photonEnergy = obj["photon_energy"]
            d0.adddata(cspad.raw)
            d0.save(xtc2file)
            current_timestamp = obj["timestamp"]

    xtc2file.close()

if __name__ == '__main__':
    main()
    
