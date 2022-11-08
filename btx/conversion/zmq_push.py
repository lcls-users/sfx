import numpy as np
import argparse
from btx.interfaces.psana_interface import *
from btx.conversion.zmqhelper import ZmqSender

"""
This script is based on Mona's zmq_pushall.py script in the xtc1to2 repository:
https://github.com/monarin/xtc1to2/blob/master/examples/zmq_pushall.py
"""

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', help='Experiment name', required=True, type=str)
    parser.add_argument('-r', '--run', help='Run number', required=True, type=int)
    parser.add_argument('-d', '--det_type', help='Detector name, e.g epix10k2M or jungfrau4M',  required=True, type=str)
    parser.add_argument('--max_events', help='Max events per run to retrieve', required=True, type=int, default=-1)
    
    return parser.parse_args()

def retrieve_detector_info(detector, run):
    """
    Retrieve detector information required by peak finding but 
    likely inaccessible through psana2.
    
    Parameters
    ----------
    detector : Detector.AreaDetector.AreaDetector
        instance of a psana area detector
    run : int
        run number
        
    Returns
    -------
    d_det : dict
        store of detector information
    """
    d_det = dict()
    d_det['pixel_index_map'] = retrieve_pixel_index_map(detector.geometry(run)).astype(np.int16)
    d_det['iX'] = detector.indexes_x(run).astype(np.int16)
    d_det['iY'] = detector.indexes_y(run).astype(np.int16)
    d_det['ipx'], d_det['ipy'] = detector.point_indexes(run, pxy_um=(0, 0))
    d_det['det_shape'] = np.array(detector.shape()).astype(np.uint32)
    return d_det
    
def main():
    """
    Use zmq to send images and detector info retrieved from an lcls1 experiment.
    """
    
    # Initialize zmq sender
    socket = "tcp://127.0.0.1:5558"
    zmq_send = ZmqSender(socket)

    params = parse_input()
    psi = PsanaInterface(exp=params.exp, run=params.run, det_type=params.det_type)
    det_dict = retrieve_detector_info(psi.det, params.run)
    
    start_idx, end_idx = psi.counter, psi.max_events
    if params.max_events!=-1:
        end_idx = params.max_events
    
    for idx in np.arange(start_idx, end_idx):
        # retrieve calibrated image and photon energy
        evt = psi.runner.event(psi.times[idx])
        img = psi.det.calib(evt=evt)
        if img is None:
            continue
        try:
            photon_energy = 1.23984197386209e-06 / (psi.get_wavelength_evt(evt) / 10 / 1.0e9)
        except AttributeError:
            photon_energy = 1.23984197386209e-06 / (psi.get_wavelength() / 10 / 1.0e9)
            
        if idx == 0:
            # Send beginning timestamp to create config, beginrun, beginstep, and enable on the client.
            start_dict = {
                "start": True, 
                "exp": params.exp,
                "run": params.run,
                "config_timestamp": psi.times[idx].time() - 10,
                "clen" : psi.get_camera_length(pv=None)
                }
            print(start_dict)
            start_dict.update(det_dict)
            zmq_send.send_zipped_pickle(start_dict)
            print("Sent starting dict")
                
        data = {
            "calib": img,
            "photon_energy": photon_energy,
            "timestamp": psi.times[idx].time(),
        }

        zmq_send.send_zipped_pickle(data)
        print(f"Sent event {idx} of {params.max_events}")

    done_dict = {"end": True}
    zmq_send.send_zipped_pickle(done_dict)
    
if __name__ == '__main__':
    main()
