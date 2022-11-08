import numpy as np
import argparse
import itertools
import subprocess
import glob
import sys
import os
import time
import yaml
from btx.misc.shortcuts import AttrDict
from btx.misc.metrology import offset_geom
from btx.interfaces.istream import *
from btx.interfaces.ischeduler import *
from btx.processing.indexer import Indexer
from btx.processing.merge import *

class Geoptimizer:
    """
    Class for refining the geometry. 
    """
    def __init__(self, queue, task_dir, scan_dir, runs, input_geom, dx_scan, dy_scan, dz_scan):

        self.queue = queue # queue to submit jobs to, str
        self.task_dir = task_dir # path to indexing directory, str
        self.scan_dir = scan_dir # path to scan directory, str
        self.runs = runs # run(s) to process, array
        
        self.timeout = 64800 # number of seconds to allow sbatch'ed jobs to run before exiting, float
        self.frequency = 5 # frequency in seconds to check on job completion, float
        self._write_geoms(input_geom, dx_scan, dy_scan, dz_scan)
        
    def _write_geoms(self, input_geom, dx_scan, dy_scan, dz_scan):
        """
        Write a geometry file for each combination of offsets in dx,dy,dz.
        
        Parameters
        ----------
        input_geom : str
            path to original geometry
        dx_scan : numpy.array, 1d
            offsets along x (corner_x) to scan in pixels
        dy_scan : numpy.array, 1d
            offsets along y (corner_y) to scan in pixels
        dz_scan : numpy.array, 1d
            offsets along dz (coffset) to scan in mm
        """
        os.makedirs(os.path.join(self.scan_dir, "geom"), exist_ok=True)
        shifts_list = np.array(list(itertools.product(dx_scan,dy_scan,dz_scan)))

        for i,deltas in enumerate(shifts_list):
            output_geom = os.path.join(self.scan_dir, f"geom/shift{i}.geom")
            offset_geom(input_geom, output_geom, deltas[0], deltas[1], deltas[2])
            
        self.scan_results = np.zeros((len(shifts_list), 12))
        self.scan_results[:,:3] = shifts_list
        self.cols = ['dx', 'dy', 'dz']
            
    def check_status(self, statusfile, jobnames, debug=False):
        """
        Check whether all launched jobs have completed.
        
        Parameters
        ----------
        statusfile : str
            path to file that lists completed jobnames
        jobnames : list of str
            list of all jobnames launched
        """
        done = False
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            if os.path.exists(statusfile) and not done:

                with open(statusfile, "r") as f:
                    lines = f.readlines()
                    finished = [l.strip('\n') for l in lines]
                    if set(finished) == set(jobnames):
                        print(f"All done: {jobnames}")
                        done = True
                        os.remove(statusfile)
                        time.sleep(self.frequency*5)
                        break                    
                time.sleep(self.frequency)

    def launch_indexing(self, exp, det_type, params, cell_file):
        """                                                                                         
        Launch indexing jobs.                                                        

        Parameters                                                                       
        ----------
        exp : str
            name of experiment
        det_type : str
            name of detector
        params : btx.misc.shortcuts.AttrDict 
            config.index object containing indexing parameters
        cell_file : str 
            path to cell file 
        """
        jobnames = list()
        statusfile = os.path.join(self.scan_dir,"status.sh")

        for run in self.runs:

            os.makedirs(os.path.join(self.scan_dir, f"r{run:04}"), exist_ok=True)
            for num in range(self.scan_results.shape[0]):

                jobname = f"r{run}_g{num}"
                jobfile = os.path.join(self.scan_dir, f"idx_{jobname}.sh")
                stream = os.path.join(self.scan_dir, f'r{run:04}/r{run:04}_g{num}.stream')
                gfile = os.path.join(self.scan_dir, f"geom/shift{num}.geom")

                idxr = Indexer(exp=exp, run=run, det_type=det_type, tag=params.tag, tag_cxi=params.get('tag_cxi'), taskdir=self.task_dir, 
                               geom=gfile, cell=cell_file, int_rad=params.int_radius, methods=params.methods, tolerance=params.tolerance, no_revalidate=params.no_revalidate, 
                               multi=params.multi, profile=params.profile, queue=self.queue, ncores=params.get('ncores') if params.get('ncores') is not None else 64,
                               time=params.get('time') if params.get('time') is not None else '1:00:00')
                idxr.tmp_exe = jobfile
                idxr.stream = stream
                idxr.launch(addl_command=f"echo {jobname} | tee -a {statusfile}\n",
                            dont_report=True)                
                jobnames.append(jobname)

        self.check_status(statusfile, jobnames)

    def launch_stream_analysis(self, cell_file):
        """
        Compute the number of indexed crystals and standard deviations of unit
        cell parameters and concatenate, with one stream file per geometry file. 
        
        Parameters
        ----------
        cell_file : str
            path to cell file used during indexing
        """
        celldir = os.path.join(self.scan_dir, "cell")
        os.makedirs(celldir, exist_ok=True)

        for num in range(self.scan_results.shape[0]):
            stream_files = os.path.join(self.scan_dir, f"r*/r*_g{num}.stream")
            st = StreamInterface(glob.glob(stream_files), cell_only=True)
            write_cell_file(st.cell_params, os.path.join(celldir, f"g{num}.cell"), input_file=cell_file)
            stream_cat = os.path.join(self.scan_dir, f'g{num}.stream')
            os.system(f"cat {stream_files} >> {stream_cat}")
            
            self.scan_results[num,3:10] = np.append(np.array(len(st.unq_inds)), st.cell_params_std) # num indexed, cell std devs
        
        self.cols.extend(['n_indexed', 'a', 'b', 'c', 'alpha', 'beta', 'gamma'])
            
    def launch_merging(self, params):
        """
        Launch merging and hkl stats jobs.
        
        Parameters
        ----------
        params : btx.misc.shortcuts.AttrDict
            config.merge dictionary containing merging parameters
        """
        mergedir = os.path.join(self.scan_dir, "merge")
        os.makedirs(mergedir, exist_ok=True)
        for file_extension in ["*.hkl*", "*.dat"]:
            filelist = glob.glob(os.path.join(mergedir, file_extension))
            if len(filelist) > 0:
                for filename in filelist:
                    os.remove(filename)

        jobnames = list()
        statusfile = os.path.join(self.scan_dir,"status.sh")
        
        for num in range(self.scan_results.shape[0]):
                
            jobname = f"g{num}"
            jobfile = os.path.join(self.scan_dir, f"merge_{jobname}.sh")
            instream = os.path.join(self.scan_dir, f'g{num}.stream')
            cellfile = os.path.join(self.scan_dir, f"cell/g{num}.cell")
            
            stream_to_mtz = StreamtoMtz(instream, params.symmetry, mergedir, cellfile, queue=self.queue, 
                                        ncores=params.get('ncores') if params.get('ncores') is not None else 16)
            stream_to_mtz.cmd_partialator(iterations=params.iterations, model=params.model, 
                                          min_res=params.get('min_res'), push_res=params.get('push_res'))
            stream_to_mtz.cmd_compare_hkl(foms=params.foms.split(" "), nshells=1, highres=params.get('highres'))
            stream_to_mtz.cmd_get_hkl(highres=params.get('highres'))
            stream_to_mtz.js.write_main(f"echo {jobname} | tee -a {statusfile}\n")
            stream_to_mtz.launch()

            jobnames.append(jobname)
            time.sleep(self.frequency)
            
        self.check_status(statusfile, jobnames, debug=True)
        self.extract_stats(os.path.join(mergedir,"hkl"), params.foms.split(' '))
        
    def extract_stats(self, statsdir, foms):
        """
        Extract the overall figure of merit statistics.
        
        Parameters
        ----------
        statsdir : str
            directory containing results of crystfel's compare_hkl
        foms : list of str
            figures of merit that were calculated
        """
        for col,fom in enumerate(foms):
            self.cols.append(fom)
            overall_stat = np.zeros(self.scan_results.shape[0])
            
            for num in range(self.scan_results.shape[0]):
                shells_file = os.path.join(statsdir, f"g{num}_{fom}_n1.dat")
                fom_from_shell_file, stat = wrangle_shells_dat(shells_file)
                overall_stat[num] = stat
            self.scan_results[:,10+col] = overall_stat

    def _transfer(self, root_dir, tag, num):
        """
        Transfer the mtz and associated geom and cell to their 
        main associated folders.
        
        Parameters
        ----------
        root_dir : str 
             path to root directory containing geom, cell, mtz folders
        tag : str
            prefix to use when naming transferred files / sample name 
        num : int 
            index of the scan result to transfer 
        """
        geom_opt = os.path.join(self.scan_dir, f"geom/shift{num}.geom")
        geom_new = os.path.join(root_dir, "geom", f"r{self.runs[0]:04}.geom")

        cell_opt = os.path.join(self.scan_dir, f"cell/g{num}.cell")
        cell_new = os.path.join(root_dir, "cell", f"{tag}.cell")
        
        mtz_opt = os.path.join(self.scan_dir, f"merge/g{num}.mtz")
        mtz_new = os.path.join(root_dir, "solve", f"{tag}.mtz")
        os.makedirs(os.path.join(root_dir, "solve"), exist_ok=True)

        for opt,new in zip([geom_opt,cell_opt,mtz_opt],[geom_new,cell_new,mtz_new]):
            if os.path.exists(new):
                shutil.move(new, f"{new}.old")
            shutil.copy2(opt, new)

    def save_results(self, root_dir, tag, savepath=None, metric='Rsplit'):
        """
        Save the results of the scan to a text file.

        Parameters
        ----------
        root_dir : str 
             path to root directory containing geom, cell, mtz folders
        tag : str
            prefix to use when naming transferred files / sample name 
        savepath : str
            text file to save results to
        metric : str
            choose optimal mtz based on max CCstar or min Rsplit
        """
        if savepath is None:
            savepath = os.path.join(self.scan_dir, "results.txt")

        fmt=['%.2f', '%.2f', '%.4f', '%d', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f', '%.2f', '%.2f']
        np.savetxt(savepath, self.scan_results, header=' '.join(self.cols), fmt=fmt)

        col_index = self.cols.index(metric)
        if metric=='Rsplit':
            min_val = min([val for val in self.scan_results[:,col_index] if val>0])
            row_index = np.where(self.scan_results[:,col_index] == min_val)[0][0]
        elif metric == 'CCstar':
            row_index = np.nanargmax(self.scan_results[:,col_index])
        print(f"Selected g{row_index}.hkl, with {metric} of {self.scan_results[row_index,col_index]}")
        self._transfer(root_dir, tag, row_index)

def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='Config file in yaml format')
    parser.add_argument('-g', '--geom', required=True, type=str, help='CrystFEL .geom file with starting metrology')
    parser.add_argument('-q', '--queue', required=True, type=str, help='Queue to submit jobs to')
    parser.add_argument('--runs', required=True, type=int, nargs=2, help='Runs to process: [start,end]')
    parser.add_argument('--dx', required=False, type=float, nargs=3, default=[0,0,1], help='xoffset translational scan in pixels: start, stop, n_points')
    parser.add_argument('--dy', required=False, type=float, nargs=3, default=[0,0,1], help='yoffset translational scan in pixels: start, stop, n_points')
    parser.add_argument('--dz', required=False, type=float, nargs=3, default=[0,0,1], help='coffset (distance) scan in mm: start, stop, n_points')
    parser.add_argument('--merge_only', help='Runs already indexed; perform merging only', action='store_true')
    
    return parser.parse_args()

if __name__ == '__main__':

    params = parse_input()
    with open(params.config, "r") as config_file:
        config = AttrDict(yaml.safe_load(config_file))

    taskdir = os.path.join(config.setup.root_dir, "index")
    scandir = os.path.join(config.setup.root_dir, "scan")
    os.makedirs(scandir, exist_ok=True)

    geopt = Geoptimizer(params.queue,
                        taskdir,
                        scandir,
                        np.arange(params.runs[0], params.runs[1]+1),
                        params.geom,
                        np.linspace(params.dx[0], params.dx[1], int(params.dx[2])),
                        np.linspace(params.dy[0], params.dy[1], int(params.dy[2])),
                        np.linspace(params.dz[0], params.dz[1], int(params.dz[2])))
    if not params.merge_only:
        cell_file = os.path.join(config.setup.root_dir, "cell", f"{config.index.tag}.cell")
        if not os.path.isfile(cell_file):
            cell_file = config.index.get('cell')
        geopt.launch_indexing(config.setup.exp, config.setup.det_type, config.index, cell_file)
        geopt.launch_stream_analysis(cell_file)
    geopt.launch_merging(config.merge)
    geopt.save_results(config.setup.root_dir, config.merge.tag)
