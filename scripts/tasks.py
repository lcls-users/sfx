import logging
import os
import requests
import glob
import shutil
import numpy as np
import itertools
import h5py
import time
import yaml
import csv
from mpi4py import MPI
import subprocess
import socket
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch the URL to post progress update
update_url = os.environ.get('JID_UPDATE_COUNTERS')

def test(config):
    print(config)
    requests.post(update_url, json=[ { "key": "root_dir", "value": f"{config.setup.root_dir}" } ])

def fetch_mask(config):
    from btx.interfaces.imask import MaskInterface
    setup = config.setup
    task = config.fetch_mask
    """ Fetch most recent mask for this detector from mrxv. """
    if 'sdf' in setup.root_dir:
        mrxv_path = '/sdf/group/lcls/ds/tools/mrxv/masks/'
    elif 'cds' in setup.root_dir:
        mrxv_path = '/cds/sw/package/autosfx/mrxv/masks/'
    taskdir = os.path.join(setup.root_dir, 'mask')
    os.makedirs(taskdir, exist_ok=True)
    mi = MaskInterface(exp=setup.exp,
                       run=setup.run,
                       det_type=setup.det_type)
    mi.retrieve_from_mrxv(mrxv_path=mrxv_path, dataset=task.dataset)
    logger.info(f'Saving mrxv mask to {taskdir}')
    mi.save_mask(os.path.join(taskdir, f'r0000.npy'))
    logger.debug('Done!')

def fetch_geom(config):
    from btx.misc.metrology import retrieve_from_mrxv
    setup = config.setup
    task = config.fetch_geom
    """ Fetch latest geometry for this detector from mrxv. """
    taskdir = os.path.join(setup.root_dir, 'geom')
    if 'sdf' in setup.root_dir:
        mrxv_path = '/sdf/group/lcls/ds/tools/mrxv/geometries/'
    elif 'cds' in setup.root_dir:
        mrxv_path = '/cds/sw/package/autosfx/mrxv/geometries/'
    os.makedirs(taskdir, exist_ok=True)
    logger.info(f'Saving mrxv geom to {taskdir}')
    retrieve_from_mrxv(det_type=setup.det_type, out_geom=os.path.join(taskdir, f'r0000.geom'),
                       mrxv_path=mrxv_path)
    logger.debug('Done!')

def build_mask(config):
    from btx.interfaces.imask import MaskInterface
    from btx.misc.shortcuts import fetch_latest
    setup = config.setup
    task = config.build_mask
    """ Generate a mask by thresholding events from a psana run. """
    taskdir = os.path.join(setup.root_dir, 'mask')
    os.makedirs(taskdir, exist_ok=True)
    mi = MaskInterface(exp=setup.exp,
                       run=setup.run,
                       det_type=setup.det_type)
    if task.combine:
        mask_file = fetch_latest(fnames=os.path.join(taskdir, 'r*.npy'), run=setup.run)
        mi.load_mask(mask_file, mask_format='psana')
        logger.debug(f'New mask will be combined with {mask_file}')
    task.thresholds = tuple([float(elem) for elem in task.thresholds.split()])
    mi.generate_from_psana_run(thresholds=task.thresholds, n_images=task.n_images, n_edge=task.n_edge)
    logger.info(f'Saving newly generated mask to {taskdir}')
    mi.save_mask(os.path.join(taskdir, f'r{mi.psi.run:04}.npy'))
    logger.debug('Done!')

def run_analysis(config):
    from btx.interfaces.ischeduler import JobScheduler
    from btx.misc.shortcuts import fetch_latest
    setup = config.setup
    task = config.run_analysis
    """ Generate powders for a given run and plot traces of run statistics. """
    taskdir = os.path.join(setup.root_dir, 'powder')
    os.makedirs(taskdir, exist_ok=True)
    os.makedirs(os.path.join(taskdir, 'figs'), exist_ok=True)
    mask_file = fetch_latest(fnames=os.path.join(setup.root_dir, 'mask', 'r*.npy'), run=setup.run)
    script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../btx/diagnostics/run.py")
    command = f"python {script_path}"
    command += f" -e {setup.exp} -r {setup.run} -d {setup.det_type} -o {taskdir}"
    if mask_file:
        command += f" -m {mask_file}"
    if task.get('mean_threshold') is not None:
        command += f" --mean_threshold={task.mean_threshold}"
    if task.get('gain_mode') is not None:
        command += f" --gain_mode={task.gain_mode}"
    if task.get('raw_img') is not None:
        if task.raw_img:
            command += f" --raw_img"
    if task.get('outlier_threshold') is not None:
        command += f" --outlier_threshold={task.outlier_threshold}" 
    js = JobScheduler(os.path.join(".", f'ra_{setup.run:04}.sh'), 
                      queue=setup.queue,
                      ncores=task.ncores,
                      jobname=f'ra_{setup.run:04}',
                      account=setup.account,
                      reservation=setup.reservation)
    js.write_header()
    js.write_main(f"{command}\n", dependencies=['psana'])
    js.clean_up()
    js.submit()
    logger.debug('Run analysis launched!')

def opt_geom(config):
    from btx.diagnostics.geom_opt import GeomOpt
    from btx.misc.shortcuts import fetch_latest
    setup = config.setup
    task = config.opt_geom
    """ Optimize and deploy the detector geometry from a silver behenate run. """
    taskdir = os.path.join(setup.root_dir, 'geom')
    os.makedirs(taskdir, exist_ok=True)
    os.makedirs(os.path.join(taskdir, 'figs'), exist_ok=True)
    mask_file = fetch_latest(fnames=os.path.join(setup.root_dir, 'mask', 'r*.npy'), run=setup.run)
    task.dx = tuple([float(elem) for elem in task.dx.split()])
    task.dx = np.linspace(task.dx[0], task.dx[1], int(task.dx[2]))
    task.dy = tuple([float(elem) for elem in task.dy.split()])
    task.dy = np.linspace(task.dy[0], task.dy[1], int(task.dy[2]))
    centers = list(itertools.product(task.dx, task.dy))
    if type(task.n_peaks) == int:
        task.n_peaks = [int(task.n_peaks)]
    else:
        task.n_peaks = [int(elem) for elem in task.n_peaks.split()]
    if task.get('distance') is None:
        task.distance = None
    elif type(task.distance) == float or type(task.distance) == int:
        task.distance = [float(task.distance)]
    else:
        task.distance = [float(elem) for elem in task.distance.split()]
    geom_opt = GeomOpt(exp=setup.exp,
                       run=setup.run,
                       det_type=setup.det_type)
    geom_opt.opt_geom(powder=os.path.join(setup.root_dir, f"powder/r{setup.run:04}_max.npy"),
                      mask=mask_file,
                      distance=task.distance,
                      center=centers,
                      n_iterations=task.get('n_iterations'),
                      n_peaks=task.n_peaks,
                      threshold=task.get('threshold'),
                      deltas=True,
                      plot=os.path.join(taskdir, f'figs/r{setup.run:04}.png'),
                      plot_final_only=True)
    if geom_opt.rank == 0:
        try:
            geom_opt.report(update_url)
        except:
            logger.debug("Could not communicate with the elog update url")
        logger.info(f'Refined detector distance in mm: {geom_opt.distance}')
        logger.info(f'Refined detector center in pixels: {geom_opt.center}')
        logger.info(f'Detector edge resolution in Angstroms: {geom_opt.edge_resolution}')    
        geom_opt.deploy_geometry(taskdir, pv_camera_length=setup.get('pv_camera_length'))
        logger.info(f'Updated geometry files saved to: {taskdir}')
        logger.debug('Done!')

def find_peaks(config):
    from btx.processing.peak_finder import PeakFinder
    from btx.misc.shortcuts import fetch_latest
    from btx.interfaces.ielog import update_summary
    setup = config.setup
    task = config.find_peaks
    """ Perform adaptive peak finding on run. """
    taskdir = os.path.join(setup.root_dir, 'index')
    os.makedirs(taskdir, exist_ok=True)
    mask_file = fetch_latest(fnames=os.path.join(setup.root_dir, 'mask', 'r*.npy'), run=setup.run)
    pf = PeakFinder(exp=setup.exp, run=setup.run, det_type=setup.det_type, outdir=os.path.join(taskdir ,f"r{setup.run:04}"),
                    event_receiver=setup.get('event_receiver'), event_code=setup.get('event_code'), event_logic=setup.get('event_logic'),
                    tag=task.tag, mask=mask_file, psana_mask=task.psana_mask, min_peaks=task.min_peaks, max_peaks=task.max_peaks,
                    npix_min=task.npix_min, npix_max=task.npix_max, amax_thr=task.amax_thr, atot_thr=task.atot_thr,
                    son_min=task.son_min, peak_rank=task.peak_rank, r0=task.r0, dr=task.dr, nsigm=task.nsigm,
                    calibdir=task.get('calibdir'), pv_camera_length=setup.get('pv_camera_length'),
                    pypca_model_filename=task.get('pypca_model_filename'), pypca_reduced_filename=task.get('pypca_reduced_filename'))
    logger.info(f'Performing peak finding for run {setup.run} of {setup.exp}...')
    pf.find_peaks()
    pf.curate_cxi()
    pf.summarize()
    logger.info(f'Saving CXI files and summary to {taskdir}/r{setup.run:04}')
    logger.debug('Done!')

    if pf.rank == 0:
        summary_file = f'{setup.root_dir}/summary_r{setup.run:04}.json'
        update_summary(summary_file, pf.pf_summary)

def find_peaks_multiple_runs(config):
    from btx.interfaces.ischeduler import JobScheduler
    from btx.misc.shortcuts import fetch_latest
    setup = config.setup
    task = config.find_peaks
    bay_opt = config.bayesian_optimization
    """ Perform adaptive peak finding on multiple runs. """
    taskdir = os.path.join(setup.root_dir, 'index')
    os.makedirs(taskdir, exist_ok=True)
    script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../btx/processing/peak_finder.py")
    
    logger.info(f'Launching one slurm job to perform peak finding for each run in interval [{bay_opt.first_run}, {bay_opt.last_run}]') 
    for iter_run in range(bay_opt.first_run, bay_opt.last_run + 1):
        # Write the command by specifying all the arguments that can be found in the config
        command = f"python {script_path}"
        command += f" -e {setup.exp} -r {iter_run} -d {setup.det_type} -o {taskdir} -t {task.tag}"
        mask_file = fetch_latest(fnames=os.path.join(setup.root_dir, 'mask', 'r*.npy'), run=iter_run)
        if mask_file:
            command += f" -m {mask_file}"
        if task.get('event_receiver') is not None:
            command += f" --event_receiver={task.event_receiver}"
        if task.get('event_code') is not None:
            command += f" --event_code={task.event_code}"
        if task.get('event_logic') is not None:
            command += f" --event_logic={task.event_logic}"   
        if task.get('psana_mask') is not None:
            command += f" --psana_mask={task.psana_mask}"   
        if task.get('pv_camera_length') is not None:
            command += f" --pv_camera_length={task.pv_camera_length}"
        if task.get('min_peaks') is not None:
            command += f" --min_peaks={task.min_peaks}"
        if task.get('max_peaks') is not None:
            command += f" --max_peaks={task.max_peaks}"
        if task.get('npix_min') is not None:
            command += f" --npix_min={task.npix_min}"
        if task.get('npix_max') is not None:
            command += f" --npix_max={task.npix_max}"
        if task.get('amax_thr') is not None:
            command += f" --amax_thr={task.amax_thr}"
        if task.get('atot_thr') is not None:
            command += f" --atot_thr={task.atot_thr}"
        if task.get('son_min') is not None:
            command += f" --son_min={task.son_min}"
        if task.get('peak_rank') is not None:
            command += f" --peak_rank={task.peak_rank}"
        if task.get('r0') is not None:
            command += f" --r0={task.r0}"
        if task.get('dr') is not None:
            command += f" --dr={task.dr}"
        if task.get('nsigm') is not None:
            command += f" --nsigm={task.nsigm}"
        if task.get('calibdir') is not None:
            command += f" --calibdir={task.calibdir}"
        # Launch the Slurm job to perform "find_peaks" on the current run
        js = JobScheduler(os.path.join(".", f'fp_{iter_run:04}.sh'), 
                        queue=setup.queue,
                        ncores=bay_opt.ncores if bay_opt.get('ncores') is not None else 64,
                        jobname=f'fp_{iter_run:04}',
                        account=setup.account,
                        reservation=setup.reservation)
        js.write_header()
        js.write_main(f"{command}\n", dependencies=['psana'])
        js.clean_up()
        js.submit()
        logger.info(f'Launched a slurm job to perform peak finding for run {iter_run} \
                    in interval [{bay_opt.first_run}, {bay_opt.last_run}] of experiment {setup.exp}...')
    
    logger.info('All slurm jobs have been launched!')
    logger.info('Done!')
    

def index(config):
    from btx.processing.indexer import Indexer
    from btx.misc.shortcuts import fetch_latest
    setup = config.setup
    task = config.index
    """ Index run using indexamajig. """
    taskdir = os.path.join(setup.root_dir, 'index')
    geom_file = fetch_latest(fnames=os.path.join(setup.root_dir, 'geom', 'r*.geom'), run=setup.run)
    indexer_obj = Indexer(exp=config.setup.exp, run=config.setup.run, det_type=config.setup.det_type, tag=task.tag, tag_cxi=task.get('tag_cxi'), taskdir=taskdir,
                          geom=geom_file, cell=task.get('cell'), int_rad=task.int_radius, methods=task.methods, tolerance=task.tolerance, no_revalidate=task.no_revalidate,
                          multi=task.multi, profile=task.profile, queue=setup.get('queue'), ncores=task.get('ncores') if task.get('ncores') is not None else 64,
                          time=task.get('time') if task.get('time') is not None else '1:00:00', mpi_init = False, slurm_account=setup.account,
                          slurm_reservation=setup.reservation)
    logger.debug(f'Generating indexing executable for run {setup.run} of {setup.exp}...')
    indexer_obj.launch()
    logger.info(f'Indexing launched!')

def index_multiple_runs(config):
    from btx.processing.indexer import Indexer
    from btx.misc.shortcuts import fetch_latest
    setup = config.setup
    task = config.index
    bay_opt = config.bayesian_optimization
    """ Index multiple runs using indexamajig. """
    taskdir = os.path.join(setup.root_dir, 'index')

    logger.info(f'Launching indexing for each run in interval [{bay_opt.first_run}, {bay_opt.last_run}]') 
    for iter_run in range(bay_opt.first_run, bay_opt.last_run + 1):
        geom_file = fetch_latest(fnames=os.path.join(setup.root_dir, 'geom', 'r*.geom'), run=iter_run)
        indexer_obj = Indexer(exp=config.setup.exp, run=iter_run, det_type=config.setup.det_type, tag=task.tag, tag_cxi=task.get('tag_cxi'), taskdir=taskdir,
                            geom=geom_file, cell=task.get('cell'), int_rad=task.int_radius, methods=task.methods, tolerance=task.tolerance, no_revalidate=task.no_revalidate,
                            multi=task.multi, profile=task.profile, queue=setup.get('queue'), ncores=task.get('ncores') if task.get('ncores') is not None else 64,
                            time=task.get('time') if task.get('time') is not None else '1:00:00', mpi_init = False, slurm_account=setup.account,
                            slurm_reservation=setup.reservation)
        logger.debug(f'Generating indexing executable for run {iter_run} \
                     in interval [{bay_opt.first_run}, {bay_opt.last_run}] of experiment {setup.exp}...')
        indexer_obj.launch()
        logger.info(f'Indexing for run {iter_run} in interval [{bay_opt.first_run}, {bay_opt.last_run}] launched!')

    logger.info('All slurm jobs have been launched!')
    logger.info('Done!')
        


def summarize_idx(config):
    import subprocess
    from mpi4py import MPI
    from btx.interfaces.ielog import update_summary

    # Only run on rank 0
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # Pull from config yaml
        setup = config.setup
        task = config.index
        tag = task.tag
        tag_cxi = ''
        cxi = task.get('tag_cxi')
        if cxi:
            tag_cxi = cxi
        run = setup.run

        # Define paths
        taskdir = os.path.join(setup.root_dir, 'index')
        summary_file = f'{setup.root_dir}/summary_r{setup.run:04}.json'
        stream_file = os.path.join(taskdir, f'r{run:04}_{tag}.stream')
        pf_summary = os.path.join(taskdir, f'r{run:04}/peakfinding{tag_cxi}.summary')

        command1: list = ["grep", "Cell parameters", f"{stream_file}"]
        command2: list = ["grep",
                          "Number of hits found",
                          f"{pf_summary}"]

        summary_dict: dict = {}
        try:
            output: str = subprocess.check_output(command1,
                                                  universal_newlines=True)
            n_indexed: int = len(output.split('\n')[:-1])

            output: str = subprocess.check_output(command2,
                                                  universal_newlines=True)
            n_total: int = int(output.split(':')[1].split('\n')[0])

            key_strings: list = ['Number of lattices found',
                                 ('Fractional indexing rate'
                                  ' (including multiple lattices)')]
            summary_dict: dict = { key_strings[0] : f'{n_indexed}',
                                   key_strings[1] : f'{(n_indexed/n_total):.2f}' }
        except subprocess.CalledProcessError as err:
            print(err)

        update_summary(summary_file, summary_dict)
        post_to_elog(config)

def stream_analysis(config):
    from btx.interfaces.istream import launch_stream_analysis
    setup = config.setup
    task = config.stream_analysis
    """ Plot cell distribution and peakogram, write new cell file, and concatenate streams. """
    taskdir = os.path.join(setup.root_dir, 'index')
    os.makedirs(os.path.join(taskdir, 'figs'), exist_ok=True)
    os.makedirs(os.path.join(setup.root_dir, 'cell'), exist_ok=True)
    launch_stream_analysis(os.path.join(taskdir, f"r*{task.tag}.stream"),
                           os.path.join(taskdir, f"{task.tag}.stream"),
                           os.path.join(taskdir, 'figs'),
                           os.path.join(taskdir, "stream_analysis.sh"),
                           setup.queue,
                           ncores=task.get('ncores') if task.get('ncores') is not None else 6,
                           cell_only=task.get('cell_only') if task.get('cell_only') is not None else False,
                           cell_out=os.path.join(setup.root_dir, 'cell', f'{task.tag}.cell'),
                           cell_ref=task.get('ref_cell'),
                           slurm_account=setup.account,
                           slurm_reservation=setup.reservation)
    logger.info(f'Stream analysis launched')

def determine_cell(config):
    from btx.interfaces.istream import StreamInterface, write_cell_file, cluster_cell_params
    setup = config.setup
    task = config.determine_cell
    """ Cluster crystals from cell-free indexing and write most-frequently found cell to CrystFEL cell file. """
    taskdir = os.path.join(setup.root_dir, 'index')
    stream_files = os.path.join(taskdir, f"r*{task.tag}.stream")
    logger.info(f"Processing files {glob.glob(stream_files)}")
    st = StreamInterface(input_files=glob.glob(stream_files), cell_only=True)
    if st.rank == 0:
        logger.debug(f'Read stream files: {stream_files}')
        celldir = os.path.join(setup.root_dir, 'cell')
        os.makedirs(celldir, exist_ok=True)
        keys = ['a','b','c','alpha','beta','gamma']
        cell = np.array([st.stream_data[key] for key in keys])
        labels = cluster_cell_params(cell.T,
                                     os.path.join(taskdir, f"clusters_{task.tag}.txt"),
                                     os.path.join(celldir, f"{task.tag}.cell"),
                                     in_cell=task.get('input_cell'),
                                     eps=task.get('eps') if task.get('eps') is not None else 5,
                                     min_samples=task.get('min_samples') if task.get('min_samples') is not None else 5)
        logger.info(f'Wrote updated CrystFEL cell file for sample {task.tag} to {celldir}')
        logger.debug('Done!')

def merge(config):
    from btx.processing.merge import StreamtoMtz
    setup = config.setup
    task = config.merge
    """ Merge reflections from stream file and convert to mtz. """
    taskdir = os.path.join(setup.root_dir, 'merge', f'{task.tag}')
    input_stream = os.path.join(setup.root_dir, f"index/{task.tag}.stream")
    cellfile = os.path.join(setup.root_dir, f"cell/{task.tag}.cell")
    foms = task.foms.split(" ")
    stream_to_mtz = StreamtoMtz(input_stream, task.symmetry, taskdir, cellfile, queue=setup.get('queue'),
                                ncores=task.get('ncores') if task.get('ncores') is not None else 16,
                                mtz_dir=os.path.join(setup.root_dir, "solve", f"{task.tag}"),
                                anomalous=task.get('anomalous') if task.get('anomalous') is not None else False,
                                slurm_account=setup.account,
                                slurm_reservation=setup.reservation)
    stream_to_mtz.cmd_partialator(iterations=task.iterations, model=task.model,
                                  min_res=task.get('min_res'), push_res=task.get('push_res'), max_adu=task.get('max_adu'))
    for ns in [1, task.nshells]:
        stream_to_mtz.cmd_compare_hkl(foms=foms, nshells=ns, highres=task.get('highres'))
    stream_to_mtz.cmd_hkl_to_mtz(highres=task.get('highres'),
                                 space_group=task.get('space_group') if task.get('space_group') is not None else 1)
    stream_to_mtz.cmd_report(foms=foms, nshells=task.nshells)
    stream_to_mtz.launch()
    logger.info(f'Merging launched!')

def solve(config):
    from btx.interfaces.imtz import run_dimple
    setup = config.setup
    task = config.solve
    """ Run the CCP4 dimple pipeline for structure solution and refinement. """
    taskdir = os.path.join(setup.root_dir, "solve", f"{task.tag}")
    run_dimple(os.path.join(taskdir, f"{task.tag}.mtz"),
               task.pdb,
               taskdir,
               queue=setup.get('queue'),
               ncores=task.get('ncores') if task.get('ncores') is not None else 16,
               anomalous=task.get('anomalous') if task.get('anomalous') is not None else False,
               slurm_account=setup.account,
               slurm_reservation=setup.reservation)
    logger.info(f'Dimple launched!')

def refine_geometry(config, task=None):
    from btx.diagnostics.geoptimizer import Geoptimizer
    from btx.misc.shortcuts import fetch_latest, check_file_existence
    setup = config.setup
    if task is None:
        task = config.refine_geometry
        task.scan_dir = os.path.join(setup.root_dir, f'scan_{config.merge.tag}')
        task.dx = tuple([float(elem) for elem in task.dx.split()])
        task.dx = np.linspace(task.dx[0], task.dx[1], int(task.dx[2]))
        task.dy = tuple([float(elem) for elem in task.dy.split()])
        task.dy = np.linspace(task.dy[0], task.dy[1], int(task.dy[2]))
        task.dz = tuple([float(elem) for elem in task.dz.split()])
        task.dz = np.linspace(task.dz[0], task.dz[1], int(task.dz[2]))
    """ Refine detector center and/or distance based on the geometry that minimizes Rsplit. """
    taskdir = os.path.join(setup.root_dir, 'index')
    os.makedirs(task.scan_dir, exist_ok=True)
    task.runs = tuple([int(elem) for elem in task.runs.split()])
    if len(task.runs) == 2:
        task.runs = (*task.runs, 1)
    geom_file = fetch_latest(fnames=os.path.join(setup.root_dir, 'geom', 'r*.geom'), run=task.runs[0])
    cell_file = os.path.join(config.setup.root_dir, "cell", f"{config.index.tag}.cell")
    if not os.path.isfile(cell_file):
        cell_file = config.index.get('cell')
    logger.info(f'Scanning around geometry file {geom_file}')
    geopt = Geoptimizer(setup.queue,
                        taskdir,
                        task.scan_dir,
                        np.arange(task.runs[0], task.runs[1]+1, task.runs[2]),
                        geom_file,
                        task.dx,
                        task.dy,
                        task.dz,
                        slurm_account=setup.account,
                        slurm_reservation=setup.reservation
    )
    geopt.launch_indexing(setup.exp, setup.det_type, config.index, cell_file)
    geopt.launch_stream_wrangling(config.stream_analysis)
    geopt.launch_merging(config.merge)
    geopt.save_results(setup.root_dir, config.merge.tag)
    check_file_existence(os.path.join(task.scan_dir, "results.txt"), geopt.timeout)
    logger.debug('Done!')

def refine_center(config):
    """ Wrapper for the refine_geometry task, searching for the detector center. """
    setup = config.setup
    task = config.refine_center
    task.scan_dir = os.path.join(setup.root_dir, f'scan_center_{config.merge.tag}')
    task.dx = tuple([float(elem) for elem in task.dx.split()])
    task.dx = np.linspace(task.dx[0], task.dx[1], int(task.dx[2]))
    task.dy = tuple([float(elem) for elem in task.dy.split()])
    task.dy = np.linspace(task.dy[0], task.dy[1], int(task.dy[2]))
    task.dz = [0]
    refine_geometry(config, task)

def refine_distance(config):
    """ Wrapper for the refine_geometry task, searching for the detector distance. """
    setup = config.setup
    task = config.refine_distance
    task.scan_dir = os.path.join(setup.root_dir, f'scan_distance_{config.merge.tag}')
    task.dx, task.dy = [0], [0]
    task.dz = tuple([float(elem) for elem in task.dz.split()])
    task.dz = np.linspace(task.dz[0], task.dz[1], int(task.dz[2]))
    refine_geometry(config, task)

def elog_display(config):
    from btx.interfaces.ielog import eLogInterface
    setup = config.setup
    """ Updates the summary page in the eLog with most recent results. """
    logger.info(f'Updating the reports in the eLog summary tab.')
    eli = eLogInterface(setup)
    eli.update_summary(plot_type='holoviews')
    logger.debug('Done!')

def post_to_elog(config):
    from btx.interfaces.ielog import elog_report_post
    setup = config.setup
    root_dir = setup.root_dir
    run = setup.run

    summary_file = f'{root_dir}/summary_r{run:04}.json'
    url = os.environ.get('JID_UPDATE_COUNTERS')
    if url:
        elog_report_post(summary_file, url)

def visualize_sample(config):
    from btx.misc.visuals import VisualizeSample
    setup = config.setup
    task = config.visualize_sample
    """ Plot per-run cell parameters and peak-finding/indexing statistics. """
    logger.info(f'Extracting statistics from stream and summary files.')
    vs = VisualizeSample(os.path.join(setup.root_dir, "index"),
                         task.tag,
                         save_plots=True)
    vs.plot_cell_trajectory()
    vs.plot_stats()
    logger.debug('Done!')

def clean_up(config):
    setup = config.setup
    task = config.clean_up
    taskdir = os.path.join(setup.root_dir, 'index')
    if os.path.isdir(taskdir):
        os.system(f"rm -f {taskdir}/r*/*{task.tag}.cxi")
    logger.debug('Done!')

def plot_saxs(config):
    """! Plot the SAXS profile and associated diagnostic figures."""
    from btx.processing.saxs import SAXSProfiler
    setup = config.setup
    task = config.plot_saxs

    expmt = setup.exp
    run = setup.run
    detector_type = setup.det_type
    rootdir = setup.root_dir
    method = task.method

    saxs = SAXSProfiler(expmt, run, detector_type, rootdir, method)
    saxs.plot_all()

def timetool_diagnostics(config):
    """! Plot timetool diagnostic figures from data in smalldata hdf5 file."""
    from btx.io.ih5 import SmallDataReader
    setup = config.setup
    task = config.timetool_diagnostics
    savedir = os.path.join(setup.root_dir, 'timetool')

    expmt = setup.exp
    run = setup.run

    if not task.h5:
        smdr = SmallDataReader(expmt, run, savedir)
    else:
        smdr = SmallDataReader(expmt, run, savedir, task.h5)

    smdr.plot_timetool_diagnostics(output_type = 'png')

def calibrate_timetool(config):
    from btx.processing.rawimagetimetool import RawImageTimeTool
    setup = config.setup
    task = config.calibrate_timetool
    savedir = os.path.join(setup.root_dir, 'timetool')

    expmt = setup.exp
    run = task.run
    order = int(task.order)
    figs = bool(task.figs)

    tt = RawImageTimeTool(expmt, savedir)
    logger.info(f'Calibrating timetool on run {run}')
    tt.calibrate(run, order, figs)
    logger.info(f'Writing calibration data to {savedir}/calib.')
    if figs:
        logger.info(f'Writing figures to {savedir}/figs.')

def timetool_correct(config):
    from btx.processing.rawimagetimetool import RawImageTimeTool
    from btx.misc.shortcuts import fetch_latest
    setup = config.setup
    task = config.timetool_correct
    savedir = os.path.join(setup.root_dir, 'timetool')

    expmt = setup.exp
    run = task.run
    nominal = float(task.nominal_ps)
    model = task.model
    figs = bool(task.figs)
    tt = RawImageTimeTool(expmt, savedir)

    logger.info('Attempting to correct nominal delays using the timetool data.')
    if model:
        logger.info(f'Using model {model} for the correction.')
    else:
        latest_model = fetch_latest(f'{savedir}/calib/r*.out', int(str(run).split('-')[0]))
        if latest_model:
            model = latest_model
            logger.info(f'Most recent calibration model, {model}, will be used for timetool correction.')
        else:
            logger.info('No model found! Will return the nominal delay uncorrected!')

    tt.timetool_correct(run, nominal, model, figs)

def update_plot(ax, x, y, label):
    ax.clear()
    ax.plot(x, y, label=label)
    ax.legend()
    ax.set_title('Live Plot')
    ax.set_xlabel('Run')
    ax.set_ylabel(label)

def pipca_run(config):
    from btx.processing.pipca import PiPCA
    from btx.misc.pipca_visuals import compute_compression_loss
    from btx.processing.pipca import append_to_dataset
    from btx.processing.pipca import compute_norm_difference
    from btx.processing.pipca import remove_file_with_timeout
    from btx.processing.pipca import initialize_matrices

    setup = config.setup
    task = config.pipca_run
    exp = setup.exp
    run = task.run
    det_type = setup.det_type
    start_offset = task.start_offset
    num_images = task.num_images
    num_components = task.num_components
    batch_size = task.batch_size
    tag = task.tag
    path = task.path
    offline = task.offline

    overwrite = True

    filename_with_tag = f"{path}pipca_model_{tag}.h5"
    
    remove_file_with_timeout(filename_with_tag, overwrite, timeout=10)

    previous_U, previous_S, previous_mu_tot, previous_var_tot = initialize_matrices(filename_with_tag)

    # Initialize number of runs
    # list_runs = get_runs(tag) function to code later

    if offline:
        start_run = run
        num_run = task.num_run
        
    else:
        start_run = run
        num_run = 1

    # Iterate through runs 
    for run in range(start_run, start_run + num_run):
        
        # Create a PiPCA instance for the current run
        pipca = PiPCA(
            exp=exp,
            run=run,
            det_type=det_type,
            num_images=num_images,
            num_components=num_components,
            batch_size=batch_size,
            priming=False,
            downsample=False,
            bin_factor=2,
            filename = filename_with_tag
        )

        # Run iPCA for the current run
        pipca.run_model_full(previous_U, previous_S, previous_mu_tot, previous_var_tot)

        with h5py.File(filename_with_tag, 'r') as f:
            previous_U = f['U'][:]
            previous_S = f['S'][:]
            previous_mu_tot = f['mu'][:]
            previous_var_tot = f['total_variance'][:]

def ipca_pytorch(config):
    from btx.processing.pipca import iPCA_Pytorch
    from btx.processing.pipca import append_to_dataset
    from btx.processing.pipca import compute_norm_difference
    from btx.processing.pipca import remove_file_with_timeout
    from btx.processing.pipca import initialize_matrices

    setup = config.setup
    task = config.ipca_pytorch
    exp = setup.exp
    run = task.run
    det_type = setup.det_type
    start_offset = task.start_offset
    num_images = task.num_images
    num_components = task.num_components
    batch_size = task.batch_size
    tag = task.tag
    path = task.path
    offline = task.offline

    overwrite = True

    filename_with_tag = f"{path}ipca_model_{tag}.h5"
    
    remove_file_with_timeout(filename_with_tag, overwrite, timeout=10)

    # Initialize number of runs
    # list_runs = get_runs(tag) function to code later

    if offline:
        start_run = run
        num_run = task.num_run
        
    else:
        start_run = run
        num_run = 1

    # Iterate through runs 
    for run in range(start_run, start_run + num_run):
        
        # Create a IPCA instance for the current run
        ipca = iPCA_Pytorch(
            exp=exp,
            run=run,
            det_type=det_type,
            num_images=num_images,
            num_components=num_components,
            batch_size=batch_size,
            filename = filename_with_tag
        )

        # Run iPCA for the current run
        ipca.run()

def bayesian_optimization(config):
    from btx.diagnostics.bayesian_optimization import BayesianOptimization
    """ Perform an iteration of the Bayesian optimization. """
    logger.info('Running an iteration of the Bayesian Optimization.')
    BayesianOptimization.run_bayesian_opt(config)
    logger.info('Done!')
    
def bo_init_samples_configs(config):
    from btx.diagnostics.bayesian_optimization import BayesianOptimization
    """ Generates the config files that will be used to generate the initial samples for the Bayesian optimization. """
    BayesianOptimization.init_samples_configs(config, logger)

def bo_aggregate_init_samples(config):
    from btx.diagnostics.bayesian_optimization import BayesianOptimization
    """ Aggregates the scores and parameters of the initial samples of the Bayesian optimization. """
    BayesianOptimization.aggregate_init_samples(config, logger)

def t_sne(config):
    from btx.interfaces.ischeduler import JobScheduler
    from btx.misc.get_max_events import main as compute_max_events
    setup = config.setup
    task = config.t_sne
    num_images = task.num_images
    num_gpus = task.num_gpus
    filename = task.filename
    num_tries = task.num_tries
    threshold = task.threshold
    num_runs = 0
    distribution_images = []
    exp = setup.exp
    run = task.run
    grid_size = task.grid_size
    det_type = setup.det_type
    copy_num_images = num_images
    while num_images > 0:
        max_event = compute_max_events(exp, run+num_runs, det_type)
        images_for_run = min(max_event, num_images)
        distribution_images.append(images_for_run)
        num_images -= images_for_run
        num_runs += 1

    print(f"Number of runs: {num_runs}")
    num_images_str = json.dumps(distribution_images)
    print(f"Number of images: {num_images_str}")

    if task.get('loading_batch_size') is not None:
        loading_batch_size = task.loading_batch_size
    else:
        loading_batch_size = 2000

    server_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../btx/interfaces/iserver.py")
    client_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../btx/misc/t_snes.py")

    command = "which python; ulimit -n 4096;"
    command += f"python {server_path} & echo 'Server is running'"
    command += f"; echo 'Number of images: {num_images}'"
    command += "; sleep 10"
    command += ";conda deactivate; echo 'Server environment deactivated'"
    command += "; conda activate /sdf/group/lcls/ds/tools/conda_envs/py3.11-nopsana-torch-rapids; which python; echo 'Client environment activated'"
    command += f"; python {client_path} --filename {filename} --num_images {num_images_str} --loading_batch_size {loading_batch_size} --num_tries {num_tries} --threshold {threshold} --num_runs {num_runs} --grid_size {grid_size}"

    js = JobScheduler(os.path.join(".", f't_snes_{copy_num_images}.sh'),queue = 'ampere', ncores=  1, jobname=f't_snes_{copy_num_images}',logdir='/sdf/home/n/nathfrn/btx/scripts',account='lcls',mem = '200G',num_gpus = num_gpus) ##
    js.write_header()
    js.write_main(f"{command}\n", dependencies=['psana'],find_python_path=False)
    js.clean_up()
    js.submit()
    print('All done!')

def create_pypca(config, num_nodes = 1, id_current_node = 0):
    from btx.interfaces.ischeduler import JobScheduler
    from btx.misc.get_max_events import main as compute_max_events

    setup = config.setup
    task = config.create_pypca_multinodes
    exp = setup.exp
    run = task.run
    det_type = setup.det_type
    start_offset = task.start_offset
    num_images = task.num_images
    num_tot_images = num_images
    distribution_images = [] 

    ## Computes number of runs and number of images per run
    num_runs = 0
    while num_images > 0:
        max_event = compute_max_events(exp, run+num_runs, det_type)
        images_for_run = min(max_event, num_images)
        distribution_images.append(images_for_run)
        num_images -= images_for_run
        num_runs += 1
    ##

    print(f"Number of runs: {num_runs}")
    num_images_str = json.dumps(distribution_images)
    num_components = task.num_components
    batch_size = task.batch_size
    path = task.path
    tag = task.tag

    if num_nodes >1:
        tag = f"{tag}_node_{id_current_node}"

    num_gpus = task.num_gpus
    training_percentage = task.training_percentage
    smoothing_function = task.smoothing_function
    compute_loss = task.compute_loss
    comm = MPI.COMM_WORLD
    ncores = comm.Get_size()
    compute_projected_images = task.compute_projected_images

    if task.get('loading_batch_size') is not None:
        loading_batch_size = task.loading_batch_size
    else:
        loading_batch_size = 2000

    server_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../btx/interfaces/iserver.py")
    client_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../btx/interfaces/iclient.py")

    command = "which python; ulimit -n 4096;"
    command += f"python {server_path} & echo 'Server is running'"
    command += f"; echo 'Number of images: {num_tot_images}'; echo 'Number of events to collect per run: {num_images_str}'"
    command += "; sleep 10"
    command += ";conda deactivate; echo 'Server environment deactivated'"
    command += "; conda activate /sdf/group/lcls/ds/tools/conda_envs/py3.11-nopsana-torch-rapids; which python; echo 'Client environment activated'"
    command += f"; python {client_path} -e {exp} -r {run} -d {det_type} --start_offset {start_offset} --num_images '{num_images_str}' --loading_batch_size {loading_batch_size} --num_components {num_components} --batch_size {batch_size} --path {path} --tag {tag} --training_percentage {training_percentage} --smoothing_function {smoothing_function} --num_gpus {num_gpus} --compute_loss {compute_loss} --num_runs {num_runs} --compute_projected_images {compute_projected_images} --num_nodes {num_nodes} --id_current_node {id_current_node}"

    js = JobScheduler(os.path.join(".", f'create_pypca_{num_components}_{num_tot_images}_{batch_size}_node_{id_current_node}.sh'),queue = 'ampere', ncores=  1, jobname=f'create_pypca_{num_components}_{num_tot_images}_{batch_size}_node_{id_current_node}',logdir='/sdf/home/n/nathfrn/btx/scripts',account='lcls',mem = '200G',num_gpus = num_gpus)
    js.write_header()
    js.write_main(f"{command}\n", dependencies=['psana'],find_python_path=False)
    js.clean_up()
    js.submit()
    print('All done!')

def create_pypca_multinodes(config):
    num_nodes = config.create_pypca_multinodes.num_nodes
    """if num_nodes ==1:
        create_pypca(config)
    else:"""
    import multiprocessing
    from btx.misc.clean_pypca import clean_pypca
    algo_start_time = time.time()
    with multiprocessing.Pool(processes=num_nodes) as pool:
        args = [(config, num_nodes, node) for node in range(num_nodes)]
        pool.starmap(create_pypca, args)
    algo_end_time = time.time()
    print(f"Algorithm time: {algo_end_time - algo_start_time}")
    
    clean_pypca(config.create_pypca_multinodes.path, config.create_pypca_multinodes.tag, num_nodes)

    print('All nodes done!')

def update_pypca(config,num_nodes = 1, id_current_node = 0):
    from btx.interfaces.ischeduler import JobScheduler
    from btx.misc.get_max_events import main as compute_max_events

    setup = config.setup
    task = config.update_pypca_multinodes
    exp = setup.exp
    run = task.run
    det_type = setup.det_type
    start_offset = task.start_offset
    num_images = task.num_images
    num_tot_images = num_images
    lower_bound = task.lower_bound
    upper_bound = task.upper_bound

    distribution_images = [] 
    ##
    num_runs = 0
    while num_images > 0:
        max_event = compute_max_events(exp, run+num_runs, det_type)
        images_for_run = min(max_event, num_images)
        distribution_images.append(images_for_run) #-1 enlevé là
        num_images -= images_for_run
        num_runs += 1
    ##
    print(f"Number of runs: {num_runs}")
    num_images_str = json.dumps(distribution_images)
    batch_size = task.batch_size
    num_gpus = task.num_gpus
    model = task.model

    comm = MPI.COMM_WORLD
    ncores = comm.Get_size()

    if task.get('loading_batch_size') is not None:
        loading_batch_size = task.loading_batch_size
    else:
        loading_batch_size = 2000

    server_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../btx/interfaces/iserver.py")
    client_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../btx/processing/update_pipca.py")

    command = "which python; ulimit -n 4096;"
    command += f"python {server_path} & echo 'Server is running'"
    command += f"; echo 'Number of images: {num_tot_images}'; echo 'Number of events to collect per run: {num_images_str}'"
    command += "; sleep 10"
    command += ";conda deactivate; echo 'Server environment deactivated'"
    command += "; conda activate /sdf/group/lcls/ds/tools/conda_envs/py3.11-nopsana-torch-rapids; which python; echo 'Client environment activated'"
    command += f"; python {client_path} -e {exp} -r {run} -d {det_type} --start_offset {start_offset} --num_images '{num_images_str}' --loading_batch_size {loading_batch_size} --batch_size {batch_size} --num_runs {num_runs} --lower_bound {lower_bound} --upper_bound {upper_bound} --model {model} --num_gpus {num_gpus} --num_nodes {num_nodes} --id_current_node {id_current_node}"

    js = JobScheduler(os.path.join(".", f'update_pypca_{num_tot_images}_{batch_size}_node_{id_current_node}.sh'),queue = 'ampere', ncores=  1, jobname=f'update_pypca_{num_tot_images}_{batch_size}_node_{id_current_node}',logdir='/sdf/home/n/nathfrn/btx/scripts',account='lcls',mem = '200G',num_gpus = num_gpus)
    js.write_header()
    js.write_main(f"{command}\n", dependencies=['psana'],find_python_path=False)
    js.clean_up()
    js.submit()
    print('All done!')

def update_pypca_multinodes(config):
    num_nodes = config.update_pypca_multinodes.num_nodes
    if num_nodes ==1:
        update_pypca(config)
    else:
        import multiprocessing
        from btx.misc.clean_pypca import clean_pypca
        algo_start_time = time.time()
        with multiprocessing.Pool(processes=num_nodes) as pool:
            args = [(config, num_nodes, node) for node in range(num_nodes)]
            pool.starmap(update_pypca, args)
        algo_end_time = time.time()
        print(f"Algorithm time: {algo_end_time - algo_start_time}")
        
        model_path = os.path.dirname(config.reduce_pypca_multinodes.model)
        tag = config.reduce_pypca_multinodes.model.split('/')[-1]
        clean_pypca(model_path, tag, num_nodes,mode='update')

    print('All nodes done!')

def reduce_pypca(config,num_nodes = 1, id_current_node = 0):
    from btx.interfaces.ischeduler import JobScheduler
    from btx.misc.get_max_events import main as compute_max_events

    setup = config.setup
    task = config.reduce_pypca_multinodes
    exp = setup.exp
    run = task.run
    det_type = setup.det_type
    start_offset = task.start_offset
    num_images = task.num_images
    num_tot_images = num_images

    distribution_images = [] 
    ##
    num_runs = 0
    while num_images > 0:
        max_event = compute_max_events(exp, run+num_runs, det_type)
        images_for_run = min(max_event, num_images)
        distribution_images.append(images_for_run)
        num_images -= images_for_run
        num_runs += 1
    ##
    print(f"Number of runs: {num_runs}")
    num_images_str = json.dumps(distribution_images)
    batch_size = task.batch_size
    num_gpus = task.num_gpus
    model = task.model

    comm = MPI.COMM_WORLD
    ncores = comm.Get_size()

    if task.get('loading_batch_size') is not None:
        loading_batch_size = task.loading_batch_size
    else:
        loading_batch_size = 2000

    server_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../btx/interfaces/iserver.py")
    client_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),  "../btx/processing/pypca_reducer.py")

    command = "which python; ulimit -n 8000;"
    command += f"python {server_path} & echo 'Server is running'"
    command += f"; echo 'Number of images: {num_tot_images}'; echo 'Number of events to collect per run: {num_images_str}'"
    command += "; sleep 10"
    command += ";conda deactivate; echo 'Server environment deactivated'"
    command += "; conda activate /sdf/group/lcls/ds/tools/conda_envs/py3.11-nopsana-torch-rapids; which python; echo 'Client environment activated'"
    command += f"; python {client_path} -e {exp} -r {run} -d {det_type} --start_offset {start_offset} --num_images '{num_images_str}' --loading_batch_size {loading_batch_size} --batch_size {batch_size} --num_runs {num_runs} --model {model} --num_gpus {num_gpus} --num_nodes {num_nodes} --id_current_node {id_current_node}"

    js = JobScheduler(os.path.join(".", f'reduce_pypca_{num_tot_images}_{batch_size}_node_{id_current_node}.sh'),queue = 'ampere', ncores=  1, jobname=f'reduce_pypca_{num_tot_images}_{batch_size}_node_{id_current_node}',logdir='/sdf/home/n/nathfrn/btx/scripts',account='lcls',mem = '200G',num_gpus = num_gpus)
    js.write_header()
    js.write_main(f"{command}\n", dependencies=['psana'],find_python_path=False)
    js.clean_up()
    js.submit()
    print('All done!')


def reduce_pypca_multinodes(config):
    num_nodes = config.reduce_pypca_multinodes.num_nodes
    if num_nodes ==1:
        reduce_pypca(config)
    else:
        import multiprocessing
        from btx.misc.clean_pypca import clean_pypca
        algo_start_time = time.time()
        with multiprocessing.Pool(processes=num_nodes) as pool:
            args = [(config, num_nodes, node) for node in range(num_nodes)]
            pool.starmap(reduce_pypca, args)
        algo_end_time = time.time()
        print(f"Algorithm time: {algo_end_time - algo_start_time}")
        
        model_path = os.path.dirname(config.reduce_pypca_multinodes.model)
        tag = f"projected_images_{config.setup.exp}_start_run_{config.reduce_pypca_multinodes.run}_num_images_{config.reduce_pypca_multinodes.num_images}"
        clean_pypca(model_path, tag, num_nodes,mode='reduce')

    print('All nodes done!')


def average_img_cluster(config):
    average_img_cluster = config.average_img_cluster
    exp = config.setup.exp
    run = average_img_cluster.run
    det_type = config.setup.det_type

    from btx.processing.average_img_cluster import AverageImgCluster