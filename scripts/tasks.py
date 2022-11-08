import logging
import os
import requests
import glob
import shutil
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Fetch the URL to post progress update
update_url = os.environ.get('JID_UPDATE_COUNTERS')

def test(config):
    print(config)
    requests.post(update_url, json=[ { "key": "root_dir", "value": f"{config.setup.root_dir}" } ])

def fetch_mask(config):
    from btx.interfaces.mask_interface import MaskInterface
    setup = config.setup
    task = config.fetch_mask
    """ Fetch most recent mask for this detector from mrxv. """
    taskdir = os.path.join(setup.root_dir, 'mask')
    os.makedirs(taskdir, exist_ok=True)
    mi = MaskInterface(exp=setup.exp,
                       run=setup.run,
                       det_type=setup.det_type)
    mi.retrieve_from_mrxv(dataset=task.dataset)
    logger.info(f'Saving mrxv mask to {taskdir}')
    mi.save_mask(os.path.join(taskdir, f'r0000.npy'))
    logger.debug('Done!')

def fetch_geom(config):
    from btx.misc.metrology import retrieve_from_mrxv
    setup = config.setup
    task = config.fetch_geom
    """ Fetch latest geometry for this detector from mrxv. """
    taskdir = os.path.join(setup.root_dir, 'geom')
    os.makedirs(taskdir, exist_ok=True)
    logger.info(f'Saving mrxv geom to {taskdir}')
    retrieve_from_mrxv(det_type=setup.det_type, out_geom=os.path.join(taskdir, f'r0000.geom'))
    logger.debug('Done!')

def build_mask(config):
    from btx.interfaces.mask_interface import MaskInterface
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
    from btx.diagnostics.run import RunDiagnostics
    from btx.misc.shortcuts import fetch_latest
    setup = config.setup
    task = config.run_analysis
    """ Generate the max, avg, and std powders for a given run. """
    taskdir = os.path.join(setup.root_dir, 'powder')
    os.makedirs(taskdir, exist_ok=True)
    os.makedirs(os.path.join(taskdir, 'figs'), exist_ok=True)
    mask_file = fetch_latest(fnames=os.path.join(setup.root_dir, 'mask', 'r*.npy'), run=setup.run)
    logger.debug(f'Applying mask: {mask_file}...')
    rd = RunDiagnostics(exp=setup.exp,
                        run=setup.run,
                        det_type=setup.det_type)
    logger.debug(f'Computing Powder for run {setup.run} of {setup.exp}...')
    rd.compute_run_stats(max_events=task.max_events, mask=mask_file, threshold=task.get('mean_threshold'))
    logger.info(f'Saving Powders and plots to {taskdir}')
    rd.save_powders(taskdir)
    rd.visualize_powder(output=os.path.join(taskdir, f"figs/powder_r{rd.psi.run:04}.png"))
    rd.visualize_stats(output=os.path.join(taskdir, f"figs/stats_r{rd.psi.run:04}.png"))
    logger.debug('Done!')
    
def opt_geom(config):
    from btx.diagnostics.geom_opt import GeomOpt
    from btx.misc.shortcuts import fetch_latest
    setup = config.setup
    task = config.opt_geom
    """ Optimize and deploy the detector geometry from a silver behenate run. """
    if task.get('center') is not None:
        task.center = tuple([float(elem) for elem in task.center.split()])
    else:
        task.center = None
    taskdir = os.path.join(setup.root_dir, 'geom')
    os.makedirs(taskdir, exist_ok=True)
    os.makedirs(os.path.join(taskdir, 'figs'), exist_ok=True)
    geom_opt = GeomOpt(exp=setup.exp,
                       run=setup.run,
                       det_type=setup.det_type)
    if geom_opt.rank == 0:
        mask_file = fetch_latest(fnames=os.path.join(setup.root_dir, 'mask', 'r*.npy'), run=setup.run)
        logger.debug(f'Optimizing detector distance for run {setup.run} of {setup.exp}...')
        geom_opt.opt_geom(powder=os.path.join(setup.root_dir, f"powder/r{setup.run:04}_max.npy"),
                          mask=mask_file,
                          distance=task.get('distance'),
                          center=task.center,
                          n_iterations=task.get('n_iterations'), 
                          n_peaks=task.get('n_peaks'), 
                          threshold=task.get('threshold'),
                          plot=os.path.join(taskdir, f'figs/r{setup.run:04}.png'))
        try:
            geom_opt.report(update_url)
        except:
            logger.debug("Could not communicate with the elog update url")
        logger.info(f'Detector distance in mm inferred from powder rings: {geom_opt.distance}')
        logger.info(f'Detector center in pixels inferred from powder rings: {geom_opt.center}')
        logger.info(f'Detector edge resolution in Angstroms: {geom_opt.edge_resolution}')    
        geom_opt.deploy_geometry(taskdir)
        logger.info(f'Updated geometry files saved to: {taskdir}')
        logger.debug('Done!')

def find_peaks(config):
    from btx.processing.peak_finder import PeakFinder
    from btx.misc.shortcuts import fetch_latest
    setup = config.setup
    task = config.find_peaks
    """ Perform adaptive peak finding on run. """
    taskdir = os.path.join(setup.root_dir, 'index')
    os.makedirs(taskdir, exist_ok=True)
    mask_file = fetch_latest(fnames=os.path.join(setup.root_dir, 'mask', 'r*.npy'), run=setup.run)
    pf = PeakFinder(exp=setup.exp, run=setup.run, det_type=setup.det_type, xtc_dir=setup.xtc_dir, outdir=os.path.join(taskdir ,f"r{setup.run:04}"),
                    tag=task.tag, mask=mask_file, min_peaks=task.min_peaks, max_peaks=task.max_peaks,
                    npix_min=task.npix_min, npix_max=task.npix_max, amax_thr=task.amax_thr, atot_thr=task.atot_thr, 
                    son_min=task.son_min, peak_rank=task.peak_rank, r0=task.r0, dr=task.dr, nsigm=task.nsigm,
                    calibdir=task.get('calibdir'))
    logger.debug(f'Performing peak finding for run {setup.run} of {setup.exp}...')
    pf.find_peaks()
    pf.curate_cxi()
    pf.summarize() 
    try:
        pf.report(update_url)
    except:
        logger.debug("Could not communicate with the elog update url")
    logger.info(f'Saving CXI files and summary to {taskdir}/r{setup.run:04}')
    logger.debug('Done!')

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
                          time=task.get('time') if task.get('time') is not None else '1:00:00')
    logger.debug(f'Generating indexing executable for run {setup.run} of {setup.exp}...')
    indexer_obj.launch()
    logger.info(f'Indexing launched!')

def stream_analysis(config):
    from btx.interfaces.istream import StreamInterface, write_cell_file
    setup = config.setup
    task = config.stream_analysis
    """ Diagnostics including cell distribution and peakogram. Concatenate streams. """
    taskdir = os.path.join(setup.root_dir, 'index')
    os.makedirs(os.path.join(taskdir, 'figs'), exist_ok=True)
    stream_files = os.path.join(taskdir, f"r*{task.tag}.stream")
    st = StreamInterface(input_files=glob.glob(stream_files), cell_only=task.get('cell_only') if task.get('cell_only') is not None else False)
    if st.rank == 0:
        logger.debug(f'Read stream files: {stream_files}')
        st.report()
        st.plot_cell_parameters(output=os.path.join(taskdir, f"figs/cell_{task.tag}.png"))
        if not st.cell_only:
            st.plot_peakogram(output=os.path.join(taskdir, f"figs/peakogram_{task.tag}.png"))
        logger.info(f'Cell distribution and possibly peakogram generated for sample {task.tag}')
        celldir = os.path.join(setup.root_dir, 'cell')
        os.makedirs(celldir, exist_ok=True)
        write_cell_file(st.cell_params, os.path.join(celldir, f"{task.tag}.cell"), input_file=setup.get('cell'))
        logger.info(f'Wrote updated CrystFEL cell file to {celldir}')
        stream_cat = os.path.join(taskdir, f"{task.tag}.stream")
        os.system(f"cat {stream_files} > {stream_cat}")
        logger.info(f'Concatenated all stream files to {task.tag}.stream')
        logger.debug('Done!')

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
        cell = st.stream_data[:,2:]
        cell[:,:3] *= 10
        labels = cluster_cell_params(cell, 
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
                                mtz_dir=os.path.join(setup.root_dir, "solve", f"{task.tag}"))
    stream_to_mtz.cmd_partialator(iterations=task.iterations, model=task.model, 
                                  min_res=task.get('min_res'), push_res=task.get('push_res'), max_adu=task.get('max_adu'))
    for ns in [1, task.nshells]:
        stream_to_mtz.cmd_compare_hkl(foms=foms, nshells=ns, highres=task.get('highres'))
    stream_to_mtz.cmd_get_hkl(highres=task.get('highres'))
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
               ncores=task.get('ncores') if task.get('ncores') is not None else 16)
    logger.info(f'Dimple launched!')

def refine_geometry(config, task=None):
    from btx.diagnostics.geoptimizer import Geoptimizer
    from btx.misc.shortcuts import fetch_latest, check_file_existence
    setup = config.setup
    if task is None:
        task = config.refine_geometry
        task.scan_dir = os.path.join(setup.root_dir, f'scan_{config.merge.tag}')
        for var in [task.dx, task.dy, task.dz]:
            var = tuple([float(elem) for elem in var.split()])
            var = np.linspace(var[0], var[1], int(var[2]))
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
                        task.dz)
    geopt.launch_indexing(setup.exp, setup.det_type, config.index, cell_file)
    geopt.launch_stream_analysis(config.index.cell)    
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
    eli.update_summary()
    logger.debug('Done!')

def clean_up(config):
    setup = config.setup
    task = config.clean_up
    taskdir = os.path.join(setup.root_dir, 'index')
    if os.path.isdir(taskdir):
        os.system(f"rm -f {taskdir}/r*/*{task.tag}.cxi")
    logger.debug('Done!')