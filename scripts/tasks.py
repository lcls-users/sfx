import logging
import os
import requests
import glob
import shutil
import numpy as np
import itertools
import yaml
import csv
import time
import pickle
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch the URL to post progress update
update_url = os.environ.get("JID_UPDATE_COUNTERS")


def test(config):
    print(config)
    requests.post(
        update_url, json=[{"key": "root_dir", "value": f"{config.setup.root_dir}"}]
    )


def fetch_mask(config):
    from btx.interfaces.imask import MaskInterface

    setup = config.setup
    task = config.fetch_mask
    """ Fetch most recent mask for this detector from mrxv. """
    if "sdf" in setup.root_dir:
        mrxv_path = "/sdf/group/lcls/ds/tools/mrxv/masks/"
    elif "cds" in setup.root_dir:
        mrxv_path = "/cds/sw/package/autosfx/mrxv/masks/"
    taskdir = os.path.join(setup.root_dir, "mask")
    os.makedirs(taskdir, exist_ok=True)
    mi = MaskInterface(exp=setup.exp, run=setup.run, det_type=setup.det_type)
    mi.retrieve_from_mrxv(mrxv_path=mrxv_path, dataset=task.dataset)
    logger.info(f"Saving mrxv mask to {taskdir}")
    mi.save_mask(os.path.join(taskdir, f"r0000.npy"))
    logger.debug("Done!")


def fetch_geom(config):
    from btx.misc.metrology import retrieve_from_mrxv

    setup = config.setup
    task = config.fetch_geom
    """ Fetch latest geometry for this detector from mrxv. """
    taskdir = os.path.join(setup.root_dir, "geom")
    if "sdf" in setup.root_dir:
        mrxv_path = "/sdf/group/lcls/ds/tools/mrxv/geometries/"
    elif "cds" in setup.root_dir:
        mrxv_path = "/cds/sw/package/autosfx/mrxv/geometries/"
    os.makedirs(taskdir, exist_ok=True)
    logger.info(f"Saving mrxv geom to {taskdir}")
    retrieve_from_mrxv(
        det_type=setup.det_type,
        out_geom=os.path.join(taskdir, f"r0000.geom"),
        mrxv_path=mrxv_path,
    )
    logger.debug("Done!")


def build_mask(config):
    from btx.interfaces.imask import MaskInterface
    from btx.misc.shortcuts import fetch_latest

    setup = config.setup
    task = config.build_mask
    """ Generate a mask by thresholding events from a psana run. """
    taskdir = os.path.join(setup.root_dir, "mask")
    os.makedirs(taskdir, exist_ok=True)
    mi = MaskInterface(exp=setup.exp, run=setup.run, det_type=setup.det_type)
    if task.combine:
        mask_file = fetch_latest(fnames=os.path.join(taskdir, "r*.npy"), run=setup.run)
        mi.load_mask(mask_file, mask_format="psana")
        logger.debug(f"New mask will be combined with {mask_file}")
    task.thresholds = tuple([float(elem) for elem in task.thresholds.split()])
    mi.generate_from_psana_run(
        thresholds=task.thresholds, n_images=task.n_images, n_edge=task.n_edge
    )
    logger.info(f"Saving newly generated mask to {taskdir}")
    mi.save_mask(os.path.join(taskdir, f"r{mi.psi.run:04}.npy"))
    logger.debug("Done!")


def run_analysis(config):
    from btx.interfaces.ischeduler import JobScheduler
    from btx.misc.shortcuts import fetch_latest

    setup = config.setup
    task = config.run_analysis
    """ Generate powders for a given run and plot traces of run statistics. """
    taskdir = os.path.join(setup.root_dir, f"powder/{setup.exp}/r{setup.run:04}")
    os.makedirs(taskdir, exist_ok=True)
    os.makedirs(os.path.join(taskdir, "figs"), exist_ok=True)
    mask_file = fetch_latest(
        fnames=os.path.join(setup.root_dir, "mask", "r*.npy"), run=setup.run
    )
    script_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../btx/diagnostics/run.py"
    )
    command = f"python {script_path}"
    command += f" -e {setup.exp} -r {setup.run} -d {setup.det_type} -o {taskdir}"
    if mask_file:
        command += f" -m {mask_file}"
    if task.get("mean_threshold") is not None:
        command += f" --mean_threshold={task.mean_threshold}"
    if task.get("gain_mode") is not None:
        command += f" --gain_mode={task.gain_mode}"
    if task.get("raw_img") is not None:
        if task.raw_img:
            command += f" --raw_img"
    if task.get("outlier_threshold") is not None:
        command += f" --outlier_threshold={task.outlier_threshold}"
    if task.get("assemble") is not None:
        if not task.assemble:
            command += f" --assemble"
    js = JobScheduler(
        os.path.join(".", f"ra_{setup.run:04}.sh"),
        queue=setup.queue,
        ncores=task.ncores if task.get("ncores") is not None else 120,
        jobname=f"ra_{setup.run:04}",
        account=setup.account,
        reservation=setup.reservation,
    )
    js.write_header()
    js.write_main(f"{command}\n", dependencies=["psana"])
    js.clean_up()
    js.submit()
    logger.debug("Run analysis launched!")


def opt_geom(config):
    from btx.diagnostics.geom_opt import GeomOpt
    from btx.misc.shortcuts import fetch_latest
    from btx.misc.shortcuts import TaskTimer

    setup = config.setup
    task = config.opt_geom
    task_durations = dict({})
    """ Optimize and deploy the detector geometry from a silver behenate run. """
    with TaskTimer(task_durations, "total duration"):
        taskdir = os.path.join(setup.root_dir, "geom")
        os.makedirs(taskdir, exist_ok=True)
        os.makedirs(os.path.join(taskdir, "figs"), exist_ok=True)
        mask_file = fetch_latest(
            fnames=os.path.join(setup.root_dir, "mask", "r*.npy"), run=setup.run
        )
        task.dx = tuple([float(elem) for elem in task.dx.split()])
        task.dx = np.linspace(task.dx[0], task.dx[1], int(task.dx[2]))
        task.dy = tuple([float(elem) for elem in task.dy.split()])
        task.dy = np.linspace(task.dy[0], task.dy[1], int(task.dy[2]))
        centers = list(itertools.product(task.dx, task.dy))
        if type(task.n_peaks) == int:
            task.n_peaks = [int(task.n_peaks)]
        else:
            task.n_peaks = [int(elem) for elem in task.n_peaks.split()]
        if task.get("distance") is None:
            task.distance = None
        elif type(task.distance) == float or type(task.distance) == int:
            task.distance = [float(task.distance)]
        else:
            task.distance = [float(elem) for elem in task.distance.split()]
        geom_opt = GeomOpt(exp=setup.exp, run=setup.run, det_type=setup.det_type)
        geom_opt.opt_geom(
            powder=os.path.join(setup.root_dir, f"powder/r{setup.run:04}_max.npy"),
            mask=mask_file,
            distance=task.distance,
            center=centers,
            n_iterations=task.get("n_iterations"),
            n_peaks=task.n_peaks,
            threshold=task.get("threshold"),
            deltas=True,
            plot=os.path.join(taskdir, f"figs/r{setup.run:04}.png"),
            plot_final_only=True,
        )
        if geom_opt.rank == 0:
            try:
                geom_opt.report(update_url)
            except:
                logger.debug("Could not communicate with the elog update url")
            logger.info(f"Refined detector distance in mm: {geom_opt.distance}")
            logger.info(f"Refined detector center in pixels: {geom_opt.center}")
            logger.info(
                f"Detector edge resolution in Angstroms: {geom_opt.edge_resolution}"
            )
            geom_opt.deploy_geometry(
                taskdir, pv_camera_length=setup.get("pv_camera_length")
            )
            logger.info(f"Geometry shift in microns: {geom_opt.shift}")
            logger.info(f"Updated geometry files saved to: {taskdir}")
            logger.debug("Done!")

    logger.info(f"Total duration: {task_durations['total duration'][0]} seconds")


def grid_search_pyFAI_geom(config):
    from btx.diagnostics.pyFAI_geom_opt import GridSearchGeomOpt
    from btx.diagnostics.converter import CrystFELtoPyFAI, PsanatoCrystFEL
    from btx.misc.shortcuts import TaskTimer

    setup = config.setup
    task = config.grid_search_pyFAI_geom
    task_durations = dict({})
    """ Grid search for geometry. """
    with TaskTimer(task_durations, "total duration"):
        taskdir = os.path.join(setup.root_dir, "grid_search")
        os.makedirs(taskdir, exist_ok=True)
        geomfile = task.get("geomfile")
        if geomfile != '':
            logger.info(f"Using {geomfile} as input geometry")
        else:
            logger.info(f"No geometry files provided: using calibration data as input geometry")
            geomfile = f'/sdf/data/lcls/ds/mfx/{setup.exp}/calib/*/geometry/0-end.data'
        PsanatoCrystFEL(geomfile, geomfile.replace(".data", ".geom"), det_type=setup.det_type)
        conv = CrystFELtoPyFAI(geomfile.replace(".data", ".geom"), psana_file=geomfile, det_type=setup.det_type)
        det = conv.detector
        geom_opt = GridSearchGeomOpt(exp=setup.exp, run=setup.run, det_type=setup.det_type, detector=det, calibrant=task.calibrant, Imin=task.Imin)
        powder = task.get("powder")
        task.poni = tuple([float(elem) for elem in task.poni.split()])
        bounds = {'poni1':(task.poni[0], task.poni[1], int(task.poni[2])), 'poni2':(task.poni[0], task.poni[1], int(task.poni[2]))}
        dist = task.distance
        cx, cy, y = geom_opt.grid_search_geom_opt(
            powder=powder,
            bounds=bounds,
            dist=dist,
        )
        np.save(os.path.join(taskdir, f"grid_search_{os.path.splitext(os.path.basename(powder))[0]}_#ncp.npy"), y)
        logger.debug("Done!")
    logger.warning(f"Total duration: {task_durations['total duration'][0]} seconds")


def bayes_pyFAI_geom(config):
    from btx.diagnostics.pyFAI_geom_opt import BayesGeomOpt
    from btx.diagnostics.converter import PsanaToPyFAI, PyFAIToCrystFEL, CrystFELToPsana
    from btx.misc.shortcuts import TaskTimer

    setup = config.setup
    task = config.bayes_pyFAI_geom
    task_durations = dict({})
    """ Bayesian Optimization for geometry. """
    with TaskTimer(task_durations, "total duration"):
        in_file = task.get("in_file")
        if in_file != '':
            logger.warning(f"Using {in_file} as input geometry")
        else:
            logger.warning(f"No geometry files provided: using calibration data as input geometry")
            in_file = f'/sdf/data/lcls/ds/mfx/{setup.exp}/calib/*/geometry/0-end.data'
        psana_to_pyfai = PsanaToPyFAI(in_file, det_type=setup.det_type)
        detector = psana_to_pyfai.detector
        powder = task.get("powder")
        Imin = task.get("Imin", 'max')
        n_samples = task.get("n_samples", 50)
        num_iterations = task.get("num_iterations", 50)
        af = task.get("af", 'ucb')
        prior = task.get("prior", True)
        seed = task.get("seed")
        dist = tuple([float(elem) for elem in task.dist.split()])
        poni1 = tuple([float(elem) for elem in task.poni1.split()])
        poni2 = tuple([float(elem) for elem in task.poni2.split()])
        bounds = {'dist':(dist[0], dist[1]),'poni1':(poni1[0], poni1[1], poni1[2]), 'poni2':(poni2[0], poni2[1], poni2[2])}
        geom_opt = BayesGeomOpt(
            exp=setup.exp,
            run=setup.run,
            det_type=setup.det_type,
            detector=detector,
            calibrant=task.calibrant,
        )
        geom_opt.bayes_opt_geom(
            powder=powder,
            bounds=bounds,
            Imin=Imin,
            n_samples=n_samples,
            num_iterations=num_iterations,
            af=af,
            prior=prior,
            seed=seed,
            )
        if geom_opt.rank == 0:
            logger.warning(f"Refined PONI distance in m: {geom_opt.params[0]:.2e}")
            logger.warning(f"Refined detector PONI in m: {geom_opt.params[1]:.2e}, {geom_opt.params[2]:.2e}")
            logger.warning(f"Refined detector rotations in rad: \u03B8x = {geom_opt.params[3]:.2e}, \u03B8y = {geom_opt.params[4]:.2e}, \u03B8z = {geom_opt.params[5]:.2e}")
            logger.warning(f"Final score: {geom_opt.residuals:.2e}")
            logger.warning(f"Saving geometry to {in_file.replace('0-end.data', f'r{setup.run:04}.geom')}")
            PyFAIToCrystFEL(detector=geom_opt.detector, params=geom_opt.params, psana_file=in_file, out_file=in_file.replace("0-end.data", f"r{setup.run:04}.geom"))
            logger.warning(f"Saving geometry to {in_file.replace('0-end.data', f'{setup.run}-end.data')}")
            CrystFELToPsana(in_file=in_file.replace("0-end.data", f"r{setup.run:04}.geom"), det_type=setup.det_type, out_file=in_file.replace("0-end.data", f"{setup.run}-end.data"))
            psana_to_pyfai = PsanaToPyFAI(in_file.replace("0-end.data", f"{setup.run}-end.data"), det_type=setup.det_type)
            detector = psana_to_pyfai.detector
            plot = f'{setup.root_dir}/figs/bayes_opt/bayes_opt_geom_r{setup.run:04}.png'
            geom_opt.visualize_results(powder=geom_opt.powder_img, bo_history=geom_opt.bo_history, detector=detector, params=geom_opt.params, plot=plot)
            logger.debug("Done!")
    logger.warning(f"Total duration: {task_durations['total duration'][0]} seconds")


def hooke_jeeves_pyFAI_geom(config):
    from btx.diagnostics.pyFAI_geom_opt import HookeJeevesGeomOpt
    from btx.diagnostics.converter import CrystFELtoPyFAI, PsanatoCrystFEL
    from btx.misc.shortcuts import TaskTimer

    setup = config.setup
    task = config.hooke_jeeves_pyFAI_geom
    task_durations = dict({})
    """ Grid search for geometry. """
    with TaskTimer(task_durations, "total duration"):
        geomfile = task.get("geomfile")
        if geomfile != '':
            logger.info(f"Using {geomfile} as input geometry")
        else:
            logger.info(f"No geometry files provided: using calibration data as input geometry")
            geomfile = f'/sdf/data/lcls/ds/mfx/{setup.exp}/calib/*/geometry/0-end.data'
        PsanatoCrystFEL(geomfile, geomfile.replace(".data", ".geom"), det_type=setup.det_type)
        conv = CrystFELtoPyFAI(geomfile.replace(".data", ".geom"), psana_file=geomfile, det_type=setup.det_type)
        det = conv.detector
        det.set_pixel_corners(conv.corner_array)
        fix = ['rot1', 'rot2', 'rot3']
        geom_opt = HookeJeevesGeomOpt(exp=setup.exp, run=setup.run, det_type=setup.det_type, detector=det, calibrant=task.calibrant, fix=fix, Imin=task.Imin)
        powder = task.get("powder")
        fix = ['wavelength']
        step_size = task.get("step_size")
        tol = task.get("tol")
        bo_history, best_idx, best_score = geom_opt.hooke_jeeves_geom_opt(
            powder=powder,
            fix=fix,
            mask=None,
            step_size=step_size,
            tol=tol,
        )
        logger.info(f"Refined PONI distance in m: {geom_opt.dist}")
        logger.info(f"Refined detector PONI in m: {geom_opt.poni1:.2e}, {geom_opt.poni2:.2e}")
        logger.info(f"Refined detector rotations in rad: \u03B8x = {geom_opt.rot1}, \u03B8y = {geom_opt.rot2}, \u03B8z = {geom_opt.rot3}")
        Xc = geom_opt.poni1+geom_opt.dist*(np.tan(geom_opt.rot2)/np.cos(geom_opt.rot1))
        Yc = geom_opt.poni2-geom_opt.dist*(np.tan(geom_opt.rot1))
        Zc = geom_opt.dist/(np.cos(geom_opt.rot1)*np.cos(geom_opt.rot2))
        logger.info(f"Refined detector distance in m: {Zc:.2e}")
        logger.info(f"Refined detector center in m: {Xc:.2e}, {Yc:.2e}")
        grid_search = np.load(os.path.join(setup.root_dir, f"grid_search/{setup.exp}/grid_search_calib_max_flat_{setup.run:04}_quad2.npy"))
        plot = os.path.join(setup.root_dir, f"figs/bayes_opt/bayes_opt_geom_r{setup.run:04}.png")
        geom_opt.grid_search_convergence_plot(bo_history, best_idx, grid_search, plot)
        logger.debug("Done!")
    logger.info(f"Total duration: {task_durations['total duration'][0]} seconds")


def cross_entropy_pyFAI_geom(config):
    from btx.diagnostics.pyFAI_geom_opt import CrossEntropyGeomOpt
    from btx.diagnostics.converter import CrystFELtoPyFAI, PsanatoCrystFEL
    from btx.misc.shortcuts import TaskTimer

    setup = config.setup
    task = config.cross_entropy_pyFAI_geom
    task_durations = dict({})
    """ Grid search for geometry. """
    with TaskTimer(task_durations, "total duration"):
        geomfile = task.get("geomfile")
        if geomfile != '':
            logger.info(f"Using {geomfile} as input geometry")
        else:
            logger.info(f"No geometry files provided: using calibration data as input geometry")
            geomfile = f'/sdf/data/lcls/ds/mfx/{setup.exp}/calib/*/geometry/0-end.data'
        PsanatoCrystFEL(geomfile, geomfile.replace(".data", ".geom"), det_type=setup.det_type)
        conv = CrystFELtoPyFAI(geomfile.replace(".data", ".geom"), psana_file=geomfile, det_type=setup.det_type)
        det = conv.detector
        det.set_pixel_corners(conv.corner_array)
        fix = ['rot1', 'rot2', 'rot3']
        geom_opt = CrossEntropyGeomOpt(exp=setup.exp, run=setup.run, det_type=setup.det_type, detector=det, calibrant=task.calibrant, fix=fix, Imin=task.Imin)
        powder = task.get("powder")
        fix = ['wavelength']
        task.dist = tuple([float(elem) for elem in task.dist.split()])
        task.poni = tuple([float(elem) for elem in task.poni.split()])
        bounds = {'dist':(task.dist[0], task.dist[1]), 'poni1':(task.poni[0], task.poni[1]), 'poni2':(task.poni[0], task.poni[1])}
        n_samples = task.get("n_samples")
        m_elite = task.get("m_elite")
        num_iterations = task.get("num_iterations")
        seed = task.get("seed")
        bo_history, best_idx, best_score = geom_opt.cross_entropy_geom_opt(
            powder=powder,
            fix=fix,
            mask=None,
            bounds=bounds,
            n_samples=n_samples,
            m_elite=m_elite,
            num_iterations=num_iterations,
            seed=seed,
        )
        logger.info(f"Refined PONI distance in m: {geom_opt.dist}")
        logger.info(f"Refined detector PONI in m: {geom_opt.poni1:.2e}, {geom_opt.poni2:.2e}")
        logger.info(f"Refined detector rotations in rad: \u03B8x = {geom_opt.rot1}, \u03B8y = {geom_opt.rot2}, \u03B8z = {geom_opt.rot3}")
        Xc = geom_opt.poni1+geom_opt.dist*(np.tan(geom_opt.rot2)/np.cos(geom_opt.rot1))
        Yc = geom_opt.poni2-geom_opt.dist*(np.tan(geom_opt.rot1))
        Zc = geom_opt.dist/(np.cos(geom_opt.rot1)*np.cos(geom_opt.rot2))
        logger.info(f"Refined detector distance in m: {Zc:.2e}")
        logger.info(f"Refined detector center in m: {Xc:.2e}, {Yc:.2e}")
        grid_search = np.load(os.path.join(setup.root_dir, f"grid_search/{setup.exp}/grid_search_calib_max_flat_{setup.run:04}_quad2.npy"))
        plot = os.path.join(setup.root_dir, f"figs/bayes_opt/bayes_opt_geom_r{setup.run:04}.png")
        geom_opt.grid_search_convergence_plot(bo_history, best_idx, grid_search, plot)
        logger.debug("Done!")
    logger.info(f"Total duration: {task_durations['total duration'][0]} seconds")


def grid_search_geom(config):
    from btx.misc.shortcuts import fetch_latest
    from btx.diagnostics.run import RunDiagnostics
    from btx.diagnostics.ag_behenate import AgBehenate
    from btx.interfaces.ipsana import assemble_image_stack_batch
    from btx.misc.radial import radial_profile, pix2q, q2pix
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    from btx.misc.shortcuts import TaskTimer
    import scipy as scp
    from scipy.stats import norm

    setup = config.setup
    task_durations = dict({})
    """ Grid for geometry. """
    with TaskTimer(task_durations, "total duration"):
        taskdir = os.path.join(setup.root_dir, "geom")
        os.makedirs(taskdir, exist_ok=True)
        os.makedirs(os.path.join(taskdir, "figs"), exist_ok=True)
        powder_file = os.path.join(setup.root_dir, f"powder/r{setup.run:04}_max.npy")
        powder_img = np.load(powder_file)

        diagnostics = RunDiagnostics(
            exp=setup.exp, run=setup.run, det_type=setup.det_type
        )
        pixel_size = diagnostics.psi.get_pixel_size()
        wavelength = diagnostics.psi.get_wavelength()

        mask_file = fetch_latest(
            fnames=os.path.join(setup.root_dir, "mask", "r*.npy"), run=setup.run
        )
        mask_img = np.load(mask_file)
        if diagnostics.psi.det_type != "Rayonix":
            mask_img = assemble_image_stack_batch(mask_img, diagnostics.pixel_index_map)

        ag_behenate = AgBehenate(
            powder=powder_img,
            mask=mask_img,
            pixel_size=pixel_size,
            wavelength=wavelength,
        )

        height, width = powder_img.shape[:2]
        # Initial guess for the distance
        distance_i = diagnostics.psi.estimate_distance()

        length_fraction = 128
        x_MIN = width // 2 - width // length_fraction
        x_MAX = width // 2 + width // length_fraction
        y_MIN = height // 2 - height // length_fraction
        y_MAX = height // 2 + height // length_fraction

        resolution = 0.5  # pixels

        output_file_path = os.path.join(
            setup.root_dir,
            f"grid_geom_{length_fraction//2}th_length_square_res_{resolution}.csv",
        )

        # Create/Empty the .csv file
        with open(output_file_path, "w", newline="") as file:
            writer = csv.writer(file, delimiter=",")
            # Write a header row with column names
            writer.writerow(["cx", "cy", "score"])

        with open(output_file_path, "a", newline="") as file:
            writer = csv.writer(file, delimiter=",")

            for cx in np.arange(x_MIN, x_MAX + resolution, resolution):
                for cy in np.arange(y_MIN, y_MAX + resolution, resolution):

                    iprofile = radial_profile(
                        data=powder_img, center=(cx, cy), mask=mask_img
                    )
                    peaks_observed, properties = scp.signal.find_peaks(
                        iprofile, prominence=1, distance=10
                    )
                    qprofile = pix2q(
                        np.arange(iprofile.shape[0]), wavelength, distance_i, pixel_size
                    )

                    # optimize the detector distance based on inter-peak distances
                    rings, scores = ag_behenate.ideal_rings(qprofile[peaks_observed])
                    peaks_predicted = q2pix(rings, wavelength, distance_i, pixel_size)

                    # Write the data
                    data_to_write = [cx, cy, float(np.min(scores))]
                    writer.writerow(data_to_write)

        logger.debug("Done!")
    logger.info(f"Total duration: {task_durations['total duration'][0]} seconds")


def find_peaks(config):
    from btx.processing.peak_finder import PeakFinder
    from btx.misc.shortcuts import fetch_latest
    from btx.interfaces.ielog import update_summary

    setup = config.setup
    task = config.find_peaks
    """ Perform adaptive peak finding on run. """
    taskdir = os.path.join(setup.root_dir, "index")
    os.makedirs(taskdir, exist_ok=True)
    mask_file = fetch_latest(
        fnames=os.path.join(setup.root_dir, "mask", "r*.npy"), run=setup.run
    )
    pf = PeakFinder(
        exp=setup.exp,
        run=setup.run,
        det_type=setup.det_type,
        outdir=os.path.join(taskdir, f"r{setup.run:04}"),
        event_receiver=setup.get("event_receiver"),
        event_code=setup.get("event_code"),
        event_logic=setup.get("event_logic"),
        tag=task.tag,
        mask=mask_file,
        psana_mask=task.psana_mask,
        min_peaks=task.min_peaks,
        max_peaks=task.max_peaks,
        npix_min=task.npix_min,
        npix_max=task.npix_max,
        amax_thr=task.amax_thr,
        atot_thr=task.atot_thr,
        son_min=task.son_min,
        peak_rank=task.peak_rank,
        r0=task.r0,
        dr=task.dr,
        nsigm=task.nsigm,
        calibdir=task.get("calibdir"),
        pv_camera_length=setup.get("pv_camera_length"),
    )
    logger.info(f"Performing peak finding for run {setup.run} of {setup.exp}...")
    pf.find_peaks()
    pf.curate_cxi()
    pf.summarize()
    logger.info(f"Saving CXI files and summary to {taskdir}/r{setup.run:04}")
    logger.debug("Done!")

    if pf.rank == 0:
        summary_file = f"{setup.root_dir}/summary_r{setup.run:04}.json"
        update_summary(summary_file, pf.pf_summary)


def find_peaks_multiple_runs(config):
    from btx.interfaces.ischeduler import JobScheduler
    from btx.misc.shortcuts import fetch_latest

    setup = config.setup
    task = config.find_peaks
    bay_opt = config.bayesian_optimization
    """ Perform adaptive peak finding on multiple runs. """
    taskdir = os.path.join(setup.root_dir, "index")
    os.makedirs(taskdir, exist_ok=True)
    script_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../btx/processing/peak_finder.py"
    )

    logger.info(
        f"Launching one slurm job to perform peak finding for each run in interval [{bay_opt.first_run}, {bay_opt.last_run}]"
    )
    for iter_run in range(bay_opt.first_run, bay_opt.last_run + 1):
        # Write the command by specifying all the arguments that can be found in the config
        command = f"python {script_path}"
        command += f" -e {setup.exp} -r {iter_run} -d {setup.det_type} -o {taskdir} -t {task.tag}"
        mask_file = fetch_latest(
            fnames=os.path.join(setup.root_dir, "mask", "r*.npy"), run=iter_run
        )
        if mask_file:
            command += f" -m {mask_file}"
        if task.get("event_receiver") is not None:
            command += f" --event_receiver={task.event_receiver}"
        if task.get("event_code") is not None:
            command += f" --event_code={task.event_code}"
        if task.get("event_logic") is not None:
            command += f" --event_logic={task.event_logic}"
        if task.get("psana_mask") is not None:
            command += f" --psana_mask={task.psana_mask}"
        if task.get("pv_camera_length") is not None:
            command += f" --pv_camera_length={task.pv_camera_length}"
        if task.get("min_peaks") is not None:
            command += f" --min_peaks={task.min_peaks}"
        if task.get("max_peaks") is not None:
            command += f" --max_peaks={task.max_peaks}"
        if task.get("npix_min") is not None:
            command += f" --npix_min={task.npix_min}"
        if task.get("npix_max") is not None:
            command += f" --npix_max={task.npix_max}"
        if task.get("amax_thr") is not None:
            command += f" --amax_thr={task.amax_thr}"
        if task.get("atot_thr") is not None:
            command += f" --atot_thr={task.atot_thr}"
        if task.get("son_min") is not None:
            command += f" --son_min={task.son_min}"
        if task.get("peak_rank") is not None:
            command += f" --peak_rank={task.peak_rank}"
        if task.get("r0") is not None:
            command += f" --r0={task.r0}"
        if task.get("dr") is not None:
            command += f" --dr={task.dr}"
        if task.get("nsigm") is not None:
            command += f" --nsigm={task.nsigm}"
        if task.get("calibdir") is not None:
            command += f" --calibdir={task.calibdir}"
        # Launch the Slurm job to perform "find_peaks" on the current run
        js = JobScheduler(
            os.path.join(".", f"fp_{iter_run:04}.sh"),
            queue=setup.queue,
            ncores=bay_opt.ncores if bay_opt.get("ncores") is not None else 64,
            jobname=f"fp_{iter_run:04}",
            account=setup.account,
            reservation=setup.reservation,
        )
        js.write_header()
        js.write_main(f"{command}\n", dependencies=["psana"])
        js.clean_up()
        js.submit()
        logger.info(
            f"Launched a slurm job to perform peak finding for run {iter_run} \
                    in interval [{bay_opt.first_run}, {bay_opt.last_run}] of experiment {setup.exp}..."
        )

    logger.info("All slurm jobs have been launched!")
    logger.info("Done!")


def index(config):
    from btx.processing.indexer import Indexer
    from btx.misc.shortcuts import fetch_latest

    setup = config.setup
    task = config.index
    """ Index run using indexamajig. """
    taskdir = os.path.join(setup.root_dir, "index")
    geom_file = fetch_latest(
        fnames=os.path.join(setup.root_dir, "geom", "r*.geom"), run=setup.run
    )
    indexer_obj = Indexer(
        exp=config.setup.exp,
        run=config.setup.run,
        det_type=config.setup.det_type,
        tag=task.tag,
        tag_cxi=task.get("tag_cxi"),
        taskdir=taskdir,
        geom=geom_file,
        cell=task.get("cell"),
        int_rad=task.int_radius,
        methods=task.methods,
        tolerance=task.tolerance,
        no_revalidate=task.no_revalidate,
        multi=task.multi,
        profile=task.profile,
        queue=setup.get("queue"),
        ncores=task.get("ncores") if task.get("ncores") is not None else 64,
        time=task.get("time") if task.get("time") is not None else "1:00:00",
        mpi_init=False,
        slurm_account=setup.account,
        slurm_reservation=setup.reservation,
    )
    logger.debug(
        f"Generating indexing executable for run {setup.run} of {setup.exp}..."
    )
    indexer_obj.launch()
    logger.info(f"Indexing launched!")


def index_multiple_runs(config):
    from btx.processing.indexer import Indexer
    from btx.misc.shortcuts import fetch_latest

    setup = config.setup
    task = config.index
    bay_opt = config.bayesian_optimization
    """ Index multiple runs using indexamajig. """
    taskdir = os.path.join(setup.root_dir, "index")

    logger.info(
        f"Launching indexing for each run in interval [{bay_opt.first_run}, {bay_opt.last_run}]"
    )
    for iter_run in range(bay_opt.first_run, bay_opt.last_run + 1):
        geom_file = fetch_latest(
            fnames=os.path.join(setup.root_dir, "geom", "r*.geom"), run=iter_run
        )
        indexer_obj = Indexer(
            exp=config.setup.exp,
            run=iter_run,
            det_type=config.setup.det_type,
            tag=task.tag,
            tag_cxi=task.get("tag_cxi"),
            taskdir=taskdir,
            geom=geom_file,
            cell=task.get("cell"),
            int_rad=task.int_radius,
            methods=task.methods,
            tolerance=task.tolerance,
            no_revalidate=task.no_revalidate,
            multi=task.multi,
            profile=task.profile,
            queue=setup.get("queue"),
            ncores=task.get("ncores") if task.get("ncores") is not None else 64,
            time=task.get("time") if task.get("time") is not None else "1:00:00",
            mpi_init=False,
            slurm_account=setup.account,
            slurm_reservation=setup.reservation,
        )
        logger.debug(
            f"Generating indexing executable for run {iter_run} \
                     in interval [{bay_opt.first_run}, {bay_opt.last_run}] of experiment {setup.exp}..."
        )
        indexer_obj.launch()
        logger.info(
            f"Indexing for run {iter_run} in interval [{bay_opt.first_run}, {bay_opt.last_run}] launched!"
        )

    logger.info("All slurm jobs have been launched!")
    logger.info("Done!")


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
        tag_cxi = ""
        cxi = task.get("tag_cxi")
        if cxi:
            tag_cxi = cxi
        run = setup.run

        # Define paths
        taskdir = os.path.join(setup.root_dir, "index")
        summary_file = f"{setup.root_dir}/summary_r{setup.run:04}.json"
        stream_file = os.path.join(taskdir, f"r{run:04}_{tag}.stream")
        pf_summary = os.path.join(taskdir, f"r{run:04}/peakfinding{tag_cxi}.summary")

        command1: list = ["grep", "Cell parameters", f"{stream_file}"]
        command2: list = ["grep", "Number of hits found", f"{pf_summary}"]

        summary_dict: dict = {}
        try:
            output: str = subprocess.check_output(command1, universal_newlines=True)
            n_indexed: int = len(output.split("\n")[:-1])

            output: str = subprocess.check_output(command2, universal_newlines=True)
            n_total: int = int(output.split(":")[1].split("\n")[0])

            key_strings: list = [
                "Number of lattices found",
                ("Fractional indexing rate" " (including multiple lattices)"),
            ]
            summary_dict: dict = {
                key_strings[0]: f"{n_indexed}",
                key_strings[1]: f"{(n_indexed/n_total):.2f}",
            }
        except subprocess.CalledProcessError as err:
            print(err)

        update_summary(summary_file, summary_dict)
        post_to_elog(config)


def stream_analysis(config):
    from btx.interfaces.istream import launch_stream_analysis

    setup = config.setup
    task = config.stream_analysis
    """ Plot cell distribution and peakogram, write new cell file, and concatenate streams. """
    taskdir = os.path.join(setup.root_dir, "index")
    os.makedirs(os.path.join(taskdir, "figs"), exist_ok=True)
    os.makedirs(os.path.join(setup.root_dir, "cell"), exist_ok=True)
    launch_stream_analysis(
        os.path.join(taskdir, f"r*{task.tag}.stream"),
        os.path.join(taskdir, f"{task.tag}.stream"),
        os.path.join(taskdir, "figs"),
        os.path.join(taskdir, "stream_analysis.sh"),
        setup.queue,
        ncores=task.get("ncores") if task.get("ncores") is not None else 6,
        cell_only=task.get("cell_only") if task.get("cell_only") is not None else False,
        cell_out=os.path.join(setup.root_dir, "cell", f"{task.tag}.cell"),
        cell_ref=task.get("ref_cell"),
        slurm_account=setup.account,
        slurm_reservation=setup.reservation,
    )
    logger.info(f"Stream analysis launched")


def determine_cell(config):
    from btx.interfaces.istream import (
        StreamInterface,
        write_cell_file,
        cluster_cell_params,
    )

    setup = config.setup
    task = config.determine_cell
    """ Cluster crystals from cell-free indexing and write most-frequently found cell to CrystFEL cell file. """
    taskdir = os.path.join(setup.root_dir, "index")
    stream_files = os.path.join(taskdir, f"r*{task.tag}.stream")
    logger.info(f"Processing files {glob.glob(stream_files)}")
    st = StreamInterface(input_files=glob.glob(stream_files), cell_only=True)
    if st.rank == 0:
        logger.debug(f"Read stream files: {stream_files}")
        celldir = os.path.join(setup.root_dir, "cell")
        os.makedirs(celldir, exist_ok=True)
        keys = ["a", "b", "c", "alpha", "beta", "gamma"]
        cell = np.array([st.stream_data[key] for key in keys])
        labels = cluster_cell_params(
            cell.T,
            os.path.join(taskdir, f"clusters_{task.tag}.txt"),
            os.path.join(celldir, f"{task.tag}.cell"),
            in_cell=task.get("input_cell"),
            eps=task.get("eps") if task.get("eps") is not None else 5,
            min_samples=(
                task.get("min_samples") if task.get("min_samples") is not None else 5
            ),
        )
        logger.info(
            f"Wrote updated CrystFEL cell file for sample {task.tag} to {celldir}"
        )
        logger.debug("Done!")


def merge(config):
    from btx.processing.merge import StreamtoMtz

    setup = config.setup
    task = config.merge
    """ Merge reflections from stream file and convert to mtz. """
    taskdir = os.path.join(setup.root_dir, "merge", f"{task.tag}")
    input_stream = os.path.join(setup.root_dir, f"index/{task.tag}.stream")
    cellfile = os.path.join(setup.root_dir, f"cell/{task.tag}.cell")
    foms = task.foms.split(" ")
    stream_to_mtz = StreamtoMtz(
        input_stream,
        task.symmetry,
        taskdir,
        cellfile,
        queue=setup.get("queue"),
        ncores=task.get("ncores") if task.get("ncores") is not None else 16,
        mtz_dir=os.path.join(setup.root_dir, "solve", f"{task.tag}"),
        anomalous=task.get("anomalous") if task.get("anomalous") is not None else False,
        slurm_account=setup.account,
        slurm_reservation=setup.reservation,
    )
    stream_to_mtz.cmd_partialator(
        iterations=task.iterations,
        model=task.model,
        min_res=task.get("min_res"),
        push_res=task.get("push_res"),
        max_adu=task.get("max_adu"),
    )
    for ns in [1, task.nshells]:
        stream_to_mtz.cmd_compare_hkl(
            foms=foms, nshells=ns, highres=task.get("highres")
        )
    stream_to_mtz.cmd_hkl_to_mtz(
        highres=task.get("highres"),
        space_group=(
            task.get("space_group") if task.get("space_group") is not None else 1
        ),
    )
    stream_to_mtz.cmd_report(foms=foms, nshells=task.nshells)
    stream_to_mtz.launch()
    logger.info(f"Merging launched!")


def solve(config):
    from btx.interfaces.imtz import run_dimple

    setup = config.setup
    task = config.solve
    """ Run the CCP4 dimple pipeline for structure solution and refinement. """
    taskdir = os.path.join(setup.root_dir, "solve", f"{task.tag}")
    run_dimple(
        os.path.join(taskdir, f"{task.tag}.mtz"),
        task.pdb,
        taskdir,
        queue=setup.get("queue"),
        ncores=task.get("ncores") if task.get("ncores") is not None else 16,
        anomalous=task.get("anomalous") if task.get("anomalous") is not None else False,
        slurm_account=setup.account,
        slurm_reservation=setup.reservation,
    )
    logger.info(f"Dimple launched!")


def refine_geometry(config, task=None):
    from btx.diagnostics.geoptimizer import Geoptimizer
    from btx.misc.shortcuts import fetch_latest, check_file_existence

    setup = config.setup
    if task is None:
        task = config.refine_geometry
        task.scan_dir = os.path.join(setup.root_dir, f"scan_{config.merge.tag}")
        task.dx = tuple([float(elem) for elem in task.dx.split()])
        task.dx = np.linspace(task.dx[0], task.dx[1], int(task.dx[2]))
        task.dy = tuple([float(elem) for elem in task.dy.split()])
        task.dy = np.linspace(task.dy[0], task.dy[1], int(task.dy[2]))
        task.dz = tuple([float(elem) for elem in task.dz.split()])
        task.dz = np.linspace(task.dz[0], task.dz[1], int(task.dz[2]))
    """ Refine detector center and/or distance based on the geometry that minimizes Rsplit. """
    taskdir = os.path.join(setup.root_dir, "index")
    os.makedirs(task.scan_dir, exist_ok=True)
    task.runs = tuple([int(elem) for elem in task.runs.split()])
    if len(task.runs) == 2:
        task.runs = (*task.runs, 1)
    geom_file = fetch_latest(
        fnames=os.path.join(setup.root_dir, "geom", "r*.geom"), run=task.runs[0]
    )
    cell_file = os.path.join(config.setup.root_dir, "cell", f"{config.index.tag}.cell")
    if not os.path.isfile(cell_file):
        cell_file = config.index.get("cell")
    logger.info(f"Scanning around geometry file {geom_file}")
    geopt = Geoptimizer(
        setup.queue,
        taskdir,
        task.scan_dir,
        np.arange(task.runs[0], task.runs[1] + 1, task.runs[2]),
        geom_file,
        task.dx,
        task.dy,
        task.dz,
        slurm_account=setup.account,
        slurm_reservation=setup.reservation,
    )
    geopt.launch_indexing(setup.exp, setup.det_type, config.index, cell_file)
    geopt.launch_stream_wrangling(config.stream_analysis)
    geopt.launch_merging(config.merge)
    geopt.save_results(setup.root_dir, config.merge.tag)
    check_file_existence(os.path.join(task.scan_dir, "results.txt"), geopt.timeout)
    logger.debug("Done!")


def refine_center(config):
    """Wrapper for the refine_geometry task, searching for the detector center."""
    setup = config.setup
    task = config.refine_center
    task.scan_dir = os.path.join(setup.root_dir, f"scan_center_{config.merge.tag}")
    task.dx = tuple([float(elem) for elem in task.dx.split()])
    task.dx = np.linspace(task.dx[0], task.dx[1], int(task.dx[2]))
    task.dy = tuple([float(elem) for elem in task.dy.split()])
    task.dy = np.linspace(task.dy[0], task.dy[1], int(task.dy[2]))
    task.dz = [0]
    refine_geometry(config, task)


def refine_distance(config):
    """Wrapper for the refine_geometry task, searching for the detector distance."""
    setup = config.setup
    task = config.refine_distance
    task.scan_dir = os.path.join(setup.root_dir, f"scan_distance_{config.merge.tag}")
    task.dx, task.dy = [0], [0]
    task.dz = tuple([float(elem) for elem in task.dz.split()])
    task.dz = np.linspace(task.dz[0], task.dz[1], int(task.dz[2]))
    refine_geometry(config, task)


def elog_display(config):
    from btx.interfaces.ielog import eLogInterface

    setup = config.setup
    """ Updates the summary page in the eLog with most recent results. """
    logger.info(f"Updating the reports in the eLog summary tab.")
    eli = eLogInterface(setup)
    eli.update_summary(plot_type="holoviews")
    logger.debug("Done!")


def post_to_elog(config):
    from btx.interfaces.ielog import elog_report_post

    setup = config.setup
    root_dir = setup.root_dir
    run = setup.run

    summary_file = f"{root_dir}/summary_r{run:04}.json"
    url = os.environ.get("JID_UPDATE_COUNTERS")
    if url:
        elog_report_post(summary_file, url)


def visualize_sample(config):
    from btx.misc.visuals import VisualizeSample

    setup = config.setup
    task = config.visualize_sample
    """ Plot per-run cell parameters and peak-finding/indexing statistics. """
    logger.info(f"Extracting statistics from stream and summary files.")
    vs = VisualizeSample(
        os.path.join(setup.root_dir, "index"), task.tag, save_plots=True
    )
    vs.plot_cell_trajectory()
    vs.plot_stats()
    logger.debug("Done!")


def clean_up(config):
    setup = config.setup
    task = config.clean_up
    taskdir = os.path.join(setup.root_dir, "index")
    if os.path.isdir(taskdir):
        os.system(f"rm -f {taskdir}/r*/*{task.tag}.cxi")
    logger.debug("Done!")


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
    savedir = os.path.join(setup.root_dir, "timetool")

    expmt = setup.exp
    run = setup.run

    if not task.h5:
        smdr = SmallDataReader(expmt, run, savedir)
    else:
        smdr = SmallDataReader(expmt, run, savedir, task.h5)

    smdr.plot_timetool_diagnostics(output_type="png")


def calibrate_timetool(config):
    from btx.processing.rawimagetimetool import RawImageTimeTool

    setup = config.setup
    task = config.calibrate_timetool
    savedir = os.path.join(setup.root_dir, "timetool")

    expmt = setup.exp
    run = task.run
    order = int(task.order)
    figs = bool(task.figs)

    tt = RawImageTimeTool(expmt, savedir)
    logger.info(f"Calibrating timetool on run {run}")
    tt.calibrate(run, order, figs)
    logger.info(f"Writing calibration data to {savedir}/calib.")
    if figs:
        logger.info(f"Writing figures to {savedir}/figs.")


def timetool_correct(config):
    from btx.processing.rawimagetimetool import RawImageTimeTool
    from btx.misc.shortcuts import fetch_latest

    setup = config.setup
    task = config.timetool_correct
    savedir = os.path.join(setup.root_dir, "timetool")

    expmt = setup.exp
    run = task.run
    nominal = float(task.nominal_ps)
    model = task.model
    figs = bool(task.figs)
    tt = RawImageTimeTool(expmt, savedir)

    logger.info("Attempting to correct nominal delays using the timetool data.")
    if model:
        logger.info(f"Using model {model} for the correction.")
    else:
        latest_model = fetch_latest(
            f"{savedir}/calib/r*.out", int(str(run).split("-")[0])
        )
        if latest_model:
            model = latest_model
            logger.info(
                f"Most recent calibration model, {model}, will be used for timetool correction."
            )
        else:
            logger.info("No model found! Will return the nominal delay uncorrected!")

    tt.timetool_correct(run, nominal, model, figs)


### Bayesian Optimization


def bayesian_optimization(config):
    from btx.diagnostics.bayesian_optimization import BayesianOptimization

    """ Perform an iteration of the Bayesian optimization. """
    logger.info("Running an iteration of the Bayesian Optimization.")
    BayesianOptimization.run_bayesian_opt(config)
    logger.info("Done!")


def bo_init_samples_configs(config):
    from btx.diagnostics.bayesian_optimization import BayesianOptimization

    """ Generates the config files that will be used to generate the initial samples for the Bayesian optimization. """
    BayesianOptimization.init_samples_configs(config, logger)


def bo_aggregate_init_samples(config):
    from btx.diagnostics.bayesian_optimization import BayesianOptimization

    """ Aggregates the scores and parameters of the initial samples of the Bayesian optimization. """
    BayesianOptimization.aggregate_init_samples(config, logger)


### BO Benchmark


def check_status(statusfile, jobnames, frequency=None):
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
    timeout = (
        1200  # number of seconds to allow sbatch'ed jobs to run before exiting, float
    )
    if frequency is None:
        frequency = 5  # frequency in seconds to check on job completion, float

    while time.time() - start_time < timeout:
        if os.path.exists(statusfile) and not done:

            with open(statusfile, "r") as f:
                lines = f.readlines()
                finished = [l.strip("\n") for l in lines]
                if set(finished) == set(jobnames):
                    print(f"All done: {jobnames}")
                    done = True
                    os.remove(statusfile)
                    time.sleep(frequency * 5)
                    break
            time.sleep(frequency)
    return done


atot_thr_MIN = 100
atot_thr_MAX = 150
atot_thr_step = 10

son_min_MIN = 5
son_min_MAX = 15
son_min_step = 1

run_MIN = 16
run_MAX = 25


def fp_idx_bo_benchmark(config):
    from btx.interfaces.ischeduler import JobScheduler
    from btx.processing.indexer import Indexer
    from btx.misc.shortcuts import fetch_latest

    setup = config.setup
    task = config.find_peaks
    bay_opt = config.bayesian_optimization
    """ Perform adaptive peak finding on run. A grid of parameters is used to benchmark the Bayesian Optimization applied to peak finding. """
    taskdir = os.path.join(setup.root_dir, "index")
    os.makedirs(taskdir, exist_ok=True)
    script_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../btx/processing/peak_finder.py"
    )
    cell_dir_path = os.path.join(setup.root_dir, "cell")

    statusfile = os.path.join(setup.root_dir, "status.sh")
    # Empty the status file
    with open(statusfile, "w") as file:
        pass

    logger.info(
        f"Launching slurm jobs to perform peak finding then indexing with parameters \
                atot_thr in range({atot_thr_MIN}, {atot_thr_MAX} + {atot_thr_step}, {atot_thr_step})\n \
                and son_min in range({son_min_MIN}, {son_min_MAX} + {son_min_step}, {son_min_step})"
    )

    for atot_thr in np.arange(
        atot_thr_MIN, atot_thr_MAX + atot_thr_step, atot_thr_step
    ):
        for son_min in np.arange(son_min_MIN, son_min_MAX + son_min_step, son_min_step):
            # for atot_thr in [100]:
            # for son_min in np.arange(son_min_MIN, son_min_MAX + son_min_step, son_min_step):

            new_tag = f"{task.tag}_atot_thr_{atot_thr:04}_son_min_{son_min:04}"

            jobnames = []

            # STEP 1: Launch peak finding for every run
            task = config.find_peaks
            for iter_run in np.arange(run_MIN, run_MAX + 1):
                jobname = f"r{iter_run:04}_fp"
                # Write the command by specifying all the arguments that can be found in the config
                command = f"python {script_path}"
                command += f" -e {setup.exp} -r {iter_run} -d {setup.det_type} -o {os.path.join(taskdir ,f'r{iter_run:04}')} -t {new_tag}"
                mask_file = fetch_latest(
                    fnames=os.path.join(setup.root_dir, "mask", "r*.npy"), run=iter_run
                )
                if mask_file:
                    command += f" -m {mask_file}"
                if task.get("event_receiver") is not None:
                    command += f" --event_receiver={task.event_receiver}"
                if task.get("event_code") is not None:
                    command += f" --event_code={task.event_code}"
                if task.get("event_logic") is not None:
                    command += f" --event_logic={task.event_logic}"
                if task.get("psana_mask") is not None:
                    command += f" --psana_mask={task.psana_mask}"
                if task.get("pv_camera_length") is not None:
                    command += f" --pv_camera_length={task.pv_camera_length}"
                if task.get("min_peaks") is not None:
                    command += f" --min_peaks={task.min_peaks}"
                if task.get("max_peaks") is not None:
                    command += f" --max_peaks={task.max_peaks}"
                if task.get("npix_min") is not None:
                    command += f" --npix_min={task.npix_min}"
                if task.get("npix_max") is not None:
                    command += f" --npix_max={task.npix_max}"
                if task.get("amax_thr") is not None:
                    command += f" --amax_thr={task.amax_thr}"
                if task.get("atot_thr") is not None:
                    command += f" --atot_thr={atot_thr}"
                if task.get("son_min") is not None:
                    command += f" --son_min={son_min}"
                if task.get("peak_rank") is not None:
                    command += f" --peak_rank={task.peak_rank}"
                if task.get("r0") is not None:
                    command += f" --r0={task.r0}"
                if task.get("dr") is not None:
                    command += f" --dr={task.dr}"
                if task.get("nsigm") is not None:
                    command += f" --nsigm={task.nsigm}"
                if task.get("calibdir") is not None:
                    command += f" --calibdir={task.calibdir}"
                # Add the jobname in the statusfile
                addl_command = f"echo {jobname} | tee -a {statusfile}\n"
                command += f"\n{addl_command}"
                # Create the cell file
                shutil.copy(
                    os.path.join(cell_dir_path, f"{task.tag}.cell"),
                    os.path.join(cell_dir_path, f"{new_tag}.cell"),
                )
                logger.info(f"[LOOP] Cell file copied.")
                # Launch the Slurm job to perform "find_peaks" on the current run
                js = JobScheduler(
                    os.path.join(".", f"fp_{new_tag}.sh"),
                    queue=setup.queue,
                    ncores=bay_opt.ncores if bay_opt.get("ncores") is not None else 120,
                    jobname=f"fp_{new_tag}",
                    account=setup.account,
                    reservation=setup.reservation,
                )
                js.write_header()
                js.write_main(f"{command}\n", dependencies=["psana"])
                js.clean_up()
                js.submit(wait=False)
                jobnames.append(jobname)
                logger.info(
                    f"[LOOP] Launched a slurm job to perform peak finding for run {iter_run} of experiment {setup.exp} \
                                with parameters atot_thr = {atot_thr} and son_min = {son_min}..."
                )

            # Check that all peak finding slurm jobs have finished
            fp_finished = check_status(statusfile, jobnames)
            # Empty the status file
            with open(statusfile, "w") as file:
                pass

            # STEP 2: Launch indexing for every run
            task = config.index
            if fp_finished:
                jobnames = []
                for iter_run in np.arange(run_MIN, run_MAX + 1):
                    jobname = f"r{iter_run:04}_idx"
                    geom_file = fetch_latest(
                        fnames=os.path.join(setup.root_dir, "geom", "r*.geom"),
                        run=iter_run,
                    )
                    indexer_obj = Indexer(
                        exp=config.setup.exp,
                        run=iter_run,
                        det_type=config.setup.det_type,
                        tag=new_tag,
                        tag_cxi=new_tag,
                        taskdir=taskdir,
                        geom=geom_file,
                        cell=task.get("cell"),
                        int_rad=task.int_radius,
                        methods=task.methods,
                        tolerance=task.tolerance,
                        no_revalidate=task.no_revalidate,
                        multi=task.multi,
                        profile=task.profile,
                        queue=setup.get("queue"),
                        ncores=(
                            task.get("ncores")
                            if task.get("ncores") is not None
                            else 120
                        ),
                        time=(
                            task.get("time")
                            if task.get("time") is not None
                            else "1:00:00"
                        ),
                        mpi_init=False,
                        slurm_account=setup.account,
                        slurm_reservation=setup.reservation,
                    )
                    indexer_obj.launch(
                        addl_command=f"echo {jobname} | tee -a {statusfile}\n",
                        dont_report=True,
                        wait=False,
                    )
                    jobnames.append(jobname)
                    logger.info(
                        f"[LOOP] Launched a slurm job to perform indexing for run {iter_run} of experiment {setup.exp} \
                                    with parameters atot_thr = {atot_thr} and son_min = {son_min}..."
                    )

            # Check that all indexing slurm jobs have finished
            idx_finished = check_status(statusfile, jobnames)
            # Empty the status file
            with open(statusfile, "w") as file:
                pass

            # STEP 3: remove all .cxi files
            for iter_run in np.arange(run_MIN, run_MAX + 1):
                pattern = os.path.join(taskdir, f"r{iter_run:04}", f"*_{new_tag}.cxi")
                matching_files = glob.glob(pattern)
                for file in matching_files:
                    os.remove(file)
                logger.info(
                    f"[LOOP] All .cxi files have been removed for run {iter_run} of experiment {setup.exp} \
                                    with parameters atot_thr = {atot_thr} and son_min = {son_min}..."
                )

    logger.info("All slurm jobs have been launched!")
    logger.info("Done!")


def stream_analysis_bo_benchmark(config):
    from btx.interfaces.istream import launch_stream_analysis

    setup = config.setup
    task = config.stream_analysis
    """ Plot cell distribution and peakogram, write new cell file, and concatenate streams. """
    taskdir = os.path.join(setup.root_dir, "index")
    os.makedirs(os.path.join(taskdir, "figs"), exist_ok=True)
    os.makedirs(os.path.join(setup.root_dir, "cell"), exist_ok=True)

    logger.info(
        f"Launching stream analysis for find_peaks with parameters atot_thr in range({atot_thr_MIN}, {atot_thr_MAX} + {atot_thr_step}, {atot_thr_step})\n \
                                                                    and son_min in range({son_min_MIN}, {son_min_MAX} + {son_min_step}, {son_min_step})"
    )
    for atot_thr in np.arange(
        atot_thr_MIN, atot_thr_MAX + atot_thr_step, atot_thr_step
    ):
        for son_min in np.arange(son_min_MIN, son_min_MAX + son_min_step, son_min_step):
            new_tag = f"{task.tag}_atot_thr_{atot_thr:04}_son_min_{son_min:04}"
            launch_stream_analysis(
                os.path.join(taskdir, f"r*{new_tag}.stream"),
                os.path.join(taskdir, f"{new_tag}.stream"),
                os.path.join(taskdir, "figs"),
                os.path.join(taskdir, "stream_analysis.sh"),
                setup.queue,
                ncores=task.get("ncores") if task.get("ncores") is not None else 12,
                cell_only=(
                    task.get("cell_only")
                    if task.get("cell_only") is not None
                    else False
                ),
                cell_out=os.path.join(setup.root_dir, "cell", f"{new_tag}.cell"),
                cell_ref=task.get("ref_cell"),
                slurm_account=setup.account,
                slurm_reservation=setup.reservation,
                wait=False,
            )
            logger.info(
                f"[LOOP] Launched a slurm job to perform stream analysis for run {setup.run} of experiment {setup.exp} \
                            with parameters atot_thr = {atot_thr} and son_min = {son_min}..."
            )
    logger.info(f"All stream analysis have been launched")
    logger.info(f"Done!")


def merge_bo_benchmark(config):
    from btx.processing.merge import StreamtoMtz

    setup = config.setup
    task = config.merge
    """ Merge reflections from stream file and convert to mtz. """
    cell_dir_path = os.path.join(setup.root_dir, "cell")

    statusfile = os.path.join(setup.root_dir, "status.sh")
    # Empty the status file
    with open(statusfile, "w") as file:
        pass

    logger.info(
        f"Launching merge for find_peaks with parameters atot_thr in range({atot_thr_MIN}, {atot_thr_MAX} + {atot_thr_step}, {atot_thr_step})\n \
                                                                    and son_min in range({son_min_MIN}, {son_min_MAX} + {son_min_step}, {son_min_step})"
    )

    for atot_thr in np.arange(
        atot_thr_MIN, atot_thr_MAX + atot_thr_step, atot_thr_step
    ):
        for son_min in np.arange(son_min_MIN, son_min_MAX + son_min_step, son_min_step):
            new_tag = f"{task.tag}_atot_thr_{atot_thr:04}_son_min_{son_min:04}"

            taskdir = os.path.join(setup.root_dir, "merge", f"{new_tag}")
            input_stream = os.path.join(setup.root_dir, f"index/{new_tag}.stream")
            cellfile = os.path.join(setup.root_dir, f"cell/{new_tag}.cell")
            foms = task.foms.split(" ")
            stream_to_mtz = StreamtoMtz(
                input_stream,
                task.symmetry,
                taskdir,
                cellfile,
                queue=setup.get("queue"),
                ncores=task.get("ncores") if task.get("ncores") is not None else 32,
                mtz_dir=os.path.join(setup.root_dir, "solve", f"{new_tag}"),
                anomalous=(
                    task.get("anomalous")
                    if task.get("anomalous") is not None
                    else False
                ),
                slurm_account=setup.account,
                slurm_reservation=setup.reservation,
            )
            stream_to_mtz.cmd_partialator(
                iterations=task.iterations,
                model=task.model,
                min_res=task.get("min_res"),
                push_res=task.get("push_res"),
                max_adu=task.get("max_adu"),
            )
            for ns in [1, task.nshells]:
                stream_to_mtz.cmd_compare_hkl(
                    foms=foms, nshells=ns, highres=task.get("highres")
                )
            stream_to_mtz.cmd_hkl_to_mtz(
                highres=task.get("highres"),
                space_group=(
                    task.get("space_group")
                    if task.get("space_group") is not None
                    else 1
                ),
            )
            stream_to_mtz.cmd_report(foms=foms, nshells=task.nshells)
            stream_to_mtz.launch(wait=False)
            logger.info(
                f"[LOOP] Launched a slurm job to perform merge for run {setup.run} of experiment {setup.exp} \
                            with parameters atot_thr = {atot_thr} and son_min = {son_min}..."
            )

    # # Delete the cell files
    # for atot_thr in np.arange(atot_thr_MIN, atot_thr_MAX + atot_thr_step, atot_thr_step):
    #     for son_min in np.arange(son_min_MIN, son_min_MAX + son_min_step, son_min_step):
    #         new_tag = f"{task.tag}_atot_thr_{atot_thr:04}_son_min_{son_min:04}"
    #         os.remove(os.path.join(cell_dir_path, f"{new_tag}.cell"))
    # logger.info(f'Cell files deleted.')

    logger.info(f"All merge tasks have been launched")
    logger.info(f"Done!")


def bay_opt_geom(config):
    from btx.misc.shortcuts import TaskTimer
    from btx.interfaces.ischeduler import JobScheduler

    setup = config.setup
    script_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../btx/diagnostics/bay_opt_geom.py",
    )
    outdir = os.path.join(setup.root_dir, "bay_opt_geom")
    os.makedirs(outdir, exist_ok=True)
    statusfile = os.path.join(setup.root_dir, "status.sh")
    task_durations = dict({})
    """ Optimize and deploy the detector geometry from a silver behenate run using Bayesian Optimization. """

    # Main parameters
    length_fraction = 64
    resolution = 0.5  # pixels
    n_samples = 5
    num_iterations = 30

    num_jobs = 5

    with TaskTimer(task_durations, "total duration"):

        # Empty the status file
        with open(statusfile, "w") as file:
            pass

        jobnames = []

        for job_index in range(num_jobs):

            jobname = f"bay_opt_geom_job_{job_index}"

            # Write the command
            command = f"python {script_path}"
            command += f" --job_index {job_index}"
            command += f" --root_dir {setup.root_dir} -o {outdir} -r {setup.run} -e {setup.exp} -d {setup.det_type}"
            command += f" --length_fraction {length_fraction} --resolution {resolution}"
            command += f" --n_samples {n_samples} --num_iterations {num_iterations}"

            # Add the jobname in the statusfile
            addl_command = f"echo {jobname} | tee -a {statusfile}\n"
            command += f"\n{addl_command}"

            # Launch the Slurm job
            js = JobScheduler(
                os.path.join(".", f"{jobname}.sh"),
                queue=setup.queue,
                ncores=1,
                jobname=jobname,
                account=setup.account,
                reservation=setup.reservation,
            )
            js.write_header()
            js.write_main(f"{command}\n", dependencies=["psana"])
            js.clean_up()
            js.submit(wait=False)
            jobnames.append(jobname)
            logger.info(f"[LOOP] Launched job number {job_index}")

        # Check that all peak finding slurm jobs have finished
        jobs_finished = check_status(statusfile, jobnames, frequency=1)

        results_dict = {"scores": [], "cx_min": [], "cy_min": [], "opt_distances": []}
        for job_index in range(num_jobs):
            with open(
                os.path.join(
                    outdir,
                    f"bay_opt_geom_{length_fraction//2}th_length_{n_samples}_points_{num_iterations}_iter_job_{job_index}_results.yaml",
                ),
                "r",
            ) as yaml_file:
                results = yaml.load(yaml_file, Loader=yaml.Loader)
                results_dict["scores"].append(results["best_score"])
                results_dict["cx_min"].append(results["cx_min"])
                results_dict["cy_min"].append(results["cy_min"])
                results_dict["opt_distances"].append(results["opt_distance"])

        best_index = np.argmax(np.array(results_dict["scores"]))
        best_score = -results_dict["scores"][best_index]
        cx_min = results_dict["cx_min"][best_index]
        cy_min = results_dict["cy_min"][best_index]
        opt_distance = results_dict["opt_distances"][best_index]
        print("best_score:", best_score)
        print("cx_min:", cx_min)
        print("cy_min:", cy_min)
        print("opt_distance:", opt_distance)

        # TODO: Generate the .geom file

        logger.debug("Done!")

    logger.info(f"Total duration: {task_durations['total duration'][0]} seconds")
