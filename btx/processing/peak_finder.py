import numpy as np
import argparse
import h5py
import os
import requests
from mpi4py import MPI
from btx.interfaces.psana_interface import *
from psana.peakFinder.pypsalg import *
import matplotlib.pyplot as plt

class PeakFinder:
    
    """
    Perform adaptive peak-finding on a psana run and save the results to cxi format. 
    Adapted from psocake. More information about peak finding is available below:
    https://confluence.slac.stanford.edu/display/PSDM/Hit+and+Peak+Finding+Algorithms
    """
    
    def __init__(self, exp, run, det_type, xtc_dir, outdir, clen=None, tag='', mask=None, 
                 min_peaks=2, max_peaks=2048, npix_min=2, npix_max=30, amax_thr=80., 
                 atot_thr=120., son_min=7.0, peak_rank=3, r0=3.0, dr=2.0, nsigm=7.0):
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.n_hits_per_rank = []
        self.n_hits_total = 0
        if self.size > 2:
            self.root = 2
        else:
            self.root = 0
        
        # peak-finding algorithm parameters
        self.npix_min = npix_min # int, min number of pixels in peak
        self.npix_max = npix_max # int, max number of pixels in peak
        self.amax_thr = amax_thr # float, threshold on pixel amplitude
        self.atot_thr = atot_thr # float, threshold on total amplitude
        self.son_min = son_min # float, minimal S/N in peak; in psocake, nsigm=son_min
        self.peak_rank = peak_rank # radius in which central pix is a local max, int
        self.r0 = r0 # radius in pixels of ring for background evaluation, float
        self.dr = dr # width in pixels of ring for background evaluation, float
        self.nsigm = nsigm # intensity threshold to include pixel in connected group, float
        self.min_peaks = min_peaks # int, min number of peaks per image
        self.max_peaks = max_peaks # int, max number of peaks per image
        self.clen = clen # float, clen distance in mm
        self.outdir = outdir # str, path for saving cxi files

        # set up class
        self._set_up(exp, run, det_type, xtc_dir, mask)
        self._initialize_cxi(tag)
        
    def _set_up(self, exp, run, det_type, xtc_dir, mask_file=None):
        """
        Set up PsanaInterface object and distribute events between ranks.
        
        Parameters
        ----------
        exp : str
            experiment name
        run : int
            run number
        det_type : str
            detector name, e.g. jungfrau4M or epix10k2M
        xtc_dir : str
            path to xtc2 files
        mask_file : str
            path to mask in shape of unassembled detector, optional
        """
        # set up psana interface and its detector
        print(f"xtc_dir: {xtc_dir}")
        self.psi = Psana2Interface(exp, run, det_type, xtc_dir)
        self.iX = self.psi.det.iX.astype(np.int64)
        self.iY = self.psi.det.iY.astype(np.int64)
        if det_type == 'Rayonix':
            self.iX = np.expand_dims(self.iX, axis=0)
            self.iY = np.expand_dims(self.iY, axis=0)            
        self.ipx, self.ipy = self.psi.det.ipx, self.psi.det.ipy
        self.clen = self.psi.det.clen

        self._generate_mask(mask_file)
        self.n_hits, self.n_events = 0, 0
        self.powder_hits, self.powder_misses = np.zeros(self.psi.det.shape), np.zeros(self.psi.det.shape)      

    def _generate_mask(self, mask_file=None):
        """
        Generate mask.
        
        Parameters
        ----------
        mask_file : str
            path to mask in shape of unassembled detector, optional
        """
        self.mask = np.ones(self.psi.det.shape).astype(np.uint16)  
        if mask_file is not None:
            self.mask *= np.load(mask_file).astype(np.uint16)
        
    def _initialize_cxi(self, tag=''):
        """
        Set up the CXI files to which peak finding results will be saved.
        
        Parameters
        ----------
        tag : str
            file nomenclature suffix, optional
        """
        os.makedirs(self.outdir, exist_ok=True)
        if (tag != '') and (tag[0]!='_'):
            tag = '_' + tag
        self.tag = tag # track for writing summary file
        self.fname = os.path.join(self.outdir, f'{self.psi.exp}_r{self.psi.run:04}_{self.rank}{tag}.cxi')
        
        outh5 = h5py.File(self.fname, 'w')
        
        # entry_1 dataset for downstream processing with CrystFEL
        entry_1 = outh5.create_group("entry_1")
        keys = ['nPeaks', 'peakXPosRaw', 'peakYPosRaw', 'rcent', 'ccent', 'rmin',
                'rmax', 'cmin', 'cmax', 'peakTotalIntensity', 'peakMaxIntensity', 'peakRadius']
        ds_expId = entry_1.create_dataset("experimental_identifier", (10000,), maxshape=(None,), dtype=int)
        ds_expId.attrs["axes"] = "experiment_identifier"
        
        # for storing images in crystFEL format
        det_shape = self.psi.det.shape
        if self.psi.det_type == 'Rayonix':
            dim0, dim1 = det_shape[0], det_shape[1]
        else:
            dim0, dim1 = det_shape[0] * det_shape[1], det_shape[2]
        data_1 = entry_1.create_dataset('/entry_1/data_1/data', (10000, dim0, dim1), chunks=(1, dim0, dim1),
                                        maxshape=(None, dim0, dim1),dtype=np.float32)
        data_1.attrs["axes"] = "experiment_identifier"
        
        for key in ['powderHits', 'powderMisses', 'mask', 'powderHitsRank', 'powderMissesRank']:
            entry_1.create_dataset(f'/entry_1/data_1/{key}', (dim0, dim1), chunks=(dim0, dim1), maxshape=(dim0, dim1), dtype=float)
                
        # peak-related keys
        for key in keys:
            if key == 'nPeaks':
                ds_x = outh5.create_dataset(f'/entry_1/result_1/{key}', (10000,), maxshape=(None,), dtype=int)
                ds_x.attrs['minPeaks'] = self.min_peaks
                ds_x.attrs['maxPeaks'] = self.max_peaks
            else:
                ds_x = outh5.create_dataset(f'/entry_1/result_1/{key}', (10000,self.max_peaks), 
                                            maxshape=(None,self.max_peaks), chunks=(1,self.max_peaks), dtype=float)
            ds_x.attrs["axes"] = "experiment_identifier:peaks"
            
        # LCLS dataset to track event timestamps
        lcls_1 = outh5.create_group("LCLS")
        keys = ['eventNumber', 'photon_energy_eV']
        for key in keys:
            if key == 'photon_energy_eV':
                ds_x = lcls_1.create_dataset(f'{key}', (10000,), maxshape=(None,), dtype=float)
            else:
                ds_x = lcls_1.create_dataset(f'{key}', (10000,), maxshape=(None,), dtype=int)
            ds_x.attrs["axes"] = "experiment_identifier"
            
        ds_x = outh5.create_dataset('/LCLS/detector_1/EncoderValue', (10000,), maxshape=(None,), dtype=float)
        ds_x.attrs["axes"] = "experiment_identifier"

        outh5.close()
    
    def store_event(self, outh5, img, peaks, phot_energy, timestamp):
        """
        Store event's peaks in CXI file, converting to Cheetah conventions.
        
        Parameters
        ----------
        outh5 : h5py._hl.files.File
            open h5 file for storing output for this rank
        img : numpy.ndarray, shape (n_panels, n_panels_fs, n_panels_ss)
            calibrated detector data in shape of unassembled detector
        peaks : numpy.ndarray, shape (n_peaks, 17)
            results of peak finding algorithm for a single event
        phot_energy : float
            photon energy in eV
        timestamp : int
            timestamp of event
        """
        ch_rows = peaks[:,0] * img.shape[1] + peaks[:,1]
        ch_cols = peaks[:,2]
        
        # entry_1 entries for crystFEL processing
        outh5['/entry_1/data_1/data'][self.n_hits,:,:] = img.reshape(-1, img.shape[-1]) 
        outh5['/entry_1/result_1/nPeaks'][self.n_hits] = peaks.shape[0] 
        outh5['/entry_1/result_1/peakXPosRaw'][self.n_hits,:peaks.shape[0]] = ch_cols.astype('int')
        outh5['/entry_1/result_1/peakYPosRaw'][self.n_hits,:peaks.shape[0]] = ch_rows.astype('int')
        
        outh5['/entry_1/result_1/rcent'][self.n_hits,:peaks.shape[0]] = peaks[:,6] # row center of gravity
        outh5['/entry_1/result_1/ccent'][self.n_hits,:peaks.shape[0]] = peaks[:,7] # col center of gravity
        outh5['/entry_1/result_1/rmin'][self.n_hits,:peaks.shape[0]] = peaks[:,10] # minimal row of pixel group in the peak
        outh5['/entry_1/result_1/rmax'][self.n_hits,:peaks.shape[0]] = peaks[:,11] # maximal row of pixel group in the peak
        outh5['/entry_1/result_1/cmin'][self.n_hits,:peaks.shape[0]] = peaks[:,12] # minimal col pixel group in the peak
        outh5['/entry_1/result_1/cmax'][self.n_hits,:peaks.shape[0]] = peaks[:,13] # maximal col of pixel group in the peak
        
        outh5['/entry_1/result_1/peakTotalIntensity'][self.n_hits,:peaks.shape[0]] = peaks[:,5]
        outh5['/entry_1/result_1/peakMaxIntensity'][self.n_hits,:peaks.shape[0]] = peaks[:,4]
        outh5['/entry_1/result_1/peakRadius'][self.n_hits,:peaks.shape[0]] = self._compute_peak_radius(peaks)
        
        # LCLS dataset - currently omitting timetool information
        outh5['/LCLS/eventNumber'][self.n_hits] = timestamp
        outh5['/LCLS/photon_energy_eV'][self.n_hits] = phot_energy

    def curate_cxi(self):
        """
        Curate the CXI file by reshaping the keys to the number of hits
        and adding powders.
        """
        outh5 = h5py.File(self.fname,"r+")
        
        # resize the CrystFEL keys
        data_shape = outh5["/entry_1/data_1/data"].shape
        outh5['/entry_1/data_1/data'].resize((self.n_hits, data_shape[1], data_shape[2]))
        outh5[f'/entry_1/result_1/nPeaks'].resize((self.n_hits,))

        for key in ['peakXPosRaw', 'peakYPosRaw', 'rcent', 'ccent', 'rmin', 'rmax', 
                    'cmin', 'cmax', 'peakTotalIntensity', 'peakMaxIntensity', 'peakRadius']:
            outh5[f'/entry_1/result_1/{key}'].resize((self.n_hits, self.max_peaks))
            
        # add clen distance, then crop the LCLS keys
        outh5['/LCLS/detector_1/EncoderValue'][:] = self.clen
        for key in ['eventNumber', 'photon_energy_eV', 'detector_1/EncoderValue']:
            outh5[f'/LCLS/{key}'].resize((self.n_hits,))

        # add powders and mask, reshaping to match crystfel conventions
        outh5["/entry_1/data_1/powderHitsRank"][:] = self.powder_hits.reshape(-1, self.powder_hits.shape[-1])
        outh5["/entry_1/data_1/powderMissesRank"][:] = self.powder_misses.reshape(-1, self.powder_misses.shape[-1])
        outh5["/entry_1/data_1/mask"][:] = (1-self.mask).reshape(-1, self.mask.shape[-1]) # crystfel expects inverted values

        outh5.close()
    
    def _compute_peak_radius(self, peaks):
        """
        Compute radii of peaks based on their constituent pixels.
        
        Parameters
        ----------
        peaks : numpy.ndarray, shape (n_peaks, 17)
            results of peak finding algorithm for a single event
            
        Returns 
        -------
        radius : numpy.ndarray, shape (n_peaks)
            radii of peaks in pixels
        """
        cenX = self.iX[np.array(peaks[:, 0], dtype=np.int64),
                       np.array(peaks[:, 1], dtype=np.int64),
                       np.array(peaks[:, 2], dtype=np.int64)] + 0.5 - self.ipx
        cenY = self.iY[np.array(peaks[:, 0], dtype=np.int64),
                       np.array(peaks[:, 1], dtype=np.int64),
                       np.array(peaks[:, 2], dtype=np.int64)] + 0.5 - self.ipy
        return np.sqrt((cenX ** 2) + (cenY ** 2))
        
    def find_peaks_event(self, img):
        """
        Find peaks on a single image.
        
        Parameters
        ----------
        img : numpy.ndarray, shape (n_panels, n_panels_fs, n_panels_ss)
            calibrated detector data in shape of unassembled detector
            
        Returns
        -------
        peaks : numpy.ndarray, shape (n_peaks, 17)
            results of peak finding algorithm for this image
        """
        peaks = peaks_adaptive(img, self.mask, rank=self.peak_rank, r0=self.r0, dr=self.dr, 
                               nsigm=self.nsigm, npix_min=self.npix_min, npix_max=self.npix_max, 
                               amax_thr=self.amax_thr, atot_thr=self.atot_thr, son_min=self.son_min)
        return np.array(list_of_peak_parameters(peaks))
    
    def find_peaks(self):
        """
        Find all peaks in the images assigned to this rank.
        """
        
        outh5 = h5py.File(self.fname,"r+")
        empty_images = 0

        for nevt, evt in enumerate(self.psi.runner.events()):
            img = self.psi.detector.raw.raw(evt)
            if img is None:
                empty_images += 1
                continue
                
            peaks = self.find_peaks_event(img)
            if (peaks.shape[0] >= self.min_peaks) and (peaks.shape[0] <= self.max_peaks):
                phot_energy = self.psi.detector.raw.photonEnergy(evt)
                self.store_event(outh5, img, peaks, phot_energy, evt.timestamp)
                self.n_hits+=1 
                
            self.n_events += 1
            if peaks.shape[0] >= self.min_peaks:
                if self.powder_hits is None:
                    self.powder_hits = img
                else:
                    self.powder_hits = np.maximum(self.powder_hits, img)
            else:
                if self.powder_misses is None:
                    self.powder_misses = img
                else:
                    self.powder_misses = np.maximum(self.powder_misses, img)
        
        if empty_images != 0:
            print(f"Rank {self.rank} encountered {empty_images} empty images.")
        outh5.close()
        self.comm.Barrier()
                
    def summarize(self):
        """
        Summarize results and write to peakfinding.summary file.
        """
        # grab summary stats
        self.n_hits_per_rank = self.comm.gather(self.n_hits, root=self.root)
        self.n_hits_total = self.comm.reduce(self.n_hits, MPI.SUM, root=self.root)
        self.n_events_per_rank = self.comm.gather(self.n_events, root=self.root)
        powder_hits_all = np.max(np.array(self.comm.gather(self.powder_hits, root=self.root)), axis=0)
        powder_misses_all = np.max(np.array(self.comm.gather(self.powder_misses, root=self.root)), axis=0)

        if self.rank == 2:
            # write summary file
            with open(os.path.join(self.outdir, f'peakfinding{self.tag}.summary'), 'w') as f:
                f.write(f"Number of events processed: {np.sum(self.n_events_per_rank)}\n")
                f.write(f"Number of hits found: {self.n_hits_total}\n")
                f.write(f"Fractional hit rate: {self.n_hits_total/np.sum(self.n_events_per_rank)}\n")
                f.write(f'No. hits per rank: {self.n_hits_per_rank}')

            # add final powders, only needed in first and virtual cxis
            outh5 = h5py.File(self.fname, "r+")
            outh5["/entry_1/data_1/powderHits"][:] = powder_hits_all.reshape(-1, powder_hits_all.shape[-1])
            outh5["/entry_1/data_1/powderMisses"][:] = powder_misses_all.reshape(-1, powder_misses_all.shape[-1])
            outh5.close()

            # generate virtual dataset and list for
            vfname = os.path.join(self.outdir, f'{self.psi.exp}_r{self.psi.run:04}{self.tag}.cxi')
            self.generate_vds(vfname)

    def report(self, update_url):
        """
        Post summary to elog.
        Parameters
        ----------
        update_url : str
            elog URL for posting progress update
        """
        if self.rank == 2:
            requests.post(update_url, json=[{ "key": "Number of events processed", "value": f"{self.n_events_per_rank[-1]}" },
                                            { "key": "Number of hits found", "value": f"{self.n_hits_total}"},
                                            { "key": "Fractional hit rate", "value": f"{(self.n_hits_total/self.n_events_per_rank[-1]):.2f}"}, ])

    def add_virtual_dataset(self, vfname, fnames, dname, shape, dtype, mode='a'):
        """
        Add a virtual dataset to the hdf5 file.
        Parameters
        ----------
        vfname : str
            filename for virtual CXI file
        fnames : list of str
            list of individual CXI file names
        dname : str
            dataset path within hdf5 file
        shape : tuple
            shape of virtual dataset
        dtype : type
            dataset type, e.g. int or float
        mode : str
            'w' if first dataset, 'a' otherwise
        """
        layout = h5py.VirtualLayout(shape=(self.n_hits_total,) + shape[1:], dtype=dtype)
        cursor = 0
        for i,fn in enumerate(fnames):
            vsrc = h5py.VirtualSource(fn, dname, shape=(self.n_hits_per_rank[i],) + shape[1:])
            if len(shape) == 1:
                layout[cursor : cursor + self.n_hits_per_rank[i]] = vsrc
            else:
                layout[cursor : cursor + self.n_hits_per_rank[i], :] = vsrc
            cursor += self.n_hits_per_rank[i]
        with h5py.File(vfname, mode, libver="latest") as f:
            f.create_virtual_dataset(dname, layout, fillvalue=-1)

    def generate_vds(self, vfname):
        """
        Generate a virtual dataset to map all individual files for this run.
        Parameters
        ----------
        vfname : str
            filename for virtual CXI file    
        """
        if not hasattr(h5py, 'VirtualLayout'):
            raise Exception("HDF5>=1.10 and h5py>=2.9 are required to generate a virtual dataset")
            
        # retrieve list of file names
        fnames = []
        for fi in range(self.size):
            if self.n_hits_per_rank[fi] > 0:
                fnames.append(os.path.join(self.outdir, f'{self.psi.exp}_r{self.psi.run:04}_{fi}{self.tag}.cxi'))
        if len(fnames) == 0:
            sys.exit("No hits found")
        print(f"Files with peaks: {fnames}")
        with open(os.path.join(self.outdir, f'r{self.psi.run:04}{self.tag}.lst'), 'w') as f:
            for fn in fnames:
                f.write(f"{fn}\n")

        # retrieve datasets to populate in virtual hdf5
        dname_list, key_list, shape_list, dtype_list = [], [], [], []
        datasets = ['/entry_1/result_1', '/LCLS/detector_1', '/LCLS', '/entry_1/data_1']
        f = h5py.File(self.fname, "r")
        for dname in datasets:
            dset = f[dname]
            for key in dset.keys():
                if f'{dname}/{key}' not in datasets:
                    dname_list.append(dname)
                    key_list.append(key)
                    shape_list.append(dset[key].shape)
                    dtype_list.append(dset[key].dtype)
        f.close()    

        # populate virtual dataset
        for dnum in range(len(dname_list)):
            mode = 'a'
            if dnum == 0: 
                mode = 'w'
            dname = f'{dname_list[dnum]}/{key_list[dnum]}'
            if key_list[dnum] not in ['mask', 'powderHits', 'powderMisses']:
                self.add_virtual_dataset(vfname, fnames, dname, shape_list[dnum], dtype_list[dnum], mode=mode)
            else:
                layout = h5py.VirtualLayout(shape=shape_list[dnum], dtype=dtype_list[dnum])
                vsrc = h5py.VirtualSource(fnames[0], dname, shape=shape_list[dnum])
                layout[:] = vsrc
                with h5py.File(vfname, "a", libver="latest") as f:
                     f.create_virtual_dataset(dname, layout, fillvalue=-1)

def visualize_hits(fname, exp, run, det_type, savepath=None, vmax_ind=3, vmax_powder=5):
    """
    Visualize a random selection of hits stored in the CXI file.
    
    Parameters
    ----------
    fname : str
        path to CXI file
    exp : str
        experiment name
    run : int
        run number 
    det_type : str
        detector type
    savename : str
        directory to which to save plots
    vmax_ind : float
        vmax for plot of individual hits set to vmax_ind * mean(image)
    vmax_powder : float
        vmax for plot of powder hits set to vmax_powder * mean(image)
    """
    psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
    pixel_index_map = retrieve_pixel_index_map(psi.det.geometry(psi.run))
    
    f = h5py.File(fname, 'r')
    mask = 1 - f["/entry_1/data_1/mask"][:] # need to invert from CrystFEL convention
    powder_hits = f['entry_1/data_1/powderHits'][:]
    powder_misses = f['entry_1/data_1/powderMisses'][:]
    shape = f['entry_1/data_1/data'].shape
    indices = np.sort(np.random.randint(low=0, high=shape[0], size=9))
    hits = f['entry_1/data_1/data'][indices]
    
    if det_type is not 'Rayonix':
        hits = hits.reshape(hits.shape[0], *psi.det.shape())
        hits = assemble_image_stack_batch(hits, pixel_index_map)
        mask = assemble_image_stack_batch(mask.reshape(psi.det.shape()), pixel_index_map)
        powder_hits = assemble_image_stack_batch(powder_hits.reshape(psi.det.shape()), pixel_index_map)
        powder_misses = assemble_image_stack_batch(powder_misses.reshape(psi.det.shape()), pixel_index_map)
    
    # individual hits
    fig1, axs = plt.subplots(nrows=3, ncols=3, figsize=(6,6),dpi=180)
    n_hits = hits.shape[0]
    rand_sel = np.random.randint(0, high=n_hits, size=9)
    
    k=0
    for i in np.arange(3):
        for j in np.arange(3):
            if k >= n_hits:
                break
            axs[i,j].imshow(hits[k] * mask, vmin=0,vmax=vmax_ind*hits[k].mean(),cmap='Greys')
            axs[i,j].axis('off')
            npeaks = f['entry_1/result_1/nPeaks'][rand_sel[k]]
            axs[i,j].set_title(f'# of peaks: {npeaks}')
            for ipeak in np.arange(npeaks):
                panel_num = f['entry_1/result_1/peakYPosRaw'][rand_sel[k]] // psi.det.shape()[1]
                panel_row = f['entry_1/result_1/peakYPosRaw'][rand_sel[k]] % psi.det.shape()[1]
                panel_col = f['entry_1/result_1/peakXPosRaw'][rand_sel[k]]
                pixel = pixel_index_map[int(panel_num[ipeak]), int(panel_row[ipeak]), int(panel_col[ipeak])]
                circle = plt.Circle((pixel[1],pixel[0]),20, color='blue', alpha=0.2)
                axs[i,j].add_patch(circle)
            k+=1
            
    if savepath is not None:
        fig1.savefig(os.path.join(savepath, "peakfinding_hits.png"), bbox_inches='tight', dpi=300)
        
    # 'powder' hits and misses
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    
    ax1.imshow(mask*powder_hits, vmin=0, vmax=vmax_powder)
    ax2.imshow(mask*powder_misses, vmin=0, vmax=vmax_powder)
    ax1.set_title("Powder Hits", fontsize=18)
    ax2.set_title("Powder Misses", fontsize=18)
    
    if savepath is not None:
        fig2.savefig(os.path.join(savepath, "peakfinding_powderhits.png"), bbox_inches='tight', dpi=300)
    
    f.close()

#### For command line use ####
            
def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp', help='Experiment name', required=True, type=str)
    parser.add_argument('-r', '--run', help='Run number', required=True, type=int)
    parser.add_argument('-d', '--det_type', help='Detector name, e.g epix10k2M or jungfrau4M',  required=True, type=str)
    parser.add_argument('-x', '--xtc_dir', help='Path to xtc2 folder', required=True, type=str)
    parser.add_argument('-o', '--outdir', help='Output directory for cxi files', required=True, type=str)
    parser.add_argument('-c', '--clen', help='Distance between source and camera', required=False, type=float)
    parser.add_argument('-t', '--tag', help='Tag to append to cxi file names', required=False, type=str, default='')
    parser.add_argument('-m', '--mask', help='Binary mask', required=False, type=str)
    parser.add_argument('--min_peaks', help='Minimum number of peaks per image', required=False, type=int, default=2)
    parser.add_argument('--max_peaks', help='Maximum number of peaks per image', required=False, type=int, default=2048)
    parser.add_argument('--npix_min', help='Minimum number of pixels per peak', required=False, type=int, default=2)
    parser.add_argument('--npix_max', help='Maximum number of pixels per peak', required=False, type=int, default=30)
    parser.add_argument('--amax_thr', help='Minimum intensity threshold for starting a peak', required=False, type=float, default=80.)
    parser.add_argument('--atot_thr', help='Minimum summed intensity threshold for pixel collection', required=False, type=float, default=120.)
    parser.add_argument('--son_min', help='Minimum signal-to-noise ratio to be considered a peak', required=False, type=float, default=7.0)
    parser.add_argument('--peak_rank', help='Radius in which central peak pixel is a local maximum', required=False, type=int, default=3)
    parser.add_argument('--r0', help='Radius of ring for background evaluation in pixels', required=False, type=float, default=3.0)
    parser.add_argument('--dr', help='Width of ring for background evaluation in pixels', required=False, type=float, default=2.0)
    parser.add_argument('--nsigm', help='Intensity threshold to include pixel in connected group', required=False, type=float, default=7.0)
    parser.add_argument('--calibdir', help='Alternative calibration directory', required=False, type=str)
    
    return parser.parse_args()

if __name__ == '__main__':
    
    params = parse_input()
    pf = PeakFinder(exp=params.exp, run=params.run, det_type=params.det_type, xtc_dir=params.xtc_dir, outdir=params.outdir,
                    clen=params.clen, tag=params.tag, mask=params.mask, min_peaks=params.min_peaks, max_peaks=params.max_peaks,
                    npix_min=params.npix_min, npix_max=params.npix_max, amax_thr=params.amax_thr, atot_thr=params.atot_thr, 
                    son_min=params.son_min, peak_rank=params.peak_rank, r0=params.r0, dr=params.dr, nsigm=params.nsigm)
    pf.find_peaks()
    pf.curate_cxi()
    pf.summarize()
