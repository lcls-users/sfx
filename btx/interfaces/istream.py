import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from btx.misc.xtal import compute_resolution
import glob
import argparse
import os
import requests
from btx.interfaces.ischeduler import JobScheduler
from btx.interfaces.ielog import update_summary, elog_report_post

class StreamInterface:
    
    def __init__(self, input_files, cell_only=False, mpi_init=True):
        if mpi_init:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.rank = 0
            self.size = 1

        self.cell_only = cell_only # bool, if True only extract unit cell params
        self.input_files = input_files # list of stream file(s)
        self.stream_data, self.file_limits_cell, self.file_limits_refn = self.read_all_streams(self.input_files)
        if self.rank == 0:
            self.compute_cell()
            self.store_stats()
            if not self.cell_only:
                self.compute_resolution()
    
    def read_all_streams(self, input_files):
        """
        Read stream file(s), distributing files across ranks if available.
        
        Parameters
        ----------
        input_files : list of str
            stream file(s) to parse
        
        Returns
        -------
        stream_data : dict
            cell parameters and (if not self.cell_only) reflection information
        file_limits : numpy.ndarray, shape (n_files)
            indices of stream_data's first dimension that indicate start/end of each file
        """
        # processing all files given to each rank
        stream_data_rank = {}
        file_limits_rank_cell = []
        file_limits_rank_refn = []
        
        input_sel = self.distribute_streams(input_files) 
        if len(input_sel) != 0:
            for ifile in input_sel:
                single_stream_data = self.read_stream(ifile)
                file_limits_rank_cell.append(len(single_stream_data['a']))
                if not self.cell_only:
                    file_limits_rank_refn.append(len(single_stream_data['h']))
                if len(stream_data_rank) == 0:
                    stream_data_rank = single_stream_data
                else:
                    [stream_data_rank[key].extend(single_stream_data[key]) for key in stream_data_rank.keys()]

        # amassing files from different ranks
        stream_data = {}
        for key in stream_data_rank.keys():
            stream_data[key] = self.comm.gather(stream_data_rank[key], root=0)
        file_limits_cell = self.comm.gather(file_limits_rank_cell, root=0)
        file_limits_refn = self.comm.gather(file_limits_rank_refn, root=0)
        
        if self.rank == 0:
            for key in stream_data.keys():
                if key == 'n_crystal':
                    stream_data[key] = [np.array(arr) for arr in stream_data[key]]
                    for narr in range(1, len(stream_data[key])):
                        stream_data[key][narr] += stream_data[key][narr-1][-1]+1
                stream_data[key] = np.concatenate(np.array(stream_data[key], dtype=object))
                if key in ['n_crystal','n_chunk', 'n_crystal_cell', 'n_lattice', 'image_num', 'h', 'k', 'l']:
                    stream_data[key] = stream_data[key].astype(int)
                else:
                    stream_data[key] = stream_data[key].astype(float)
                if key in ['a','b','c']:
                    stream_data[key] *= 10.0 # convert from nm to Angstrom
            file_limits_cell = np.cumsum(np.append(np.array([0]), np.concatenate(np.array(file_limits_cell, dtype=object))))
            file_limits_refn = np.cumsum(np.append(np.array([0]), np.concatenate(np.array(file_limits_refn, dtype=object))))

        return stream_data, file_limits_cell, file_limits_refn
        
    def distribute_streams(self, input_files):
        """
        Evenly distribute stream files among available ranks.
        
        Parameters
        ----------
        input_files : list of str
            list of input stream files to read
    
        Returns
        -------
        input_sel : list of str
            select list of input stream files for this rank
        """
        self.n_crystal = -1
        
        # divvy up files
        n_files = len(input_files)
        split_indices = np.zeros(self.size)
        for r in range(self.size):
            num_per_rank = n_files // self.size
            if r < (n_files % self.size):
                num_per_rank += 1
            split_indices[r] = num_per_rank
        split_indices = np.append(np.array([0]), np.cumsum(split_indices)).astype(int) 
        return input_files[split_indices[self.rank]:split_indices[self.rank+1]]
    
    def read_stream(self, input_file=None):
        """
        Read a single stream file. Function possibly adapted from CrystFEL.
        
        Parameters
        ----------
        input_file : str
            stream file to parse
        
        Returns
        -------
        single_stream_data : dict
            cell parameters and (if not self.cell_only) reflection information
        """
        # set up storage arrays
        single_stream_data = {}
        keys = ['a','b','c','alpha','beta','gamma','n_crystal','n_chunk', 
                'n_crystal_cell','n_lattice','image_num','highres_A',
                'residual','det_shift_x','det_shift_y']
        if not self.cell_only:
            keys.extend(['h','k','l','sumI','sigI','maxI'])
        for key in keys:
            single_stream_data[key] = []
        
        # parse stream file
        if input_file is not None:
            n_chunk = -1
            n_crystal_cell = -1
            in_refl = False

            f = open(input_file)
            for lc,line in enumerate(f):
                if line.find("Begin chunk") != -1:
                    n_lattice = 0
                    n_chunk += 1
                    if in_refl:
                        in_refl = False
                        print(f"Warning! Line {lc} associated with chunk {n_chunk} is problematic: {line}")
                        
                if line.find("Image serial number") != -1:
                    single_stream_data['image_num'].append(int(line.split()[-1]))

                if line.find("Cell parameters") != -1:
                    cell = line.split()[2:5] + line.split()[6:9]
                    for ival,param in enumerate(['a','b','c','alpha','beta','gamma']):
                        single_stream_data[param].append(float(cell[ival]))
                    self.n_crystal+=1
                    n_crystal_cell += 1
                    single_stream_data['n_crystal_cell'].append(n_crystal_cell)
                    single_stream_data['n_chunk'].append(n_chunk)
                    if self.cell_only:
                        single_stream_data['n_crystal'].append(self.n_crystal)
                    n_lattice += 1
                    
                if line.find("diffraction_resolution_limit") != -1:
                    single_stream_data['highres_A'].append(float(line.split()[-2]))

                if line.find("final_residual") != -1:
                    single_stream_data['residual'].append(float(line.split()[-1]))
                    
                if line.find("det_shift") != -1:
                    single_stream_data['det_shift_x'].append(float(line.split()[3]))
                    single_stream_data['det_shift_y'].append(float(line.split()[-2]))
                    
                if line.find("End chunk") != -1:
                    single_stream_data['n_lattice'].append(n_lattice)

                if not self.cell_only:
                    if line.find("Reflections measured after indexing") != -1:
                        in_refl = True
                        continue

                    if line.find("End of reflections") != -1:
                        in_refl = False
                        continue

                    if in_refl:
                        if line.find("h    k    l") == -1:
                            try:
                                reflection = [val for val in line.split()]
                                for ival,param in enumerate(['h','k','l','sumI','sigI','maxI']):
                                    if ival < 3:
                                        single_stream_data[param].append(int(reflection[ival]))
                                    else:
                                        single_stream_data[param].append(float(reflection[ival]))
                                single_stream_data['n_crystal'].append(self.n_crystal)
                            except ValueError:
                                print(f"Couldn't parse line {lc}: {line}")
                            continue
            f.close()

            if not self.cell_only:
                if len(single_stream_data['h']) == 0:
                    print(f"Warning: no indexed reflections found in {input_file}!")
                
        return single_stream_data
    
    def compute_cell(self):
        """ 
        Compute the mean and std. dev. of the unit cell parameters in A/degrees. 
        """
        keys = ['a','b','c','alpha','beta','gamma']
        self.cell_params = np.array([np.mean(self.stream_data[key]) for key in keys])
        self.cell_params_std = np.array([np.std(self.stream_data[key]) for key in keys])
        
    def store_stats(self):
        """ Store statistics regarding indexing rate as self variables."""
        counts = np.bincount(self.stream_data['n_lattice'])
        self.n_indexed = np.sum(counts[1:]) # no. indexed events, including those with multiple lattices
        self.n_unindexed = counts[0] # no. of events processed from peak finder
        self.n_multiple = np.sum(counts[2:]) # no. events with multiple lattices

    def compute_resolution(self):
        """
        Compute the resolution in inverse Angstrom of each reflection.
        """
        cell = self.get_cell_parameters()
        hkl = np.array([self.stream_data[key] for key in ['h','k','l']]).T
        cindex, ccounts = np.unique(self.stream_data['n_crystal'], return_counts=True)
        cell = np.repeat(cell, ccounts, axis=0)
        self.stream_data['d'] = compute_resolution(cell, hkl)
        
    def get_cell_parameters(self):
        """ Retrieve unit cell parameters: [a,b,c,alpha,beta,gamma] in A/degrees. """
        keys = ['a','b','c','alpha','beta','gamma']
        return np.array([self.stream_data[key] for key in keys]).T

    def plot_cell_parameters(self, output=None):
        """
        Plot histograms of the unit cell parameters, indicating the median
        cell axis and angle values.
        
        Parameters
        ----------
        output : string, default=None
            if supplied, path for saving png of unit cell parameters
        """
        cell = self.get_cell_parameters()
        
        f, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(12,5))

        labels = ["$a$", "$b$", "$c$", r"$\alpha$", r"$\beta$", r"$\gamma$"]
        for i,ax in enumerate([ax1,ax2,ax3,ax4,ax5,ax6]):
            ax.hist(cell[:,i], bins=100, color='black')
            if i<3: 
                ax.set_title(labels[i] + f"={np.mean(cell[:,i]):.3f}" + " ${\mathrm{\AA}}$")
            else: 
                ax.set_title(labels[i] + f"={np.mean(cell[:,i]):.3f}" + "$^{\circ}$")
            if i == 0 or i == 3: 
                ax.set_ylabel("No. crystals")

        f.subplots_adjust(hspace=0.4)
        
        if output is not None:
            f.savefig(output, dpi=300)
            
    def plot_peakogram(self, output=None):
        """
        Generate a peakogram of the stream data.
        
        Parameters
        ----------
        output : string, default=None
            if supplied, path for saving png of peakogram
        """
        if self.cell_only:
            print("Cannot plot peakogram because only cell parameters were extracted")
            return
        
        figsize = 8
        peakogram_bins = [500, 500]

        fig = plt.figure(figsize=(figsize, figsize), dpi=300)
        gs = fig.add_gridspec(2, 2)

        irow = 0
        ax1 = fig.add_subplot(gs[irow, 0:])

        n_neg_peaks = len(np.where(self.stream_data['maxI']<0)[0])
        if n_neg_peaks != 0:
            print(f"Warning: {100*n_neg_peaks/len(self.stream_data['maxI']):.1f}% of reflections have negative intensity.")
            print("These will be excluded from the peakogram.")
        ax1.set_title(f"Peakogram ({len(self.stream_data['h']) - n_neg_peaks} reflections)")

        H, xedges, yedges = np.histogram2d(np.log10(self.stream_data['maxI'][self.stream_data['maxI']>0]),
                                           self.stream_data['d'][self.stream_data['maxI']>0],
                                           bins=peakogram_bins)
        im = ax1.pcolormesh(yedges, xedges, H, cmap='gray', norm=LogNorm())
        plt.colorbar(im)
        ax1.set_xlabel("1/d (${\mathrm{\AA}}$$^{-1}$)")
        ax1.set_ylabel(f"log(peak intensity)")

        irow += 1
        ax2 = fig.add_subplot(gs[irow, 0])
        im = ax2.hexbin(self.stream_data['sumI'], self.stream_data['maxI'], 
                        gridsize=100, mincnt=1, norm=LogNorm(), cmap='gray')

        ax2.set_xlabel(f'sum in peak')
        ax2.set_ylabel(f'max in peak')

        ax3 = fig.add_subplot(gs[irow, 1])
        im = ax3.hexbin(self.stream_data['sigI'], self.stream_data['maxI'], 
                        gridsize=100, mincnt=1, norm=LogNorm(), cmap='gray')
        ax3.set_xlabel('sig(I)')
        ax3.set_yticks([])
        plt.colorbar(im, label='No. reflections')
        
        if output is not None:
            fig.savefig(output, dpi=300)

    def plot_hits_distribution(self, output=None):
        """ 
        Plot the distribution of unindexed, single hit, and multi-lattice events.

        Parameters
        ----------
        output : string, default=None  
            if supplied, path for saving png file
        """
        f, ax1 = plt.subplots(figsize=(6,4))

        counts = np.bincount(self.stream_data['n_lattice'])
        ax1.bar(range(0, len(counts)), counts, color='black')
        ax1.set_xlabel("Number of lattices", fontsize=14)
        ax1.set_ylabel('Number of images', fontsize=14)
        ax1.set_title("Distribution of single vs. multi-hits", fontsize=14)

        if output is not None:
            fig.savefig(output, dpi=300)

    def report(self, tag=None):
        """
        Summarize the cell parameters and optionally report to the elog.
    
        Parameters
        ----------
        tag : str
            suffix for naming summary file
        """
        # write summary file
        if tag is not None:
            tag = "_" + tag
        else:
            tag = ""

        summary_file = os.path.join(os.path.dirname(self.input_files[0]), f"stream{tag}.summary")
        with open(summary_file, 'w') as f:
            f.write("Cell mean: " + " ".join(f"{self.cell_params[i]:.3f}" for i in range(self.cell_params.shape[0])) + "\n")
            f.write("Cell std: " + " ".join(f"{self.cell_params_std[i]:.3f}" for i in range(self.cell_params.shape[0])) + "\n")
            f.write(f"Number of indexed events: {self.n_indexed}" + "\n")
            f.write(f"Fractional indexing rate: {self.n_indexed/(self.n_indexed+self.n_unindexed):.2f}" + "\n")
            f.write(f"Fraction of indexed with multiple lattices: {self.n_multiple/self.n_indexed:.2f}" + "\n")
            
        # report to elog
        update_url = os.environ.get('JID_UPDATE_COUNTERS')
        if update_url is not None:
            labels = ["a", "b", "c", "alpha", "beta", "gamma"]
            elog_json = [{"key": labels[i], "value": f"{self.cell_params[i]:.3f} +/- {self.cell_params_std[i]:.3f}"} for i in range(len(labels))]
            elog_json.append({'key': 'Number of indexed events', 'value': f'{self.n_indexed}'})
            elog_json.append({'key': 'Fractional indexing rate', 'value': f'{self.n_indexed/(self.n_indexed+self.n_unindexed):.2f}'})
            elog_json.append({'key': 'Fraction of indexed with multiple lattices', 'value': f'{self.n_multiple/self.n_indexed:.2f}'})
            requests.post(update_url, json=elog_json)

    @property
    def stream_summary(self) -> dict:
        """! Return a dictionary of key/values to post to the eLog.

        @return (dict) summary_dit Key/values parsed by eLog posting function.
        """
        summary_dict: dict = {}
        key_strings: list = ['(SA) Cell mean:',
                             '(SA) Cell std:',
                             '(SA) Number of indexed events:',
                             '(SA) Fractional indexing rate:',
                             '(SA) Fraction of indexed with multiple lattices:']
        summary_dict.update({
            key_strings[0] : ' '.join(f'{self.cell_params[i]:.3f}'
                                      for i in range(self.cell_params.shape[0])),
            key_strings[1] : ' '.join(f'{self.cell_params_std[i]:.3f}'
                                      for i in range(self.cell_params.shape[0])),
            key_strings[2] : f'{self.n_indexed}',
            key_strings[3] : f'{self.n_indexed/(self.n_indexed+self.n_unindexed):.2f}',
            key_strings[4] : f'{self.n_multiple/self.n_indexed:.2f}'
        })
        return summary_dict

    def copy_from_stream(self, stream_file, chunk_indices, crystal_indices, output):
        """
        Add the indicated crystals from the input to the output stream.
        There may be multiple crystals per chunk with multi-lattice indexing.
        
        Parameters
        ----------
        stream_file : str
            input stream file
        chunk_indices : list
            indices of the chunks from stream file to transfer
        crystal_indices : list
            indices of the crystals from stream file to transfer
        output : string
            path to output .stream file     
        """

        f_out = open(output, "a")
        f_in = open(stream_file, "r")
        
        n_chunk = -1
        n_crystal = -1
        in_refl = False
        write = False
        in_header = True
        
        target = [tuple((chunk_indices[i], crystal_indices[i])) for i in range(len(crystal_indices))]

        for lc,line in enumerate(f_in):

            # copy header of stream file before first chunk begins
            if in_header:
                if line.find("Indexing methods") != -1:
                    in_header = False
                f_out.write(line)
                
            # copy over relevant chunks
            if line.find("Begin chunk") != -1:
                if in_refl:
                    in_refl = False
                    print(f"Warning! Line {lc} associated with chunk {n_chunk} is problematic: {line}")
                n_chunk += 1

                #if (n_chunk in chunk_indices) and (n_crystal in crystal_indices):
                if (n_chunk, n_crystal+1) in target:
                    write = True    
                    
            if line.find("Cell parameters") != -1:
                cell = line.split()[2:5] + line.split()[6:9]
                n_crystal += 1
                
                if (n_chunk, n_crystal+1) not in target:
                    write = False

            if write:
                f_out.write(line)

            if line.find("End chunk") != -1:
                write=False
            
        f_in.close()
        f_out.close()
            
    def write_stream(self, selection, output):
        """
        Write a new stream file from a selection of crystals from the input
        stream file(s).
        
        Parameters
        ----------
        selection : numpy.ndarray, 1d
            indices of the crystals in self.stream_data to retain
        output : string
            path to output .stream file 
        """
        for nf,infile in enumerate(self.input_files):
            lower, upper = self.file_limits_cell[nf], self.file_limits_cell[nf+1]
            idx = np.where((selection>=lower) & (selection<upper))[0]
            sel_indices = selection[idx] - lower
            if len(sel_indices) > 0:
                print(f"Copying {len(sel_indices)} crystals from file {infile}")
                chunk_indices = self.stream_data['n_chunk'][lower:upper][sel_indices]
                crystal_indices = self.stream_data['n_crystal_cell'][lower:upper][sel_indices]
                self.copy_from_stream(infile, chunk_indices, crystal_indices, output)

def read_cell_file(cell_file):
    """
    Extract cell parameters from a CrystFEL cell file.
    
    Parameters
    ----------
    cell_file : str
        CrystFEL cell file
        
    Returns
    -------
    array of cell parameters: [a,b,c,alpha,beta,gamma] in Angstrom/degrees
    """
    with open(cell_file, "r") as f:
        params = f.readlines()[-6:]
        return [float(line.split("=")[1].split()[0]) for line in params]

def write_cell_file(cell, output_file, input_file=None):
    """
    Write a new CrystFEL-style cell file with the same lattice type as
    the input file (or primitive triclinic if none given) and the cell 
    parameters changed to input cell.
    
    Parameters
    ----------
    cell : np.array, 1d
        unit cell parameters [a,b,c,alpha,beta,gamma], in Å/degrees
    output_file : str
        output CrystFEL unit cell file
    input_file : str
        input CrystFEL unit cell file, optional
    """
    from itertools import islice

    if input_file is not None:
        with open(input_file, "r") as in_cell:
            header = in_cell.readlines()
            index = [i for i,x in enumerate(header) if "a =" in x]
            header = header[:index[0]]
    else:
        header = ['CrystFEL unit cell file version 1.0\n',
                  '\n',
                  'lattice_type = triclinic\n',
                  'centering = P\n']

    outfile = open(output_file, "w")
    for item in header:
        outfile.write(item)
    outfile.write(f'a = {cell[0]:.3f} A\n')
    outfile.write(f'b = {cell[1]:.3f} A\n')
    outfile.write(f'c = {cell[2]:.3f} A\n')
    outfile.write(f'al = {cell[3]:.3f} deg\n')
    outfile.write(f'be = {cell[4]:.3f} deg\n')
    outfile.write(f'ga = {cell[5]:.3f} deg\n')
    outfile.close()

def cluster_cell_params(cell, out_clusters, out_cell, in_cell=None, eps=5, min_samples=5):
    """
    Apply DBScan clustering to unit cell parameters and write the most
    prevalent unit cell parameters to a CrystFEL cell file. Unit cell
    parameters from all valid clusters are written to a separate file.
    
    Parameters
    ----------
    cell : numpy.ndarray, shape (n_crystals, 6)
        [a,b,c,alpha,beta,gamma] in Angstrom and degrees
    out_clusters : str
        file to write clustering results to
    out_cell : str
        output CrystFEL unit cell file
    in_cell : str
        input CrystFEL unit cell file to copy lattice type from, optional
    eps : float
        max distance for points to be considered neighbors
    min_samples : int
        minimum number of samples for a cluster to be valid
        
    Returns
    -------
    labels : numpy.ndarray, shape (n_crystals,)
        cluster labels for input cell array
    """
    from sklearn.cluster import DBSCAN
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(cell)
    counts = np.bincount(clustering.labels_[clustering.labels_!=-1])
    sort_idx = np.argsort(counts)[::-1]
    
    cols = ['n_crystals', 'fraction', 'a', 'b', 'c', 'alpha', 'beta', 'gamma']
    results = np.array([np.concatenate((np.array([counts[nc], counts[nc]/np.sum(counts)]), 
                                        np.median(cell[clustering.labels_==nc], axis=0))) for nc in sort_idx])
    
    if len(results) == 0:
        print(f"No clusters found from {cell.shape[0]} crystals.")
        return

    fmt=['%d', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f', '%.2f']
    np.savetxt(out_clusters, results, header=' '.join(cols), fmt=fmt)
    
    write_cell_file(results[0][2:], out_cell, input_file=in_cell) 
    
    return clustering.labels_

def launch_stream_analysis(in_stream, out_stream, fig_dir, tmp_exe, queue, ncores, 
                           cell_only=False, cell_out=None, cell_ref=None, addl_command=None,
                           slurm_account="lcls"):
                           
    """
    Launch stream analysis task using iScheduler.
    
    Parameters
    ----------
    in_stream : str
        glob-compatible path to input stream file(s)
    out_stream : str
        name of output concatenated stream file
    fig_dir : str
        directory to write peakogram and cell distribution plot to
    tmp_exe : str
        name of temporary executable file
    queue : str 
        queue to submit job to
    ncores : int
        minimum of number of cores and number of stream files
    cell_only : bool
        if True, do not plot peakogram
    cell_out : str
        output CrystFEL-style cell file
    cell_ref : str
        CrystFEL cell file to copy symmetry from
    addl_command : str
        additional command to add to end of job to launch
    """
    ncores_max = len(glob.glob(in_stream))
    if ncores > ncores_max:
        ncores = ncores_max
    tag = out_stream.split("/")[-1].split(".")[0]

    script_path = os.path.abspath(__file__)
    command = f"python {script_path} -i='{in_stream}' -o {fig_dir} -t {tag}"
    if cell_only:
        command += " --cell_only"
    if cell_out is not None:
        command += f" --cell_out={cell_out}"
        if cell_ref is not None:
            command += f" --cell_ref={cell_ref}"
        
    js = JobScheduler(tmp_exe, ncores=ncores, jobname=f'stream_analysis', queue=queue,
                      account=slurm_account)
    js.write_header()
    js.write_main(f"{command}\n")
    js.write_main(f"cat {in_stream} > {out_stream}\n")
    if addl_command is not None:
        js.write_main(f"{addl_command}\n")
    js.clean_up()
    js.submit()

def get_most_recent_run(streams: list) -> int:
    """ From a list of stream files, get the most recent run.

    @param streams (list[str]) List of full paths to stream files.
    @return run (int) Most recent run in the list of streams.
    """
    run: int = 0
    for streampath in streams:
        filename: str = streampath.split('/')[-1]
        current_run = int(filename[1:5])
        if current_run > run:
            run = current_run
    return run

#### For command line use ####
            
def parse_input():
    """
    Parse command line input.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', help='Input stream file(s) in glob-readable format', required=True, type=str)
    parser.add_argument('-o', '--outdir', help='Output directory for peakogram and cell plots', required=True, type=str)
    parser.add_argument('-t', '--tag', help='Sample tag', required=True, type=str)
    parser.add_argument('--cell_only', help='Only read unit cell parameters, not reflections', action='store_true')
    parser.add_argument('--cell_out', help='Path to output cell file', required=False, type=str)
    parser.add_argument('--cell_ref', help='Path to reference cell file (for symmetry information)', required=False, type=str)

    return parser.parse_args()

if __name__ == '__main__':

    params = parse_input()
    stream_path: str = params.inputs
    streams: list = glob.glob(stream_path)
    st = StreamInterface(input_files=streams, cell_only=params.cell_only)
    if st.rank == 0:
        st.plot_cell_parameters(output=os.path.join(params.outdir, f"{params.tag}_cell.png"))
        if not params.cell_only:
            st.plot_peakogram(output=os.path.join(params.outdir, f"{params.tag}_peakogram.png"))

        run: int = get_most_recent_run(streams)
        indexdir: str = stream_path[:-len(stream_path.split('/')[-1])]
        rootdir: str = indexdir[:-7]
        summary_file: str = f'{rootdir}/summary_r{run:04}.json'
        update_summary(summary_file, st.stream_summary)
        elog_report_post(summary_file)
        if params.cell_out is not None:
            write_cell_file(st.cell_params, params.cell_out, input_file=params.cell_ref)
