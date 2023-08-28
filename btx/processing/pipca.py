import os, csv, h5py, argparse

import numpy as np
from mpi4py import MPI

from matplotlib import pyplot as plt
from matplotlib import colors

import holoviews as hv
hv.extension('bokeh')
from holoviews.streams import Params

import panel as pn
import panel.widgets as pnw

from btx.misc.shortcuts import TaskTimer

from btx.interfaces.ipsana import (
    PsanaInterface,
    bin_data,
    bin_pixel_index_map,
    retrieve_pixel_index_map,
    assemble_image_stack_batch,
)

class PiPCA:

    """Parallelized Incremental Principal Component Analysis."""

    def __init__(
        self,
        exp,
        run,
        det_type,
        start_offset=0,
        num_images=10,
        num_components=10,
        batch_size=10,
        priming=False,
        downsample=False,
        bin_factor=2,
        output_dir="",
    ):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
        self.psi.counter = start_offset
        self.start_offset = start_offset

        self.priming = priming
        self.downsample = downsample
        self.bin_factor = bin_factor
        self.output_dir = output_dir

        (
            self.num_images,
            self.num_components,
            self.batch_size,
            self.num_features,
        ) = self.set_params(num_images, num_components, batch_size, bin_factor)

        self.split_indices, self.split_counts = distribute_indices_over_ranks(
            self.num_features, self.size
        )

        self.task_durations = dict({})

        self.num_incorporated_images = 0
        self.outliers, self.pc_data = [], []

    def get_params(self):
        """
        Method to retrieve iPCA params.

        Returns
        -------
        num_incorporated_images : int
            number of images used to build model
        num_components : int
            number of components maintained in model
        batch_size : int
            batch size used in model updates
        num_features : int
            dimensionality of incorporated images
        """
        return (
            self.num_incorporated_images,
            self.num_components,
            self.batch_size,
            self.num_features,
        )

    def set_params(self, num_images, num_components, batch_size, bin_factor):
        """
        Method to initialize iPCA parameters.

        Parameters
        ----------
        num_images : int
            Desired number of images to incorporate into model.
        num_components : int
            Desired number of components for model to maintain.
        batch_size : int
            Desired size of image block to be incorporated into model at each update.
        bin_factor : int
            Factor to bin data by.

        Returns
        -------
        num_images : int
            Number of images to incorporate into model.
        num_components : int
            Number of components for model to maintain.
        batch_size : int
            Size of image block to be incorporated into model at each update.
        num_features : int
            Number of features (dimension) in each image.
        """
        max_events = self.psi.max_events
        downsample = self.downsample

        num_images = min(num_images, max_events) if num_images != -1 else max_events
        num_components = min(num_components, num_images)
        batch_size = min(batch_size, num_images)

        # set d
        det_shape = self.psi.det.shape()
        num_features = np.prod(det_shape).astype(int)

        if downsample:
            if det_shape[-1] % bin_factor or det_shape[-2] % bin_factor:
                print("Invalid bin factor, toggled off downsampling.")
                self.downsample = False
            else:
                num_features = int(num_features / bin_factor**2)

        return num_images, num_components, batch_size, num_features

    def run(self):
        """
        Perform iPCA on run subject to initialization parameters.
        """
        m = self.batch_size
        num_images = self.num_images

        # initialize and prime model, if specified
        if self.priming:
            img_batch = self.get_formatted_images(
                self.num_components, 0, self.num_features
            )
            self.prime_model(img_batch)
        else:
            self.U = np.zeros((self.split_counts[self.rank], self.num_components))
            self.S = np.ones(self.num_components)
            self.mu = np.zeros((self.split_counts[self.rank], 1))
            self.total_variance = np.zeros((self.split_counts[self.rank], 1))

        # divide remaining number of images into batches
        # will become redundant in a streaming setting, need to change
        rem_imgs = num_images - self.num_incorporated_images
        batch_sizes = np.array(
            [m] * np.floor(rem_imgs / m).astype(int)
            + ([rem_imgs % m] if rem_imgs % m else [])
        )

        # define batch indices based on batch sizes
        self.batch_indices = self.distribute_images_over_batches(batch_sizes)
        self.batch_number = 0

        # update model with remaining batches
        for batch_size in batch_sizes:
            self.fetch_and_update_model(batch_size)
            
        self.comm.Barrier()
        
        U, S, V, _, _ = self.get_model()
        
        if self.rank == 0:  
            print("Model complete")
        
            # save model to an hdf5 file
            filename = 'pipca_model.h5'
            with h5py.File(filename, 'w') as f:
                f.create_dataset('exp', data=self.psi.exp)
                f.create_dataset('run', data=self.psi.run)
                f.create_dataset('det_type', data=self.psi.det_type)
                f.create_dataset('start_offset', data=self.start_offset)
                f.create_dataset('loadings', data=self.pc_data)
                f.create_dataset('U', data=U)
                f.create_dataset('S', data=S)
                f.create_dataset('V', data=V)
                print(f'Model saved to {filename}')

    def get_formatted_images(self, n, start_index, end_index):
        """
        Fetch n - x image segments from run, where x is the number of 'dead' images.

        Parameters
        ----------
        n : int
            number of images to retrieve
        start_index : int
            start index of subsection of data to retrieve
        end_index : int
            end index of subsection of data to retrieve

        Returns
        -------
        ndarray, shape (end_index-start_index, n-x)
            n-x retrieved image segments of dimension end_index-start_index
        """

        bin_factor = self.bin_factor
        downsample = self.downsample

        # may have to rewrite eventually when number of images becomes large,
        # i.e. streamed setting, either that or downsample aggressively
        imgs = self.psi.get_images(n, assemble=False)

        if downsample:
            imgs = bin_data(imgs, bin_factor)

        imgs = imgs[
            [i for i in range(imgs.shape[0]) if not np.isnan(imgs[i : i + 1]).any()]
        ]

        num_valid_imgs, p, x, y = imgs.shape
        formatted_imgs = np.reshape(imgs, (num_valid_imgs, p * x * y)).T

        return formatted_imgs[start_index:end_index, :]

    def prime_model(self, X):
        """
        Initialize model on sample of data using batch PCA.

        Parameters
        ----------
        X : ndarray, shape (d x n)
            set of n (d x 1) observations
        """

        d, n = X.shape

        if self.rank == 0:
            print(f"Priming model with {n} samples...")


        mu_full, total_variance_full = self.calculate_sample_mean_and_variance(X)

        start, end = self.split_indices[self.rank], self.split_indices[self.rank+1]
        self.mu = mu_full[start:end]
        self.total_variance = total_variance_full[start:end]
        
        centered_data = X - np.tile(mu_full, n)

        U, self.S, V_T = np.linalg.svd(centered_data, full_matrices=False)
        self.U = U[start:end, :]
        self.V = V_T.T

        self.num_incorporated_images += n

    def fetch_and_update_model(self, n):
        """
        Fetch images and update model.

        Parameters
        ----------
        n : int
            number of images to incorporate
        """

        rank = self.rank
        start_index, end_index = self.split_indices[rank], self.split_indices[rank + 1]

        img_batch = self.get_formatted_images(n, start_index, end_index)

        self.update_model(img_batch)

    def update_model(self, X):
        """
        Update model with new batch of observations using iPCA.

        Parameters
        ----------
        X : ndarray, shape (d x m)
            batch of m (d x 1) observations

        Notes
        -----
        Implementation of iPCA algorithm from [1].

        References
        ----------
        [1] Ross DA, Lim J, Lin RS, Yang MH. Incremental learning for robust visual tracking.
        International journal of computer vision. 2008 May;77(1):125-41.
        """
        _, m = X.shape
        n = self.num_incorporated_images
        q = self.num_components

        with TaskTimer(self.task_durations, "total update"):

            if self.rank == 0:
                print(
                    "Factoring {m} sample{s} into {n} sample, {q} component model...".format(
                        m=m, s="s" if m > 1 else "", n=n, q=q
                    )
                )

            with TaskTimer(self.task_durations, "update mean and variance"):
                mu_n = self.mu
                mu_m, s_m = self.calculate_sample_mean_and_variance(X)

                self.total_variance = self.update_sample_variance(
                    self.total_variance, s_m, mu_n, mu_m, n, m
                )
                self.mu = self.update_sample_mean(mu_n, mu_m, n, m)

            with TaskTimer(
                self.task_durations, "center data and compute augment vector"
            ):
                X_centered = X - np.tile(mu_m, m)
                mean_augment_vector = np.sqrt(n * m / (n + m)) * (mu_m - mu_n)

                X_augmented = np.hstack((X_centered, mean_augment_vector))

            with TaskTimer(self.task_durations, "first matrix product U@S"):
                US = self.U @ np.diag(self.S)

            with TaskTimer(self.task_durations, "QR concatenate"):
                A = np.hstack((US, X_augmented))

            with TaskTimer(self.task_durations, "parallel QR"):
                Q_r, U_tilde, S_tilde = self.parallel_qr(A)

            with TaskTimer(self.task_durations, "compute local U_prime"):
                self.U = Q_r @ U_tilde[:, :q]
                self.S = S_tilde[:q]
                
            with TaskTimer(self.task_durations, "update V"):
                self.update_V(X, self.U, self.S)

            with TaskTimer(self.task_durations, "record pc data"):
                self.record_loadings()

            self.num_incorporated_images += m
            self.batch_number += 1


    def calculate_sample_mean_and_variance(self, imgs):
        """
        Compute the sample mean and variance of a flattened stack of n images.

        Parameters
        ----------
        imgs : ndarray, shape (d x n)
            horizonally stacked batch of flattened images

        Returns
        -------
        mu_m : ndarray, shape (d x 1)
            mean of imgs
        su_m : ndarray, shape (d x 1)
            sample variance of imgs (1 dof)
        """
        d, m = imgs.shape

        mu_m = np.reshape(np.mean(imgs, axis=1), (d, 1))
        s_m = np.zeros((d, 1))

        if m > 1:
            s_m = np.reshape(np.var(imgs, axis=1, ddof=1), (d, 1))

        return mu_m, s_m

    def parallel_qr(self, A):
        """
        Perform parallelized qr factorization on input matrix A.

        Parameters
        ----------
        A : ndarray, shape (_ x q+m+1)
            Input data to be factorized.

        Returns
        -------
        q_fin : ndarray, shape (_, q+m+1)
            Q_{r,1} from TSQR algorithm, where r = self.rank + 1
        U_tilde : ndarray, shape (q+m+1, q+m+1)
            Q_{r,2} from TSQR algorithm, where r = self.rank + 1
        S_tilde : ndarray, shape (q+m+1)
            R_tilde from TSQR algorithm, where r = self.rank + 1

        Notes
        -----
        Parallel QR algorithm implemented from [1], with additional elements from [2]
        sprinkled in to record elements for iPCA using SVD, etc.

        References
        ----------
        [1] Benson AR, Gleich DF, Demmel J. Direct QR factorizations for tall-and-skinny
        matrices in MapReduce architectures. In2013 IEEE international conference on
        big data 2013 Oct 6 (pp. 264-272). IEEE.

        [2] Ross DA, Lim J, Lin RS, Yang MH. Incremental learning for robust visual tracking.
        International journal of computer vision. 2008 May;77(1):125-41.

        [3] Maulik, R., & Mengaldo, G. (2021, November). PyParSVD: A streaming, distributed and
        randomized singular-value-decomposition library. In 2021 7th International Workshop on
        Data Analysis and Reduction for Big Scientific Data (DRBSD-7) (pp. 19-25). IEEE.
        """
        _, x = A.shape
        q = self.num_components
        m = x - q - 1

        with TaskTimer(self.task_durations, "qr - local qr"):
            Q_r1, R_r = np.linalg.qr(A, mode="reduced")

        self.comm.Barrier()

        with TaskTimer(self.task_durations, "qr - r_tot gather"):
            if self.rank == 0:
                R = np.empty((self.size * (q + m + 1), q + m + 1))
            else:
                R = None

            self.comm.Gather(R_r, R, root=0)

        if self.rank == 0:
            with TaskTimer(self.task_durations, "qr - global qr"):
                Q_2, R_tilde = np.linalg.qr(R, mode="reduced")

            with TaskTimer(self.task_durations, "qr - global svd"):
                U_tilde, S_tilde, _ = np.linalg.svd(R_tilde)
        else:
            U_tilde = np.empty((q + m + 1, q + m + 1))
            S_tilde = np.empty(q + m + 1)
            Q_2 = None

        self.comm.Barrier()

        with TaskTimer(self.task_durations, "qr - scatter q_tot"):
            Q_r2 = np.empty((q + m + 1, q + m + 1))
            self.comm.Scatter(Q_2, Q_r2, root=0)

        with TaskTimer(self.task_durations, "qr - local matrix build"):
            Q_r = Q_r1 @ Q_r2

        self.comm.Barrier()

        with TaskTimer(self.task_durations, "qr - bcast S_tilde"):
            self.comm.Bcast(S_tilde, root=0)

        self.comm.Barrier()

        with TaskTimer(self.task_durations, "qr - bcast U_tilde"):
            self.comm.Bcast(U_tilde, root=0)

        return Q_r, U_tilde, S_tilde

    def update_sample_mean(self, mu_n, mu_m, n, m):
        """
        Compute combined mean of two blocks of data.

        Parameters
        ----------
        mu_n : ndarray, shape (d x 1)
            mean of first block of data
        mu_m : ndarray, shape (d x 1)
            mean of second block of data
        n : int
            number of observations in first block of data
        m : int
            number of observations in second block of data

        Returns
        -------
        mu_nm : ndarray, shape (d x 1)
            combined mean of both blocks of input data
        """
        mu_nm = mu_m

        if n != 0:
            mu_nm = (1 / (n + m)) * (n * mu_n + m * mu_m)

        return mu_nm

    def update_sample_variance(self, s_n, s_m, mu_n, mu_m, n, m):
        """
        Compute combined sample variance of two blocks
        of data described by input parameters.

        Parameters
        ----------
        s_n : ndarray, shape (d x 1)
            sample variance of first block of data
        s_m : ndarray, shape (d x 1)
            sample variance of second block of data
        mu_n : ndarray, shape (d x 1)
            mean of first block of data
        mu_m : ndarray, shape (d x 1)
            mean of second block of data
        n : int
            number of observations in first block of data
        m : int
            number of observations in second block of data

        Returns
        -------
        s_nm : ndarray, shape (d x 1)
            combined sample variance of both blocks of data described by input
            parameters
        """
        s_nm = s_m

        if n != 0:
            s_nm = (((n - 1) * s_n + (m - 1) * s_m)
                    + (n * m * (mu_n - mu_m) ** 2) / (n + m)) / (n + m - 1)

        return s_nm
    
    def update_V(self, X, U, S):
        """
        Updates current V with shape (n x q) to shape (n + m, q)
        based on the previous and updated U and S.
        
        Parameters
        ----------
        X : ndarray, shape (_ x m)
            sliced image batch on self.rank
        U : ndarray, shape (d x q)
            principle components of the updated model
        S : ndarray, shape (q,)
            singular values of the updated model
        """
        _, m = X.shape
        n = self.num_incorporated_images
        j = self.batch_number
        q = self.num_components
        start_indices = self.split_indices[:-1]
        
        # Gather all X and U across all ranks
        if self.rank == 0:
            X_tot = np.empty((self.num_features, m))
            U_tot = np.empty((self.num_features, q))
        else:
            X_tot, U_tot = None, None
            V = np.empty((n + m, q))
        
        self.comm.Barrier()

        with TaskTimer(self.task_durations, "V - gather X and U"):
            self.comm.Gatherv(
                X.flatten(),
                [
                    X_tot,
                    self.split_counts * m,
                    start_indices * m,
                    MPI.DOUBLE,
                ],
                root=0,
            )

            self.comm.Gatherv(
                U.flatten(),
                [
                    U_tot,
                    self.split_counts * q,
                    start_indices * q,
                    MPI.DOUBLE,
                ],
                root=0,
            )

        if self.rank == 0:
            X_tot = np.reshape(X_tot, (self.num_features, m))
            U_tot = np.reshape(U_tot, (self.num_features, q))

            with TaskTimer(self.task_durations, "V - compute updated V"):
                if n > 0:
                    # U_prev and S_prev aren't instantiated on first primed batch
                    if self.priming and j == 0:
                        self.U_prev = U_tot
                        self.S_prev = S
                    
                    # Update each previous V_i from the previous batches
                    V = self.V
                    B = np.diag(self.S_prev) @ self.U_prev.T @ U_tot @ np.linalg.inv(np.diag(S))
                    for i in range(j):
                        start, end = self.batch_indices[i], self.batch_indices[i+1]
                        V[start:end] = V[start:end] @ B

                    # Compute new V_m from current batch with standard SVD
                    V_m = X_tot.T @ U_tot @ np.linalg.inv(np.diag(S))
                    V = np.concatenate((V, V_m))
                else:
                    # Instantiate self.V with standard SVD
                    V = X_tot.T @ U_tot @ np.linalg.inv(np.diag(S))
            
        self.comm.Barrier()

        with TaskTimer(self.task_durations, "V - bcast V"):
            self.comm.Bcast(V, root=0)

        self.V = V
        self.U_prev = U_tot
        self.S_prev = S

    def get_model(self):
        """
        Method to retrieve model parameters.

        Returns
        -------
        U_tot : ndarray, shape (d x q)
            iPCA principal axes from model.
        S_tot : ndarray, shape (1 x q)
            iPCA singular values from model.
        V_tot : ndarray, shape (n x q)
            iPCA eigenimages from model.
        mu_tot : ndarray, shape (1 x d)
            Data mean computed from all input images.
        var_tot : ndarray, shape (1 x d)
            Sample data variance computed from all input images.
        """
        if self.rank == 0:
            U_tot = np.empty(self.num_features * self.num_components)
            mu_tot = np.empty((self.num_features, 1))
            var_tot = np.empty((self.num_features, 1))
        else:
            U_tot, mu_tot, var_tot = None, None, None

        start_indices = self.split_indices[:-1]

        self.comm.Gatherv(
            self.U.flatten(),
            [
                U_tot,
                self.split_counts * self.num_components,
                start_indices * self.num_components,
                MPI.DOUBLE,
            ],
            root=0,
        )

        if self.rank == 0:
            U_tot = np.reshape(U_tot, (self.num_features, self.num_components))

        self.comm.Gatherv(
            self.mu,
            [
                mu_tot,
                self.split_counts * self.num_components,
                start_indices,
                MPI.DOUBLE,
            ],
            root=0,
        )
        self.comm.Gatherv(
            self.total_variance,
            [
                var_tot,
                self.split_counts * self.num_components,
                start_indices,
                MPI.DOUBLE,
            ],
            root=0,
        )

        S_tot = self.S
        V_tot = self.V

        return U_tot, S_tot, V_tot, mu_tot, var_tot

    def get_outliers(self):
        """
        Method to retrieve and print outliers on root process.
        """

        if self.rank == 0:
            print(self.outliers)

    def record_loadings(self):
        """
        Method to store current all loadings, ΣV^T,
        up to the current batch.
        """
        U, S, V, mu, _ = self.get_model()

        if self.rank == 0:
            j = self.batch_number
            pcs = np.empty((self.num_components, self.batch_indices[j+1]))
            
            for i in range(j):
                start, end = self.batch_indices[i], self.batch_indices[i+1]
                batch_size = end - start
                centered_model = U @ np.diag(S) @ V[start:end].T - np.tile(mu, (1, batch_size))
                pcs[:, start:end] = U.T @ centered_model 
            
            self.pc_data = pcs

            pc_dist = np.linalg.norm(pcs[:self.num_components], axis=0)
            std = np.std(pc_dist)
            mu = np.mean(pc_dist)

            index_offset = self.start_offset + self.num_incorporated_images
            self.outliers = np.where(np.abs(pc_dist - mu) > 2 * std)[0] + index_offset

    def display_image(self, idx, output_dir="", save_image=False):
        """
        Method to retrieve single image from run subject to model binning constraints.

        Parameters
        ----------
        idx : int
            Run index of image to be retrieved.
        output_dir : str, optional
            File path to output directory, by default ""
        save_image : bool, optional
            Whether to save image to file, by default False
        """

        U, S, _, mu, var = self.get_model()

        if self.rank != 0:
            return

        bin_factor = 1
        if self.downsample:
            bin_factor = self.bin_factor

        n, q, m, d = self.get_params()

        a, b, c = self.psi.det.shape()
        b = int(b / bin_factor)
        c = int(c / bin_factor)

        fig, ax = plt.subplots(1)

        counter = self.psi.counter
        self.psi.counter = idx
        img = self.get_formatted_images(1, 0, d)
        self.psi.counter = counter

        img = img - mu
        img = np.reshape(img, (a, b, c))

        pixel_index_map = retrieve_pixel_index_map(self.psi.det.geometry(self.psi.run))
        binned_pim = bin_pixel_index_map(pixel_index_map, bin_factor)

        img = assemble_image_stack_batch(img, binned_pim)

        vmax = np.max(img.flatten())
        ax.imshow(
            img,
            norm=colors.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=0, vmax=vmax),
            interpolation=None
        )

        if save_image:
            plt.savefig(output_dir)

        plt.show()

        
    def display_dashboard(self):
        """
        Displays a pipca dashboard with a PC plot and intensity heatmap.
        """
        U, S, V, _, _ = self.get_model()
        start_img = self.start_offset

        if self.rank != 0:
            return
        
        # Create PC dictionary and widgets
        PCs = {f'PC{i}' : v for i, v in enumerate(self.pc_data, start=1)}
        PC_options = list(PCs)
        
        PCx = pnw.Select(name='X-Axis', value='PC1', options=PC_options)
        PCy = pnw.Select(name='Y-Axis', value='PC2', options=PC_options)
        widgets_scatter = pn.WidgetBox(PCx, PCy, width=150)
        
        PC_scree = pnw.Select(name='Component Cut-off', value=f'PC{len(PCs)}', options=PC_options)
        widgets_scree = pn.WidgetBox(PC_scree, width=150)
        
        tap_source = None
        posxy = hv.streams.Tap(source=tap_source, x=0, y=0)
        
        # Create PC scatter plot
        @pn.depends(PCx.param.value, PCy.param.value)
        def create_scatter(PCx, PCy):
            img_index_arr = np.arange(start_img, start_img + len(PCs[PCx]))
            scatter_data = {**PCs, 'Image': img_index_arr}
            
            opts = dict(width=400, height=300, color='Image', cmap='rainbow', colorbar=True,
                        show_grid=True, shared_axes=False, toolbar='above', tools=['hover'])
            scatter = hv.Points(scatter_data, kdims=[PCx, PCy], vdims=['Image'], 
                                label="%s vs %s" % (PCx.title(), PCy.title())).opts(**opts)
            
            posxy.source = scatter
            return scatter
        
        # Create scree plot
        @pn.depends(PC_scree.param.value)
        def create_scree(PC_scree):
            q = int(PC_scree[2:])
            components = np.arange(1, q + 1)
            singular_values = S[:q]
            bars_data = np.stack((components, singular_values)).T
            
            opts = dict(width=400, height=300, show_grid=True, show_legend=False,
                        shared_axes=False, toolbar='above', default_tools=['hover','save','reset'],
                        xlabel='Components', ylabel='Singular Values')
            scree = hv.Bars(bars_data, label="Scree Plot").opts(**opts)
            
            return scree
        
        # Define function to compute heatmap based on tap location
        def tap_heatmap(x, y, pcx, pcy, pcscree):
            # Finds the index of image closest to the tap location
            img_source = closest_image_index(x, y, PCs[pcx], PCs[pcy])
            
            counter = self.psi.counter
            self.psi.counter = start_img + img_source
            img = self.psi.get_images(1)
            self.psi.counter = counter
            img = img.squeeze()
            
            # Downsample so heatmap is at most 100 x 100
            hm_data = construct_heatmap_data(img, 100)
        
            opts = dict(width=400, height=300, cmap='plasma', colorbar=True, shared_axes=False, toolbar='above')
            heatmap = hv.HeatMap(hm_data, label="Source Image %s" % (start_img+img_source)).aggregate(function=np.mean).opts(**opts)
            
            return heatmap
        
        # Define function to compute reconstructed heatmap based on tap location
        def tap_heatmap_reconstruct(x, y, pcx, pcy, pcscree):
            # Finds the index of image closest to the tap location
            img_source = closest_image_index(x, y, PCs[pcx], PCs[pcy])
            
            # Calculate and format reconstructed image
            p, x, y = self.psi.det.shape()
            pixel_index_map = retrieve_pixel_index_map(self.psi.det.geometry(self.psi.run))
            
            q = int(pcscree[2:])
            img = U[:, :q] @ np.diag(S[:q]) @ np.array([V[img_source][:q]]).T
            img = img.reshape((p, x, y))
            img = assemble_image_stack_batch(img, pixel_index_map)
            
            # Downsample so heatmap is at most 100 x 100
            hm_data = construct_heatmap_data(img, 100)
            
            opts = dict(width=400, height=300, cmap='plasma', colorbar=True, shared_axes=False, toolbar='above')
            heatmap_reconstruct = hv.HeatMap(hm_data, label="PiPCA Image %s" % (start_img+img_source)).aggregate(function=np.mean).opts(**opts)
            
            return heatmap_reconstruct
            
        # Connect the Tap stream to the heatmap callbacks
        stream1 = [posxy]
        stream2 = Params.from_params({'pcx': PCx.param.value, 'pcy': PCy.param.value, 'pcscree': PC_scree.param.value})
        tap_dmap = hv.DynamicMap(tap_heatmap, streams=stream1+stream2)
        tap_dmap_reconstruct = hv.DynamicMap(tap_heatmap_reconstruct, streams=stream1+stream2)
        
        return pn.Column(pn.Row(widgets_scatter, create_scatter, tap_dmap),
                         pn.Row(widgets_scree, create_scree, tap_dmap_reconstruct)).servable('PiPCA Dashboard')
    
    def display_error_plots(self):
        """
        Displays error plots comparing the computed U, S, V versus the true U, S, V.
        Expected to output a correlation of positive and negative 1 for U and V,
        and strictly positive one for S.
        """
        U, _, _, _, _ = self.get_model()
        
        if self.rank != 0:
            return
        
        # Find true V matrix
        n = self.num_images
        d = self.num_features
        q = self.num_components
        
        assert n * d < 4e8, 'image and feature dimensions are too high to test errors'
        assert n == q, 'number of images must equal the number of componets to commpare with np.linalg.svd() output'
        
        self.psi.counter = self.start_offset
        X = self.get_formatted_images(n, 0, d)
        
        mu_full, _ = self.calculate_sample_mean_and_variance(X)
        
        centered_data = X - np.tile(mu_full, n)

        U_true, S_true, V_true_T = np.linalg.svd(centered_data, full_matrices=False)
        
        # Create eigenimage dictionary and widgets
        eigenimages = {f'PC{i}' : v for i, v in enumerate(U.T, start=1)}
        PC_options = list(eigenimages)
        
        split_indices, _ = distribute_indices_over_ranks(d, d // 10000)
        batch_options = list(range(1, len(split_indices)))
        
        component = pnw.Select(name='Components', value='PC1', options=PC_options)
        batch_slider = pnw.DiscreteSlider(name='Batch Slider', value=1, options=batch_options)
        widgets_scatter = pn.WidgetBox(component, batch_slider, width=400)
        
        # Create scatter plots
        @pn.depends(component.param.value, batch_slider.param.value)
        def create_U_scatter(component, batch_index):
            component_index = int(component[2:]) - 1
            start, end = split_indices[batch_index - 1], split_indices[batch_index]
            scatter_data = dict(true=U_true.T[component_index][start:end], calc=eigenimages[component][start:end], index=np.arange(end-start))
            
            opts = dict(width=400, height=300, show_grid=True, toolbar='above',
                        color='index', colorbar=True, tools=['hover'], shared_axes=False)
            scatter = hv.Points(scatter_data, kdims=['true', 'calc'], vdims=['index'], label=f"True U vs Computed U").opts(**opts)
            
            return scatter
        
        def create_S_scatter():
            scatter_data = dict(true=S_true, calc=self.S, index=np.arange(q))
            
            opts = dict(width=400, height=300, show_grid=True, toolbar='above',
                        color='index', colorbar=True, tools=['hover'], shared_axes=False)
            scatter = hv.Points(scatter_data, kdims=['true', 'calc'], vdims=['index'], label=f"True S vs Computed S").opts(**opts)
            
            return scatter
        
        def create_V_scatter():
            scatter_data = dict(true=V_true_T.flatten(), calc=self.V.T.flatten(), index=np.arange(len(V_true_T.flatten())))
            
            opts = dict(width=400, height=300, show_grid=True, ylim=(-1,1), toolbar='above',
                        color='index', colorbar=True, tools=['hover'], shared_axes=False)
            scatter = hv.Points(scatter_data, kdims=['true', 'calc'], vdims=['index'], label=f"True V vs Computed V").opts(**opts)
            
            return scatter
        
        return pn.Column(widgets_scatter, pn.Row(create_U_scatter, create_S_scatter, create_V_scatter)).servable('PiPCA Model Error Plots')
    
    def display_eigenimages(self):
        """
        Displays a PC selector widget and a heatmap of the
        eigenimage corresponding to the selected PC.
        """
        U, _, _, _, _ = self.get_model()

        if self.rank != 0:
            return
        
        # Create eigenimage dictionary and widget
        eigenimages = {f'PC{i}' : v for i, v in enumerate(U.T, start=1)}
        PC_options = list(eigenimages)
        
        component = pnw.Select(name='Components', value='PC1', options=PC_options)
        widget_heatmap = pn.WidgetBox(component, width=150)
        
        # Define function to compute heatmap
        @pn.depends(component.param.value)
        def create_heatmap(component):
            # Reshape selected eigenimage to work with construct_heatmap_data()
            p, x, y = self.psi.det.shape()
            pixel_index_map = retrieve_pixel_index_map(self.psi.det.geometry(self.psi.run))
            
            img = eigenimages[component]
            img = img.reshape((p, x, y))
            img = assemble_image_stack_batch(img, pixel_index_map)
            
            # Downsample so heatmap is at most 100 x 100
            hm_data = construct_heatmap_data(img, 100)
        
            opts = dict(width=400, height=300, cmap='BrBG', colorbar=True,
                        symmetric=True, shared_axes=False, toolbar='above')
            heatmap = hv.HeatMap(hm_data, label="%s Eigenimage" % (component.title())).aggregate(function=np.mean).opts(**opts)
            
            return heatmap
        
        return pn.Row(widget_heatmap, create_heatmap).servable('PiPCA Eigenimages')
    
    def distribute_images_over_batches(self, batch_sizes):
        """
        Returns 1D array of batch indices depending on batch sizes
        and number of incorporated images.
        
        Parameters
        ----------
        batch_sizes : ndarray, shape (b,)
            number of images in each batch,
            where b is the number of batches
        
        Returns
        -------
        batch_indices : ndarray, shape (b+1,)
            division indices between batches
        """
        
        total_indices = self.num_incorporated_images
        batch_indices = [self.num_incorporated_images]
        
        for size in batch_sizes:
            total_indices += size
            batch_indices.append(total_indices)
        
        return np.array(batch_indices)

def distribute_indices_over_ranks(d, size):
    """

    Parameters
    ----------
    d : int
        total number of dimensions
    size : int
        number of ranks in world

    Returns
    -------
    split_indices : ndarray, shape (size+1 x 1)
        division indices between ranks
    split_counts : ndarray, shape (size x 1)
        number of dimensions allocated per rank
    """

    total_indices = 0
    split_indices, split_counts = [0], []

    for r in range(size):
        num_per_rank = d // size
        if r < (d % size):
            num_per_rank += 1

        split_counts.append(num_per_rank)

        total_indices += num_per_rank
        split_indices.append(total_indices)

    split_indices = np.array(split_indices)
    split_counts = np.array(split_counts)

    return split_indices, split_counts

def closest_image_index(x, y, PCx_vector, PCy_vector):
    """
    Finds the index of image closest to the tap location
    
    Parameters:
    -----------
    x : float
        tap location on x-axis of tap source plot
    y : float
        tap location on y-axis of tap source plot
    PCx_vector : ndarray, shape (d,)
        principle component chosen for x-axis
    PCx_vector : ndarray, shape (d,)
        principle component chosen for y-axis
        
    Returns:
    --------
    img_source:
        index of the image closest to tap location
    """
    img_source = None
    min_diff = None
    square_diff = None
    
    for i, (xv, yv) in enumerate(zip(PCx_vector, PCy_vector)):    
        square_diff = (x - xv) ** 2 + (y - yv) ** 2
        if (min_diff is None or square_diff < min_diff):
            min_diff = square_diff
            img_source = i
    
    return img_source

def construct_heatmap_data(img, max_pixels):
    """
    Formats img to properly be displayed by hv.Heatmap()
    
    Parameters:
    -----------
    img : ndarray, shape (x_pixels x y_pixels)
        single image we want to display on a heatmap
    max_pixels: ing
        max number of pixels on x and y axes of heatmap
    
    Returns:
    --------
    hm_data : ndarray, shape ((x_pixels*y__pixels) x 3)
        coordinates to be displayed by hv.Heatmap (row, col, color)
    """
    x_pixels, y_pixels = img.shape
    bin_factor_x = int(x_pixels / max_pixels)
    bin_factor_y = int(y_pixels / max_pixels)
    
    while x_pixels % bin_factor_x != 0:
        bin_factor_x += 1
    while y_pixels % bin_factor_y != 0:
        bin_factor_y += 1
    
    img = img.reshape((x_pixels, y_pixels))
    binned_img = img.reshape(int(x_pixels / bin_factor_x),
                                bin_factor_x,
                                int(y_pixels / bin_factor_y),
                                bin_factor_y).mean(-1).mean(1)
    
    # Creates hm_data array for heatmap
    bin_x_pixels, bin_y_pixels = binned_img.shape
    rows = np.tile(np.arange(bin_x_pixels).reshape((bin_x_pixels, 1)), bin_y_pixels).flatten()
    cols = np.tile(np.arange(bin_y_pixels), bin_x_pixels)
    
    hm_data = np.stack((rows, cols, binned_img.flatten()))
    hm_data = hm_data.T.reshape((bin_x_pixels * bin_y_pixels, 3))
    
    return hm_data

#### for command line use ###


def parse_input():
    """
    Parse command line input.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp", help="Experiment name.", required=True, type=str)
    parser.add_argument("-r", "--run", help="Run number.", required=True, type=int)
    parser.add_argument(
        "-d",
        "--det_type",
        help="Detector name, e.g epix10k2M or jungfrau4M.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--start_offset",
        help="Run index of first image to be incorporated into iPCA model.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--num_components",
        help="Number of principal components to compute and maintain.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        help="Size of image batch incorporated in each model update.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--num_images",
        help="Total number of images to be incorporated into model.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--output_dir",
        help="Path to output directory for recording task duration data.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--priming",
        help="Initialize model with PCA.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--downsample",
        help="Enable downsampling of images.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "--bin_factor",
        help="Bin factor if using downsizing.",
        required=False,
        type=int,
    )

    return parser.parse_args()


if __name__ == "__main__":

    params = parse_input()
    kwargs = {k: v for k, v in vars(params).items() if v is not None}

    pipca = PiPCA(**kwargs)
    pipca.run()
    pipca.get_outliers()
