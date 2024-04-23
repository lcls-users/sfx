import os, csv, h5py, argparse
import math
import logging
import numpy as np
import time
from mpi4py import MPI

from matplotlib import pyplot as plt
from matplotlib import colors
from scipy.linalg import qr

import holoviews as hv
hv.extension('bokeh')
from holoviews.streams import Params

import panel as pn
import panel.widgets as pnw
import statistics

from btx.misc.shortcuts import TaskTimer

from btx.misc.pipca_visuals import compute_compression_loss

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
        filename='pipca.model_h5',
    ):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
        self.psi.counter = start_offset
        self.start_offset = start_offset
        #self.psi.turn_calibration_off()

        self.priming = priming
        self.downsample = downsample
        self.bin_factor = bin_factor
        self.output_dir = output_dir
        self.filename = filename

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

    def run_model_full(self, previous_U = None, previous_S = None, previous_mu = None, previous_var = None):
        """
        Perform iPCA on run subject to initialization parameters.
        """
        start_time = time.time()

        batch_size = self.batch_size
        num_images = self.num_images

        # initialize and prime model, if specified
        if self.priming:
            img_batch = self.get_formatted_images(
                self.num_components, 0, self.num_features
            )
            self.prime_model(img_batch)

        with TaskTimer(self.task_durations, "Initialize matrices"):
            if previous_U is not None:

                self.U = previous_U[self.split_indices[self.rank]:self.split_indices[self.rank +1]]
                self.S = previous_S
                self.mu = previous_mu[self.split_indices[self.rank]:self.split_indices[self.rank +1]]
                self.total_variance = previous_var[self.split_indices[self.rank]:self.split_indices[self.rank +1]]

            else:
                self.U = np.zeros((self.split_counts[self.rank], self.num_components))
                self.S = np.ones(self.num_components)
                self.mu = np.zeros((self.split_counts[self.rank], 1))
                self.total_variance = np.zeros((self.split_counts[self.rank], 1))
        
        # divide remaining number of images into batches
        # will become redundant in a streaming setting, need to change
        rem_imgs = num_images - self.num_incorporated_images
        batch_sizes = np.array(
            [batch_size] * np.floor(rem_imgs / batch_size).astype(int)
            + ([rem_imgs % batch_size] if rem_imgs % batch_size else [])
        )

        # define batch indices based on batch sizes
        with TaskTimer(self.task_durations, "distribute images over batches"):
            self.batch_indices = self.distribute_images_over_batches(batch_sizes)
            self.batch_number = 0

        # update model with remaining batches
        with TaskTimer(self.task_durations, "formatting and update model"):
            for batch_size in batch_sizes:
                with TaskTimer(self.task_durations, "get images on each rank - equiv to old get_formatted_img"):
                    if self.rank==0:
                        with TaskTimer(self.task_durations, "get formatted images rank 0"):
                            imgs = self.psi.get_images(batch_size,assemble=False)
                            with TaskTimer(self.task_durations, "distribute pixels"):
                                img_batch = self.get_formatted_images(imgs)
                    else :
                        img_batch = None
                    
                    self.comm.Barrier()

                    with TaskTimer(self.task_durations, "scattering data to all ranks"):
                        formatted_imgs = self.comm.scatter(img_batch,root=0) 
                    
                with TaskTimer(self.task_durations, "update model"):
                    self.update_model(formatted_imgs)

        self.comm.Barrier()
        
        with TaskTimer(self.task_durations, "gather matrices end"):
            U = self.gather_U()
            S = self.S
            V = self.V
            mu_tot = self.gather_mu()
            var_tot = self.gather_var()
        
        end_time = time.time()
        logging.basicConfig(level=logging.INFO)

        if self.rank == 0:  
            logging.info("Model complete")
            execution_time = end_time - start_time  # Calculate the execution time
            frequency = self.num_incorporated_images/execution_time

            # save model to an hdf5 file
            with TaskTimer(self.task_durations, "save inputs file h5"):
                with h5py.File(self.filename, 'a') as f:
                    if 'exp' not in f or 'det_type' not in f or 'start_offset' not in f:
                        # Create datasets only if they don't exist
                        f.create_dataset('exp', data=self.psi.exp)
                        f.create_dataset('det_type', data=self.psi.det_type)
                        f.create_dataset('start_offset', data=self.start_offset)

                    create_or_update_dataset(f,'loadings', data=self.pc_data)
                    create_or_update_dataset(f, 'run', self.psi.run)
                    create_or_update_dataset(f, 'U', U)
                    create_or_update_dataset(f, 'S', S)
                    create_or_update_dataset(f, 'V', V)
                    create_or_update_dataset(f, 'mu', mu_tot)
                    create_or_update_dataset(f, 'total_variance', var_tot)

                    append_to_dataset(f, 'frequency', data=frequency)
                    append_to_dataset(f, 'execution_times', data=execution_time)
                    logging.info(f'Model saved to {self.filename}')

            # Print the mean duration and standard deviation for each task
            for task, durations in self.task_durations.items():
                durations = [float(round(float(duration), 2)) for duration in durations]  # Convert to float and round to 2 decimal places
                if len(durations) == 1:
                    logging.info(f"Task: {task}, Duration: {durations[0]:.2f} (Only 1 duration)")
                else:
                    mean_duration = np.mean(durations)
                    std_deviation = statistics.stdev(durations)
                    logging.info(f"Task: {task}, Mean Duration: {mean_duration:.2f}, Standard Deviation: {std_deviation:.2f}")

        self.comm.Barrier()

    def get_formatted_images(self, imgs):
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
        if downsample:
            imgs = bin_data(imgs, bin_factor)

        imgs = imgs[
            [i for i in range(imgs.shape[0]) if not np.isnan(imgs[i : i + 1]).any()]
        ]

        num_valid_imgs, p, x, y = imgs.shape
        formatted_imgs = np.reshape(imgs, (num_valid_imgs, p * x * y)).T

        formatted_imgs_to_scatter = []

        for i in range(0,self.comm.Get_size()):
            formatted_imgs_to_scatter.append(formatted_imgs[self.split_indices[i]:self.split_indices[i + 1],:])

        return formatted_imgs_to_scatter

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

        with TaskTimer(self.task_durations, "get formatted images"):
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
        num_incorporated_images = self.num_incorporated_images
        num_components = self.num_components

        with TaskTimer(self.task_durations, "total update"):

            if self.rank == 0:
                print(
                    "Factoring {m} sample{s} into {n} sample, {q} component model...".format(
                        m=m, s="s" if m > 1 else "", n=num_incorporated_images, q=num_components
                    )
                )

            with TaskTimer(self.task_durations, "update mean and variance"):
                mu_n = self.mu
                mu_m, s_m = self.calculate_sample_mean_and_variance(X)
                self.total_variance = self.update_sample_variance(
                    self.total_variance, s_m, mu_n, mu_m, num_incorporated_images, m
                )
                self.mu = self.update_sample_mean(mu_n, mu_m, num_incorporated_images, m)

            with TaskTimer(
                self.task_durations, "center data and compute augment vector"
            ):
                X_centered = X - np.tile(mu_m, m)

                mean_augment_vector = np.sqrt(num_incorporated_images * m / (num_incorporated_images + m)) * (mu_m - mu_n)

                X_augmented = np.hstack((X_centered, mean_augment_vector))

            with TaskTimer(self.task_durations, "first matrix product U@S"):
                US = self.U @ np.diag(self.S)

            with TaskTimer(self.task_durations, "QR concatenate"):
                A = np.hstack((US, X_augmented))

            with TaskTimer(self.task_durations, "parallel QR"):
                Q_r, U_tilde, S_tilde = self.parallel_qr(A)

            with TaskTimer(self.task_durations, "compute local U_prime"):
                self.U = Q_r @ U_tilde[:, :num_components]
                self.S = S_tilde[:num_components]
                
            with TaskTimer(self.task_durations, "update V"):
                self.update_V(X, self.S)

            self.num_incorporated_images += m
            self.batch_number += 1

            with TaskTimer(self.task_durations, "record pc data"):
                self.record_loadings()
                

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
        num_components = self.num_components
        m = x - num_components - 1

        with TaskTimer(self.task_durations, "qr - local qr"):
            # Principal computational time bottleneck
            Q_r1, R_r = np.linalg.qr(A, mode="reduced")

        self.comm.Barrier()

        with TaskTimer(self.task_durations, "qr - r_tot gather"):
            if self.rank == 0:
                R = np.empty((self.size * (num_components + m + 1), num_components + m + 1))
            else:
                R = None

            self.comm.Gather(R_r, R, root=0)

        if self.rank == 0:
            with TaskTimer(self.task_durations, "qr - global qr"):
                Q_2, R_tilde = np.linalg.qr(R, mode="reduced")

            with TaskTimer(self.task_durations, "qr - global svd"):
                U_tilde, S_tilde, _ = np.linalg.svd(R_tilde)
        else:
            U_tilde = np.empty((num_components + m + 1, num_components + m + 1))
            S_tilde = np.empty(num_components + m + 1)
            Q_2 = None

        self.comm.Barrier()

        with TaskTimer(self.task_durations, "qr - scatter q_tot"):
            Q_r2 = np.empty((num_components + m + 1, num_components + m + 1))
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

    def update_V(self, X, S):
        """
        Updates current V with shape (n x q) to shape (n + m, q)
        based on the previous and updated U and S.

        Parameters
        ----------
        X : ndarray, shape (_ x m)
            sliced image batch on self.rank
        S : ndarray, shape (q,)
            singular values of the updated model
        """
        num_incorporated_images = self.num_incorporated_images
        batch_number = self.batch_number
        _, m = X.shape
        num_components = self.num_components

        self.comm.Barrier()

        # Gather all X and U across all ranks
        X_tot = self.gather_X(X)
        U_tot = self.gather_U()

        V = np.empty((num_incorporated_images + m, num_components))
        
        self.comm.Barrier()

        with TaskTimer(self.task_durations, "V - compute updated V"):
            if num_incorporated_images > 0:
                # U_prev and S_prev aren't instantiated on the first primed batch
                if self.priming and batch_number == 0:
                    self.U_prev = U_tot
                    self.S_prev = S

                # Update each previous V_i from the previous batches
                V = self.V
                B = np.diag(self.S_prev) @ self.U_prev.T @ U_tot @ np.linalg.inv(np.diag(S))
                for i in range(batch_number):
                    start, end = self.batch_indices[i], self.batch_indices[i + 1]
                    V[start:end] = V[start:end] @ B

                # Compute new V_m from the current batch with standard SVD
                V_m = X_tot.T @ U_tot @ np.linalg.inv(np.diag(S))
                V = np.concatenate((V, V_m))
            else:
                # Instantiate self.V with standard SVD
                V = X_tot.T @ U_tot @ np.linalg.inv(np.diag(S))

        self.comm.Barrier()

        self.comm.Bcast(V, root=0)

        self.V = V
        self.U_prev = U_tot
        self.S_prev = S

    def gather_X(self, X):
        """
        Gather and return the X_tot variable.
        """
        _, m = X.shape
        if self.rank == 0:
            X_tot = np.empty((self.num_features, m))
        else:
            X_tot = np.empty((self.num_features, m))

        start_indices = self.split_indices[:-1]

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

        if self.rank == 0:
            X_tot = np.reshape(X_tot, (self.num_features, m))

        return X_tot 
    
    def gather_U(self):
        """
        Gather and return the U_tot variable.
        """
        if self.rank == 0:
            U_tot = np.empty((self.num_features, self.num_components))
        else:
            U_tot = np.empty((self.num_features, self.num_components))

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

        return U_tot 

    def gather_mu(self):
        """
        Gather and return the mu_tot variable.
        """
        if self.rank == 0:
            mu_tot = np.empty((self.num_features, 1))
        else:
            mu_tot = np.empty((self.num_features, 1))

        start_indices = self.split_indices[:-1]
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
        return mu_tot 

    def gather_var(self):
        """
        Gather and return the var_tot variable.
        """
        if self.rank == 0:
            var_tot = np.empty((self.num_features, 1))
        else:
            var_tot = np.empty((self.num_features, 1))

        start_indices = self.split_indices[:-1]
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
        return var_tot 

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
        U_tot = self.gather_U()
        S_tot = self.S
        V_tot = self.V
        mu_tot = self.gather_mu()
        var_tot = self.gather_var()
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
        if self.num_images == self.num_incorporated_images:

            U = self.gather_U()
            S = self.S
            V = self.V
            mu = self.gather_mu()

            self.comm.Barrier()

            if self.rank == 0:
                with TaskTimer(self.task_durations, 'Record Loadings'):
                    pcs = np.diag(S) @ V.T - np.tile((mu.T @ U).T, (1, self.num_images))

                self.pc_data = pcs

                pc_dist = np.linalg.norm(self.pc_data[:self.num_components], axis=0)
                std = np.std(pc_dist)
                mu = np.mean(pc_dist)

                index_offset = self.start_offset + self.num_incorporated_images
                self.outliers = np.where(np.abs(pc_dist - mu) > 2 * std)[0] + index_offset
            
    def record_loadings_per_batch(self):
        """
        Method to store current all loadings, ΣV^T,
        up to the current batch.
        """
        U = self.gather_U()
        S = self.S
        V = self.V
        mu = self.gather_mu()

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

        mu = self.gather_mu()

        if self.rank != 0:
            return

        bin_factor = 1
        if self.downsample:
            bin_factor = self.bin_factor

        num_features = self.num_features

        a, b, c = self.psi.det.shape()
        b = int(b / bin_factor)
        c = int(c / bin_factor)

        _, ax = plt.subplots(1)

        counter = self.psi.counter
        self.psi.counter = idx
        img = self.get_formatted_images(1, 0, num_features)
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

    def display_error_plots(self):
        """
        Displays error plots comparing the computed U, S, V versus the true U, S, V.
        Expected to output a correlation of positive and negative 1 for U and V,
        and strictly positive one for S.
        """
        U = self.gather_U()
        
        if self.rank != 0:
            return
        
        # Find true V matrix
        num_images = self.num_images
        num_features = self.num_features
        num_components = self.num_components
        
        try:
            assert num_images * num_features < 4e8, 'image and feature dimensions are too high to test errors'
            assert num_images == num_components, 'number of images must equal the number of components to compare with np.linalg.svd() output'
        except AssertionError as e:
            print(f"Assertion Error: {e}")
        
        self.psi.counter = self.start_offset
        X = self.get_formatted_images(num_images, 0, num_features)
        
        mu_full, _ = self.calculate_sample_mean_and_variance(X)
        
        centered_data = X - np.tile(mu_full, num_images)

        U_true, S_true, V_true_T = np.linalg.svd(centered_data, full_matrices=False)
        
        # Create eigenimage dictionary and widgets
        eigenimages = {f'PC{i}' : v for i, v in enumerate(U.T, start=1)}
        PC_options = list(eigenimages)
        
        split_indices, _ = distribute_indices_over_ranks(num_features, num_features // 10000)
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
            scatter_data = dict(true=S_true, calc=self.S, index=np.arange(num_components))
            
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

def append_to_dataset(f, dataset_name, data):
    if dataset_name not in f:
        f.create_dataset(dataset_name, data=np.array(data))
    else:
        if isinstance(f[dataset_name], h5py.Dataset) and f[dataset_name].shape == ():
            # Scalar dataset, convert to array
            existing_data = np.atleast_1d(f[dataset_name][()])
        else:
            # Non-scalar dataset, use slicing
            existing_data = f[dataset_name][:]

        new_data = np.atleast_1d(np.array(data))
        data_combined = np.concatenate([existing_data, new_data])
        del f[dataset_name]
        f.create_dataset(dataset_name, data=data_combined)

def compute_norm_difference(current, previous=None):
    if previous is not None:
        diff = current - previous
    else:
        diff = current
    return np.linalg.norm(diff)

def create_or_update_dataset(f, name, data):
    if name in f:
        del f[name]
    f.create_dataset(name, data=data)

def compute_and_save_metrics(filename, num_components, S, previous_S, mu, previous_mu, total_variance, previous_var, complete_compression):

    # Compute differences for each list
    diff_S = compute_norm_difference(S, previous_S)
    diff_mu = compute_norm_difference(mu, previous_mu)
    diff_var = compute_norm_difference(total_variance, previous_var)

    # Compute compression losses
    start_time = time.time()
    compression_loss_random = compute_compression_loss(filename, num_components, True, 3)[0]
    end_time = time.time()
    time_random = end_time- start_time
    print(f"Random: {compression_loss_random}, {time_random}")

    if complete_compression:
        start_time = time.time()
        compression_loss_full = compute_compression_loss(filename, num_components)[0]
        end_time = time.time()
        time_full = end_time- start_time
        print(f"Full: {compression_loss_full}, {time_full}")

    with h5py.File(filename, 'a') as f:
        append_to_dataset(f, 'compression_loss_random_values', data=compression_loss_random)
        if complete_compression:
            append_to_dataset(f, 'compression_loss_full_values', data=compression_loss_full)
        append_to_dataset(f, 'norm_diff_S_list', data=diff_S)
        append_to_dataset(f, 'norm_diff_mu_list', data=diff_mu)
        append_to_dataset(f, 'norm_diff_var_list', data=diff_var)

    print("Metrics saved to HDF5 file.")

def remove_file_with_timeout(filename_with_tag, overwrite=True, timeout=10):
    """
    Remove the file specified by filename_with_tag if it exists.
    
    Parameters:
        filename_with_tag (str): The name of the file to remove.
        overwrite (bool): Whether to attempt removal if the file exists (default is True).
        timeout (int): Maximum time allowed for attempting removal (default is 10 seconds).
    """
    start_time = time.time()  # Record the start time

    while overwrite and os.path.exists(filename_with_tag):
        # Check if the loop has been running for more than the timeout period
        if time.time() - start_time > timeout:
            break  # Exit the loop
            
        try:
            os.remove(filename_with_tag)
        except FileNotFoundError:
            break  # Exit the loop

def initialize_matrices(filename_with_tag):
    """
    Initialize matrices U, S, mu, and var based on whether the file exists.

    Parameters:
        filename_with_tag (str): The name of the file to check for existence.

    Returns:
        Tuple: Tuple containing U, S, mu, and var matrices.
    """
    if os.path.exists(filename_with_tag):
        with h5py.File(filename_with_tag, 'r') as f:
            previous_U = np.asarray(f.get('U'))
            previous_S = np.asarray(f.get('S'))
            previous_mu_tot = np.asarray(f.get('mu'))
            previous_var_tot = np.asarray(f.get('total_variance'))
    else:
        previous_U = None
        previous_S = None
        previous_mu_tot = None
        previous_var_tot = None 

    return previous_U, previous_S, previous_mu_tot, previous_var_tot

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

