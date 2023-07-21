import os, csv, argparse

import numpy as np
from mpi4py import MPI

from matplotlib import pyplot as plt
from matplotlib import colors

from btx.misc.shortcuts import TaskTimer

from btx.interfaces.ipsana import (
    PsanaInterface,
    bin_data,
    bin_pixel_index_map,
    retrieve_pixel_index_map,
    assemble_image_stack_batch,
)

###########################################
#John Imports
from numpy import zeros, sqrt, dot, diag
from numpy.linalg import svd, LinAlgError
from scipy.linalg import svd as scipy_svd
import numpy as np

import time

from datetime import datetime
currRun = datetime.now().strftime("%y%m%d%H%M%S")

import h5py
#############################################

class FreqDir:

    """Parallel Frequent Directions."""

    def __init__(
        self,
        john_start,
        tot_imgs,
        ell, 
        alpha,
        exp,
        run,
        det_type,
        rankAdapt,
        merger=False,
        mergerFeatures=0,
        downsample=False,
        bin_factor=2,
        output_dir="",
    ):


        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if not merger:
            self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
            self.psi.counter = john_start + tot_imgs*self.rank//self.size

            self.downsample = downsample
            self.bin_factor = bin_factor
            self.output_dir = output_dir

            (
                self.num_images,
                _,
                self.num_features,
            ) = self.set_params(tot_imgs, ell, bin_factor)

            self.task_durations = dict({})

            self.num_incorporated_images = 0
        else:
            #JOHN: NEED TO IMPROVE. CURRENTLY, NEED TO MANUALLY SET d, WHICH IS UNACCEPTABLE. 
            self.num_features = mergerFeatures
            self.task_durations = dict({})
            self.num_incorporated_images = 0

        self.d = self.num_features
        self.ell = ell
        self.m = 2*self.ell
        self.sketch = zeros( (self.m, self.d) ) 
        self.nextZeroRow = 0
        self.alpha = alpha

        self.noImgsToProcess = tot_imgs//self.size

        self.rankAdapt = rankAdapt
        self.increaseEll = False

    def set_params(self, num_images, num_components, bin_factor):
        """
        Method to initialize FreqDir parameters.

        Parameters
        ----------
        num_images : int
            Desired number of images to incorporate into model.
        num_components : int
            Desired number of components for model to maintain.
        bin_factor : int
            Factor to bin data by.

        Returns
        -------
        num_images : int
            Number of images to incorporate into model.
        num_components : int
            Number of components for model to maintain.
        num_features : int
            Number of features (dimension) in each image.
        """

        max_events = self.psi.max_events
        downsample = self.downsample

        num_images = min(num_images, max_events) if num_images != -1 else max_events
        num_components = min(num_components, num_images)

        # set d
        det_shape = self.psi.det.shape()
        num_features = np.prod(det_shape).astype(int)

        if downsample:
            if det_shape[-1] % bin_factor or det_shape[-2] % bin_factor:
                print("Invalid bin factor, toggled off downsampling.")
                self.downsample = False
            else:
                num_features = int(num_features / bin_factor**2)

        return num_images, num_components, num_features

    def run(self):
        """
        Perform frequent directions matrix sketching
        on run subject to initialization parameters.
        """

        for batch in range(0,self.noImgsToProcess,self.ell):
            self.fetch_and_update_model(self.ell)

        self.comm.Barrier()

    def get_formatted_images(self, n):
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

        return formatted_imgs

    def fetch_and_update_model(self, n):
        """
        Fetch images and update model.

        Parameters
        ----------
        n : int
            number of images to incorporate
        """

        img_batch = self.get_formatted_images(n)


        self.john_update_model(img_batch)


    def john_update_model(self, X):
        """
        Update matrix sketch with new batch of observations
        
        Parameters
        ----------
        X: ndarray
            data to update matrix sketch with
        """

        _, numIncorp = X.shape
#        n = self.num_incorporated_images
#        q = self.ell
#
        with TaskTimer(self.task_durations, "total update"):

#            if self.rank == 0:
#                print(
#                    "Factoring {m} sample{s} into {n} sample, {q} component model...".format(
#                        m=numIncorp, s="s" if numIncorp > 1 else "", n=n, q=q
#                    )
#                )
            for row in X.T:
                canRankAdapt = numIncorp > (self.ell + 15)
                if self.nextZeroRow >= self.m:
                    if self.increaseEll and canRankAdapt and self.rankAdapt:
                        self.ell = self.ell + 10
                        self.m = 2*self.ell
                        self.sketch = np.vstack((*self.sketch, np.zeros((20, self.d))))
                        self.increaseEll = False
                    else:
                        copyBatch = self.sketch[self.ell:,:].copy()
                        self.john_rotate()
                        if canRankAdapt and self.rankAdapt:
                            reconError = self.lowMemoryReconstructionErrorUnscaled(copyBatch)
                            if (np.sqrt(reconError) > 0.08):
                                self.increaseEll = True
                self.sketch[self.nextZeroRow,:] = row 
                self.nextZeroRow += 1
                self.num_incorporated_images += 1
                numIncorp -= 1
#            if self.rank==0:
#                print(f'{self.lowMemoryReconstructionErrorUnscaled():.6f}')
    
    def john_rotate(self):
        """ 
        Apply Frequent Directions Algorithm to 
        current matrix sketch and adjoined buffer

        Notes
        -----
        Based on [1] and [2]. 

        [1] Frequent Directions: Simple and Deterministic Matrix 
        Sketching Mina Ghashami, Edo Liberty, Jeff M. Phillips, and 
        David P. Woodruff SIAM Journal on Computing 2016 45:5, 1762-1792

        [2] Ghashami, M., Desai, A., Phillips, J.M. (2014). Improved 
        Practical Matrix Sketching with Guarantees. In: Schulz, A.S., 
        Wagner, D. (eds) Algorithms - ESA 2014. ESA 2014. Lecture Notes 
        in Computer Science, vol 8737. Springer, Berlin, Heidelberg. 
        https://doi.org/10.1007/978-3-662-44777-2_39
        """

        try:
            [_,s,Vt] = svd(self.sketch , full_matrices=False)
        except LinAlgError as err:
            [_,s,Vt] = scipy_svd(self.sketch, full_matrices = False)

        if len(s) >= self.ell:
            sCopy = s.copy()
            
            toShrink = s[:self.ell]**2 - s[self.ell-1]**2
            #John: Explicitly set this value to be 0, since sometimes it is negative
            # or even turns to NaN due to roundoff error
            toShrink[-1] = 0
            toShrink = sqrt(toShrink)
            
            toShrink[:int(self.ell*(1-self.alpha))] = sCopy[:int(self.ell*(1-self.alpha))]

            self.sketch[:self.ell:,:] = dot(diag(toShrink), Vt[:self.ell,:])
            self.sketch[self.ell:,:] = 0
            self.nextZeroRow = self.ell
        else:
            self.sketch[:len(s),:] = dot(diag(s), Vt[:len(s),:])
            self.sketch[len(s):,:] = 0
            self.nextZeroRow = len(s)

    def john_reconstructionError(self, matrixCentered):
        """ 
        Compute the reconstruction error of the matrix sketch
        against given data

        Parameters
        ----------
        matrixCentered: ndarray
           Data to compare matrix sketch to 
       """

        matSketch = self.sketch
        k = 10
        matrixCenteredT = matrixCentered.T
        matSketchT = matSketch.T
        U, S, Vt = np.linalg.svd(matSketchT)
        G = U[:,:k]
        UA, SA, VtA = np.linalg.svd(matrixCenteredT)
        UAk = UA[:,:k]
        SAk = np.diag(SA[:k])
        VtAk = VtA[:k]
        Ak = UAk @ SAk @ VtAk
        return (np.linalg.norm(
        	matrixCenteredT - G @ G.T @ matrixCenteredT, 'fro')**2)/(
                (np.linalg.norm(matrixCenteredT - Ak, 'fro'))**2) 

    def lowMemoryReconstructionErrorUnscaled(self, matrixCentered):
        """ 
        Compute the low memory reconstruction error of the matrix sketch
        against given data. This si the same as john_reconstructionError,
        but estimates the norm computation and does not scale by the matrix. 

        Parameters
        ----------
        matrixCentered: ndarray
           Data to compare matrix sketch to 
       """

        matSketch = self.sketch
        k = 10
        matrixCenteredT = matrixCentered.T
        matSketchT = matSketch.T
        U, S, Vt = np.linalg.svd(matSketchT)
        G = U[:,:k]
        return estimFrobNormSquared(matrixCenteredT, [G,G.T,matrixCenteredT], 100)

    def estimFrobNormSquared(self, addMe, arrs, its):
        """ 
        Estimate the Frobenius Norm of product of arrs matrices 
        plus addME matrix using its iterations. 

        Parameters
        ----------
        arrs: list of ndarray
           Matrices to multiply together

        addMe: ndarray
            Matrix to add to others

        its: int
            Number of iterations to average over

        Returns
        -------
        sumMe/its*no_rows : float
            Estimate of frobenius norm of produce 
            of arrs matrices plus addMe matrix

        Notes
        -----
        Frobenius estimation is the expected value of matrix
        multiplied by random vector from multivariate normal distribution
        based on [1]. 

        [1] Norm and Trace Estimation with Random Rank-one Vectors 
        Zvonimir Bujanovic and Daniel Kressner SIAM Journal on Matrix 
        Analysis and Applications 2021 42:1, 202-223
       """

        no_rows = arrs[-1].shape[1]
        v = np.random.normal(size=no_rows)
        v_hat = v / np.linalg.norm(v)
        sumMe = 0
        for j in range(its):
            v = np.random.normal(size=no_rows)
            v_hat = v / np.linalg.norm(v)
            v_addMe = addMe @ v_hat
            for arr in arrs[::-1]:
                v_hat = arr @ v_hat
            sumMe = sumMe + (np.linalg.norm(v_addMe - v_hat))**2
        return sumMe/its*no_rows


    def gatherFreqDirs(self):
        """
        Gather local matrix sketches to root node and
        merge local sketches together. 
        """
        sendbuf = self.ell
        buffSizes = np.array(self.comm.allgather(sendbuf))

        if self.rank==0:
            origMatSketch = self.sketch.copy()
            origNextZeroRow = self.nextZeroRow
            self.nextZeroRow = self.ell
            counter = 0
            for proc in range(1, self.size):
                bufferMe = np.empty(buffSizes[self.rank]*self.d, dtype=np.double)
                self.comm.Recv(bufferMe, source=proc, tag=13)
                bufferMe = np.reshape(bufferMe, (buffSizes[self.rank], self.d))
                for row in bufferMe:
                    if(np.any(row)):
                        if self.nextZeroRow >= self.m:
                            self.john_rotate()
                    self.sketch[self.nextZeroRow,:] = row 
                    self.nextZeroRow += 1
                    counter += 1
#                    print("DATA PROCESSED: {}".format(counter))
            toReturn = self.sketch.copy()
            print("COMPLETED MERGE PROCESS: ", toReturn)
            self.sketch = origMatSketch
            return toReturn
        else:
            bufferMe = self.sketch[:self.ell, :].copy().flatten()
            self.comm.Send(bufferMe, dest=0, tag=13)
            return 

    def get(self):
        return self.sketch[:self.ell, :]

class MergeTree:

    """Frequent Directions Merging Object."""

    def __init__(self, divBy, readFile, dataSetName):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.divBy = divBy
        
        with h5py.File(readFile, 'r') as hf:
            self.data = hf[dataSetName][:]

        print("AOIDJWOIJA", self.rank, self.data.shape)

        self.fd = FreqDir(0, 0, rankAdapt=False, exp='0', run='0', det_type='0', ell=self.data.shape[0], alpha=0.2, downsample=False, bin_factor=0, merger=True, mergerFeatures = self.data.shape[1]) 

        self.fd.d = self.data.shape[1]

        sendbuf = self.data.shape[0]
        self.buffSizes = np.array(self.comm.allgather(sendbuf))
        print(self.buffSizes)

        #JOHN: MUST CHECK THAT THIS ACTION ONLY FILLS UP THE SKETCH WITH THE CURRENT SKETCH FROM THE DATA
        self.fd.john_update_model(self.data)


    def merge(self):

        """
        Merge Frequent Direction Components in a tree-like fashion. 
        Returns
        -------
        finalSketch : ndarray
            Merged matrix sketch of cumulative data


        """
        powerNum = 1
        while(powerNum < self.size):
            powerNum = powerNum * self.divBy
        if powerNum != size:
            raise ValueError('NUMBER OF CORES WOULD LEAD TO INBALANCED MERGE TREE. ENDING PROGRAM.')
            return

        level = 0
        while((self.divBy ** level) < self.size):
            jump = self.divBy ** level
            if(self.rank%jump ==0):
                root = self.rank - (self.rank%(jump*self.divBy))
                grouping = [j for j in range(root, root + jump*self.divBy, jump)]
                print(grouping)
#                if self.rank==root:
#                    for proc in grouping[1:]:
#                        bufferMe = np.empty(self.data.shape[0] * self.data.shape[1], dtype=np.double)
#                        comm.Recv(bufferMe, source=proc, tag=17)
#                        bufferMe = np.reshape(bufferMe, (self.data.shape[0], self.data.shape[1]))
#                        self.fd.john_update_model(bufferMe.T)
#                        print(level, data)
#                else:
#                    bufferMe = self.fd.get().copy().flatten()
#                    comm.Send(bufferMe, dest=root, tag=17)
            level += 1
        if self.rank==0:
            finalSketch = self.fd.get()
            return finalSketch
        else:
            return


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
