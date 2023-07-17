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

import cProfile, sys

#############################################

class FreqDir:

    """Frequent Directions."""

    def __init__(
        self,
        john_start,
        tot_imgs,
        ell, 
        alpha,
        exp,
        run,
        det_type,
        batch_size=10,
        downsample=False,
        bin_factor=2,
        output_dir="",
    ):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
        self.psi.counter = john_start + tot_imgs*self.rank//self.size

        self.downsample = downsample
        self.bin_factor = bin_factor
        self.output_dir = output_dir

        (
            self.num_images,
            _,
            self.batch_size,
            self.num_features,
        ) = self.set_params(tot_imgs, ell, batch_size, bin_factor)

        self.task_durations = dict({})

        self.num_incorporated_images = 0
        self.outliers, self.pc_data = [], []

        self.d = self.num_features
        self.ell = ell
        self.m = 2*self.ell
        self.sketch = zeros( (self.m, self.d) ) 
        self.nextZeroRow = 0
        self.alpha = alpha

        self.dataseen = []
        
        self.noImgsToProcess = tot_imgs//self.size

        print("MY RANK IS: {}".format(self.rank))

        if self.rank==0:
            self.totPSI = PsanaInterface(exp=exp, run=run, det_type=det_type)
            self.totPSI.counter = john_start

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
            self.ell,
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
        # update model with remaining batches
        
        for batch in range(0,self.noImgsToProcess,self.batch_size):
            self.fetch_and_update_model(self.batch_size)


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
        """

#        pr = cProfile.Profile()
#        pr.enable()

        _, numIncorp = X.shape
        n = self.num_incorporated_images
        q = self.ell

        with TaskTimer(self.task_durations, "total update"):

            if self.rank == 0:
                print(
                    "Factoring {m} sample{s} into {n} sample, {q} component model...".format(
                        m=numIncorp, s="s" if numIncorp > 1 else "", n=n, q=q
                    )
                )
            for row in X.T:
                if self.nextZeroRow >= self.m:
                    self.john_rotate()
                self.sketch[self.nextZeroRow,:] = row 
                self.nextZeroRow += 1
                self.num_incorporated_images += 1
                if self.rank==0:
                    self.dataseen.append(row)
            
#            if self.rank==0:
#                print(f'{self.lowMemoryReconstructionErrorUnscaled():.6f}')

#        pr.disable()
#        # Dump results:
#        # - for binary dump
#        pr.dump_stats('logging/{0}_rank_{1}.prof'.format(currRun, self.rank))
#        # - for text dump
#        with open( 'logging/{0}_rank_{1}.txt'.format(currRun, self.rank), 'w') as output_file:
#            sys.stdout = output_file
#            pr.print_stats( sort='time' )
#            sys.stdout = sys.__stdout__

    
    def john_rotate(self):
        try:
            [_,s,Vt] = svd(self.sketch , full_matrices=False)
        except LinAlgError as err:
            [_,s,Vt] = scipy_svd(self.sketch, full_matrices = False)

        if len(s) >= self.ell:
            sCopy = s.copy()
            
            sShrunk = s[:self.ell]**2 - s[self.ell-1]**2
            #John: Explicitly set this value to be 0, since sometimes it is negative
            # or even turns to NaN due to roundoff error
            sShrunk[-1] = 0
            sShrunk = sqrt(sShrunk)
            
            sShrunk[:int(self.ell*(1-self.alpha))] = sCopy[:int(self.ell*(1-self.alpha))]

            self.sketch[:self.ell:,:] = dot(diag(sShrunk), Vt[:self.ell,:])
            self.sketch[self.ell:,:] = 0
            self.nextZeroRow = self.ell
        else:
            self.sketch[:len(s),:] = dot(diag(s), Vt[:len(s),:])
            self.sketch[len(s):,:] = 0
            self.nextZeroRow = len(s)

    def john_reconstructionError(self):
        matrixCentered = np.array(self.dataseen)
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

    def lowMemoryReconstructionErrorUnscaled(self):
        matrixCentered = np.array(self.dataseen)
        matSketch = self.sketch
        k = 10
        matrixCenteredT = matrixCentered.T
        matSketchT = matSketch.T
        U, S, Vt = np.linalg.svd(matSketchT)
        G = U[:,:k]
        return estimFrobNormSquared(matrixCenteredT, [G,G.T,matrixCenteredT], 100)

    def estimFrobNormSquared(addMe, arrs, its):
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
        sendbuf = self.sketch[:self.ell,:]
        recvbuf = None
        if self.rank == 0:
            recvbuf = np.empty(
                    [self.size, self.ell, self.d], dtype=np.float64)
        self.comm.Gather(sendbuf, recvbuf, root=0)
        if self.rank==0:
            origMatSketch = self.sketch.copy()
            for j in range(1, self.size):
                for row in recvbuf[j]:
                    if(np.any(row)):
                        if self.nextZeroRow >= self.m:
                            self.john_rotate()
                        self.sketch[self.nextZeroRow,:] = row 
                        self.nextZeroRow += 1
            toReturn = self.sketch.copy()
            self.sketch = origMatSketch
            print(toReturn)
            return toReturn
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
    pipca.get_outliers()

