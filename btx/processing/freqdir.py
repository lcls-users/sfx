import sys
sys.path.append("/sdf/home/w/winnicki/btx/")
from btx.processing.dimRed import *

import os, csv, argparse
import math
import time
import random
from collections import Counter
import h5py

import numpy as np
from numpy import zeros, sqrt, dot, diag
from numpy.linalg import svd, LinAlgError
from scipy.linalg import svd as scipy_svd
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
import heapq

from mpi4py import MPI

from matplotlib import pyplot as plt
from matplotlib import colors

# from btx.misc.shortcuts import TaskTimer
#
# from btx.interfaces.ipsana import (
#     PsanaInterface,
#     bin_data,
#     bin_pixel_index_map,
#     retrieve_pixel_index_map,
#     assemble_image_stack_batch,
# )

from PIL import Image
from io import BytesIO
import base64

from datetime import datetime

#import umap
#import hdbscan
from sklearn.cluster import OPTICS, cluster_optics_dbscan

from matplotlib import colors
import matplotlib as mpl
from matplotlib import cm

from bokeh.plotting import figure, show, output_file, save
from bokeh.models import HoverTool, CategoricalColorMapper, LinearColorMapper, ColumnDataSource, CustomJS, Slider, RangeSlider, Toggle, RadioButtonGroup, Range1d, Label
from bokeh.palettes import Viridis256, Cividis256, Turbo256, Category20, Plasma3
from bokeh.layouts import column, row

import cProfile
import string

class FreqDir(DimRed):

    """
    Parallel Rank Adaptive Frequent Directions.
    
    Based on [1] and [2]. Frequent Directions is a matrix sketching algorithm used to
    approximate large data sets. The basic goal of matrix sketching is to process an
    n x d matrix A to somehow represent a matrix B so that ||A-B|| or covariance error
    is small. Frequent Directions provably acheives a spectral bound on covariance 
    error and greatly outperforms comparable existing sketching techniques. It acheives
    similar runtime and performance to incremental SVD as well. 

    In this module we implement the frequent directions algorithm. This is the first of
    three modules in this data processing pipeline, and it produces a sketch of a subset
    of the data into an h5 file. The "Merge Tree" module will be responsible for merging
    each of the sketches together, parallelizing the process, and the apply compression
    algorithm will be responsible for using the full matrix sketch projecting the 
    original data to low dimensional space for data exploration. 

    One novel feature of this implementation is the rank adaption feature: users have the
    ability to select the approximate reconstruction error they want the sketch to operate
    over, and the algorithm will adjust the rank of the sketch to meet this error bound
    as data streams in. The module also gives users the ability to perform the sketching
    process over thresholded and non-zero image data.

    [1] Frequent Directions: Simple and Deterministic Matrix 
    Sketching Mina Ghashami, Edo Liberty, Jeff M. Phillips, and 
    David P. Woodruff SIAM Journal on Computing 2016 45:5, 1762-1792

    [2] Ghashami, M., Desai, A., Phillips, J.M. (2014). Improved 
    Practical Matrix Sketching with Guarantees. In: Schulz, A.S., 
    Wagner, D. (eds) Algorithms - ESA 2014. ESA 2014. Lecture Notes 
    in Computer Science, vol 8737. Springer, Berlin, Heidelberg. 
    https://doi.org/10.1007/978-3-662-44777-2_39

    Attributes
    ----------
       start_offset: starting index of images to process
       num_imgs: total number of images to process
       ell: number of components of matrix sketch
       alpha: proportion  of components to not rotate in frequent directions algorithm
       exp, run, det_type: experiment properties
       rankAdapt: indicates whether to perform rank adaptive FD
       increaseEll: internal variable indicating whether ell should be increased for rank adaption
       output_dir: directory to write output
       merger: indicates whether object will be used to merge other FD objects
       mergerFeatures: used if merger is true and indicates number of features of local matrix sketches
       downsample, bin_factor: whether data should be downsampled and by how much
       threshold: whether data should be thresholded (zero if less than threshold amount)
       normalizeIntensity: whether data should be normalized to have total intensity of one
       noZeroIntensity: whether data with low total intensity should be discarded
       d: number of features (pixels) in data
       m: internal frequent directions variable recording total number of components used in algorithm
       sketch: numpy array housing current matrix sketch
       mean: geometric mean of data processed
       num_incorporated_images: number of images processed so far
       imgsTracked: indices of images processed so far
       currRun: Current datetime used to identify run
       samplingFactor: Proportion of batch data to process based on Priority Sampling Algorithm
    """

    def __init__(
        self,
        comm,
        rank,
        size,
        start_offset,
        num_imgs,
        exp,
        run,
        det_type,
        output_dir,
        currRun,
        imgData,
        imgsTracked,
        alpha,
        rankAdapt,
        rankAdaptMinError,
        merger,
        mergerFeatures,
        downsample,
        bin_factor,
        samplingFactor, 
        num_components,
        psi,
        usePSI
    ):

########################
        if usePSI:
            super().__init__(exp=exp, run=run, det_type=det_type, start_offset=start_offset,
                    num_images=num_imgs, num_components=num_components, batch_size=0, priming=False,
                    downsample=downsample, bin_factor=bin_factor, output_dir=output_dir, psi=psi)
        else:
            self.start_offset = start_offset
            self.downsample = False
            self.bin_factor = 0
            self.output_dir = output_dir
            self.num_components = num_components
            self.num_features,self.num_images = imgData.shape 
            print("NUM IMAGES: ", self.num_images)

            self.task_durations = dict({})

            self.num_incorporated_images = 0
            self.outliers, self.pc_data = [], []
########################
        self.comm = comm
        self.rank= rank
        self.size = size

        self.currRun = currRun

        self.output_dir = output_dir

        self.merger = merger

        if self.merger:
            self.num_features = mergerFeatures

        self.num_incorporated_images = 0

        self.d = self.num_features
        self.ell = num_components
        self.m = 2*self.ell
        self.sketch = zeros( (self.m, self.d) ) 
        self.nextZeroRow = 0
        self.alpha = alpha
#        self.mean = None

        self.rankAdapt = rankAdapt
        self.rankAdaptMinError = rankAdaptMinError
        self.increaseEll = False

        self.samplingFactor = samplingFactor

        self.imgData = imgData
        self.imgsTracked = imgsTracked

    def run(self):
        """
        Perform frequent directions matrix sketching
        on run subject to initialization parameters.
        """
        img_batch = self.imgData
        if self.samplingFactor <1:
            psamp = PrioritySampling(int((img_batch.shape[1])*self.samplingFactor), self.d)
            for row in img_batch.T:
                psamp.update(row)
            img_batch = np.array(psamp.sketch.get()).T
        self.update_model(img_batch)
#        if self.mean is None:
#            self.mean = np.mean(img_batch, axis=1)
#        else:
##            self.mean = (self.mean*self.num_incorporated_images + np.sum(img_batch.T, axis=0))/(
##                    self.num_incorporated_images + (img_batch.shape[1]))
#             self.mean = (self.mean*self.num_incorporated_images + np.sum(img_batch, axis=1, dtype=np.double))/(
#                    self.num_incorporated_images + (img_batch.shape[1]))
#        self.update_model((img_batch.T - self.mean).T)

    def update_model(self, X):
        """
        Update matrix sketch with new batch of observations. 

        The matrix sketch array is of size 2*ell. The first ell rows maintained
        represent the current matrix sketch. The next ell rows form a buffer.
        Each row of the data is added to the buffer until ell rows have been
        accumulated. Then, we apply the rotate function to the buffer, which
        incorporates the buffer data into the matrix sketch. 
        
        Following the rotation step, it is checked if rank adaption is enabled. Then,
        is checked if there is enough data to perform one full rotation/shrinkage
        step. Without this check, one runs the risk of having zero rows in the
        sketch, which is innaccurate in representing the data one has seen.
        If one can increase the rank, the increaseEll flag is raised, and once sufficient
        data has been accumulated in the buffer, the sketch and buffer size is increased.
        This happens when we check if increaseEll, canRankAdapt, and rankAdapt are all true,
        whereby we check if we should be increasing the rank due to high error, we
        have sufficient incoming data to do so (to avoid zero rows in the matrix sketch), 
        and the user would like for the rank to be adaptive, respectively. 
        
        Parameters
        ----------
        X: ndarray
            data to update matrix sketch with
        """
        _, numIncorp  = X.shape
        origNumIncorp = numIncorp
        with TaskTimer(self.task_durations, "total update"):
            if self.rank==0 and not self.merger:
                print(
                    "Factoring {m} sample{s} into {n} sample, {q} component model...".format(
                        m=numIncorp, s="s" if numIncorp > 1 else "", n=self.num_incorporated_images, q=self.ell
                    )
                )
            for row in X.T:
                canRankAdapt = numIncorp > (self.ell + 15)
                if self.nextZeroRow >= self.m:
                    if self.increaseEll and canRankAdapt and self.rankAdapt:
                        self.ell = self.ell + 10
                        self.m = 2*self.ell
                        self.sketch = np.vstack((*self.sketch, np.zeros((20, self.d))))
                        self.increaseEll = False
                        print("Increasing rank of process {} to {}".format(self.rank, self.ell))
                    else:
                        copyBatch = self.sketch[self.ell:,:].copy()
                        self.rotate()
                        if canRankAdapt and self.rankAdapt:
                            reconError = np.sqrt(self.lowMemoryReconstructionErrorScaled(copyBatch))
                            print("RANK ADAPT RECON ERROR: ", reconError)
                            if (reconError > self.rankAdaptMinError):
                                self.increaseEll = True
                self.sketch[self.nextZeroRow,:] = row 
                self.nextZeroRow += 1
                self.num_incorporated_images += 1
                numIncorp -= 1
    
    def rotate(self):
        """ 
        Apply Frequent Directions rotation/shrinkage step to current matrix sketch and adjoined buffer. 

        The Frequent Directions algorithm is inspired by the well known Misra Gries Frequent Items
        algorithm. The Frequent Items problem is informally as follows: given a sequence of items, find the items which occur most frequently. The Misra Gries Frequent Items algorithm maintains a dictionary of <= k items and counts. For each item in a sequence, if the item is in the dictionary, increase its count. if the item is not in the dictionary and the size of the dictionary is <= k, then add the item with a count of 1 to the dictionary. Otherwise, decrease all counts in the dictionary by 1 and remove any items with 0 count. Every item which occurs more than n/k times is guaranteed to appear in the output array.

        The Frequent Directions Algorithm works in an analogous way for vectors: in the same way that Frequent Items periodically deletes ell different elements, Frequent Directions periodically "shrinks? ell orthogonal vectors by roughly the same amount. To do so, at each step: 1) Data is appended to the matrix sketch (whereby the last ell rows form a buffer and are zeroed at the start of the algorithm and after each rotation). 2) Matrix Sketch is rotated from left via SVD so that its rows are orthogonal and in descending magnitude order. 3) Norm of sketch rows are shrunk so that the smallest direction is set to 0.

        This function performs the rotation and shrinkage step by performing SVD and left multiplying by the unitary U matrix, followed by a subtraction. This particular implementation follows the alpha FD algorithm, which only performs the shrinkage step on the first alpha rows of the sketch, which has been shown to perform better than vanilla FD in [2]. 

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
        [_,S,Vt] = np.linalg.svd(self.sketch , full_matrices=False)
        ssize = S.shape[0]
        if ssize >= self.ell:
            sCopy = S.copy()
           #JOHN: I think actually this should be ell+1 and ell. We lose a component otherwise.
            toShrink = S[:self.ell]**2 - S[self.ell-1]**2
            #John: Explicitly set this value to be 0, since sometimes it is negative
            # or even turns to NaN due to roundoff error
            toShrink[-1] = 0
            toShrink = sqrt(toShrink)
            toShrink[:int(self.ell*(1-self.alpha))] = sCopy[:int(self.ell*(1-self.alpha))]
            #self.sketch[:self.ell:,:] = dot(diag(toShrink), Vt[:self.ell,:]) #JOHN: Removed this extra colon 10/01/2023
            self.sketch[:self.ell,:] = dot(diag(toShrink), Vt[:self.ell,:])
            self.sketch[self.ell:,:] = 0
            self.nextZeroRow = self.ell
        else:
            print(S.shape, self.ell)
            self.sketch[:ssize,:] = diag(s) @ Vt[:ssize,:]
            self.sketch[ssize:,:] = 0
            self.nextZeroRow = ssize

    def reconstructionError(self, matrixCentered):
        """ 
        Compute the reconstruction error of the matrix sketch
        against given data

        Parameters
        ----------
        matrixCentered: ndarray
           Data to compare matrix sketch to 

        Returns
        -------
        float,
            Data subtracted by data projected onto sketched space, scaled by minimum theoretical sketch
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

#    def lowMemoryReconstructionErrorScaled(self, matrixCentered):
#        """ 
#        Compute the low memory reconstruction error of the matrix sketch
#        against given data. This is the same as reconstructionError,
#        but estimates the norm computation and does not scale by the 
#        minimum projection matrix, but rather by the matrix norm itself. 
#
#        Parameters
#        ----------
#        matrixCentered: ndarray
#           Data to compare matrix sketch to 
#
#        Returns
#        -------
#        float,
#            Data subtracted by data projected onto sketched space, scaled by matrix elements
#       """
#        matSketch = self.sketch[:self.ell, :]
#        print("RANK ADAPTIVE SHAPE:",matrixCentered.shape, matSketch.shape)
##        k = 10
#        matrixCenteredT = matrixCentered.T
#        matSketchT = matSketch.T
#        U, S, Vt = np.linalg.svd(matSketchT, full_matrices=False)
##        G = U[:,:k]
#        G = U
#        return (self.estimFrobNormSquared(matrixCenteredT, [G,G.T,matrixCenteredT], 50)/
#                np.linalg.norm(matrixCenteredT, 'fro')**2)

    def lowMemoryReconstructionErrorScaled(self, matrixCentered):
        matSketch = self.sketch[:self.ell, :]
        matrixCenteredT = matrixCentered.T
        matSketchT = matSketch.T
        U, S, Vt = np.linalg.svd(matSketchT, full_matrices=False)
        G = U
        return self.estimFrobNormJ(matrixCenteredT, [G,G.T,matrixCenteredT], 20)/np.linalg.norm(matrixCenteredT, 'fro')

    def estimFrobNormJ(self, addMe, arrs, k):
        m, n = addMe.shape
        randMat = np.random.normal(0, 1, size=(n, k))
        minusMe = addMe @ randMat
        sumMe = 0
        for arr in arrs[::-1]:
            randMat = arr @ randMat
        sumMe += math.sqrt(1/k) * np.linalg.norm(randMat - minusMe, 'fro')
        return sumMe

#    def estimFrobNormSquared(self, addMe, arrs, its):
#        """ 
#        Estimate the Frobenius Norm of product of arrs matrices 
#        plus addME matrix using its iterations. 
#
#        Parameters
#        ----------
#        arrs: list of ndarray
#           Matrices to multiply together
#
#        addMe: ndarray
#            Matrix to add to others
#
#        its: int
#            Number of iterations to average over
#
#        Returns
#        -------
#        sumMe/its*no_rows : float
#            Estimate of frobenius norm of product
#            of arrs matrices plus addMe matrix
#
#        Notes
#        -----
#        Frobenius estimation is the expected value of matrix
#        multiplied by random vector from multivariate normal distribution
#        based on [1]. 
#
#        [1] Norm and Trace Estimation with Random Rank-one Vectors 
#        Zvonimir Bujanovic and Daniel Kressner SIAM Journal on Matrix 
#        Analysis and Applications 2021 42:1, 202-223
#       """
#        no_rows = arrs[-1].shape[1]
#        v = np.random.normal(size=no_rows)
#        v_hat = v / np.linalg.norm(v)
#        sumMe = 0
#        for j in range(its):
#            v = np.random.normal(size=no_rows)
#            v_hat = v / np.linalg.norm(v)
#            v_addMe = addMe @ v_hat
#            for arr in arrs[::-1]:
#                v_hat = arr @ v_hat
#            sumMe = sumMe + (np.linalg.norm(v_addMe - v_hat))**2
#        return sumMe/its*no_rows


    def gatherFreqDirsSerial(self):
        """
        Gather local matrix sketches to root node and
        merge local sketches together in a serial fashion. 

        Returns
        -------
        toReturn : ndarray
            Sketch of all data processed by all cores
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
                            self.rotate()
                    self.sketch[self.nextZeroRow,:] = row 
                    self.nextZeroRow += 1
                    counter += 1
            toReturn = self.sketch.copy()
            self.sketch = origMatSketch
            return toReturn
        else:
            bufferMe = self.sketch[:self.ell, :].copy().flatten()
            self.comm.Send(bufferMe, dest=0, tag=13)
            return 

    def get(self):
        """
        Fetch matrix sketch

        Returns
        -------
        self.sketch[:self.ell,:] : ndarray
            Sketch of data locally processed
        """
        return self.sketch[:self.ell, :]

    def write(self):
        """
        Write matrix sketch to h5 file. 

        Returns
        -------
        filename : string
            Name of h5 file where sketch, mean of data, and indices of data processed is written
        """
#        self.comm.barrier()
        filename = self.output_dir + '{}_sketch_{}.h5'.format(self.currRun, self.rank)
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset("sketch",  data=self.sketch[:self.ell, :])
#            hf.create_dataset("mean", data=self.mean)
            hf.create_dataset("imgsTracked", data=np.array(self.imgsTracked))
            hf["sketch"].attrs["numImgsIncorp"] = self.num_incorporated_images
        print(self.rank, "CREATED FILE: ", filename)
        self.comm.barrier()
        return filename 


class MergeTree:

    """
    Class used to efficiently merge Frequent Directions Matrix Sketches

    The Frequent Directions matrix sketch has the special property that it is a mergeable
    summary. This means it can be merged easily and retain the same theoretical guarantees
    by stacking two sketches ontop of one another and applying the algorithm again.

    We can perform this merging process in a tree-like fashion in order to merge any 
    number of sketches in log number of applications of the frequent directions algorithm. 

    The class is designed to take in local sketches of data from h5 files produced by 
    the FreqDir class (where local refers to the fact that a subset of the total number
    of images has been processed by the algorithm in a single core and saved to its own h5 file).

    Attributes
    ----------
    divBy: Factor to merge by at each step: number of sketches must be a power of divBy 
    readFile: File name of local sketch for this particular core to process
    dir: directory to write output
    allWriteDirecs: all file names of local sketches
    currRun: Current datetime used to identify run
    """

    def __init__(self, comm, rank, size, exp, run, det_type, divBy, readFile, output_dir, allWriteDirecs, currRun, psi, usePSI):
        self.comm = comm
        self.rank = rank
        self.size = size
        
        self.divBy = divBy
        
        time.sleep(10)
        with h5py.File(readFile, 'r') as hf:
            self.data = hf["sketch"][:]

        self.fd = FreqDir(comm=comm, rank=rank, size=size, num_imgs=0, start_offset=0, currRun = currRun, rankAdapt=False, rankAdaptMinError=1, exp=exp, run=run, det_type=det_type, num_components=self.data.shape[0], alpha=0.2, merger=True, mergerFeatures = self.data.shape[1], output_dir=output_dir, imgData = np.random.rand(2, 2), imgsTracked=None, downsample=False, bin_factor=1, samplingFactor=1, psi=psi, usePSI=usePSI)

        sendbuf = self.data.shape[0]
        self.buffSizes = np.array(self.comm.allgather(sendbuf))
#        print(self.buffSizes)

        self.fd.update_model(self.data.T)

        self.output_dir = output_dir

        self.allWriteDirecs = allWriteDirecs


        self.fullMean = None
        self.fullNumIncorp = 0
        self.fullImgsTracked = []

        self.currRun = currRun

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
        if powerNum != self.size:
            raise ValueError('NUMBER OF CORES WOULD LEAD TO INBALANCED MERGE TREE. ENDING PROGRAM.')
            return

        level = 0
        while((self.divBy ** level) < self.size):
            jump = self.divBy ** level
            if(self.rank%jump ==0):
                root = self.rank - (self.rank%(jump*self.divBy))
                grouping = [j for j in range(root, root + jump*self.divBy, jump)]
                if self.rank==root:
                    for proc in grouping[1:]:
                        bufferMe = np.empty(self.buffSizes[proc] * self.data.shape[1], dtype=np.double)
                        self.comm.Recv(bufferMe, source=proc, tag=17)
                        bufferMe = np.reshape(bufferMe, (self.buffSizes[proc], self.data.shape[1]))
                        self.fd.update_model(np.hstack((bufferMe.T, np.zeros((bufferMe.shape[1],1)))))
#                        self.fd.update_model(bufferMe.T)
                else:
                    bufferMe = self.fd.get().copy().flatten()
                    self.comm.Send(bufferMe, dest=root, tag=17)
            level += 1
        if self.rank==0:
            fullLen = len(self.allWriteDirecs)
            for readMe in self.allWriteDirecs:
                with h5py.File(readMe, 'r') as hf:
#                    if self.fullMean is None:
#                        self.fullMean = hf["mean"][:]
                    if self.fullNumIncorp==0:
                        self.fullNumIncorp = hf["sketch"].attrs["numImgsIncorp"]
                        self.fullImgsTracked = hf["imgsTracked"][:]
                    else:
#                        self.fullMean =  (self.fullMean*self.fullNumIncorp + hf["mean"][:])/(self.fullNumIncorp
#                                + hf["sketch"].attrs["numImgsIncorp"])
                        self.fullNumIncorp += hf["sketch"].attrs["numImgsIncorp"]
                        self.fullImgsTracked = np.vstack((self.fullImgsTracked,  hf["imgsTracked"][:]))
            return self.fd.get()
        else:
            return

    def write(self):
        """
        Write merged matrix sketch to h5 file
        """
        filename = self.output_dir + '{}_merge.h5'.format(self.currRun)

        if self.rank==0:
            for ind in range(self.size):
                filename2 = filename[:-3] + "_"+str(ind)+".h5"
                with h5py.File(filename2, 'w') as hf:
                    hf.create_dataset("sketch",  data=self.fd.sketch[:self.fd.ell, :])
#                    hf.create_dataset("mean",  data=self.fullMean)
                    hf["sketch"].attrs["numImgsIncorp"] = self.fullNumIncorp
                    hf.create_dataset("imgsTracked",  data=self.fullImgsTracked)
                self.comm.send(filename2, dest=ind, tag=ind)
        else:
            print("{} RECEIVED FILE NAME: {}".format(self.rank, self.comm.recv(source=0, tag=self.rank)))
        self.comm.barrier()
        return filename

class ApplyCompression:
    """
    Compute principal components of matrix sketch and apply to data

    Attributes
    ----------
    start_offset: starting index of images to process
    num_imgs: total number of images to process
    exp, run, det_type: experiment properties
    dir: directory to write output
    downsample, bin_factor: whether data should be downsampled and by how much
    threshold: whether data should be thresholded (zero if less than threshold amount)
    normalizeIntensity: whether data should be normalized to have total intensity of one
    noZeroIntensity: whether data with low total intensity should be discarded
    readFile: H5 file with matrix sketch
    data: numpy array housing current matrix sketch
    mean: geometric mean of data processed
    num_incorporated_images: number of images processed so far
    imgageIndicesProcessed: indices of images processed so far
    currRun: Current datetime used to identify run
    imgGrabber: FD object used solely to retrieve data from psana
    grabberToSaveImages: FD object used solely to retrieve 
    non-downsampled data for thumbnail generation
    components: Principal Components of matrix sketch
    processedData: Data projected onto matrix sketch range
    """

    def __init__(
        self,
        comm,
        rank,
        size,
        start_offset,
        num_imgs,
        exp,
        run,
        det_type,
        readFile,
        output_dir,
        currRun,
        imgData, 
    ):

        self.comm = comm
        self.rank = rank
        self.size= size

        self.output_dir = output_dir

        self.num_imgs = num_imgs

        self.currRun = currRun

        self.num_incorporated_images = 0

        readFile2 = readFile[:-3] + "_"+str(self.rank)+".h5"

#        print("FOR RANK {}, READFILE: {} HAS THE CURRENT EXISTENCE STATUS {}".format(self.rank, readFile2, os.path.isfile(readFile2)))
#        while(not os.path.isfile(readFile2)):
#            print("{} DOES NOT CURRENTLY EXIST FOR {}".format(readFile2, self.rank))
        time.sleep(10)
        with h5py.File(readFile2, 'r') as hf:
            self.data = hf["sketch"][:]
#            self.mean = hf["mean"][:]
        
        U, S, Vt = np.linalg.svd(self.data, full_matrices=False)
        self.components = Vt
        
        self.processedData = None

        self.imageIndicesProcessed = []

        self.imgData = imgData


    def run(self):
        """
        Retrieve sketch, project images onto new coordinates. Save new coordinates to h5 file.

        Note: If-Else statement is from previous/future work enabling streaming processing. 
        """
        self.apply_compression(self.imgData)
        return self.data

    def apply_compression(self, X):
        """
        Project data X onto matrix sketch space. 

        Parameters
        ----------
        X: ndarray
            data to project
        """
        if self.processedData is None:
            self.processedData = np.dot(X.T, self.components.T)
        else:
            self.processedData = np.vstack((self.processedData, np.dot(X.T, self.components.T)))

    def write(self):
        """
        Write projected data and downsampled data to h5 file
        """
        filename = self.output_dir + '{}_ProjectedData_{}.h5'.format(self.currRun, self.rank)
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset("ProjectedData",  data=self.processedData)
#        print("CREATED FILE: ", filename)
        self.comm.barrier()
        return filename


class CustomPriorityQueue:
    """
    Custom Priority Queue. 

    Maintains a priority queue of items based on user-inputted priority for said items. 
    """
    def __init__(self, max_size):
        self.queue = []
        self.index = 0  # To handle items with the same priority
        self.max_size = max_size

    def push(self, item, priority, origWeight):
        if len(self.queue) >= self.max_size:
            self.pop()  # Remove the lowest-priority item if queue is full
        heapq.heappush(self.queue, (priority, self.index, (item, priority, origWeight)))
        self.index += 1

    def pop(self):
        return heapq.heappop(self.queue)[-1]

    def is_empty(self):
        return len(self.queue) == 0

    def size(self):
        return len(self.queue)

    def get(self):
        ret = []
        while self.queue:
            curr = heapq.heappop(self.queue)[-1]
            #ret.append(curr[0]*max(curr[1], curr[2])/curr[2])
            ret.append(curr[0])
        return ret

class PrioritySampling:
    """
    Priority Sampling. 

    Based on [1] and [2]. Frequent Directions is a sampling algorithm that, 
    given a high-volume stream of weighted items, creates a generic sample 
    of a certain limited size that can later be used to estimate the total 
    weight of arbitrary subsets. In our case, we use Priority Sampling to
    generate a matrix sketch based, sampling rows of our data using the
    2-norm as weights. Priority Sampling "first assigns each element i a random 
    number u_i ∈ Unif(0, 1). This implies a priority p_i = w_i/u_i , based 
    on its weight w_i (which for matrix rows w_i = ||a||_i^2). We then simply 
    retain the l rows with largest priorities, using a priority queue of size l."

    [1] Nick Duffield, Carsten Lund, and Mikkel Thorup. 2007. Priority sampling for 
    estimation of arbitrary subset sums. J. ACM 54, 6 (December 2007), 32–es. 
    https://doi.org/10.1145/1314690.1314696

    Attributes
    ----------
    ell: Number of components to keep
    d: Number of features of each datapoint
    sketch: Matrix Sketch maintained by Priority Queue

    """
    def __init__(self, ell, d):
        self.ell = ell
        self.d = d
        self.sketch = CustomPriorityQueue(self.ell)

    def update(self, vec):
        ui = random.random()
        wi = np.linalg.norm(vec)**2
        pi = wi/ui
        self.sketch.push(vec, pi, wi)


class visualizeFD:
    """
    Visualize FD Dimension Reduction using UMAP and DBSCAN
    """
    def __init__(self, inputFile, outputFile, numImgsToUse, nprocs, includeABOD, userGroupings, 
            skipSize, umap_n_neighbors, umap_random_state, hdbscan_min_samples, hdbscan_min_cluster_size,
            optics_min_samples, optics_xi, optics_min_cluster_size, outlierQuantile):
        self.inputFile = inputFile
        self.outputFile = outputFile
        output_file(filename=outputFile, title="Static HTML file")
        self.viewResults = None
        self.numImgsToUse = numImgsToUse
        self.nprocs = nprocs
        self.includeABOD = includeABOD
        self.userGroupings = userGroupings
        self.skipSize = skipSize
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_random_state = umap_random_state
        self.hdbscan_min_samples=hdbscan_min_samples
        self.hdbscan_min_cluster_size=hdbscan_min_cluster_size
        self.optics_min_samples=optics_min_samples
        self.optics_xi = optics_xi
        self.optics_min_cluster_size = optics_min_cluster_size
        self.outlierQuantile = outlierQuantile

    def embeddable_image(self, data):
        img_data = np.uint8(cm.jet(data/max(data.flatten()))*255)
#        image = Image.fromarray(img_data, mode='RGBA').resize((75, 75), Image.Resampling.BICUBIC)
        image = Image.fromarray(img_data, mode='RGBA')
        buffer = BytesIO()
        image.save(buffer, format='png')
        for_encoding = buffer.getvalue()
        return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode('utf-8')

    def random_unique_numbers_from_range(self, start, end, count):
        all_numbers = list(range(start, end + 1))
        random.shuffle(all_numbers)
        return all_numbers[:count]

    def compute_medoid(self, points):
        return points[np.argmin(euclidean_distances(points).sum(axis=0))]

    def genMedoids(self, medoidLabels, clusterPoints):
        dictMe = {}
        for j in set(medoidLabels):
            dictMe[j] = []
        for index, class_name in enumerate(medoidLabels):
            dictMe[class_name].append((index, clusterPoints[index, 0], clusterPoints[index, 1]))
        medoid_lst = []
        for k, v in dictMe.items():
            lst = [(x[1], x[2]) for x in v]
            medoid_point = self.compute_medoid(lst)
            for test_index, test_point in enumerate(lst):
                if math.isclose(test_point[0],medoid_point[0]) and math.isclose(test_point[1], medoid_point[1]):
                    fin_ind = test_index
            medoid_lst.append((k, v[fin_ind][0]))
        return medoid_lst

    def relabel_to_closest_zero(self, labels):
        unique_labels = sorted(set(labels))
        relabel_dict = {label: new_label for new_label, label in enumerate(unique_labels)}
        relabeled = [relabel_dict[label] for label in labels]
        return relabeled

    def regABOD(self, pts):
        abofs = []
        for a in range(len(pts)):
            test_list = [x for x in range(len(pts)) if x != a]
            otherPts = [(d, e) for idx, d in enumerate(test_list) for e in test_list[idx + 1:]]
            outlier_factors = []
            for b, c in otherPts:
                apt = pts[a]
                bpt = pts[b]
                cpt = pts[c]
                ab = bpt - apt
                ac = cpt - apt
                outlier_factors.append(np.dot(ab, ac)/((np.linalg.norm(ab)**2) * (np.linalg.norm(ac))))
            abofs.append(np.var(np.array(outlier_factors)))
        return abofs

    def fastABOD(self, pts, nsamples):
        nbrs = NearestNeighbors(n_neighbors=nsamples, algorithm='ball_tree').fit(pts)
        k_inds = nbrs.kneighbors(pts)[1]
        abofs = []
        count = 0
        for a in range(len(pts)):
            test_list = k_inds[a][1:]
            otherPts = [(d, e) for idx, d in enumerate(test_list) for e in test_list[idx + 1:]]
            outlier_factors = []
            for (b, c) in otherPts:
                apt = pts[a]
                bpt = pts[b]
                cpt = pts[c]
                ab = bpt - apt
                ac = cpt - apt
                if math.isclose(np.linalg.norm(ab), 0.0) or math.isclose(np.linalg.norm(ac), 0.0):
                    count += 1
#                    print("TOO CLOSE")
                    continue
                outlier_factors.append(np.dot(ab, ac)/((np.linalg.norm(ab)**2) * (np.linalg.norm(ac))))
#            print("CURRENT POINT: ", pts[a], test_list, outlier_factors, np.var(np.array(outlier_factors)))
            if(len(outlier_factors)==0):
                abofs.append(np.inf)
            else:
                abofs.append(np.var(np.array(outlier_factors)))
        return abofs

    def getOutliers(self, lst):
#        lstCopy = lst.copy()
#        lstCopy.sort()
#        quart10 = lstCopy[len(lstCopy)//divBy]

        lstQuant = np.quantile(np.array(lst), self.outlierQuantile)
#        print("AIDJWOIJDAOWIDJWAOIDJAWOIDWJA", lstQuant, lst)
        outlierInds = []
        notOutlierInds = []
        for j in range(len(lst)):
            if lst[j]>lstQuant:
                outlierInds.append(j)
            else:
                notOutlierInds.append(j)
#        print("OUTLIER INDS: ", outlierInds)
#        print("NOT OUTLIER INDS: ", notOutlierInds)
        return np.array(outlierInds), np.array(notOutlierInds)

    def genHist(self, vals, endClass):
        totNum = endClass + 1
        countVals = Counter(vals)
        hist = [0]*(totNum)
        for val in set(countVals):
            hist[val] = countVals[val]
        maxval = max(countVals.values())
        return hist, maxval

    def genLeftRight(self, endClass):
        return [*range(endClass+1)], [*range(1, endClass+2)]

    def genUMAP(self):
        import umap
        import hdbscan

        imgs = None
        projections = None
        for currRank in range(self.nprocs):
            with h5py.File(self.inputFile+"_"+str(currRank)+".h5", 'r') as hf:
                if imgs is None:
                    imgs = hf["SmallImages"][:]
                    projections = hf["ProjectedData"][:]
                else:
                    imgs = np.concatenate((imgs, hf["SmallImages"][:]), axis=0)
                    projections = np.concatenate((projections, hf["ProjectedData"][:]), axis=0)

        print("AOIDWJOIAWDJ", len(imgs), len(projections))

        intensities = []
        for img in imgs:
            intensities.append(np.sum(img.flatten()))
        intensities = np.array(intensities)

        self.imgs = imgs[:self.numImgsToUse:self.skipSize]
        self.projections = projections[:self.numImgsToUse:self.skipSize]
        self.intensities = intensities[:self.numImgsToUse:self.skipSize]

        self.numImgsToUse = int(self.numImgsToUse/self.skipSize)

        if len(self.imgs)!= self.numImgsToUse:
            raise TypeError("NUMBER OF IMAGES REQUESTED ({}) EXCEEDS NUMBER OF DATA POINTS PROVIDED ({})".format(len(self.imgs), self.numImgsToUse))

        self.clusterable_embedding = umap.UMAP(
            n_neighbors=self.umap_n_neighbors,
            random_state=self.umap_random_state,
            n_components=2,
#            min_dist=0.25,
            min_dist=0.1,
        ).fit_transform(self.projections)

        self.labels = hdbscan.HDBSCAN(
            min_samples = self.hdbscan_min_samples,
            min_cluster_size = self.hdbscan_min_cluster_size
        ).fit_predict(self.clusterable_embedding)
        exclusionList = np.array([])
        self.clustered = np.isin(self.labels, exclusionList, invert=True)

        self.opticsClust = OPTICS(min_samples=self.optics_min_samples, xi=self.optics_xi, min_cluster_size=self.optics_min_cluster_size)
        self.opticsClust.fit(self.clusterable_embedding)
        self.opticsLabels = cluster_optics_dbscan(
            reachability=self.opticsClust.reachability_,
            core_distances=self.opticsClust.core_distances_,
            ordering=self.opticsClust.ordering_,
            eps=2.5,
        )
#        self.opticsLabels = self.opticsClust.labels_

        self.experData_df = pd.DataFrame({'x':self.clusterable_embedding[self.clustered, 0],'y':self.clusterable_embedding[self.clustered, 1]})
        self.experData_df['image'] = list(map(self.embeddable_image, self.imgs[self.clustered]))
        self.experData_df['imgind'] = np.arange(self.numImgsToUse)*self.skipSize

    def genABOD(self):
        if self.includeABOD:
            abod = self.fastABOD(self.projections, 10)
            outliers, notOutliers = self.getOutliers(abod)
        else:
            outliers = []
            notOutliers = []
        outlierLabels = []
        for j in range(self.numImgsToUse):
            if j in outliers:
                outlierLabels.append(str(6))
            else:
                outlierLabels.append(str(0))
        self.experData_df['anomDet'] = outlierLabels
        self.experData_df['anom_backgroundColor'] = [Category20[20][int(x)] for x in outlierLabels]

    def setUserGroupings(self, userGroupings):
        """
        Set User Grouping. An adjustment is made at the beginning of this function,
        whereby 1 is added to each label. This is because internally, the clusters are stored
        starting at -1 rather than 0.
        """
        self.userGroupings = [[x-1 for x in grouping] for grouping in userGroupings]

    def genLabels(self):
        newLabels = []
        for j in self.labels[self.clustered]:
            doneChecking = False
            for grouping in self.userGroupings:
                if j in grouping and not doneChecking:
                    newLabels.append(min(grouping))
                    doneChecking=True
            if not doneChecking:
                newLabels.append(j)
        newLabels = list(np.array(newLabels) + 1)
        self.newLabels = np.array(self.relabel_to_closest_zero(newLabels))
        self.experData_df['cluster'] = [str(x) for x in self.newLabels[self.clustered]]
        self.experData_df['ptColor'] = [x for x in self.experData_df['cluster']]
        self.experData_df['dbscan_backgroundColor'] = [Category20[20][x] for x in self.newLabels]
        self.experData_df['backgroundColor'] = [Category20[20][x] for x in self.newLabels]
        medoid_lst = self.genMedoids(self.newLabels, self.clusterable_embedding)
        self.medoidInds = [x[1] for x in medoid_lst]
        medoidBold = []
        for ind in range(self.numImgsToUse):
            if ind in self.medoidInds:
                medoidBold.append(12)
            else:
                medoidBold.append(4)
        self.experData_df['medoidBold'] = medoidBold

        opticsNewLabels = []
        for j in self.opticsLabels[self.clustered]:
            doneChecking = False
            for grouping in self.userGroupings:
                if j in grouping and not doneChecking:
                    opticsNewLabels.append(min(grouping))
                    doneChecking=True
            if not doneChecking:
                opticsNewLabels.append(j)
        opticsNewLabels = list(np.array(opticsNewLabels) + 1)
        self.opticsNewLabels = np.array(self.relabel_to_closest_zero(opticsNewLabels))
#        self.experData_df['optics_backgroundColor'] = [Category20[20][x] for x in self.opticsNewLabels[self.opticsClust.ordering_]]
        self.experData_df['optics_backgroundColor'] = [Category20[20][x] for x in self.opticsNewLabels]

    def genHTML(self):
        datasource = ColumnDataSource(self.experData_df)
        color_mapping = CategoricalColorMapper(factors=[str(x) for x in list(set(self.newLabels))],palette=Category20[20])
        plot_figure = figure(
            title='UMAP projection with DBSCAN clustering of the LCLS dataset',
            tools=('pan, wheel_zoom, reset'),
            width = 2000, height = 600
        )
        plot_figure.add_tools(HoverTool(tooltips="""
        <div style="width: 170; height: 64; background-color:@backgroundColor; margin: 5px 0px 5px 0px">
            <div style='width: 64; height: 64; float: left;'>
                <img src='@image'; float: left;'/>
            </div>
            <div style="height: 64;">
                <div style='margin-left: 75; margin-top: 10'>
                    <span style='font-size: 15px; color: #224499'>Cluster </span>
                    <span style='font-size: 15px'>@cluster</span>
                </div>
                <div style='margin-left: 75; margin-top: 10'>
                    <span style='font-size: 15px; color: #224499'>Image </span>
                    <span style='font-size: 15px'>@imgind</span>
                </div>
            </div>
        </div>
        """))
        plot_figure.circle(
            'x',
            'y',
            source=datasource,
            color=dict(field='ptColor', transform=color_mapping),
            line_alpha=0.6,
            fill_alpha=0.6,
            size='medoidBold',
            legend_field='cluster'
        )
        plot_figure.sizing_mode = 'scale_both'
        plot_figure.legend.location = "bottom_right"
        plot_figure.legend.title = "Clusters"

        vals = [x for x in self.newLabels]
        trueSource = ColumnDataSource(data=dict(vals = vals))
        hist, maxCount = self.genHist(vals, max(vals))
        left, right = self.genLeftRight(max(vals))
        histsource = ColumnDataSource(data=dict(hist=hist, left=left, right=right))
        p = figure(width=2000, height=450, toolbar_location=None,
                   title="Histogram Testing")
        p.quad(source=histsource, top='hist', bottom=0, left='left', right='right',
                 fill_color='skyblue', line_color="white")
        p.y_range = Range1d(0, maxCount)
        p.x_range = Range1d(0, max(vals)+1)
        p.xaxis.axis_label = "Cluster Label"
        p.yaxis.axis_label = "Count"

        indexCDS = ColumnDataSource(dict(
            index=[*range(0, self.numImgsToUse, 2)]
            )
        )
        cols = RangeSlider(title="ET",
                start=0,
                end=self.numImgsToUse,
                value=(0, self.numImgsToUse-1),
                step=1, sizing_mode="stretch_width")
        callback = CustomJS(args=dict(cols=cols, trueSource = trueSource,
                                      histsource = histsource, datasource=datasource, indexCDS=indexCDS), code="""
        function countNumbersAtIndices(numbers, startInd, endInd, smallestVal, largestVal) {
            let counts = new Array(largestVal-smallestVal); for (let i=0; i<largestVal-smallestVal; ++i) counts[i] = 0;
            for (let i = Math.round(startInd); i <= Math.round(endInd); i++) {
                let numMe = numbers[i];
                if (typeof counts[numMe] === 'undefined') {
                  counts[numMe] = 1;
                } else {
                  counts[numMe]++;
                }
            }
            return counts;
            }
        const vals = trueSource.data.vals
        const leftVal = cols.value[0]
        const rightVal = cols.value[1]
        const oldhist = histsource.data.hist
        const left = histsource.data.left
        const right = histsource.data.right
        const hist = countNumbersAtIndices(vals, leftVal, rightVal, left[0], right.slice(-1))
        histsource.data = { hist, left, right }
        let medoidBold = new Array(datasource.data.medoidBold.length); for (let i=0; i<datasource.data.medoidBold.length; ++i) medoidBold[i] = 0;
                for (let i = Math.round(leftVal); i < Math.round(rightVal); i++) {
            medoidBold[i] = 5
        }
        const x = datasource.data.x
        const y = datasource.data.y
        const image = datasource.data.image
        const cluster = datasource.data.cluster
        const ptColor = datasource.data.ptColor
        const anomDet = datasource.data.anomDet
        const imgind = datasource.data.imgind
        const backgroundColor = datasource.data.backgroundColor
        const dbscan_backgroundColor = datasource.data.dbscan_backgroundColor
        const anom_backgroundColor = datasource.data.anom_backgroundColor
        const optics_backgroundColor = datasource.data.optics_backgroundColor
        datasource.data = { x, y, image, cluster, medoidBold, ptColor, anomDet, imgind, backgroundColor, dbscan_backgroundColor, anom_backgroundColor, optics_backgroundColor}
        """)
        cols.js_on_change('value', callback)


        imgsPlot = figure(width=2000, height=150, toolbar_location=None)
        imgsPlot.image(image=[self.imgs[imgindMe][::-1] for imgindMe in self.medoidInds],
                x=[0.25+xind for xind in range(len(self.medoidInds))],
                y=0,
                dw=0.5, dh=1,
                palette="Turbo256", level="image")
        imgsPlot.axis.visible = False
        imgsPlot.grid.visible = False
        for xind in range(len(self.medoidInds)):
            mytext = Label(x=0.375+xind, y=-0.25, text='Cluster {}'.format(xind))
            imgsPlot.add_layout(mytext)
        imgsPlot.y_range = Range1d(-0.3, 1.1)
        imgsPlot.x_range = Range1d(0, max(vals)+1)

        toggl = Toggle(label='► Play',active=False)
        toggl_js = CustomJS(args=dict(slider=cols,indexCDS=indexCDS),code="""
        // https://discourse.bokeh.org/t/possible-to-use-customjs-callback-from-a-button-to-animate-a-slider/3985/3
            var check_and_iterate = function(index){
                var slider_val0 = slider.value[0];
                var slider_val1 = slider.value[1];
                var toggle_val = cb_obj.active;
                if(toggle_val == false) {
                    cb_obj.label = '► Play';
                    clearInterval(looop);
                    }
                else if(slider_val1 >= index[index.length - 1]) {
//                    cb_obj.label = '► Play';
                    slider.value = [0, slider_val1-slider_val0];
//                   cb_obj.active = false;
//                    clearInterval(looop);
                    }
                else if(slider_val1 !== index[index.length - 1]){
                    slider.value = [index.filter((item) => item > slider_val0)[0], index.filter((item) => item > slider_val1)[0]];
                    }
                else {
                clearInterval(looop);
                    }
            }
            if(cb_obj.active == false){
                cb_obj.label = '► Play';
                clearInterval(looop);
            }
            else {
                cb_obj.label = '❚❚ Pause';
                var looop = setInterval(check_and_iterate, 0.1, indexCDS.data['index']);
            };
        """)
        toggl.js_on_change('active',toggl_js)

        reachabilityDiag = figure(
            title='OPTICS Reachability Diag',
            tools=('pan, wheel_zoom, reset'),
            width = 2000, height = 400
        )
        space = np.arange(self.numImgsToUse)
        reachability = self.opticsClust.reachability_[self.opticsClust.ordering_]
#        reachability = self.opticsClust.reachability_
        opticsData_df = pd.DataFrame({'x':space,'y':reachability})
        opticsData_df['clusterForScatterPlot'] = [str(x) for x in self.opticsNewLabels]
        opticsData_df['cluster'] = [str(x) for x in self.opticsNewLabels[self.opticsClust.ordering_]]
        opticsData_df['ptColor'] = [x for x in opticsData_df['cluster']]
        color_mapping2 = CategoricalColorMapper(factors=[str(x) for x in list(set(self.opticsNewLabels))],
                                               palette=Category20[20])
        opticssource = ColumnDataSource(opticsData_df)
        reachabilityDiag.circle(
            'x',
            'y',
            source=opticssource,
            color=dict(field='ptColor', transform=color_mapping2),
            line_alpha=0.6,
            fill_alpha=0.6,
            legend_field='cluster'
        )
        reachabilityDiag.line([0, len(opticsData_df['ptColor'])], [2, 2], line_width=2, color="black", line_dash="dashed")
        reachabilityDiag.y_range = Range1d(-1, 10)

        LABELS = ["DBSCAN Clustering", "OPTICS Clustering", "Anomaly Detection"]
        radio_button_group = RadioButtonGroup(labels=LABELS, active=0)
        radioGroup_js = CustomJS(args=dict(datasource=datasource, opticssource=opticssource), code="""
            const x = datasource.data.x
            const y = datasource.data.y
            const image = datasource.data.image
            const medoidBold = datasource.data.medoidBold
            const cluster = datasource.data.cluster
            const anomDet = datasource.data.anomDet
            const imgind = datasource.data.imgind
            const dbscan_backgroundColor = datasource.data.dbscan_backgroundColor
            const anom_backgroundColor = datasource.data.anom_backgroundColor
            const optics_backgroundColor = datasource.data.optics_backgroundColor

            const opticsClust = opticssource.data.clusterForScatterPlot

            let ptColor = null
            let backgroundColor = null

            if (cb_obj.active==0){
                ptColor = cluster
                backgroundColor = dbscan_backgroundColor
            }
            else if (cb_obj.active==1){
                ptColor = opticsClust
                backgroundColor = optics_backgroundColor
            }
            else{
                ptColor = anomDet
                backgroundColor = anom_backgroundColor
            }
            datasource.data = { x, y, image, cluster, medoidBold, ptColor, anomDet, imgind, backgroundColor, dbscan_backgroundColor, anom_backgroundColor, optics_backgroundColor}
        """)
        radio_button_group.js_on_change("active", radioGroup_js)

        self.viewResults = column(plot_figure, p, imgsPlot, row(cols, toggl, radio_button_group), reachabilityDiag)

    def fullVisualize(self):
        self.genUMAP()
        self.genABOD()
        self.genLabels()
        self.genHTML()

    def updateLabels(self):
        self.genLabels()
        self.genHTML()

    def userSave(self):
        save(self.viewResults)

    def userShow(self):
        from IPython.display import display, HTML
        display(HTML("<style>.container { width:100% !important; }</style>"))
        display(HTML("<style>.output_result { max-width:100% !important; }</style>"))
        display(HTML("<style>.container { height:100% !important; }</style>"))
        display(HTML("<style>.output_result { max-height:100% !important; }</style>"))
        from bokeh.io import output_notebook
        output_notebook()
        show(self.viewResults)

class WrapperFullFD:
    """
    Frequent Directions Data Processing Wrapper Class.
    """
    from btx.interfaces.ipsana import PsanaInterface
    def __init__(self, exp, run, det_type, start_offset, num_imgs, writeToHere, grabImgSteps, num_components, alpha, rankAdapt, rankAdaptMinError, downsample, bin_factor, threshold, eluThreshold, eluAlpha, normalizeIntensity, noZeroIntensity, minIntensity, samplingFactor, divBy, thresholdQuantile, usePSI=True):
        self.start_offset = start_offset
        self.num_imgs = num_imgs
        self.exp = exp
        self.run = run
        self.det_type = det_type
        self.writeToHere = writeToHere
        self.num_components=num_components
        self.alpha = alpha
        self.rankAdapt = rankAdapt
        self.rankAdaptMinError = rankAdaptMinError
        self.downsample=downsample
        self.bin_factor= bin_factor
        self.threshold= threshold
        self.eluThreshold = eluThreshold
        self.eluAlpha = eluAlpha
        self.normalizeIntensity=normalizeIntensity
        self.noZeroIntensity=noZeroIntensity
        self.minIntensity = minIntensity
        self.samplingFactor=samplingFactor
        self.divBy = divBy 
        self.thresholdQuantile = thresholdQuantile

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.imgsTracked = []
        self.grabImgSteps = grabImgSteps

        self.usePSI = usePSI
        if usePSI:
            self.psi = self.PsanaInterface(exp=exp, run=run, det_type=det_type)
            self.psi.counter = self.start_offset + self.num_imgs*self.rank//self.size
        else:
            self.psi = None

        if self.rank==0:
            self.currRun = run #datetime.now().strftime("%y%m%d%H%M%S")
        else:
            self.currRun = None
        self.currRun = self.comm.bcast(self.currRun, root=0)

        self.imageProcessor = FD_ImageProcessing(threshold = self.threshold, eluThreshold = self.eluThreshold, eluAlpha = self.eluAlpha, noZeroIntensity = self.noZeroIntensity, normalizeIntensity=self.normalizeIntensity, minIntensity=self.minIntensity, thresholdQuantile=self.thresholdQuantile)
        self.imgRetriever = SinglePanelDataRetriever(exp=exp, det_type=det_type, run=run, downsample=downsample, bin_factor=bin_factor, imageProcessor = self.imageProcessor, thumbnailHeight = 150, thumbnailWidth = 150)

#    def lowMemoryReconstructionErrorScaled(self, matrixCentered, matSketch):
#        """ 
#        Compute the low memory reconstruction error of the matrix sketch
#        against given data. This is the same as reconstructionError,
#        but estimates the norm computation and does not scale by the 
#        minimum projection matrix, but rather by the matrix norm itself. 
#
#        Parameters
#        ----------
#        matrixCentered: ndarray
#           Data to compare matrix sketch to 
#
#        Returns
#        -------
#        float,
#            Data subtracted by data projected onto sketched space, scaled by matrix elements
#       """
##        k = 10
#        matrixCenteredT = matrixCentered.T
#        matSketchT = matSketch.T
#        U, S, Vt = np.linalg.svd(matSketchT, full_matrices=False)
##        G = U[:,:k]
#        G = U
#        return (self.estimFrobNormSquared(matrixCenteredT, [G,G.T,matrixCenteredT], 50)/
#                np.linalg.norm(matrixCenteredT, 'fro')**2)
#
#    def estimFrobNormSquared(self, addMe, arrs, its):
#        """ 
#        Estimate the Frobenius Norm of product of arrs matrices 
#        plus addME matrix using its iterations. 
#
#        Parameters
#        ----------
#        arrs: list of ndarray
#           Matrices to multiply together
#
#        addMe: ndarray
#            Matrix to add to others
##
#        its: int
#            Number of iterations to average over
#
#        Returns
#        -------
#        sumMe/its*no_rows : float
#            Estimate of frobenius norm of product
#            of arrs matrices plus addMe matrix
#
#        Notes
#        -----
#        Frobenius estimation is the expected value of matrix
#        multiplied by random vector from multivariate normal distribution
#        based on [1]. 
#
#        [1] Norm and Trace Estimation with Random Rank-one Vectors 
#        Zvonimir Bujanovic and Daniel Kressner SIAM Journal on Matrix 
#        Analysis and Applications 2021 42:1, 202-223
#       """
#        no_rows = arrs[-1].shape[1]
#        v = np.random.normal(size=no_rows)
#        v_hat = v / np.linalg.norm(v)
#        sumMe = 0
#        for j in range(its):
#            v = np.random.normal(size=no_rows)
#            v_hat = v / np.linalg.norm(v)
#            v_addMe = addMe @ v_hat
#            for arr in arrs[::-1]:
#                v_hat = arr @ v_hat
#            sumMe = sumMe + (np.linalg.norm(v_addMe - v_hat))**2
#        return sumMe/its*no_rows

    def lowMemoryReconstructionErrorScaled(self, matrixCentered, matSketch):
        matrixCenteredT = matrixCentered.T
        matSketchT = matSketch.T
        U, S, Vt = np.linalg.svd(matSketchT, full_matrices=False)
        G = U
        return self.estimFrobNormJ(matrixCenteredT, [G,G.T,matrixCenteredT], 20)/np.linalg.norm(matrixCenteredT, 'fro')

    def estimFrobNormJ(self, addMe, arrs, k):
        m, n = addMe.shape
        randMat = np.random.normal(0, 1, size=(n, k))
        minusMe = addMe @ randMat
        sumMe = 0
        for arr in arrs[::-1]:
            randMat = arr @ randMat
        sumMe += math.sqrt(1/k) * np.linalg.norm(randMat - minusMe, 'fro')
        return sumMe

    def retrieveImages(self):
        startingPoint = self.start_offset + self.num_imgs*self.rank//self.size
        self.fullImgData, self.imgsTracked = self.imgRetriever.get_formatted_images(startInd=startingPoint, n=self.num_imgs//self.size, num_steps=self.grabImgSteps, getThumbnails=False)

    def genSynthData(self):
        self.fullImgData = np.random.rand(70000, 100000//self.size)
        self.imgsTracked = [(0, self.rank)]

#    def genDecayingSVD(self):
#        numFeats = 70000
#        numSamps = 100000//self.size
#        A = np.random.rand(matrixSize, matrixSize)
##        A = A.T @ A
#        eigVals, eigVecs = np.linalg.eig(A)
#        diag_entries = list(np.random.rand(matrixSize))
##        diag_entries.sort()
#        multMe = np.ones(numSamps)
##        diag_entries = np.array(diag_entries[::-1])
#        D = np.diag(diag_entries) + np.eye(matrixSize)
#        return (eigVecs @ (D) @ eigVecs.T)

    def compDecayingSVD(self, seedMe, a, b):
        numFeats = a
        numSamps = b//self.size
        perturbation = np.random.rand(numSamps, numFeats)*0.1
        np.random.seed(seedMe)
        A1 = np.random.rand(numSamps, numFeats) 
        Q1, R1 = np.linalg.qr(A1)
        Q1 = Q1 + perturbation
        A2 = np.random.rand(numFeats, numFeats) #Modify
        Q2, R2 = np.linalg.qr(A2)
        S = list(np.random.rand(numFeats)) #Modify
        S.sort()
        S = S[::-1]
        for j in range(len(S)): #Modify
            S[j] = (2**(-16*(j+1)/len(S)))*S[j]
        self.fullImgData = (Q1 @ np.diag(S) @ Q2).T
        self.imgsTracked = [(0, numSamps)]

    def runMe(self):

        stfull = time.perf_counter()

        #DATA RETRIEVAL STEP
        ##########################################################################################
#        if self.usePSI:
#            self.retrieveImages()
#        else:
#            self.compDecayingSVD()
##            self.genSynthData()
#        et = time.perf_counter()
#        print("Estimated time for data retrieval for rank {0}/{1}: {2}".format(self.rank, self.size, et - stfull))

        #SKETCHING STEP
        ##########################################################################################
        freqDir = FreqDir(comm= self.comm, rank=self.rank, size = self.size, start_offset=self.start_offset, num_imgs=self.num_imgs, exp=self.exp, run=self.run,
                det_type=self.det_type, output_dir=self.writeToHere, num_components=self.num_components, alpha=self.alpha, rankAdapt=self.rankAdapt, rankAdaptMinError = self.rankAdaptMinError,
                merger=False, mergerFeatures=0, downsample=self.downsample, bin_factor=self.bin_factor,
                currRun = self.currRun, samplingFactor=self.samplingFactor, imgData = self.fullImgData, imgsTracked = self.imgsTracked, psi=self.psi, usePSI=self.usePSI)
        print("{} STARTING SKETCHING FOR {}".format(self.rank, self.currRun))
        st1 = time.perf_counter()
        freqDir.run()
        localSketchFilename = freqDir.write()
        et1 = time.perf_counter()
        print("Estimated time for frequent directions rank {0}/{1}: {2}".format(self.rank, self.size, et1 - st1))

        #MERGING STEP
        ##########################################################################################
        if freqDir.rank<10:
            fullSketchFilename = localSketchFilename[:-4]
        else:
            fullSketchFilename = localSketchFilename[:-5]
        allNames = []
        for j in range(freqDir.size):
            allNames.append(fullSketchFilename + str(j) + ".h5")
        mergeTree = MergeTree(comm=self.comm, rank=self.rank, size=self.size, exp=self.exp, run=self.run, det_type=self.det_type, divBy=self.divBy, readFile = localSketchFilename,
                output_dir=self.writeToHere, allWriteDirecs=allNames, currRun = self.currRun, psi=self.psi, usePSI=self.usePSI)
        st2 = time.perf_counter()
        mergeTree.merge()
        mergedSketchFilename = mergeTree.write()
        et2 = time.perf_counter()
        print("Estimated time merge tree for rank {0}/{1}: {2}".format(self.rank, self.size, et2 - st2))

        #PROJECTION STEP
        ##########################################################################################
        appComp = ApplyCompression(comm=self.comm, rank = self.rank, size=self.size, start_offset=self.start_offset, num_imgs=self.num_imgs, exp=self.exp, run=self.run,det_type=self.det_type, readFile = mergedSketchFilename, output_dir = self.writeToHere, currRun = self.currRun, imgData = self.fullImgData)
        st3 = time.perf_counter()
        self.matSketch = appComp.run()
        appComp.write()
        et3 = time.perf_counter()
        print("Estimated time projection for rank {0}/{1}: {2}".format(self.rank, self.size, et3 - st3))
        print("Estimated full processing time for rank {0}/{1}: {2}, {3}".format(self.rank, self.size, (et1 + et2 + et3 - st1 - st2 - st3), et3 - stfull))
        self.addThumbnailsToProjectH5()
        return (et1 + et2 + et3 - st1 - st2 - st3)

#        self.comm.barrier()
#        self.comm.Barrier()
#        filenameTest3 = random.randint(0, 10)
#        filenameTest3 = self.comm.allgather(filenameTest3)
#        print("TEST 3: ", self.rank, filenameTest3)

    def addThumbnailsToProjectH5(self):
#        print("Gathering thumbnails")
        startingPoint = self.start_offset + self.num_imgs*self.rank//self.size
        _,self.fullThumbnailData,_ = self.imgRetriever.get_formatted_images(startInd=startingPoint, n=self.num_imgs//self.size, num_steps=self.grabImgSteps, getThumbnails=True)
        file_name = self.writeToHere+"{}_ProjectedData_{}.h5".format(self.currRun, self.rank)
        f1 = h5py.File(file_name, 'r+')
        f1.create_dataset("SmallImages",  data=self.fullThumbnailData)
        f1.close()
        self.comm.barrier()

class FD_ImageProcessing:
    def __init__(self, threshold, eluThreshold, eluAlpha, noZeroIntensity, normalizeIntensity, minIntensity, thresholdQuantile):
        self.threshold = threshold
        self.eluThreshold = eluThreshold
        self.eluAlpha = eluAlpha
        self.noZeroIntensity = noZeroIntensity
        self.normalizeIntensity = normalizeIntensity
        self.minIntensity = minIntensity
        self.thresholdQuantile = thresholdQuantile

    def processImg(self, nimg, currIntensity):
        if self.threshold:
            nimg = self.thresholdFunc(nimg)
        if self.eluThreshold:
            nimg = self.eluThresholdFunc(nimg)
        if self.noZeroIntensity:
            nimg = self.removeZeroIntensityFunc(nimg, currIntensity)
        if self.normalizeIntensity:
            nimg = self.normalizeIntensityFunc(nimg, currIntensity)
        return nimg

    def elu(self,x):
        if x > 0:
            return x
        else:
            return self.eluAlpha*(math.exp(x)-1)

    def eluThresholdFunc(self, img):
        if img is None:
            return img
        else:
            elu_v = np.vectorize(self.elu)
            secondQuartile = np.quantile(img, self.thresholdQuantile)
            return(elu_v(img-secondQuartile)+secondQuartile)

    def thresholdFunc(self, img):
        if img is None:
            return img
        else:
            secondQuartile = np.quantile(img, self.thresholdQuantile)
            return (img>secondQuartile)*img

    def removeZeroIntensityFunc(self, img, currIntensity):
        if currIntensity<self.minIntensity:
            return None
        else:
            return img

    def normalizeIntensityFunc(self, img, currIntensity):
        if img is None:
            return img
        elif currIntensity<self.minIntensity:
            return np.zeros(img.shape)+1
        else:
            return img/np.sum(img.flatten(), dtype=np.double)


class DataRetriever:
    from btx.interfaces.ipsana import (
        PsanaInterface,
        bin_data,
        retrieve_pixel_index_map,
        assemble_image_stack_batch,
    )
    def __init__(self, exp, det_type, run, downsample, bin_factor, imageProcessor, thumbnailHeight, thumbnailWidth):
        self.exp = exp
        self.det_type = det_type
        self.run = run
        self.downsample = downsample
        self.bin_factor = bin_factor
        self.thumbnailHeight = thumbnailHeight
        self.thumbnailWidth = thumbnailWidth

        self.psi = self.PsanaInterface(exp=exp, run=run, det_type=det_type, no_cmod=True)

        self.imageProcessor = imageProcessor

    def assembleImgsToSave(self, imgs):
        """
        Form the images from psana pixel index map and downsample images. 

        Parameters
        ----------
        imgs: ndarray
            images to downsample
        """
        pixel_index_map = self.retrieve_pixel_index_map(self.psi.det.geometry(self.psi.run))

        saveMe = []
        for img in imgs:
            imgRe = np.reshape(img, self.psi.det.shape())
            imgRe = self.assemble_image_stack_batch(imgRe, pixel_index_map)
            saveMe.append(np.array(Image.fromarray(imgRe).resize((self.thumbnailHeight, self.thumbnailWidth))))
        return np.array(saveMe)

    def split_range(self, start, end, num_tuples):
        if start==end:
            raise ValueError('Range processing error: start value equals end value, which leads to no images processed.')
            return
        total_elements = end - start
        batch_size = total_elements // num_tuples
        tuples = []
        for i in range(num_tuples - 1):
            batch_start = start + i * batch_size
            batch_end = batch_start + batch_size
            tuples.append((batch_start, batch_end))
        last_batch_start = start + (num_tuples - 1) * batch_size
        last_batch_end = end
        tuples.append((last_batch_start, last_batch_end))
        return tuples    

    def get_formatted_images(self, startInd, n, num_steps, getThumbnails):
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
        fullimgs = None
        fullthumbnails = None
        imgsTracked = []
        runs = self.split_range(startInd, startInd+n, num_steps)
        print(runs) 
        for runStart, runEnd in runs:
#            print("RETRIEVING: [", runStart, ":", runEnd,"]")
            self.psi.counter = runStart
            imgsTracked.append((runStart, runEnd))

#            print("getting images")
            imgs = self.psi.get_images(runEnd-runStart, assemble=False)

#            print("Removing nan images")
            imgs = imgs[
                [i for i in range(imgs.shape[0]) if not np.isnan(imgs[i : i + 1]).any()]
            ]

            if getThumbnails:
#                print("Assembling thumbnails")
                thumbnails = self.assembleImgsToSave(imgs)

            if self.downsample:
#                print("Downsampling images")
                imgs = self.bin_data(imgs, self.bin_factor)
#            print("Flattening images")
            num_valid_imgs, p, x, y = imgs.shape
            img_batch = np.reshape(imgs, (num_valid_imgs, p * x * y)).T
#            print("Image values less than 0 setting to 0")
            img_batch[img_batch<0] = 0
    
            if getThumbnails:
#                print("FLattening thumbnails")
                num_valid_thumbnails, tx, ty = thumbnails.shape
                thumbnail_batch = np.reshape(thumbnails, (num_valid_thumbnails, tx*ty)).T

            if getThumbnails:
                nimg_batch = []
                nthumbnail_batch = []
                for img, thumbnail in zip(img_batch.T, thumbnail_batch.T):
                    currIntensity = np.sum(img.flatten(), dtype=np.double)
                    nimg = self.imageProcessor.processImg(img, currIntensity)
                    nthumbnail = self.imageProcessor.processImg(thumbnail, currIntensity)
                    if nimg is not None:
                        nimg_batch.append(nimg)
                        nthumbnail_batch.append(nthumbnail)
                nimg_batch = np.array(nimg_batch).T
                nthumbnail_batch = np.array(nthumbnail_batch).reshape(num_valid_thumbnails, tx, ty)
                if fullimgs is None:
                    fullimgs = nimg_batch
                    fullthumbnails = nthumbnail_batch
                else:
                    fullimgs = np.hstack((fullimgs, nimg_batch))
                    fullthumbnails = np.vstack((fullthumbnails, nthumbnail_batch))
            else:
                nimg_batch = []
                for img in img_batch.T:
                    currIntensity = np.sum(img.flatten(), dtype=np.double)
#                    print("Starting image processing of size {}".format(img_batch.T.shape))
                    nimg = self.imageProcessor.processImg(img, currIntensity)
                    if nimg is not None:
                        nimg_batch.append(nimg)
                nimg_batch = np.array(nimg_batch).T
#                print("hstacking")
                if fullimgs is None:
                    fullimgs = nimg_batch
                else:
                    fullimgs = np.hstack((fullimgs, nimg_batch))

#        print("Images tracked:", imgsTracked)
        if getThumbnails:
            return (fullimgs, fullthumbnails, imgsTracked)
        else:
            return (fullimgs, imgsTracked)


class SinglePanelDataRetriever:
    from btx.interfaces.ipsana import PsanaInterface
    def __init__(self, exp, det_type, run, downsample, bin_factor, imageProcessor, thumbnailHeight, thumbnailWidth):
        self.exp = exp
        self.det_type = det_type
        self.run = run
        self.thumbnailHeight = thumbnailHeight
        self.thumbnailWidth = thumbnailWidth

        self.psi = self.PsanaInterface(exp=exp, run=run, det_type=det_type)

        self.imageProcessor = imageProcessor

    def split_range(self, start, end, num_tuples):
        if start==end:
            raise ValueError('Range processing error: start value equals end value, which leads to no images processed.')
            return
        total_elements = end - start
        batch_size = total_elements // num_tuples
        tuples = []
        for i in range(num_tuples - 1):
            batch_start = start + i * batch_size
            batch_end = batch_start + batch_size
            tuples.append((batch_start, batch_end))
        last_batch_start = start + (num_tuples - 1) * batch_size
        last_batch_end = end
        tuples.append((last_batch_start, last_batch_end))
        return tuples    

    def get_formatted_images(self, startInd, n, num_steps, getThumbnails):
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
        fullimgs = None
        fullthumbnails = None
        imgsTracked = []
        runs = self.split_range(startInd, startInd+n, num_steps)
        print(runs) 
        for runStart, runEnd in runs:
#            print("RETRIEVING: [", runStart, ":", runEnd,"]")
            self.psi.counter = runStart
            imgsTracked.append((runStart, runEnd))

#            print("getting images")
            imgs = self.psi.get_images(runEnd-runStart, assemble=False)

#            print("Removing nan images")
            imgs = imgs[
                [i for i in range(imgs.shape[0]) if not np.isnan(imgs[i : i + 1]).any()]
            ]

            if getThumbnails:
                saveMe = []
                for img in imgs:
                    saveMe.append(np.array(Image.fromarray(img).resize((self.thumbnailHeight, self.thumbnailWidth))))
                thumbnails = np.array(saveMe)
                print("thumbaaowdijaoiajw", len(imgs), len(thumbnails))

            num_valid_imgs, x, y = imgs.shape
            img_batch = np.reshape(imgs, (num_valid_imgs, x * y)).T
#            print("Image values less than 0 setting to 0")
            img_batch[img_batch<0] = 0
    
            if getThumbnails:
#                print("FLattening thumbnails")
                num_valid_thumbnails, tx, ty = thumbnails.shape
                thumbnail_batch = np.reshape(thumbnails, (num_valid_thumbnails, tx*ty)).T

            if getThumbnails:
                nimg_batch = []
                nthumbnail_batch = []
                for img, thumbnail in zip(img_batch.T, thumbnail_batch.T):
                    currIntensity = np.sum(img.flatten(), dtype=np.double)
                    nimg = self.imageProcessor.processImg(img, currIntensity)
                    nthumbnail = self.imageProcessor.processImg(thumbnail, currIntensity)
                    if nimg is not None:
                        nimg_batch.append(nimg)
                        nthumbnail_batch.append(nthumbnail)
                nimg_batch = np.array(nimg_batch).T
                nthumbnail_batch = np.array(nthumbnail_batch).reshape(num_valid_thumbnails, tx, ty)
                if fullimgs is None:
                    fullimgs = nimg_batch
                    fullthumbnails = nthumbnail_batch
                else:
                    fullimgs = np.hstack((fullimgs, nimg_batch))
                    fullthumbnails = np.vstack((fullthumbnails, nthumbnail_batch))
            else:
                nimg_batch = []
                for img in img_batch.T:
                    currIntensity = np.sum(img.flatten(), dtype=np.double)
#                    print("Starting image processing of size {}".format(img_batch.T.shape))
                    nimg = self.imageProcessor.processImg(img, currIntensity)
                    if nimg is not None:
                        nimg_batch.append(nimg)
                nimg_batch = np.array(nimg_batch).T
#                print("hstacking")
                if fullimgs is None:

                    fullimgs = nimg_batch
                else:
                    fullimgs = np.hstack((fullimgs, nimg_batch))

#        print("Images tracked:", imgsTracked)
        if getThumbnails:
            print(fullimgs.shape, fullthumbnails.shape, imgsTracked)
            return (fullimgs, fullthumbnails, imgsTracked)
        else:
            return (fullimgs, imgsTracked)
