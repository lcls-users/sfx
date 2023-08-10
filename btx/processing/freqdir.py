import os, csv, argparse

import numpy as np
from numpy import zeros, sqrt, dot, diag
from numpy.linalg import svd, LinAlgError
from scipy.linalg import svd as scipy_svd

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

import time

import h5py
from PIL import Image
import random
import heapq

class FreqDir:

    """
    Parallel Frequent Directions.
    
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
       total_imgs: total number of images to process
       ell: number of components of matrix sketch
       alpha: proportion  of components to not rotate in frequent directions algorithm
       exp, run, det_type: experiment properties
       rankAdapt: indicates whether to perform rank adaptive FD
       increaseEll: internal variable indicating whether ell should be increased for rank adaption
       dir: directory to write output
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
        start_offset,
        total_imgs,
        exp,
        run,
        det_type,
        dir,
        currRun,
        ell=0, 
        alpha=0,
        rankAdapt=False,
        merger=False,
        mergerFeatures=0,
        downsample=False,
        bin_factor=2,
        threshold=False,
        normalizeIntensity=False,
        noZeroIntensity=False, 
        samplingFactor=1.0
    ):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.currRun = currRun

        if not self.merger:
            self.psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
            self.psi.counter = start_offset + total_imgs*self.rank//self.size
            self.downsample = downsample
            self.bin_factor = bin_factor
            (
                self.num_images,
                self.num_features,
            ) = self.set_params(total_imgs, bin_factor)
        else:
            self.num_features = mergerFeatures
        self.task_durations = dict({})
        self.num_incorporated_images = 0

        self.dir = dir
        self.d = self.num_features
        self.ell = ell
        self.m = 2*self.ell
        self.sketch = zeros( (self.m, self.d) ) 
        self.nextZeroRow = 0
        self.alpha = alpha
        self.mean = None
        self.imgsTracked = []

        self.rankAdapt = rankAdapt
        self.increaseEll = False
        self.threshold = threshold
        self.noZeroIntensity = noZeroIntensity
        self.normalizeIntensity=normalizeIntensity

        self.samplingFactor = samplingFactor

    def set_params(self, num_images, bin_factor):
        """
        Method to initialize FreqDir parameters.

        Parameters
        ----------
        num_images : int
            Desired number of images to incorporate into model.
        bin_factor : int
            Factor to bin data by.

        Returns
        -------
        num_images : int
            Number of images to incorporate into model.
        num_features : int
            Number of features (dimension) in each image.
        """

        max_events = self.psi.max_events
        downsample = self.downsample

        num_images = min(num_images, max_events) if num_images != -1 else max_events

        # set d
        det_shape = self.psi.det.shape()
        num_features = np.prod(det_shape).astype(int)

        if downsample:
            if det_shape[-1] % bin_factor or det_shape[-2] % bin_factor:
                print("Invalid bin factor, toggled off downsampling.")
                self.downsample = False
            else:
                num_features = int(num_features / bin_factor**2)

        return num_images, num_features

    def run(self):
        """
        Perform frequent directions matrix sketching
        on run subject to initialization parameters.
        """

        noImgsToProcess = self.num_images//self.size
        for batch in range(0,noImgsToProcess,int(self.ell*3//self.samplingFactor)):
            self.fetch_and_update_model(int(self.ell*3//self.samplingFactor))

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
        self.imgsTracked.append((self.psi.counter, self.psi.counter + n))

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

        img_batch = np.reshape(imgs, (num_valid_imgs, p * x * y)).T
        nimg_batch = []
        for img in img_batch.T:
            if self.threshold:
                secondQuartile = np.sort(img)[-1]//4
                nimg = (img>secondQuartile)*img
            else:
                nimg = img
            currIntensity = np.sum(nimg.flatten())
            if self.noZeroIntensity and currIntensity<1000:
                continue
            else:
                if currIntensity>10000 and self.normalizeIntensity:
                    nimg_batch.append(nimg/currIntensity)
                else:
                    nimg_batch.append(nimg)
        return np.array(nimg_batch).T

    def fetch_and_update_model(self, n):
        """
        Fetch images and update model.

        Parameters
        ----------
        n : int
            number of images to incorporate
        """
        img_batch = self.get_formatted_images(n)

        if self.samplingFactor <1:
            print("PRE PSAMP REDUCTION SHAPE: ", img_batch.shape)
            psamp = PrioritySampling(int(n*self.samplingFactor), self.d)
            for row in img_batch.T:
                psamp.update(row)
            img_batch = np.array(psamp.sketch.get()).T
            print("PSAMP REDUCTION SHAPE: ", img_batch.shape)

        if self.mean is None:
            self.mean = np.mean(img_batch, axis=1)
        else:
            self.mean = (self.mean*self.num_incorporated_images + np.sum(img_batch.T, axis=0))/(
                    self.num_incorporated_images + (img_batch.shape[1]))
        self.update_model((img_batch.T - self.mean).T)


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
        _, numIncorp = X.shape
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
                        print("INCREASING RANK OF PROCESS {} TO {}".format(self.rank, self.ell))
                    else:
                        copyBatch = self.sketch[self.ell:,:].copy()
                        self.rotate()
                        if canRankAdapt and self.rankAdapt:
                            reconError = np.sqrt(self.lowMemoryReconstructionErrorUnscaled(copyBatch))
                            if (reconError > 0.08):
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
        try:
            [_,s,Vt] = svd(self.sketch , full_matrices=False)
        except LinAlgError as err:
            [_,s,Vt] = scipy_svd(self.sketch, full_matrices = False)
        if len(s) >= self.ell:
            sCopy = s.copy()
           #JOHN: I think actually this should be ell+1 and ell. We lose a component otherwise.
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

    def lowMemoryReconstructionError(self, matrixCentered):
        """ 
        Compute the low memory reconstruction error of the matrix sketch
        against given data. This si the same as reconstructionError,
        but estimates the norm computation and does not scale by the matrix. 

        Parameters
        ----------
        matrixCentered: ndarray
           Data to compare matrix sketch to 

        Returns
        -------
        float,
            Data subtracted by data projected onto sketched space, scaled by matrix elements
       """
        matSketch = self.sketch
        k = 10
        matrixCenteredT = matrixCentered.T
        matSketchT = matSketch.T
        U, S, Vt = np.linalg.svd(matSketchT, full_matrices=False)
        G = U[:,:k]
        return (self.estimFrobNormSquared(matrixCenteredT, [G,G.T,matrixCenteredT], 10)/
                np.linalg.norm(matrixCenteredT, 'fro')**2)

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
            Estimate of frobenius norm of product
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
        filename = self.dir + '{}_sketch_{}.h5'.format(self.currRun, self.rank)
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset("sketch",  data=self.sketch[:self.ell, :])
            hf.create_dataset("mean", data=self.mean)
            hf.create_dataset("imgsTracked", data=np.array(self.imgsTracked))
            hf["sketch"].attrs["numImgsIncorp"] = self.num_incorporated_images
        self.comm.Barrier()
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

    def __init__(self, divBy, readFile, dir, allWriteDirecs, currRun):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.divBy = divBy
        
        with h5py.File(readFile, 'r') as hf:
            self.data = hf["sketch"][:]

        self.fd = FreqDir(0, 0, currRun = currRun, rankAdapt=False, exp='0', run='0', det_type='0', ell=self.data.shape[0], alpha=0.2, downsample=False, bin_factor=0, merger=True, mergerFeatures = self.data.shape[1], dir=dir) 

        sendbuf = self.data.shape[0]
        self.buffSizes = np.array(self.comm.allgather(sendbuf))
        if self.rank==0:
            print("BUFFER SIZES: ", self.buffSizes)

        self.fd.update_model(self.data.T)

        self.dir = dir

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
                        self.fd.update_model(bufferMe.T)
                else:
                    bufferMe = self.fd.get().copy().flatten()
                    self.comm.Send(bufferMe, dest=root, tag=17)
            level += 1
        if self.rank==0:
            fullLen = len(self.allWriteDirecs)
            for readMe in self.allWriteDirecs:
                with h5py.File(readMe, 'r') as hf:
                    if self.fullMean is None:
                        self.fullMean = hf["mean"][:]
                        self.fullNumIncorp = hf["sketch"].attrs["numImgsIncorp"]
                        self.fullImgsTracked = hf["imgsTracked"][:]
                    else:
                        self.fullMean =  (self.fullMean*self.fullNumIncorp + hf["mean"][:])/(self.fullNumIncorp
                                + hf["sketch"].attrs["numImgsIncorp"])
                        self.fullNumIncorp += hf["sketch"].attrs["numImgsIncorp"]
                        self.fullImgsTracked = np.vstack((self.fullImgsTracked,  hf["imgsTracked"][:]))
            return self.fd.get()
        else:
            return

    def write(self):
        """
        Write merged matrix sketch to h5 file
        """
        filename = self.dir + '{}_merge.h5'.format(self.currRun)
        if self.rank==0:
            with h5py.File(filename, 'w') as hf:
                hf.create_dataset("sketch",  data=self.fd.sketch[:self.fd.ell, :])
                hf.create_dataset("mean",  data=self.fullMean)
                hf["sketch"].attrs["numImgsIncorp"] = self.fullNumIncorp
                hf.create_dataset("imgsTracked",  data=self.fullImgsTracked)
        self.comm.Barrier()
        return filename

class ApplyCompression:
    """
    Compute principal components of matrix sketch and apply to data

    Attributes
    ----------
    start_offset: starting index of images to process
    total_imgs: total number of images to process
    exp, run, det_type: experiment properties
    dir: directory to write output
    downsample, bin_factor: whether data should be downsampled and by how much
    threshold: whether data should be thresholded (zero if less than threshold amount)
    normalizeIntensity: whether data should be normalized to have total intensity of one
    noZeroIntensity: whether data with low total intensity should be discarded
    readFile: H5 file with matrix sketch
    batchSize: Number of images to process at each iteration
    data: numpy array housing current matrix sketch
    mean: geometric mean of data processed
    num_incorporated_images: number of images processed so far
    imgageIndicesProcessed: indices of images processed so far
    currRun: Current datetime used to identify run
    imgGrabber: FD object used solely to retrieve data from psana
    components: Principal Components of matrix sketch
    processedData: Data projected onto matrix sketch range
    smallImages: Downsampled images for visualization purposes 
    """

    def __init__(
        self,
        start_offset,
        total_imgs,
        exp,
        run,
        det_type,
        readFile,
        dir,
        batchSize,
        threshold,
        noZeroIntensity,
        normalizeIntensity,
        currRun,
        downsample=False,
        bin_factor=2
    ):

        self.dir = dir

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.total_imgs = total_imgs

        self.currRun = currRun

        self.imgGrabber = FreqDir(start_offset=start_offset,total_imgs=total_imgs, currRun = currRun,
                exp=exp,run=run,det_type=det_type,dir="", downsample=downsample, bin_factor=bin_factor,
                threshold=threshold, normalizeIntensity=normalizeIntensity, noZeroIntensity=noZeroIntensity)
        self.batchSize = batchSize

        (
            self.num_images,
            self.num_features
        ) = self.imgGrabber.set_params(total_imgs, bin_factor)
        self.num_incorporated_images = 0

        with h5py.File(readFile, 'r') as hf:
            self.data = hf["sketch"][:]
            self.mean = hf["mean"][:]
        
        U, S, Vt = np.linalg.svd(self.data, full_matrices=False)
        self.components = Vt
        
        self.processedData = None
        self.smallImgs = None

        self.imageIndicesProcessed = []


    def run(self):
        """
        Retrieve sketch, project images onto new coordinates. Save new coordinates to h5 file. 
        """
        noImgsToProcess = self.num_images//self.size
        for batch in range(0,noImgsToProcess,self.batchSize):
            self.fetch_and_process_data()


    def fetch_and_process_data(self):
        """
        Fetch and downsample data, apply projection algorithm
        """
        startCounter = self.imgGrabber.psi.counter
        img_batch = self.imgGrabber.get_formatted_images(self.batchSize)
        self.imageIndicesProcessed.append((startCounter, self.imgGrabber.psi.counter))

        toSave_img_batch = self.assembleImgsToSave(img_batch)

        if self.smallImgs is None:
            self.smallImgs = toSave_img_batch
        else:
            self.smallImgs = np.concatenate((self.smallImgs, toSave_img_batch), axis=0)
        self.apply_compression((img_batch.T - self.mean).T)

    def assembleImgsToSave(self, imgs):
        """
        Form the images from psana pixel index map and downsample images. 

        Parameters
        ----------
        imgs: ndarray
            images to downsample
        """
        pixel_index_map = retrieve_pixel_index_map(self.imgGrabber.psi.det.geometry(self.imgGrabber.psi.run))

        saveMe = []
        for img in imgs.T:
            imgRe = np.reshape(img, self.imgGrabber.psi.det.shape())
            imgRe = assemble_image_stack_batch(imgRe, pixel_index_map)
            saveMe.append(np.array(Image.fromarray(imgRe).resize((150, 150))))
        saveMe = np.array(saveMe)
        return saveMe

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
        filename = self.dir + '{}_ProjectedData_{}.h5'.format(self.currRun, self.rank)
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset("ProjectedData",  data=self.processedData)
            hf.create_dataset("SmallImages", data=self.smallImgs)
        self.comm.Barrier()
        return filename


class CustomPriorityQueue:
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
    def __init__(self, ell, d):
        self.ell = ell
        self.d = d
        self.sketch = CustomPriorityQueue(self.ell)

    def update(self, vec):
        ui = random.random()
        wi = np.linalg.norm(vec)**2
        pi = wi/ui
        self.sketch.push(vec, pi, wi)
