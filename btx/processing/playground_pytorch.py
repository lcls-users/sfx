import numpy as np
import h5py
import random

from btx.interfaces.ipsana import (
    PsanaInterface,
    retrieve_pixel_index_map,
    assemble_image_stack_batch,
)

import time

from btx.misc.pipca_visuals import *
from btx.processing.PCAonGPU import gpu_pca

def test_ipca_on_pytorch(filename, num_components, batch_size):
    """
    Test the performance of PiPCA using Pytorch (can be on GPU). 
    The reconstructed images and their metadata (experiment name, run, detector, ...) are assumed to be found in the input file created by PiPCA.

    Parameters:
    -----------
    filename : string
        name of the h5 file
    num_components: int
        number of components used
    batch_size: int
        size of the batch to be processed
    
    Returns:
    --------
    Losses between PiPCA and IPCA on Pytorch
    """

    data = unpack_pipca_model_file(filename)
    
    exp, run, loadings, det_type, start_img, U, S, V = data['exp'], data['run'], data['loadings'], data['det_type'], data['start_img'], data['U'], data['S'], data['V']

    start_time = time.time()

    psi = PsanaInterface(exp=exp, run=run, det_type=det_type)
    psi.counter = start_img

    PCs = {f'PC{i}': v for i, v in enumerate(loadings, start=1)}
    eigenimages = {f'PC{i}' : v for i, v in enumerate(U.T, start=1)}

    #Get geometry
    p, x, y = psi.det.shape()
    pixel_index_map = retrieve_pixel_index_map(psi.det.geometry(psi.run))

    #Get images
    imgs = psi.get_images(len(PCs['PC1']),assemble=False)
    num_images = imgs.shape[0]

    #Just be careful, there might have been a downsampling / binning in PiPCA
    imgs = imgs[
            [i for i in range(num_images) if not np.isnan(imgs[i : i + 1]).any()]
        ]
    imgs = np.reshape(imgs, (num_images, p, x, y))

    ipca = IncrementalPCAonGPU(n_components = num_components, batch_size = batch_size)

    print("Fitting IPCA")
    ipca.fit(imgs.reshape(num_images, -1))
    print("Fitting Done")

    end_time = time.time()
    print(f"Time taken for IPCA: {end_time-start_time}")

    #Get the eigenimages
    list_eigenimages = []
    eigenimages = ipca.components_
    for i in range(num_components):
        img = components[i].reshape((p, x, y))
        img = assemble_image_stack_batch(img, pixel_index_map)
        list_eigenimages.append(img)
    
    return list_eigenimages



