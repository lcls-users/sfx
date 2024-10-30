#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import socket
import time
import requests
import io
import numpy as np
import argparse
import time
import os
import sys
import psutil
from multiprocessing import shared_memory, Pool
import torch 
import torch.nn as nn
import torch.multiprocessing as mp
import logging
import gc
import h5py
import csv
import ast

from itertools import chain

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from btx.processing.pipca_nopsana import main as run_client_task # This is the main function that runs the iPCA algorithm
from btx.processing.pipca_nopsana import remove_file_with_timeout
from btx.processing.pipca_nopsana import iPCA_Pytorch_without_Psana
from btx.processing.PCAonGPU.gpu_pca.pca_module import IncrementalPCAonGPU

class IPCRemotePsanaDataset(Dataset):
    def __init__(self, server_address, requests_list):
        """
        server_address: The address of the server. For UNIX sockets, this is the path to the socket.
                        For TCP sockets, this could be a tuple of (host, port).
        requests_list: A list of tuples. Each tuple should contain:
                       (exp, run, access_mode, detector_name, event)
        """
        self.server_address = server_address
        self.requests_list = requests_list

    def __len__(self):
        return len(self.requests_list)

    def __getitem__(self, idx):
        request = self.requests_list[idx]
        return self.fetch_event(*request)

    def fetch_event(self, exp, run, access_mode, detector_name, event):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(self.server_address)
            # Send request
            request_data = json.dumps({
                'exp'          : exp,
                'run'          : run,
                'access_mode'  : access_mode,
                'detector_name': detector_name,
                'event'        : event,
                'mode'         : 'calib',
            })
            sock.sendall(request_data.encode('utf-8'))

            # Receive and process response
            response_data = sock.recv(4096).decode('utf-8')
            response_json = json.loads(response_data)

            # Use the JSON data to access the shared memory
            shm_name = response_json['name']
            shape    = response_json['shape']
            dtype    = np.dtype(response_json['dtype'])
            fiducial = int(response_json['fiducial'])
            time = int(response_json['time'])
            nanoseconds = int(response_json['nanoseconds'])
            seconds = int(response_json['seconds'])


            # Initialize shared memory outside of try block to ensure it's in scope for finally block
            shm = None
            try:
                # Access the shared memory
                shm = shared_memory.SharedMemory(name=shm_name)
                data_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

                # Convert to numpy array (this creates a copy of the data)
                result = [np.array(data_array), fiducial, time, nanoseconds, seconds]
            finally:
                # Ensure shared memory is closed even if an exception occurs
                if shm:
                    shm.close()
                    shm.unlink()

            # Send acknowledgment after successfully accessing shared memory
            sock.sendall("ACK".encode('utf-8'))

            return result

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

def create_or_update_dataset(f, name, data):
    if name in f:
        del f[name]
    f.create_dataset(name, data=data)

def create_shared_images(images):
    shm_list = []
    for sub_imgs in images:
        chunk_size = np.prod(images[0].shape) * images[0].dtype.itemsize
        shm = shared_memory.SharedMemory(create=True, size=chunk_size)
        shm_images = np.ndarray(sub_imgs.shape, dtype=sub_imgs.dtype, buffer=shm.buf)
        np.copyto(shm_images, sub_imgs)
        shm_list.append(shm)
    return shm_list

def read_model_file(filename):
    """
    Reads PiPCA model information from h5 file and returns its contents

    Parameters
    ----------
    filename: str
        name of h5 file you want to unpack

    Returns
    -------
    data: dict
        A dictionary containing the extracted data from the h5 file.
    """
    data = {}
    with h5py.File(filename, 'r') as f:
        data['V'] = np.asarray(f.get('V'))
        data['mu'] = np.asarray(f.get('mu'))
    return data

def reduce_images(V,mu,batch_size,device_list,rank,shm_list,shape,dtype):
    """
    This function is used to update the iPCA model.

    Parameters
    ----------
    images: np.array
        The images to use for updating the model
    V: torch.Tensor
        The V matrix of the iPCA model
    mu: torch.Tensor
        The mu vector of the iPCA model

    Returns
    -------
    V: torch.Tensor
        The updated V matrix of the iPCA model
    mu: torch.Tensor
        The updated mu vector of the iPCA model
    """
    device = device_list[rank]
    V = torch.tensor(V[rank],device=device)
    mu = torch.tensor(mu[rank],device=device)

    existing_shm = shared_memory.SharedMemory(name=shm_list[rank].name)
    images = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    transformed_images = []

    for start in range(0, images.shape[0], batch_size):
        end = min(start + batch_size, images.shape[0])
        batch = images[start:end]
        batch = torch.tensor(batch.reshape(end-start,-1), device=device)
        transformed_batch = torch.mm((batch - mu).float(), V.float())
        transformed_images.append(transformed_batch)
    
    transformed_images = torch.cat(transformed_images, dim=0)
    transformed_images = transformed_images.cpu().numpy()

    return transformed_images
    
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
        "--num_images",
        help="Total number of images per run to be incorporated into model.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--loading_batch_size",
        help="Size of the batches used when loading the images on the client.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size for incremental transformation algorithm.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--num_runs",
        help="Number of runs to process.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--model",
        help="Path to the model file.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--num_gpus",
        help="Number of GPUs to use.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--num_nodes",  
        help="Number of nodes to use.",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--id_current_node",
        help="ID of the current node.",
        required=False,
        type=int,
    )

    return parser.parse_args()

if __name__ == "__main__":

    ##Ajouter parser pour les arguments
    params = parse_input()
    exp = params.exp
    init_run = params.run
    det_type = params.det_type
    start_offset = params.start_offset
    batch_size = params.batch_size
    filename = params.model
    num_gpus = params.num_gpus
    num_runs = params.num_runs
    id_current_node = params.id_current_node
    num_nodes = params.num_nodes
    num_tot_gpus = num_gpus * num_nodes

    if start_offset is None:
        start_offset = 0

    num_images = params.num_images
    num_images = json.loads(num_images)
    num_images_to_add = sum(num_images)
    loading_batch_size = params.loading_batch_size

    mp.set_start_method('spawn', force=True)    
    #Reads current model
    data = read_model_file(filename)
    V = data['V']
    mu = data['mu']
    fiducials_list, times_list, nanoseconds_list, seconds_list = [], [], [], []

    last_batch = False
    projected_images = [[] for i in range(num_gpus)]
    with Pool(processes=num_gpus) as pool:
            num_images_seen = 0
            for run in range(init_run, init_run + num_runs):
                for event in range(start_offset, start_offset + num_images[run-init_run], loading_batch_size):

                    if num_images_seen + loading_batch_size >= num_images_to_add:
                        last_batch = True

                    current_loading_batch = []
                    current_fiducials, current_times, current_nanoseconds, current_seconds = [], [], [], []
                    requests_list = [ (exp, run, 'idx', det_type, img) for img in range(event,min(event+loading_batch_size,num_images[run-init_run]))]

                    server_address = ('localhost', 5000)
                    dataset = IPCRemotePsanaDataset(server_address = server_address, requests_list = requests_list)
                    dataloader = DataLoader(dataset, batch_size=20, num_workers=4, prefetch_factor = None)
                    dataloader_iter = iter(dataloader)

                    for batch in dataloader_iter:
                        current_loading_batch.append(batch[0])
                        current_fiducials.append(batch[1])
                        current_times.append(batch[2])
                        current_nanoseconds.append(batch[3])
                        current_seconds.append(batch[4])

                        if num_images_seen + len(current_loading_batch) >= num_images_to_add and current_loading_batch != []:
                            last_batch = True
                            break

                    current_loading_batch = np.concatenate(current_loading_batch, axis=0)
                    current_fiducials = list(chain.from_iterable(current_fiducials))
                    current_times = list(chain.from_iterable(current_times))
                    current_nanoseconds = list(chain.from_iterable(current_nanoseconds))
                    current_seconds = list(chain.from_iterable(current_seconds))
                    #Remove None images
                    current_len = current_loading_batch.shape[0]
                    num_images_seen += current_len
                    print(f"Loaded {event+current_len} images from run {run}.",flush=True)
                    print("Number of images seen:",num_images_seen,flush=True)
                    current_loading_batch = current_loading_batch[[i for i in range(current_len) if not np.isnan(current_loading_batch[i : i + 1]).any()]]
                    current_fiducials = [current_fiducials[i] for i in range(current_len) if not np.isnan(current_loading_batch[i : i + 1]).any()]
                    current_times = [current_times[i] for i in range(current_len) if not np.isnan(current_loading_batch[i : i + 1]).any()]
                    current_nanoseconds = [current_nanoseconds[i] for i in range(current_len) if not np.isnan(current_loading_batch[i : i + 1]).any()]
                    current_seconds = [current_seconds[i] for i in range(current_len) if not np.isnan(current_loading_batch[i : i + 1]).any()]

                    fiducials_list.append(current_fiducials)
                    times_list.append(current_times)
                    nanoseconds_list.append(current_nanoseconds)
                    seconds_list.append(current_seconds)
                    
                    print(f"Number of non-none images in the current batch: {current_loading_batch.shape[0]}",flush=True)

                    #Split the images into batches for each GPU
                    current_loading_batch = np.split(current_loading_batch, num_tot_gpus,axis=1)
                    current_loading_batch = current_loading_batch[id_current_node*num_gpus:(id_current_node+1)*num_gpus]

                    shape = current_loading_batch[0].shape
                    dtype = current_loading_batch[0].dtype

                    #Create shared memory for each batch
                    shm_list = create_shared_images(current_loading_batch)
                    print("Images split and on shared memory",flush=True)
                    device_list = [torch.device(f'cuda:{i}' if torch.cuda.is_available() else "cpu") for i in range(num_gpus)]

                    #Compute the loss
                    results = pool.starmap(reduce_images, [(V,mu,batch_size,device_list,rank,shm_list,shape,dtype) for rank in range(num_gpus)])

                    for rank in range(num_gpus):
                        projected_images[rank].append(results[rank])
            
                    if last_batch:
                        break

                if last_batch:
                    break

    #Creates a reduced dataset
    for rank in range(num_gpus):
        projected_images[rank] = np.concatenate(projected_images[rank], axis=0)
    
    fiducials_list = np.concatenate(fiducials_list, axis=0)
    times_list = np.concatenate(times_list, axis=0)
    nanoseconds_list = np.concatenate(nanoseconds_list, axis=0)
    seconds_list = np.concatenate(seconds_list, axis=0)

    #Save the projected images
    input_path = os.path.dirname(filename)
    output_path = os.path.join(input_path, f"projected_images_{exp}_start_run_{init_run}_num_images_{num_images_to_add}_node_{id_current_node}.h5")
    with h5py.File(output_path, 'w') as f:
        append_to_dataset(f, 'projected_images', projected_images)
        append_to_dataset(f, 'fiducials', fiducials_list)
        append_to_dataset(f, 'times', times_list)
        append_to_dataset(f, 'nanoseconds', nanoseconds_list)
        append_to_dataset(f, 'seconds', seconds_list)
    
    print(f"Model saved under the name projected_images_{exp}_start_run_{init_run}_num_images_{num_images_to_add}_node_{id_current_node}.h5",flush=True)
    print("Process finished",flush=True)
