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

            # Initialize shared memory outside of try block to ensure it's in scope for finally block
            shm = None
            try:
                # Access the shared memory
                shm = shared_memory.SharedMemory(name=shm_name)
                data_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

                # Convert to numpy array (this creates a copy of the data)
                result = np.array(data_array)
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
        data['S'] = np.asarray(f.get('S'))
        data['num_images'] = f.get('num_images')[()]
        data['num_components'] = data['V'].shape[2]
    return data

def compute_loss_process(rank,model_state_dict,shm_list,device_list,shape,dtype,batch_size):

    """
    This function is used to compute the loss of the iPCA model. It is used in the multiprocessing pool.

    Parameters
    ----------
    rank: int
        The rank of the process
    V: torch.Tensor
        The V matrix of the iPCA model
    mu: torch.Tensor
        The mu vector of the iPCA model
    shm_list: list
        A list of shared memory objects containing the images
    device: torch.device
        The device to run the computation on

    Returns
    -------
    loss: float
        The loss of the iPCA model
    """

    V = model_state_dict[rank]['V']
    mu = model_state_dict[rank]['mu']

    device = device_list[rank]
    V = torch.tensor(V, device=device)
    mu = torch.tensor(mu, device=device)

    # Load the shared memory images
    existing_shm = shared_memory.SharedMemory(name=shm_list[rank].name)
    images = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

    # Compute the loss
    list_norm_diff = torch.tensor([], device=device)
    list_init_norm = torch.tensor([], device=device)
    ##

    for start in range(0, images.shape[0], batch_size):
        end = min(start + batch_size, images.shape[0])
        batch_imgs = images[start:end]
        batch_imgs = torch.tensor(batch_imgs.reshape(end-start,-1), device=device)
        initial_norm = torch.norm(batch_imgs, dim=1, p = 'fro')
        transformed_batch = torch.mm((batch_imgs.clone() - mu),V)
        reconstructed_batch = torch.mm(transformed_batch,V.T) + mu
        diff = batch_imgs - reconstructed_batch
        norm_batch = torch.norm(diff, dim=1, p = 'fro')
        list_norm_diff = torch.cat((list_norm_diff,norm_batch),dim=0)
        list_init_norm = torch.cat((list_init_norm,initial_norm),dim=0)

    list_norm_diff = list_norm_diff.cpu().detach().numpy()
    list_init_norm = list_init_norm.cpu().detach().numpy()
    """existing_shm.close()
    existing_shm.unlink()"""
    torch.cuda.empty_cache()
    gc.collect()
    
    return list_norm_diff, list_init_norm

def compute_total_loss(list_norm_diff,list_init_norm):
    all_losses = []
    for k in range(len(all_init_norm)):
        i=[0]*len(all_init_norm[k][0])
        d=[0]*len(all_norm_diff[k][0])
        for rank in range(num_gpus):
            i+= all_init_norm[k][rank]**2
            d+= all_norm_diff[k][rank]**2
        all_losses.append(np.sqrt(d)/np.sqrt(i))
    all_losses = np.concatenate(all_losses, axis=0)

    return all_losses

def indices_to_update(losses, lower_bound=0, upper_bound=1e9):
    """
    This function is used to compute the indices of the images that need to be updated.

    Parameters
    ----------
    losses: np.array
        The losses of the images
    lower_bound: float
        The lower bound of the loss
    upper_bound: float
        The upper bound of the loss

    Returns
    -------
    indices: np.array
        The indices of the images that need to be updated
    """
    indices = np.where((losses >= lower_bound) & (losses <= upper_bound))[0]
    return indices

def compute_new_model(model_state_dict,batch_size,device_list,rank,shm_list,shape,dtype,indices_to_update):
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

    print(f"Rank {rank} updating model",flush=True)

    num_components = model_state_dict[rank]['num_components']
    num_images = model_state_dict[rank]['num_images']

    device = device_list[rank]
    ipca = IncrementalPCAonGPU(n_components = num_components, batch_size = batch_size, device = device)
    ipca.components_ = torch.tensor(model_state_dict[rank]['V'].T, device=device)
    ipca.mean_ = torch.tensor(model_state_dict[rank]['mu'], device=device)
    ipca.n_samples_seen_ = num_images
    ipca.singular_values_ = torch.tensor(model_state_dict[rank]['S'], device=device)
    print(f"Rank {rank} model loaded",flush=True)
    existing_shm = shared_memory.SharedMemory(name=shm_list[rank].name)
    images = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    images = images[indices_to_update]
    print(f"Rank {rank} images to update loaded",flush=True)
    st = time.time()
    ipca.fit(images.reshape(len(indices_to_update),-1))
    print(f"Fitting done on rank {rank}. Time to fit: {time.time()-st}")

    existing_shm.close()
    existing_shm.unlink()
    torch.cuda.empty_cache()
    gc.collect()

    V = ipca.components_.T.cpu().detach().numpy()
    mu = ipca.mean_.cpu().detach().numpy()
    S = ipca.singular_values_.cpu().detach().numpy()

    model_state_dict[rank]['V'] = V
    model_state_dict[rank]['mu'] = mu
    model_state_dict[rank]['S'] = S
    model_state_dict[rank]['num_images'] += len(indices_to_update)

    print(f"\n=================\n Rank {rank} model updated",flush=True)
    torch.cuda.empty_cache()
    gc.collect()
    
    return model_state_dict

def update_model(model, model_state_dict):
    """
    This function is used to update the iPCA model.
    """
    S,V,mu = [],[],[]
    for rank in range(num_gpus):
        S.append(model_state_dict[rank]['S'])
        V.append(model_state_dict[rank]['V'])
        mu.append(model_state_dict[rank]['mu'])
    
    new_num_images = model_state_dict[0]['num_images']
    
    with h5py.File(model, 'r+') as f:
        create_or_update_dataset(f, 'V', data=V)
        create_or_update_dataset(f, 'mu', data=mu)
        create_or_update_dataset(f, 'S', data=S)
        create_or_update_dataset(f, 'num_images', data=new_num_images)
    
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
        help="Batch size for iPCA algorithm.",
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
        "--lower_bound",
        help="Lower bound for the loss.",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--upper_bound",
        help="Upper bound for the loss.",
        required=True,
        type=float,
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
    lower_bound = params.lower_bound
    upper_bound = params.upper_bound

    if start_offset is None:
        start_offset = 0

    num_images = params.num_images
    num_images = json.loads(num_images)
    num_images_to_add = sum(num_images)
    loading_batch_size = params.loading_batch_size

    mp.set_start_method('spawn', force=True)    
    #Reads current model
    data = read_model_file(filename)
    all_norm_diff = []
    all_init_norm = []
    last_batch = False
    with mp.Manager() as manager:
        model_state_dict = [manager.dict() for _ in range(num_gpus)]
        for rank in range(num_gpus):
            model_state_dict[rank]['V'] = data['V'][rank]
            model_state_dict[rank]['mu'] = data['mu'][rank]
            model_state_dict[rank]['S'] = data['S'][rank]
            model_state_dict[rank]['num_images'] = data['num_images']
            model_state_dict[rank]['num_components'] = data['num_components']
        
        print("Model loaded",flush=True)

        with Pool(processes=num_gpus) as pool:
                num_images_seen = 0
                for run in range(init_run, init_run + num_runs):
                    for event in range(start_offset, start_offset + num_images[run-init_run], loading_batch_size):

                        if num_images_seen + loading_batch_size >= num_images_to_add:
                            last_batch = True

                        current_loading_batch = []
                        all_norm_diff.append([])
                        all_init_norm.append([])
                        requests_list = [ (exp, run, 'idx', det_type, img) for img in range(event,min(event+loading_batch_size,num_images[run-init_run]))]

                        server_address = ('localhost', 5000)
                        dataset = IPCRemotePsanaDataset(server_address = server_address, requests_list = requests_list)
                        dataloader = DataLoader(dataset, batch_size=20, num_workers=4, prefetch_factor = None)
                        dataloader_iter = iter(dataloader)

                        
                        for batch in dataloader_iter:
                            current_loading_batch.append(batch)
                            if num_images_seen + len(current_loading_batch) >= num_images_to_add and current_loading_batch != []:
                                last_batch = True
                                break

                        current_loading_batch = np.concatenate(current_loading_batch, axis=0)
                        #Remove None images
                        current_len = current_loading_batch.shape[0]
                        num_images_seen += current_len
                        print(f"Loaded {event+current_len} images from run {run}.",flush=True)
                        print("Number of images seen:",num_images_seen,flush=True)
                        current_loading_batch = current_loading_batch[[i for i in range(current_len) if not np.isnan(current_loading_batch[i : i + 1]).any()]]

                        print(f"Number of non-none images in the current batch: {current_loading_batch.shape[0]}",flush=True)

                        #Split the images into batches for each GPU
                        current_loading_batch = np.split(current_loading_batch, num_gpus,axis=1)

                        shape = current_loading_batch[0].shape
                        dtype = current_loading_batch[0].dtype

                        #Create shared memory for each batch
                        shm_list = create_shared_images(current_loading_batch)
                        print("Images split and on shared memory",flush=True)
                        device_list = [torch.device(f'cuda:{i}' if torch.cuda.is_available() else "cpu") for i in range(num_gpus)]

                        #Compute the loss (or not if the bounds are -1)
                        if lower_bound == -1 and upper_bound == -1:
                            indices = range(current_len)
                        
                        else:
                            results = pool.starmap(compute_loss_process, [(rank,model_state_dict,shm_list,device_list,shape,dtype,batch_size) for rank in range(num_gpus)])
                            print("Loss computed",flush=True)
                            for rank in range(num_gpus):
                                list_norm_diff,list_init_norm = results[rank]
                                all_norm_diff[-1].append(list_norm_diff)
                                all_init_norm[-1].append(list_init_norm)

                            total_losses = compute_total_loss(all_norm_diff,all_init_norm)
                            indices = indices_to_update(total_losses,lower_bound,upper_bound)
                            all_norm_diff = []
                            all_init_norm = []
                            print(f"Number of images to update: {len(indices)}",flush=True)

                        if len(indices) > 0:
                            update_or_not = True
                            #Update the model
                            print("Updating model",flush=True)
                            results = pool.starmap(compute_new_model, [(model_state_dict,batch_size,device_list,rank,shm_list,shape,dtype,indices) for rank in range(num_gpus)])
                            print("New model computed",flush=True)
                            if last_batch:
                                print("Last batch",flush=True)
                                break
                        
                        else:
                            print("No images to update",flush=True)
                            update_or_not = False
                            if last_batch:
                                print("Last batch",flush=True)
                                break
                        
                    if last_batch:
                        break

        #Update the model
        if update_or_not:
            update_model(filename, model_state_dict)    
            print("Model updated",flush=True)   

        print("Process finished",flush=True)
