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

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from btx.processing.pipca_nopsana import main as run_client_task # This is the main function that runs the iPCA algorithm
from btx.processing.pipca_nopsana import remove_file_with_timeout
from btx.processing.pipca_nopsana import iPCA_Pytorch_without_Psana

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
        help="Total number of images to be incorporated into model.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--loading_batch_size",
        help="Size of the batches used when loading the images on the client.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--num_components",
        help="Number of principal components to retain.",
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
        "--path",
        help="Path to the output directory.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--tag",
        help="Tag to append to the output file name.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--training_percentage",
        help="Percentage of the data to be used for training.",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--smoothing_function",
        help="Can be used to apply a smoothing function to the data.",
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
        "--compute_loss",
        help="Compute the loss or not.",
        required=False,
        type=bool,
    )
    parser.add_argument(
        "--num_runs",
        help="Number of runs to process.",
        required=True,
        type=int,
    )

    return parser.parse_args()

def create_shared_images(images):
    shm_list = []
    for sub_imgs in images:
        chunk_size = np.prod(images[0].shape) * images[0].dtype.itemsize
        shm = shared_memory.SharedMemory(create=True, size=chunk_size)
        shm_images = np.ndarray(sub_imgs.shape, dtype=sub_imgs.dtype, buffer=shm.buf)
        np.copyto(shm_images, sub_imgs)
        shm_list.append(shm)
    return shm_list

def mapping_function(images, type_mapping = "id"):
    """
    Map the images to a different type.
    
    Parameters:
        images (np.array or torch.tensor): The images to map.
        type (str): The type to map to (default is "id").
        
    Returns:
        np.array or torch.tensor : The mapped images.
    """
    if isinstance(images, np.ndarray):
        if type_mapping == "id":
            return images
        elif type_mapping == "sqrt":
            return np.sign(images) * np.sqrt(np.abs(images))
        elif type_mapping == "log":
            return np.sign(images) * np.log(np.abs(images)+10**(-6))
        else:
            return images
    elif isinstance(images, torch.Tensor):
        if type_mapping == "id":
            return images
        elif type_mapping == "sqrt":
            return torch.sign(images) * torch.sqrt(torch.abs(images))
        elif type_mapping == "log":
            return torch.sign(images) * torch.log(torch.abs(images)+10**(-6))
        else:
            return images
    else:
        raise ValueError("The input type is not supported")

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

def run_batch_process(algo_state_dict, ipca_state_dict, last_batch, rank, device, shape, dtype, shm_list,ipca_instance):
    # Charge les tenseurs CUDA sur le GPU à l'intérieur du processus
    algo_state_dict[rank] = {k: v.cuda(device) if torch.is_tensor(v) else v for k, v in algo_state_dict[rank].items()}
    # Appelle la méthode de traitement
    return ipca_instance.run_batch(algo_state_dict, ipca_state_dict, last_batch, rank, device, shape, dtype, shm_list)

def compute_loss_process(rank, device_list, shape, dtype, shm_list, model_state_dict, batch_size,ipca_instance,loss_or_not):
    # Charge les tenseurs CUDA sur le GPU à l'intérieur du processus
    model_state_dict[rank] = {k:v for k, v in model_state_dict[rank].items()}
    model_state_dict[rank]['V'] = torch.tensor(model_state_dict[rank]['V'], device=device_list[rank])
    model_state_dict[rank]['mu'] = torch.tensor(model_state_dict[rank]['mu'], device=device_list[rank])
    # Appelle la méthode de traitement
    return ipca_instance.compute_loss(rank, device_list, shape, dtype, shm_list,model_state_dict, batch_size,loss_or_not)

if __name__ == "__main__":

    # Python executable location
    print("\nPython executable location from client:")
    print(sys.executable)
    

    start_time = time.time()
    params = parse_input()
    exp = params.exp
    init_run = params.run
    det_type = params.det_type
    start_offset = params.start_offset
    num_components = params.num_components
    batch_size = params.batch_size
    path = params.path
    tag = params.tag
    training_percentage = params.training_percentage
    smoothing_function = params.smoothing_function
    num_gpus = params.num_gpus
    loss_or_not = params.compute_loss
    overwrite = True
    filename_with_tag = f"{path}ipca_model_nopsana_{tag}.h5"
    remove_file_with_timeout(filename_with_tag, overwrite, timeout=10)
    average_losses=[]
    transformed_images = [[] for _ in range(num_gpus)]
    num_training_images = int(params.num_images * training_percentage)
    num_runs = params.num_runs

    if start_offset is None:
        start_offset = 0
    num_images = params.num_images
    loading_batch_size = params.loading_batch_size

    mp.set_start_method('spawn', force=True)

    #Initializes iPCA instance
    ipca_instance = iPCA_Pytorch_without_Psana(
    exp=exp,
    run=init_run,
    det_type=det_type,
    num_images=num_images,
    num_components=num_components,
    batch_size=batch_size,
    filename = filename_with_tag,
    training_percentage=training_percentage,
    num_gpus=num_gpus
    )

    l_time = 0 #loading
    t_time = 0 #treating and cleaning
    f_time = 0 #fitting

    algo_state_dict_local = ipca_instance.save_state()
    last_batch = False
    logging.basicConfig(level=logging.INFO)
    #Creates shared dictionaries for each GPU
    with mp.Manager() as manager:
        algo_state_dict = [manager.dict() for _ in range(num_gpus)]
        ipca_state_dict = [manager.dict() for _ in range(num_gpus)]
        model_state_dict = [manager.dict() for _ in range(num_gpus)]
        for key, value in algo_state_dict_local.items():
            if torch.is_tensor(value):
                for rank in range(num_gpus):
                    algo_state_dict[rank][key] = value.cpu().clone()
            else:
                for rank in range(num_gpus):
                    algo_state_dict[rank][key] = value
        #Creates a pool of processes to parallelize the loading and processing of the images
        with Pool(processes=num_gpus) as pool:
            fitting_start_time = time.time()
            for run in range(init_run, init_run + num_runs):
                for event in range(start_offset, start_offset + num_images, loading_batch_size):

                    beginning_time = time.time()

                    if event + loading_batch_size >= num_training_images + start_offset:
                        last_batch = True

                    current_loading_batch = []
                    requests_list = [ (exp, run, 'idx', det_type, img) for img in range(event,event+loading_batch_size) ]

                    server_address = ('localhost', 5000)
                    dataset = IPCRemotePsanaDataset(server_address = server_address, requests_list = requests_list)
                    dataloader = DataLoader(dataset, batch_size=20, num_workers=4, prefetch_factor = None)
                    dataloader_iter = iter(dataloader)

                    
                    for batch in dataloader_iter:
                        if event + len(current_loading_batch) > num_training_images + start_offset and current_loading_batch != []:
                            last_batch = True
                            break
                        current_loading_batch.append(batch)

                    intermediate_time = time.time()
                    l_time += intermediate_time-beginning_time

                    logging.info(f"Loaded {event+loading_batch_size} images from run {run}.")
                    current_loading_batch = np.concatenate(current_loading_batch, axis=0)
                    #Remove None images
                    current_len = current_loading_batch.shape[0]
                    current_loading_batch = current_loading_batch[[i for i in range(current_len) if not np.isnan(current_loading_batch[i : i + 1]).any()]]

                    logging.info(f"Number of non-none images: {current_loading_batch.shape[0]}")
                    #Apply the smoothing function
                    current_loading_batch = mapping_function(current_loading_batch, type_mapping = smoothing_function)

                    #Split the images into batches for each GPU
                    current_loading_batch = np.split(current_loading_batch, num_gpus,axis=1)

                    shape = current_loading_batch[0].shape
                    dtype = current_loading_batch[0].dtype

                    #Create shared memory for each batch
                    shm_list = create_shared_images(current_loading_batch)

                    device_list = [torch.device(f'cuda:{i}' if torch.cuda.is_available() else "cpu") for i in range(num_gpus)]

                    intermediate_time2 = time.time()
                    t_time += intermediate_time2-intermediate_time

                    if not last_batch:
                        #Run the batch process in parallel
                        results = pool.starmap(run_batch_process, [(algo_state_dict,ipca_state_dict,last_batch,rank,device_list,shape,dtype,shm_list,ipca_instance) for rank in range(num_gpus)])
                        logging.info("Checkpoint : Iteration done")

                        final_time = time.time()
                        f_time += final_time-intermediate_time2

                    else:
                        #Run the batch process in parallel, gather the results and update the model state dictionary
                        results = pool.starmap(run_batch_process, [(algo_state_dict,ipca_state_dict,last_batch,rank,device_list,shape,dtype,shm_list,ipca_instance) for rank in range(num_gpus)])
                        (reconstructed_images, S, V, mu, total_variance) = ([], [], [], [], [])
                        for result in results:
                            S.append(result['S'])
                            V.append(result['V'])
                            mu.append(result['mu'])
                            total_variance.append(result['total_variance'])
                        
                        for rank in range(num_gpus):
                            model_state_dict[rank]['S'] = S[rank]
                            model_state_dict[rank]['V'] = V[rank]
                            model_state_dict[rank]['mu'] = mu[rank]
                            model_state_dict[rank]['total_variance'] = total_variance[rank]
                        
                        final_time = time.time()
                        f_time += final_time-intermediate_time2
                        break
                    

                    mem = psutil.virtual_memory()
                    print("================LOADING DONE=====================",flush=True)
                    print(f"System total memory: {mem.total / 1024**3:.2f} GB",flush=True)
                    print(f"System available memory: {mem.available / 1024**3:.2f} GB",flush=True)
                    print(f"System used memory: {mem.used / 1024**3:.2f} GB",flush=True)
                    print("=====================================")

                    torch.cuda.empty_cache()
                    gc.collect()

            fitting_end_time = time.time()
            print(f"Time elapsed for fitting: {fitting_end_time - fitting_start_time} seconds.",flush=True) 
            print("=====================================\n",flush=True)
            print("FITTING : DONE\n",flush=True)
            print("Loading time: ",l_time,flush=True)
            print("Treating time: ",t_time,flush=True)
            print("Gathering+Fitting time: ",f_time,flush=True)
            print("\n=====================================",flush=True)
            loss_start_time = time.time()

            ##
            all_norm_diff = []
            all_init_norm = []
            ##

            #Compute the loss (same loading process)
            for run in range(init_run, init_run + num_runs):
                for event in range(start_offset, start_offset + num_images, loading_batch_size):
                    
                    ##
                    all_norm_diff.append([])
                    all_init_norm.append([])
                    ##

                    current_loading_batch = []
                    requests_list = [ (exp, run, 'idx', det_type, img) for img in range(event,event+loading_batch_size) ]

                    server_address = ('localhost', 5000)
                    dataset = IPCRemotePsanaDataset(server_address = server_address, requests_list = requests_list)
                    dataloader = DataLoader(dataset, batch_size=20, num_workers=4, prefetch_factor = None)
                    dataloader_iter = iter(dataloader)

                    for batch in dataloader_iter:
                        current_loading_batch.append(batch)

                    logging.info(f"Loaded {event+loading_batch_size} images from run {run}.")
                    current_loading_batch = np.concatenate(current_loading_batch, axis=0)
                    current_loading_batch = current_loading_batch[[i for i in range(loading_batch_size) if not np.isnan(current_loading_batch[i : i + 1]).any()]]

                    logging.info(f"Number of non-none images: {current_loading_batch.shape[0]}")
                    current_loading_batch = mapping_function(current_loading_batch, type_mapping = smoothing_function)
                    current_loading_batch = np.split(current_loading_batch, num_gpus,axis=1)

                    shape = current_loading_batch[0].shape
                    dtype = current_loading_batch[0].dtype

                    shm_list = create_shared_images(current_loading_batch)

                    device_list = [torch.device(f'cuda:{i}' if torch.cuda.is_available() else "cpu") for i in range(num_gpus)]

                    results = pool.starmap(compute_loss_process,[(rank,device_list,shape,dtype,shm_list,model_state_dict,batch_size,ipca_instance,loss_or_not) for rank in range(num_gpus)])
                    current_batch_loss = []
                    for rank in range(num_gpus):
                        average_loss,average_losses,batch_transformed_images,list_norm_diff,list_init_norm = results[rank]
                        current_batch_loss.append(average_loss)
                        average_losses.append(average_loss)
                        transformed_images[rank].append(batch_transformed_images)
                        all_norm_diff[-1].append(list_norm_diff)
                        all_init_norm[-1].append(list_init_norm)
                    
                    print("Batch-Averaged Loss (in %):",np.mean(current_batch_loss)*100)
                    mem = psutil.virtual_memory()
                    print("================LOADING DONE=====================",flush=True)
                    print(f"System total memory: {mem.total / 1024**3:.2f} GB",flush=True)
                    print(f"System available memory: {mem.available / 1024**3:.2f} GB",flush=True)
                    print(f"System used memory: {mem.used / 1024**3:.2f} GB",flush=True)
                    print("=====================================")
                    
                    torch.cuda.empty_cache()
                    gc.collect()

            for rank in range(num_gpus):
                transformed_images[rank] = np.concatenate(transformed_images[rank], axis=0)
            all_losses = []
            for k in range(len(all_init_norm)):
                i=[0]*len(all_init_norm[k][0])
                d=[0]*len(all_norm_diff[k][0])
                for rank in range(num_gpus):
                    i+= all_init_norm[k][rank]**2
                    d+= all_norm_diff[k][rank]**2
                all_losses.append(np.sqrt(d)/np.sqrt(i))
            all_losses = np.concatenate(all_losses, axis=0)

            if training_percentage <1:
                training_loss = (min(all_losses[:num_training_images])*100,np.mean(all_losses[:num_training_images])*100,max(all_losses[:num_training_images])*100)
                testing_loss = (min(all_losses[num_training_images:])*100,np.mean(all_losses[num_training_images:])*100,max(all_losses[num_training_images:])*100)
                print("=====================================\n",flush=True)
                print("Number of training images: ",num_training_images,flush=True)
                print("Global computation of the average training loss (in %): ",training_loss,flush=True)
                print("Global computation of the average testing loss (in %): ",testing_loss,flush=True)
                print("=====================================\n",flush=True)

            ## Optional for anomaly detection
            """threshold = 10 # in %
            for k in range(len(all_losses)):
                if all_losses[k]*100>= threshold:
                    print("Loss above threshold at index",k, all_losses[k]*100)"""
            ##
            loss_end_time = time.time()
            print("=====================================\n",flush=True)
            print("Global computation of the average loss (in %): ",np.mean(all_losses)*100,min(all_losses)*100,max(all_losses)*100,flush=True)
            print("=====================================\n",flush=True)
            
            #print("Loss distribution :",all_losses,flush=True)
            
            print("LOSS COMPUTATION : DONE IN",loss_end_time-loss_start_time,"SECONDS",flush=True)
            print("=====================================\n",flush=True)
            print("Panel-Averaged loss (in %) :",np.mean(average_losses)*100)
            print("\n=====================================",flush=True)


        #Fuse results and save the model
        saving_start_time = time.time()
        S,V,mu,total_variance = [],[],[],[]
        for rank in range(num_gpus):
            S.append(model_state_dict[rank]['S'])
            V.append(model_state_dict[rank]['V'])
            mu.append(model_state_dict[rank]['mu'])
            total_variance.append(model_state_dict[rank]['total_variance'])

    with h5py.File(filename_with_tag, 'w') as f:
                if 'exp' not in f or 'det_type' not in f or 'start_offset' not in f:
                    # Create datasets only if they don't exist
                    f.create_dataset('exp', data=exp)
                    f.create_dataset('det_type', data=det_type)
                    f.create_dataset('start_offset', data=start_offset)

                create_or_update_dataset(f, 'run', data=run)
                create_or_update_dataset(f, 'transformed_images', data=np.array(transformed_images))
                create_or_update_dataset(f, 'S', data=S)
                create_or_update_dataset(f, 'V', data=V)
                create_or_update_dataset(f, 'mu', data=mu)
                create_or_update_dataset(f, 'total_variance', data=total_variance)

    saving_end_time = time.time()

    print(f'Model saved to {filename_with_tag} in {saving_end_time-saving_start_time} seconds',flush=True) 

    print("DONE")

    #Close the server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(server_address)
        sock.sendall("DONE".encode('utf-8'))

    print('Server is shut down!')
    print('Pipca is done!')

    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time} seconds.")
