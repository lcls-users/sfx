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


if __name__ == "__main__":

    # Python executable location
    print("\nPython executable location from client:")
    print(sys.executable)
    

    start_time = time.time()
    params = parse_input()
    exp = params.exp
    run = params.run
    det_type = params.det_type
    start_offset = params.start_offset
    num_components = params.num_components
    batch_size = params.batch_size
    path = params.path
    tag = params.tag
    training_percentage = params.training_percentage
    smoothing_function = params.smoothing_function
    num_gpus = params.num_gpus
    overwrite = True
    filename_with_tag = f"{path}ipca_model_nopsana_{tag}.h5"
    remove_file_with_timeout(filename_with_tag, overwrite, timeout=10)
    all_data = []

    if start_offset is None:
        start_offset = 0
    num_images = params.num_images
    loading_batch_size = params.loading_batch_size
    ipca_state_dict = None

    mp.set_start_method('spawn', force=True)

    ipca_instance = iPCA_Pytorch_without_Psana(
    exp=exp,
    run=run,
    det_type=det_type,
    num_images=num_images,
    num_components=num_components,
    batch_size=batch_size,
    filename = filename_with_tag,
    training_percentage=training_percentage,
    num_gpus=num_gpus
    )
    
    algo_state_dict = ipca_instance.save_state()

    with Pool(processes=num_gpus) as pool:

        for event in range(start_offset, start_offset + num_images, loading_batch_size):

            current_loading_batch = []
            requests_list = [ (exp, run, 'idx', det_type, img) for img in range(event,event+loading_batch_size) ]

            server_address = ('localhost', 5000)
            dataset = IPCRemotePsanaDataset(server_address = server_address, requests_list = requests_list)
            dataloader = DataLoader(dataset, batch_size=20, num_workers=4, prefetch_factor = None)
            dataloader_iter = iter(dataloader)

            for batch in dataloader_iter:
                all_data.append(batch)
                current_loading_batch.append(batch)
            print(f"Loaded {event+loading_batch_size} images.")
            current_loading_batch = np.concatenate(current_loading_batch, axis=0)
            current_loading_batch = current_loading_batch[[i for i in range(loading_batch_size) if not np.isnan(current_loading_batch[i : i + 1]).any()]]

            logging.info(f"Number of non-none images: {current_loading_batch.shape[0]}")
            current_loading_batch = mapping_function(current_loading_batch, type_mapping = smoothing_function)
            current_loading_batch = np.split(current_loading_batch, num_gpus,axis=1)

            shape = current_loading_batch[0].shape
            dtype = current_loading_batch[0].dtype

            shm_list = create_shared_images(current_loading_batch)

            device_list = [torch.device(f'cuda:{i}' if torch.cuda.is_available() else "cpu") for i in range(num_gpus)]

            if not last_batch:
                algo_state_dict, ipca_state_dict = pool.starmap(run_batch, [(ipca_instance.algo_state_dict,ipca_state_dict,last_batch,rank,device_list,images_shape,images_dtype,shm_list) for rank in range(num_gpus)])
            else:
                results = pool.starmap(run_batch, [(ipca_instance.algo_state_dict,ipca_state_dict,last_batch,rank,device_list,images_shape,images_dtype,shm_list) for rank in range(num_gpus)])
                (reconstructed_images, S, V, mu, total_variance, losses, frequency, execution_time) = ([], [], [], [], [], [], [], [])
                for result in results:
                    reconstructed_images.append(result['reconstructed_images'])
                    S.append(result['S'])
                    V.append(result['V'])
                    mu.append(result['mu'])
                    total_variance.append(result['total_variance'])
                    losses.append(result['losses'])
                    frequency.append(result['frequency'])
                    execution_time.append(result['execution_time'])
    print("DONE DONE DOOOOOOOOOOOOOOOONE")

    """all_data = np.concatenate(all_data, axis=0) #MODIFY BECAUSE WE WANT IT TO BE INCREMENTAL, IT'S JUST A TEMPORARY THING
    mem = psutil.virtual_memory()
    print("================LOADING DONE=====================")
    print(f"System total memory: {mem.total / 1024**3:.2f} GB")
    print(f"System available memory: {mem.available / 1024**3:.2f} GB")
    print(f"System used memory: {mem.used / 1024**3:.2f} GB")
    print("=====================================")
    print(all_data.shape)
    print('Images are loaded!')"""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(server_address)
        sock.sendall("DONE".encode('utf-8'))

    print('Server is shut down!')

    """run_client_task(exp,run,det_type,num_images,num_components,batch_size,filename_with_tag,all_data,training_percentage,smoothing_function,num_gpus) # 4 is the number of gpus, will be changed to a parameter
    """
    print('Pipca is done!')

    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time} seconds.")

    """last_batch = False

    for event in range(start_offset, start_offset + num_images, loading_batch_size):

        requests_list = [ (exp, run, 'idx', det_type, img) for img in range(event,event+loading_batch_size) ]

        server_address = ('localhost', 5000)
        dataset = IPCRemotePsanaDataset(server_address = server_address, requests_list = requests_list)
        dataloader = DataLoader(dataset, batch_size=20, num_workers=4, prefetch_factor = None)
        dataloader_iter = iter(dataloader)

        print(f"Loaded {event+loading_batch_size} images.")

        for batch in dataloader_iter:
            if dataloader.batch_idx == (len(dataloader) - 1) and event + loading_batch_size >= num_images + start_offset:
                last_batch = True
                print('Images are loaded, last batch is processing!')
            run_client_task(exp,run,det_type,batch_size,num_components,batch_size,
                            filename_with_tag,batch,training_percentage,last_batch) #there will be issues with the training percentage if we do it batch by batch

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(server_address)
        sock.sendall("DONE".encode('utf-8'))

    print('Server is shut down!')

    print('Pipca is done!')

    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time} seconds.")"""


