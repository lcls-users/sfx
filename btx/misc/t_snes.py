#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
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

import cuml
from cuml.manifold import UMAP as cumlUMAP
from cuml.manifold import TSNE 
from cuml.metrics import trustworthiness as cuml_trustworthiness
import cupy as cp 


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

def process(rank, proj ,device_list,num_tries,threshold):
    proj = torch.tensor(proj,device=device_list[rank])

    torch.cuda.empty_cache()
    gc.collect()

    trustworthiness_threshold = threshold
    best_params_tsne = None
    best_score_tsne = 0
    best_params_umap = None
    best_score_umap = 0
    max_iters = num_tries

    for i in range(max_iters):
        if i%100 == 0:
            print(f"UMAP fitting on GPU {rank} iteration {i}",flush=True)

        n_neighbors = np.random.randint(5, 200)
        min_dist = np.random.uniform(0.0, 0.99) 
        umap = cumlUMAP(n_components=2,n_neighbors=n_neighbors,min_dist=min_dist)
        embedding_umap = umap.fit_transform(proj)
        
        trustworthiness_score_umap = cuml_trustworthiness(proj, embedding_umap)

        if trustworthiness_score_umap > trustworthiness_threshold:
            print(f"Trustworthiness UMAP threshold reached on GPU {rank}!",flush=True)
            best_params_umap = (n_neighbors, min_dist)
            best_score_umap = trustworthiness_score_umap
            break
        elif trustworthiness_score_umap > best_score_umap: 
            best_params_umap = (n_neighbors, min_dist)
            best_score_umap = trustworthiness_score_umap
    
    umap = cumlUMAP(n_components=2,n_neighbors=best_params_umap[0],min_dist=best_params_umap[1])
    embedding_umap = umap.fit_transform(proj)
    trustworthiness_score_umap = cuml_trustworthiness(proj, embedding_umap)

    for i in range(max_iters):
        if i%2000 == 0:
            print(f"t-SNE fitting on GPU {rank} iteration {i}",flush=True)

        perplexity = np.random.randint(5, 50)
        n_neighbors = np.random.randint(3*perplexity, 6*perplexity)
        learning_rate = np.random.uniform(10, 1000)
        tsne = TSNE(n_components=2,perplexity=perplexity,n_neighbors=n_neighbors,learning_rate=learning_rate,verbose=0)
        embedding_tsne = tsne.fit_transform(proj)

        trustworthiness_score_tsne = cuml_trustworthiness(proj, embedding_tsne)

        if trustworthiness_score_tsne > trustworthiness_threshold:
            print(f"Trustworthiness t-SNE threshold reached on GPU {rank}!", flush=True)
            best_params_tsne = (n_neighbors, perplexity)
            best_score_tsne = trustworthiness_score_tsne
            break
        elif trustworthiness_score_tsne > best_score_tsne: 
            best_params_tsne = (n_neighbors, perplexity)
            best_score_tsne = trustworthiness_score_tsne

    torch.cuda.empty_cache()
    gc.collect()

    if best_score_tsne < trustworthiness_threshold:
        tsne = TSNE(n_components=2,learning_rate_method='adaptive')
        embedding_tsne = tsne.fit_transform(proj)
        trustworthiness_score_tsne = cuml_trustworthiness(proj, embedding_tsne)

    print("=====================================\n"
          f"t-SNE and UMAP {rank} fitting done\n"
          f"Trustworthiness on t-SNE on GPU {rank}: {best_score_tsne:.4f}\n"
          f"Best parameters for t-SNE on GPU {rank}: {best_params_tsne}\n"
          f"Trustworthiness on UMAP on GPU {rank}: {best_score_umap:.4f}\n"
          f"Best parameters for UMAP on GPU {rank}: {best_params_umap}"
          "\n=====================================", flush=True)

    embedding_tsne = cp.asnumpy(embedding_tsne)
    embedding_umap = cp.asnumpy(embedding_umap)

    return embedding_tsne, embedding_umap

def get_projectors(rank,imgs,V,device_list):
    V = torch.tensor(V,device=device_list[rank])
    imgs = torch.tensor(imgs,device=device_list[rank])
    proj = torch.mm(imgs,V)
    return proj.cpu().detach().numpy()

def plot_scatters(embedding,type_of_embedding):

    if type_of_embedding == 't-SNE':
        fig = sp.make_subplots(rows=2, cols=2, subplot_titles=[f't-SNE projection (GPU {rank})' for rank in range(num_gpus)])

        for rank in range(num_gpus):
            df = pd.DataFrame({
                't-SNE1': embedding[rank][:, 0],
                't-SNE2': embedding[rank][:, 1],
                'Index': np.arange(len(embedding[rank])),
            })
            
            scatter = px.scatter(df, x='t-SNE1', y='t-SNE2', 
                                hover_data={'Index': True},
                                labels={'t-SNE1': 't-SNE1', 't-SNE2': 't-SNE2'},
                                title=f't-SNE projection (GPU {rank})')
            
            fig.add_trace(scatter.data[0], row=(rank // 2) + 1, col=(rank % 2) + 1)

        fig.update_layout(height=800, width=800, showlegend=False, title_text="t-SNE Projections Across GPUs")
        fig.show()

    else :
        fig = sp.make_subplots(rows=2, cols=2, subplot_titles=[f'UMAP projection (GPU {rank})' for rank in range(num_gpus)])

        for rank in range(num_gpus):
            df = pd.DataFrame({
                'UMAP1': embedding[rank][:, 0],
                'UMAP2': embedding[rank][:, 1],
                'Index': np.arange(len(embedding[rank])),
            })
            
            scatter = px.scatter(df, x='UMAP1', y='UMAP2', 
                                hover_data={'Index': True},
                                labels={'UMAP1': 'UMAP1', 'UMAP2': 'UMAP2'},
                                title=f'UMAP projection (GPU {rank})')
            
            fig.add_trace(scatter.data[0], row=(rank // 2) + 1, col=(rank % 2) + 1)

        fig.update_layout(height=800, width=800, showlegend=False, title_text="UMAP Projections Across GPUs")
        fig.show()

def unpack_ipca_pytorch_model_file(filename):
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
        metadata = f['metadata']
        data['exp'] = str(np.asarray(metadata.get('exp')))[2:-1]
        data['run'] = int(np.asarray(metadata.get('run')))
        data['det_type'] = str(np.asarray(metadata.get('det_type')))[2:-1]
        data['start_offset'] = int(np.asarray(metadata.get('start_offset')))
        data['S']=np.asarray(f['S'])
        data['num_images'] = int(np.asarray(metadata.get('num_images')))
    return data

def parse_input():
    """
    Parse command line input.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-n",
        "--num_images",
        required=True,
        type=int,
    )

    parser.add_argument(
        "-l",
        "--loading_batch_size",
        required=True,
        type=int
    )

    parser.add_argument(
        "--num_tries",
        type=int
    )

    parser.add_argument(
        "--threshold",
        type=float
    )

    return parser.parse_args()


if __name__ == "__main__":
    params = parse_input()
    ##
    filename = params.filename
    num_images = params.num_images
    loading_batch_size = params.loading_batch_size
    threshold = params.threshold
    num_tries = params.num_tries
    ##
    print("Unpacking model file...",flush=True)
    data = unpack_ipca_pytorch_model_file(filename)

    exp = data['exp']
    run = data['run']
    det_type = data['det_type']
    start_img = data['start_offset']
    S = data['S']
    num_images = data['num_images']

    num_gpus, num_components = S.shape
    print(S.shape,flush=True)

    mp.set_start_method('spawn', force=True)

    list_proj = []
    
    print("Unpacking done",flush=True)
    print("Gathering images...",flush=True)
    counter = start_img
    for event in range(start_img, start_img + num_images, loading_batch_size):
        requests_list = [ (exp, run, 'idx', det_type, img) for img in range(event,event+loading_batch_size) ]

        server_address = ('localhost', 5000)
        dataset = IPCRemotePsanaDataset(server_address = server_address, requests_list = requests_list)
        dataloader = DataLoader(dataset, batch_size=20, num_workers=4, prefetch_factor = None)
        dataloader_iter = iter(dataloader)
        
        list_imgs = np.array([])

        for batch in dataloader_iter:
            list_images.append(batch)

        list_images = np.concatenate(list_images, axis=0)
        list_images = list_images[[i for i in range (list_images.shape[0]) if not np.isnan(list_imgs[i : i + 1]).any()]]
        list_images = np.split(list_images,num_gpus,axis=1)

        device_list = [torch.device(f'cuda:{i}' if torch.cuda.is_available() else "cpu") for i in range(num_gpus)]

        starting_time = time.time()
        with h5py.File(filename, 'r') as f:
            V = f['V']
            with Pool(processes=num_gpus) as pool:
                proj = pool.starmap(get_projectors, [(rank,list_images[rank],V[rank,:,counter:counter+list_images.shape[1]],device_list) for rank in range(num_gpus)])
                rank_proj_list = [u for u in proj]
                list_proj.append(np.concatenate(rank_proj_list,axis=0))
        
        counter += list_images.shape[0]
    
    print("Projectors gathered",flush=True)
    print("shape of list_proj",list_proj.shape)
    print("Computing embeddings...",flush=True)

    embeddings_tsne, embeddings_umap = process(0, list_proj, device_list, num_tries, threshold)


    print(f"t-SNE and UMAP fitting done in {time.time()-starting_time} seconds",flush=True)

    data = {"embeddings_tsne": embeddings_tsne, "embeddings_umap": embeddings_umap, "S": S}
    with open(f"embedding_data_{num_components}_{num_images}.pkl", "wb") as f:
        pickle.dump(data, f)
    
    print("All done, closing server...",flush=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(server_address)
        sock.sendall("DONE".encode('utf-8'))

        



        