#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import socket
import time
import requests
import io
import numpy as np

from multiprocessing import shared_memory

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
                'mode'         : 'image',
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

def main(requests_list):
    server_address = ('localhost', 5000)

    dataset = IPCRemotePsanaDataset(server_address = server_address, requests_list = requests_list)
    dataloader = DataLoader(dataset, batch_size=20, num_workers=10, prefetch_factor = None)
    data_loader_iter = iter(data_loader)

    # Initialize an empty list to store the batches
    all_data = []

    # Iterate over the iterator to fetch all batches
    for batch in data_loader_iter:
        # Each batch is a tuple, and the first element of the tuple contains the batch data
        batch_data = batch[0]
        # Append the batch data to the list
        all_data.append(batch_data)

    # Concatenate all batches into a single array
    all_data_array = np.concatenate(all_data, axis=0)
    print("=================================== \n =================================================== \n =======================================================",all_data_array.shape,"=================================== \n =================================================== \n =======================================================")

