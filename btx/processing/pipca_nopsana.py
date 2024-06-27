import os, csv, h5py, argparse
import math
import logging
import numpy as np
import time

from matplotlib import pyplot as plt
from matplotlib import colors
from scipy.linalg import qr

import statistics

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from btx.misc.shortcuts import TaskTimer

from btx.processing.PCAonGPU.gpu_pca.pca_module import IncrementalPCAonGPU

class iPCA_Pytorch_without_Psana:

    """Incremental Principal Component Analysis, uses PyTorch. Can run on GPUs."""

    def __init__(
        self,
        exp,
        run,
        det_type,
        start_offset=0,
        num_images=10,
        num_components=10,
        batch_size=10,
        output_dir="",
        filename='pipca.model_h5',
        images=np.array([]),
        training_percentage=1.0,
        num_gpus=4
    ):

        self.start_offset = start_offset
        self.images = images
        self.output_dir = output_dir
        self.filename = filename
        
        self.num_images = num_images
        self.num_components = num_components
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.task_durations = dict({})

        self.run = run
        self.exp = exp
        self.det_type = det_type
        self.start_time = None
        self.device_list = []

        self.training_percentage = training_percentage
        self.num_training_images = math.ceil(self.num_images * self.training_percentage)
        if self.num_training_images <= self.num_components:
            self.num_training_images = self.num_components


    def run_model(self):
        """
        Run the iPCA algorithm on the given data.
        """

        self.start_time = time.time()

        logging.basicConfig(level=logging.DEBUG)

        self.images = self.images[
                [i for i in range(self.num_images) if not np.isnan(self.images[i : i + 1]).any()]
            ]
        
        self.num_images = self.images.shape[0]
        initial_shape = self.images.shape[1], self.images.shape[2], self.images.shape[3]

        logging.info(f"Number of non-none images: {self.num_images}")

        self.images = np.split(self.images, self.images.shape[1]/self.num_gpus, axis=1)

        logging.info('Images split on GPUs')

        self.device_list = [torch.device(f'cuda:{i}' if torch.cuda.is_available() else "cpu") for i in range(self.num_gpus)]
        logging.info(f"Device list: {self.device_list}")
        logging.info(f"Number of available GPUs: {torch.cuda.device_count()}")
        
        torch.cuda.init()

        mp.set_start_method('spawn', force=True)  # Ensure the start method is 'spawn'
        manager = mp.Manager()
        return_list = manager.list()  # Shared list to store results from each process
        processes = []

        for rank in range(self.num_gpus):
            p = mp.Process(target=self.process_on_gpu, args=(rank, return_list))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        results = list(return_list)

        logging.info("All processes completed")
        end_time = time.time()

        [reconstructed_images, S, V, mu, total_variance, losses, frequency, execution_time] = zip(*results)
        logging.info("Fused results from GPUs")

        """with TaskTimer(self.task_durations, "Initializing model"):
            ipca = IncrementalPCAonGPU(n_components = self.num_components, batch_size = self.batch_size)

        logging.info("Images loaded and formatted and model initialized")

        with TaskTimer(self.task_durations, "Fitting model"):
            ipca.fit(self.images.reshape(self.num_images, -1)[:self.num_training_images])
    
        logging.info(f"Model fitted on {self.num_training_images} images")

        end_time = time.time()
        execution_time = end_time - start_time  # Calculate the execution time
        frequency = self.num_images/execution_time

        reconstructed_images = np.empty((0, self.num_components))
        
        with TaskTimer(self.task_durations, "Reconstructing images"):
            for start in range(0, self.num_images, self.batch_size):
                end = min(start + self.batch_size, self.num_images)
                batch_imgs = self.images[start:end]
                reconstructed_batch = ipca._validate_data(batch_imgs.reshape(end-start, -1))
                reconstructed_batch = ipca.transform(reconstructed_batch)
                reconstructed_batch = reconstructed_batch.cpu().detach().numpy()
                reconstructed_images = np.concatenate((reconstructed_images, reconstructed_batch), axis=0)

        logging.info("Images reconstructed")

        with TaskTimer(self.task_durations, "Computing compression loss"):
            if str(torch.device("cuda" if torch.cuda.is_available() else "cpu")).strip() == "cuda" and self.num_training_images < self.num_images:
                average_training_losses = []
                for start in range(0, self.num_training_images, self.batch_size):
                    end = min(start + self.batch_size, self.num_training_images)
                    batch_imgs = self.images[start:end]
                    average_training_loss= ipca.compute_loss_pytorch(batch_imgs.reshape(end-start, -1))
                    average_training_losses.append(average_training_loss.cpu().detach().numpy())
                average_training_loss = np.mean(average_training_losses)
                logging.info(f"Average training loss: {average_training_loss*100:.3f} (in %)")
                average_evaluation_losses = []
                for start in range(self.num_training_images, self.num_images, self.batch_size):
                    end = min(start + self.batch_size, self.num_images)
                    batch_imgs = self.images[start:end]
                    average_evaluation_loss= ipca.compute_loss_pytorch(batch_imgs.reshape(end-start, -1))
                    average_evaluation_losses.append(average_evaluation_loss.cpu().detach().numpy())
                average_evaluation_loss = np.mean(average_evaluation_losses)
                logging.info(f"Average evaluation loss: {average_evaluation_loss*100:.3f} (in %)")
            elif str(torch.device("cuda" if torch.cuda.is_available() else "cpu")).strip() == "cuda":
                average_losses = []
                for start in range(0, self.num_images, self.batch_size):
                    end = min(start + self.batch_size, self.num_images)
                    batch_imgs = self.images[start:end]
                    average_loss = ipca.compute_loss_pytorch(batch_imgs.reshape(end-start, -1))
                    average_losses.append(average_loss.cpu().detach().numpy())
                average_loss = np.mean(average_losses)
                logging.info(f"Average loss: {average_loss*100:.3f} (in %)")
            else:
                RaiseError("Too long not to compute on GPU")
                        
        if str(torch.device("cuda" if torch.cuda.is_available() else "cpu")).strip() == "cuda":
            S = ipca.singular_values_.cpu().detach().numpy()
            V = ipca.components_.cpu().detach().numpy().T
            mu = ipca.mean_.cpu().detach().numpy()
            total_variance = ipca.explained_variance_.cpu().detach().numpy()
            if self.num_training_images < self.num_images:
                losses = average_training_losses, average_evaluation_losses
            else:
                losses = average_losses
        else:
            S = ipca.singular_values_
            V = ipca.components_.T
            mu = ipca.mean_
            total_variance = ipca.explained_variance_
            losses = average_loss"""

        # save model to an hdf5 file
        with TaskTimer(self.task_durations, "save inputs file h5"):
            with h5py.File(self.filename, 'a') as f:
                if 'exp' not in f or 'det_type' not in f or 'start_offset' not in f:
                    # Create datasets only if they don't exist
                    f.create_dataset('exp', data=self.exp)
                    f.create_dataset('det_type', data=self.det_type)
                    f.create_dataset('start_offset', data=self.start_offset)

                create_or_update_dataset(f, 'run', self.run)
                create_or_update_dataset(f, 'reconstructed_images', data=reconstructed_images)
                create_or_update_dataset(f, 'S', S)
                create_or_update_dataset(f, 'V', V)
                create_or_update_dataset(f, 'mu', mu)
                create_or_update_dataset(f, 'total_variance', total_variance)

                append_to_dataset(f, 'frequency', data=frequency)
                append_to_dataset(f, 'execution_times', data=execution_time)
                append_to_dataset(f, 'initial_shape', data=initial_shape)

                logging.info(f'Model saved to {self.filename}')
        
        for task, durations in self.task_durations.items():
            durations = [float(round(float(duration), 2)) for duration in durations]  # Convert to float and round to 2 decimal places
            if len(durations) == 1:
                logging.debug(f"Task: {task}, Duration: {durations[0]:.2f} (Only 1 duration)")
            else:
                mean_duration = np.mean(durations)
                std_deviation = statistics.stdev(durations)
                logging.debug(f"Task: {task}, Mean Duration: {mean_duration:.2f}, Standard Deviation: {std_deviation:.2f}")
    
        logging.info(f"Model complete in {end_time - self.start_time} seconds")
    
    def process_on_gpu(self,rank,return_list):

        device = self.device_list[rank]
        
        with TaskTimer(self.task_durations, "Initializing model"):
            ipca = IncrementalPCAonGPU(n_components = self.num_components, batch_size = self.batch_size, device = device)

        logging.info(f"Memory Allocated on GPU {rank}: {torch.cuda.memory_allocated(device)} bytes")
        logging.info(f"Memory Cached on GPU {rank}: {torch.cuda.memory_reserved(device)} bytes")
        logging.info(f"GPU {rank}: Images loaded and formatted and model initialized")

        with TaskTimer(self.task_durations, "Fitting model"):
            st = time.time()
            ipca.fit(self.images[rank].reshape(self.num_images, -1)[:self.num_training_images])
            et = time.time()
            logging.info(f"GPU {rank}: Model fitted in {et-st} seconds")
            
        logging.info(f"GPU {rank}: Model fitted on {self.num_training_images} images")
        logging.info(f"Memory Allocated on GPU {rank}: {torch.cuda.memory_allocated(device)} bytes")
        logging.info(f"Memory Cached on GPU {rank}: {torch.cuda.memory_reserved(device)} bytes")

        end_time = time.time()
        execution_time = end_time - self.start_time  # Calculate the execution time
        frequency = self.num_images/execution_time

        reconstructed_images = np.empty((0, self.num_components))
        
        with TaskTimer(self.task_durations, "Reconstructing images"):
            for start in range(0, self.num_images, self.batch_size):
                end = min(start + self.batch_size, self.num_images)
                batch_imgs = self.images[rank][start:end]
                reconstructed_batch = ipca._validate_data(batch_imgs.reshape(end-start, -1))
                reconstructed_batch = ipca.transform(reconstructed_batch)
                reconstructed_batch = reconstructed_batch.cpu().detach().numpy()
                reconstructed_images = np.concatenate((reconstructed_images, reconstructed_batch), axis=0)

        logging.info(f"GPU {rank}: Images reconstructed")

        with TaskTimer(self.task_durations, "Computing compression loss"):
            if str(torch.device("cuda" if torch.cuda.is_available() else "cpu")).strip() == "cuda" and self.num_training_images < self.num_images:
                average_training_losses = []
                for start in range(0, self.num_training_images, self.batch_size):
                    end = min(start + self.batch_size, self.num_training_images)
                    batch_imgs = self.images[rank][start:end]
                    average_training_loss= ipca.compute_loss_pytorch(batch_imgs.reshape(end-start, -1))
                    average_training_losses.append(average_training_loss.cpu().detach().numpy())
                average_training_loss = np.mean(average_training_losses)
                logging.info(f"GPU {rank}: Average training loss: {average_training_loss * 100:.3f} (in %)")
                average_evaluation_losses = []
                for start in range(self.num_training_images, self.num_images, self.batch_size):
                    end = min(start + self.batch_size, self.num_images)
                    batch_imgs = self.images[rank][start:end]
                    average_evaluation_loss= ipca.compute_loss_pytorch(batch_imgs.reshape(end-start, -1))
                    average_evaluation_losses.append(average_evaluation_loss.cpu().detach().numpy())
                average_evaluation_loss = np.mean(average_evaluation_losses)
                logging.info(f"GPU {rank}: Average evaluation loss: {average_evaluation_loss * 100:.3f} (in %)")
            elif str(torch.device("cuda" if torch.cuda.is_available() else "cpu")).strip() == "cuda":
                average_losses = []
                for start in range(0, self.num_images, self.batch_size):
                    end = min(start + self.batch_size, self.num_images)
                    batch_imgs = self.images[rank][start:end]
                    average_loss = ipca.compute_loss_pytorch(batch_imgs.reshape(end-start, -1))
                    average_losses.append(average_loss.cpu().detach().numpy())
                average_loss = np.mean(average_losses)
                logging.info(f"GPU {rank}: Average loss: {average_loss * 100:.3f} (in %)")
            else:
                RaiseError("Too long not to compute on GPU")
                        
        if str(torch.device("cuda" if torch.cuda.is_available() else "cpu")).strip() == "cuda":
            S = ipca.singular_values_.cpu().detach().numpy()
            V = ipca.components_.cpu().detach().numpy().T
            mu = ipca.mean_.cpu().detach().numpy()
            total_variance = ipca.explained_variance_.cpu().detach().numpy()
            if self.num_training_images < self.num_images:
                losses = average_training_losses, average_evaluation_losses
            else:
                losses = average_losses
        else:
            S = ipca.singular_values_
            V = ipca.components_.T
            mu = ipca.mean_
            total_variance = ipca.explained_variance_
            losses = average_loss

        return_list.append((reconstructed_images, S, V, mu, total_variance, losses, frequency, execution_time))

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

def remove_file_with_timeout(filename_with_tag, overwrite=True, timeout=10):
    """
    Remove the file specified by filename_with_tag if it exists.
    
    Parameters:
        filename_with_tag (str): The name of the file to remove.
        overwrite (bool): Whether to attempt removal if the file exists (default is True).
        timeout (int): Maximum time allowed for attempting removal (default is 10 seconds).
    """
    start_time = time.time()  # Record the start time

    while overwrite and os.path.exists(filename_with_tag):
        # Check if the loop has been running for more than the timeout period
        if time.time() - start_time > timeout:
            break  # Exit the loop
            
        try:
            os.remove(filename_with_tag)
        except FileNotFoundError:
            break  # Exit the loop

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


def main(exp,run,det_type,num_images,num_components,batch_size,filename_with_tag,images,training_percentage,smoothing_function,num_gpus):

    images = mapping_function(images, type_mapping = smoothing_function)
    print("Mapping done")
    
    ipca_instance = iPCA_Pytorch_without_Psana(
    exp=exp,
    run=run,
    det_type=det_type,
    num_images=num_images,
    num_components=num_components,
    batch_size=batch_size,
    filename = filename_with_tag,
    images = images,
    training_percentage=training_percentage,
    num_gpus=num_gpus
    )
    
    ipca_instance.run_model()