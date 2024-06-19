"""This module provides an implementation of Incremental Principal
Components Analysis (IPCA) using PyTorch for GPU acceleration.
IPCA is useful for datasets too large to fit into memory, as it
processes data in smaller chunks or batches.
"""

import os
import torch
import torch.distributed as dist
import logging
import subprocess
import cupy as cp

class IncrementalPCAonGPU():
    """
    An implementation of Incremental Principal Components Analysis (IPCA) that leverages PyTorch for GPU acceleration.

    This class provides methods to fit the model on data incrementally in batches, and to transform new data 
    based on the principal components learned during the fitting process.

    Attributes:
        n_components (int, optional): Number of components to keep. If `None`, it's set to the minimum of the 
                                      number of samples and features. Defaults to None.
        whiten (bool): When True, the `components_` vectors are divided to ensure uncorrelated outputs with 
                       unit component-wise variances. Defaults to False.
        copy (bool): If False, input data will be overwritten. Defaults to True.
        batch_size (int, optional): The number of samples to use for each batch. If `None`, it's inferred from 
                                    the data and set to `5 * n_features`. Defaults to None.
    """

    def __init__(self, n_components=None, *, whiten=False, copy=True, batch_size=None):
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size
        # Determine if there's a GPU available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.init()
        print("PyTorch is using:", device)
        print("PyTorch version:", torch.__version__)
        self.device = device

        logging.basicConfig(level=logging.INFO)

        # Configure specific loggers for numba and distributed
        logging.getLogger('numba').setLevel(logging.WARN)
        logging.getLogger('distributed').setLevel(logging.INFO)
        
        # Set n_components_ based on n_components if provided
        if n_components:
            self.n_components_ = n_components

        # Initialize attributes to avoid errors during the first call to partial_fit
        self.mean_ = None  # Will be initialized properly in partial_fit based on data dimensions
        self.var_ = None  # Will be initialized properly in partial_fit based on data dimensions
        self.n_samples_seen_ = 0

        self.components = []  # Liste pour stocker les composants
    
    def _validate_data(self, X, dtype=torch.float32, copy=True):
        """
        Validates and converts the input data `X` to the appropriate tensor format.

        This method ensures that the input data is in the form of a PyTorch tensor and resides on the correct device (CPU or GPU). 
        It also provides an option to create a copy of the tensor, which is useful when the input data should not be overwritten.

        Args:
            X (Union[np.ndarray, torch.Tensor]): Input data which can be a numpy array or a PyTorch tensor.
            dtype (torch.dtype, optional): Desired data type for the tensor. Defaults to torch.float32.
            copy (bool, optional): Whether to clone the tensor. If True, a new tensor is returned; otherwise, the original tensor 
                                   (or its device-transferred version) is returned. Defaults to True.

        Returns:
            torch.Tensor: Validated and possibly copied tensor residing on the specified device.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=dtype).to(self.device)
        elif X.device != self.device:
            X = X.to(self.device)
        if copy:
            X = X.clone()
        return X

    @staticmethod
    def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count):
        """
        Computes the incremental mean and variance for the data `X`.

        Args:
            X (torch.Tensor): The batch input data tensor with shape (n_samples, n_features).
            last_mean (torch.Tensor): The previous mean tensor with shape (n_features,).
            last_variance (torch.Tensor): The previous variance tensor with shape (n_features,).
            last_sample_count (torch.Tensor): The count tensor of samples processed before the current batch.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]: Updated mean, variance tensors, and total sample count.
        """
        if X.shape[0] == 0:
            return last_mean, last_variance, last_sample_count

        # If last_mean or last_variance is None, initialize them with zeros
        if last_mean is None:
            last_mean = torch.zeros(X.shape[1], device=X.device)
        if last_variance is None:
            last_variance = torch.zeros(X.shape[1], device=X.device)

        new_sample_count = X.shape[0]
        new_mean = torch.mean(X, dim=0)
        new_sum_square = torch.sum((X - new_mean) ** 2, dim=0)
        
        updated_sample_count = last_sample_count + new_sample_count
        
        updated_mean = (last_sample_count * last_mean + new_sample_count * new_mean) / updated_sample_count
        updated_variance = (last_variance * (last_sample_count + new_sample_count * last_mean ** 2) + new_sum_square + new_sample_count * new_mean ** 2) / updated_sample_count - updated_mean ** 2
        
        return updated_mean, updated_variance, updated_sample_count

    @staticmethod
    def _svd_flip(u, v, u_based_decision=True):
        """
        Adjusts the signs of the singular vectors from the SVD decomposition for deterministic output.

        This method ensures that the output remains consistent across different runs.

        Args:
            u (torch.Tensor): Left singular vectors tensor.
            v (torch.Tensor): Right singular vectors tensor.
            u_based_decision (bool, optional): If True, uses the left singular vectors to determine the sign flipping. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Adjusted left and right singular vectors tensors.
        """
        if u_based_decision:
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        else:
            max_abs_rows = torch.argmax(torch.abs(v), dim=1)
            signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, None]
        return u, v
    
    def fit(self, X, check_input=False):
        """
        Fits the model with data `X` using minibatches of size `batch_size`.

        Args:
            X (torch.Tensor): The input data tensor with shape (n_samples, n_features).

        Returns:
            IncrementalPCAGPU: The fitted IPCA model.
        """
        #self.client,self.cluster = self.set_up_client()

        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape
        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        for start in range(0, n_samples, self.batch_size_):
            end = min(start + self.batch_size_, n_samples)
            self.print_gpu_memory([0])
            X_batch = X[start:end]
            self.partial_fit(X_batch, check_input=True)
        return self

    def partial_fit(self, X, check_input=True):
        """
        Incrementally fits the model with batch data `X`.

        Args:
            X (torch.Tensor): The batch input data tensor with shape (n_samples, n_features).
            check_input (bool, optional): If True, validates the input. Defaults to True.

        Returns:
            IncrementalPCAGPU: The updated IPCA model after processing the batch.
        """
        first_pass = not hasattr(self, "components_")

        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape

        if first_pass:
            self.components_ = None
        if self.n_components is None:
            self.n_components_ = min(n_samples, n_features)

        col_mean, col_var, n_total_samples = self._incremental_mean_and_var(
            X, self.mean_, self.var_, torch.tensor([self.n_samples_seen_], device=self.device)
        )
        print("X before augmentation:",X.shape)
        if self.components_ is not None:
            print("Size components:",self.components_.shape)
            print("Size singular values:",self.singular_values_.shape)
            print("Size matrix on top :",(self.singular_values_.view((-1, 1)) * self.components_).shape)
        
        # Whitening
        if self.n_samples_seen_ == 0:
            X -= col_mean
        else:
            col_batch_mean = torch.mean(X, dim=0)
            X -= col_batch_mean
            mean_correction_factor = torch.sqrt(
                torch.tensor((self.n_samples_seen_ / n_total_samples.item()) * n_samples, device=self.device)
            )
            mean_correction = mean_correction_factor * (self.mean_ - col_batch_mean)

            if self.singular_values_ is not None and self.components_ is not None:
                X = torch.vstack(
                    (
                        self.singular_values_.view((-1, 1)) * self.components_,
                        X,
                        mean_correction,
                    )
                )

        # SVD of the augmented data
        print("X after augmentation:",X.shape)
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)

        U, Vt = self._svd_flip(U, Vt)
        print("U:",U.shape,"S:",S.shape,"Vt:", Vt.shape)
        explained_variance = S**2 / (n_total_samples.item() - 1)
        explained_variance_ratio = S**2 / torch.sum(col_var * n_total_samples.item())

        self.n_samples_seen_ = n_total_samples.item()
        self.components_ = Vt[: self.n_components_]
        self.singular_values_ = S[: self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[: self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[: self.n_components_]
        if self.n_components_ not in (n_samples, n_features):
            self.noise_variance_ = explained_variance[self.n_components_ :].mean().item()
        else:
            self.noise_variance_ = 0.0

        torch.cuda.empty_cache() #TEST

        return self

    def transform(self, X):
        """
        Applies dimensionality reduction to `X`.

        The input data `X` is projected on the first principal components previously extracted from a training set.

        Args:
            X (torch.Tensor): New data tensor with shape (n_samples, n_features) to be transformed.

        Returns:
            torch.Tensor: Transformed data tensor with shape (n_samples, n_components).
        """
        X = X.to(self.device)
        X -= self.mean_
        return torch.mm(X, self.components_.T)

    def print_gpu_memory(self, rank_to_print=[0,1,2,3]):
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            print("No GPU available")
        else:
            print("Available GPUs:")
            for i in range(num_gpus):
                if i in rank_to_print:
                    print(f"GPU {i}:")
                    print(f"  Name: {torch.cuda.get_device_name(i)}")

                    # Using torch.cuda.memory_allocated() and torch.cuda.memory_reserved()
                    allocated_memory = torch.cuda.memory_allocated(i)
                    reserved_memory = torch.cuda.memory_reserved(i)
                    print(f"  Memory Allocated: {allocated_memory / (1024 ** 2):.2f} MB")
                    print(f"  Memory Reserved: {reserved_memory / (1024 ** 2):.2f} MB")

                    # Using nvidia-smi
                    result = subprocess.run(['nvidia-smi', '--id='+str(i), '--query-gpu=memory.used,memory.free', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
                    used_memory, free_memory = result.stdout.decode().strip().split(',')
                    print(f"  Memory Used: {used_memory} MB")
                    print(f"  Memory Free: {free_memory} MB")
    
    def compute_loss_pytorch(self, X):
        """
        Compute the loss of the model on the input data `X`.

        Args:
            X (torch.Tensor): The input data tensor with shape (n_samples, n_features).

        Returns:
            torch.Tensor: The loss value of the model on the input data.
        """
        X = self._validate_data(X)
        X_transformed = self.transform(X)
        X_reconstructed = torch.mm(X_transformed, self.components_) + self.mean_
        print("X.shape : ",X.shape)
        print("X_reconstructed.shape : ",X_reconstructed.shape)
        norm_batch = torch.norm(X - X_reconstructed, dim = 1, p = 'fro')
        return torch.mean(norm_batch)
