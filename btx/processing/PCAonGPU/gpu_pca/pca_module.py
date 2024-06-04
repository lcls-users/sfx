"""This module provides an implementation of Incremental Principal
Components Analysis (IPCA) using PyTorch for GPU acceleration.
IPCA is useful for datasets too large to fit into memory, as it
processes data in smaller chunks or batches.
"""

import torch
import torch.nn as nn
import dask.array as da
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import cupy as cp
# Determine if there's a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch is using:", device)
print("PyTorch version:", torch.__version__)

class IncrementalPCAonGPU(nn.Module):
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
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size

        # Set n_components_ based on n_components if provided
        if n_components:
            self.n_components_ = n_components

        # Initialize attributes to avoid errors during the first call to partial_fit
        self.mean_ = None
        self.var_ = None
        self.n_samples_seen_ = 0

        num_gpus_available = torch.cuda.device_count()
        self.num_gpus_used = num_gpus_available  # Utiliser tous les GPU disponibles
        components_per_gpu = n_components // self.num_gpus_used
        if n_components % self.num_gpus_used != 0:
            components_per_gpu += 1

        self.components = []  # Liste pour stocker les composants

        for i in range(self.num_gpus_used):
            gpu_id = i % num_gpus_available  # Sélectionner un GPU différent pour chaque composant
            self.components.append(
                torch.zeros(components_per_gpu, batch_size).to(f"cuda:{gpu_id}")
            )
        

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
        current_gpu_index = torch.cuda.current_device()
        current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=current_gpu_index)

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=dtype).to(current_device)
        elif X.device != current_device:
            X = X.to(current_device)
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

        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape
        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        for start in range(0, n_samples, self.batch_size_):
            end = min(start + self.batch_size_, n_samples)
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
        current_gpu_index = torch.cuda.current_device()
        current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=current_gpu_index)

        first_pass = not hasattr(self, "components_")

        if check_input:
            X = self._validate_data(X)
        n_samples, n_features = X.shape

        if first_pass:
            self.components_ = None
        if self.n_components is None:
            self.n_components_ = min(n_samples, n_features)

        col_mean, col_var, n_total_samples = self._incremental_mean_and_var(
            X, self.mean_, self.var_, torch.tensor([self.n_samples_seen_], device=current_device)
        )

        # Whitening
        if self.n_samples_seen_ == 0:
            X -= col_mean
        else:
            col_batch_mean = torch.mean(X, dim=0)
            X -= col_batch_mean
            mean_correction_factor = torch.sqrt(
                torch.tensor((self.n_samples_seen_ / n_total_samples.item()) * n_samples, device=current_device)
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

        X_cupy = cp.asarray(X.cpu().numpy())
        cluster = LocalCUDACluster()
        client = Client(cluster)
        try:
        # Convertir les données CuPy en tableau Dask
            X_dask = da.from_array(X_cupy, asarray=True, chunks='auto')

            # Effectuer la SVD compressée avec Dask et CuPy
            U, S, Vt = da.linalg.svd_compressed(X_dask, k=self.n_components)

            # Convertir les résultats de CuPy à PyTorch
            print(type(U))
            U = U.compute()
            print(type(U))
            U = U.np()
            print(type(U))
            print(U.shape)
            U = torch.tensor(U)
            S = torch.tensor(S)
            Vt = torch.tensor(Vt)

        finally:
            # Fermer le client et le cluster
            client.close()
            cluster.close()

        U, Vt = self._svd_flip(U, Vt)

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
        current_gpu_index = torch.cuda.current_device()
        current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=current_gpu_index)

        X = X.to(current_device)
        X -= self.mean_
        return torch.mm(X, self.components_.T)
