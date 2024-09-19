import numpy as np
from matplotlib import pyplot as plt
from pyFAI.calibrant import CalibrantFactory, CALIBRANT_FACTORY
from btx.interfaces.ipsana import *
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
from skimage.morphology import dilation, skeletonize

class ePix10k2M():

    def __init__(
            self,
            pixel1=0.0001,
            pixel2=0.0001,
            n_modules=16,
            n_asics=4,
            asics_shape = (2, 2), # (rows, cols) = (ss, fs)
            fs_size=192,
            ss_size=176,
        ):
            self.n_modules = n_modules
            self.n_asics = n_asics
            self.asics_shape = asics_shape
            self.ss_size = ss_size
            self.fs_size = fs_size
            self.pixel_size = pixel1

class ePix10kaQuad():

    def __init__(
            self,
            pixel1=0.0001,
            pixel2=0.0001,
            n_modules=4,
            n_asics=4,
            asics_shape = (2, 2), # (rows, cols) = (ss, fs)
            fs_size=192,
            ss_size=176,
        ):
            self.n_modules = n_modules
            self.n_asics = n_asics
            self.asics_shape = asics_shape
            self.ss_size = ss_size
            self.fs_size = fs_size
            self.pixel_size = pixel1

class Jungfrau4M():

    def __init__(
        self,
        pixel1=0.000075,
        pixel2=0.000075,
        n_modules=8,
        n_asics=8,
        asics_shape=(2, 4), # (rows, cols) = (ss, fs)
        fs_size=256,
        ss_size=256,
    ):
        self.n_modules = n_modules
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = ss_size
        self.fs_size = fs_size
        self.pixel_size = pixel1

class Rayonix():

    def __init__(
        self,
        pixel1=0.000044,
        pixel2=0.000044,
        n_modules=1,
        n_asics=1,
        asics_shape=(1, 1),
        fs_size=1920,
        ss_size=1920,
    ):
        self.n_modules = n_modules
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = ss_size
        self.fs_size = fs_size
        self.pixel_size = pixel1

class ControlPointExtractor():
    """
    Class for extracting control points from powder diffraction data and cluster them into concentric rings and find appropriate ring index
    
    Parameters
    ----------
    exp : str
        Experiment name
    run : int
        Run number
    det_type : str
        Detector type
    powder : str
        Path to powder diffraction data
    calibrant : str
        Calibrant name
    threshold : float
        Threshold value for binarization
    filter : float
        Minimum radius gor a cluster to be considered nice
    radius_tol : float
        Absolute tolerance for merging radii of concentric rings
    ring_tol : float
        Absolute tolerance for finding ring index based on ratio of radii between data and calibrant
    """
    def __init__(self, exp, run, det_type, powder, calibrant, threshold=1, filter=100, radius_tol=20, ring_tol=0.02):
        self.exp = exp
        self.run = run
        self.det_type = det_type
        self.diagnostics = PsanaInterface(exp, run, det_type)
        self.powder = np.load(powder)
        self.calibrant = CALIBRANT_FACTORY(calibrant)
        wavelength = self.diagnostics.get_wavelength() * 1e-10
        self.calibrant.set_wavelength(wavelength)
        self.detector = self.get_detector(det_type)
        self.extract_control_points(eps_range=np.arange(20, 60, 1), threshold=threshold, filter=filter, radius_tol=radius_tol, ring_tol=ring_tol, plot=True)

    def get_detector(self, det_type):
        """
        Retrieve detector geometry info based on detector type

        Parameters
        ----------
        det_type : str
            Detector type
        """
        if det_type == "epix10k2M":
            return ePix10k2M()
        elif "Epix10kaQuad" in det_type:
            return ePix10kaQuad()
        elif det_type == "jungfrau4M":
            return Jungfrau4M()
        elif det_type == "Rayonix":
            return Rayonix()
        else:
            raise ValueError("Detector type not recognized")

    def extract(self, threshold):
        """
        Extract control points from powder

        Parameters
        ----------
        threshold : float
            Threshold value for binarization
        """
        gray_powder = (self.powder - self.powder.min()) / (self.powder.max() - self.powder.min()) * 255
        powder = gray_powder.astype(np.uint8)
        powder_smoothed = gaussian_filter(powder, sigma=1)
        gradx_powder = np.zeros_like(powder)
        grady_powder = np.zeros_like(powder)
        gradx_powder[:-1, :-1] = (powder_smoothed[1:, :-1] - powder_smoothed[:-1, :-1] + powder_smoothed[1:, 1:] - powder_smoothed[:-1, 1:]) / 2 
        grady_powder[:-1, :-1] = (powder_smoothed[:-1, 1:] - powder_smoothed[:-1, :-1] + powder_smoothed[1:, 1:] - powder_smoothed[1:, :-1]) / 2
        mag = np.sqrt(gradx_powder**2 + grady_powder**2)
        binary_powder = (mag > threshold).astype(np.uint8)
        X_total = np.nonzero(binary_powder)
        print(f'Extracted {len(X_total[0])} control points')
        structuring_element = np.ones((3, 3), dtype=np.uint8)
        dilated_powder = dilation(binary_powder, structuring_element)
        final_powder = skeletonize(dilated_powder).astype(np.uint8)
        points = np.nonzero(final_powder)
        X = np.array(list(zip(points[0], points[1])))
        N = X.shape[0]
        self.X = X
        print(f"Extracted {N} control points after dilation and thinning")

    def regroup_by_panel(self):
        """
        Regroup points by panel
        """
        self.panels = []
        self.panels_normalized = []
        for module in range(len(self.detector.n_modules)):
            panel = self.X[(self.X[:, 0] >= module * self.detector.asics_shape[0] * self.detector.ss_size) & (self.X[:, 0] < (module + 1) * self.detector.asics_shape[0] * self.detector.ss_size)]
            self.panels.append(panel)
            panel[:, 0] = panel[:, 0] - module * self.detector.asics_shape[0] * self.detector.ss_size
            self.panels_normalized.append(panel)
            print(f"Panel {module} has {len(panel)} control points")

    def clusterise(self, X, eps, min_samples):
        """
        Cluster data using Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm

        Parameters
        ----------
        X : array
            Data to cluster
        eps : float
            Maximum distance between two samples for them to be considered as in the same neighborhood
        min_samples : int
            Number of samples in a neighborhood for a point to be considered as a core point
        """
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        return db.labels_

    def fit_circles_on_clusters(self, X, labels):
        """
        Fit circles to clusterised control points

        Parameters
        ----------
        X : array
            Control points
        labels : array
            Cluster labels
        """
        def residual(params, x, y):
            cx, cy, r = params
            return np.sqrt((x-cx)**2+(y-cy)**2) - r

        def fit_circle(x, y):
            cx, cy = np.mean(x), np.mean(y) 
            r_guess = np.mean(np.sqrt((x-cx)**2+(y-cy)**2))
            initial = [cx, cy, r_guess]
            result = least_squares(residual, initial, args=(x, y))
            return result
        
        unique_labels = np.unique(labels)
        centers = []
        radii = []
        for k in unique_labels:
            if k != -1:
                class_member_mask = labels == k
                xy = X[class_member_mask]
                x = xy[:, 1]
                y = xy[:, 0]
                result = fit_circle(x, y)
                params = result.x
                centers.append(params[:2])
                radii.append(params[2])
        centers = np.array(centers)
        radii = np.array(radii)
        return centers, radii
    
    def find_nice_clusters(self, centers, radii, filter, eps=100, label=0):
        """
        Find nicely clustered control points based on fitted centers

        Parameters
        ----------
        centers : array
            Fitted centers of the clusters
        radii : array
            Fitted radii of the clusters
        filter : int
            Minimum radius gor a cluster to be considered nice
        eps : int
            Hyperparameter for DBSCAN
        label : int
            Label of the cluster to look at    
        """
        # Use DBSCAN to cluster the centers
        db = DBSCAN(eps=eps, min_samples=2).fit(centers)
        labels_c = db.labels_

        # Nice cluters are the ones with label 0
        true_centers = labels_c == label
        nice_clusters = np.arange(0, len(true_centers))[true_centers]
        nice_cluster_radii = radii[true_centers]

        # Filter out clusters where the radius is less than N pix
        filtered_indices = nice_cluster_radii >= filter
        nice_clusters = nice_clusters[filtered_indices]

        # centroid: Mean center of the true centers
        if len(nice_clusters) != 0:
            centroid = np.mean(centers[true_centers][filtered_indices], axis=0)
            if (centroid[0] >= 0 and centroid[0] < self.detector.ss_size) and (centroid[1] >= 0 and centroid[1] < self.detector.fs_size) and (len(np.unique(labels_c)) > 2):
                print(f"Found {len(np.unique(labels_c))} center clusters and label {label} is unsatisfactory, looking at label {label+1}")
                nice_clusters, centroid = self.find_nice_clusters(centers, radii, filter, 5*eps, label+1)
        else:
            centroid = np.array([0, 0])
        return nice_clusters, centroid

    def fit_concentric_rings(self, X, labels, nice_clusters, centroid):
        """
        Fit concentric rings to nicely clustered control points

        Parameters
        ----------
        X : array
            Control points
        labels : array
            Cluster labels
        nice_clusters : array
            Nice clusters list
        centroid : array
            Centroid of the nice clusters
        """
        def residual_concentric(r, cx, cy, x, y):
            return np.sqrt((x-cx)**2+(y-cy)**2) - r

        def fit_concentric_circles(cx, cy, x, y):
            r_guess = np.mean(np.sqrt((x-cx)**2+(y-cy)**2))
            result = least_squares(residual_concentric, r_guess, args=(cx, cy, x, y))
            return result
        
        cx, cy = centroid
        radii = []
        for i in nice_clusters:
            cluster = labels == i
            xy = X[cluster]
            x = xy[:, 1]
            y = xy[:, 0]
            result = fit_concentric_circles(cx, cy, x, y)
            r = result.x
            radii.append(r[0])
        return np.array(radii)
    
    def merge_rings(self, labels, nice_clusters, radii, radius_tol=20):
        """
        Merge radii of concentric rings if radii are close to each other

        Parameters
        ----------
        labels : array
            Cluster labels
        nice_clusters : array
            Nice clusters obtained after radius filtering
        radii : array
            Fitted radii of the concentric rings
        radius_tol : float
            Tolerance for merging radii
        """
        # Find close clusters based on radii
        diff_matrix = np.abs(radii[:, np.newaxis] - radii)
        close_clusters = np.argwhere((diff_matrix <= radius_tol) & (diff_matrix != 0))

        # Retrieve the pairs
        pairs = [(nice_clusters[i], nice_clusters[j]) for i, j in close_clusters if i < j]

        # Merge labels
        for i, j in pairs:
            nice_clusters[nice_clusters == j] = i
            labels[labels == j] = i
        nice_clusters = np.unique(nice_clusters)
        return labels, nice_clusters
    
    def score_clutering(self, X, labels, nice_clusters):
        """
        Score of the current clustering with given hyperparameter eps

        Parameters
        ----------
        X : array
            Control points
        labels : array
            Cluster labels
        nice_clusters : array
            Nice clusters obtained after radius filtering and merging
        """
        cp = [len(X[labels==i]) for i in nice_clusters]
        score = len(nice_clusters) + np.sum(cp)/len(X)
        return score

    def hyperparameter_eps_search(self, X, eps_range, filter, radius_tol):
        """
        Search for the best eps hyperparameter

        Parameters
        ----------
        X : array
            Control points
        eps_range : array
            Range of eps values to search
        filter : float
            Minimum radius gor a cluster to be considered nice
        radius_tol : float
            Absolute tolerance for merging radii of concentric rings
        """
        scores = []
        for eps in eps_range:
            labels = self.clusterise(X, eps=eps, min_samples=4)
            centers, radii = self.fit_circles_on_clusters(X, labels)
            nice_clusters, centroid = self.find_nice_clusters(centers, radii, filter)
            if len(nice_clusters) > 0:
                radii = self.fit_concentric_rings(X, labels, nice_clusters, centroid)
                labels, nice_clusters = self.merge_rings(labels, nice_clusters, radii, radius_tol)
                score = self.score_clutering(X, labels, nice_clusters)
            else:
                score = 0
            scores.append(score)
        return eps_range[np.argmax(scores)]
    
    def ring_indexing(self, ratio_q, ratio_radii, final_clusters, ring_tol):
        ring_index = np.full(len(final_clusters), -1)
        count = 0
        for j in range(len(ratio_radii)):
            i = np.argmin(abs(ratio_radii[j]-ratio_q))
            min_val = np.min(abs(ratio_radii[j]-ratio_q))
            if min_val < ring_tol:
                print(f'Found match for ratio idx {j} being ratio of rings {i+count}/{i+count+1}')
                ring_index[j] = i+count
                ring_index[j+1] = i+count+1
                ratio_q = np.delete(ratio_q, i)
                count += 1
            else:
                print(f'Missing one ring radius value for ratio idx {j}')
        return ring_index if len(ring_index) > 0 else None
        
    def plot_final_clustering(self, X, labels, nice_clusters, centroid, radii, ring_index, plot):
        """ 
        Plot final clustering after fitting concentric rings
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        unique_labels = np.unique(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
        for i, col in zip(unique_labels, colors):
            if i == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            cluster = labels == i

            xy = X[cluster]
            plt.plot(
                xy[:, 1],
                xy[:, 0],
                "o",
                markerfacecolor=tuple(col),
                markeredgecolor="k",
                markersize=10,
                label=i
            )
        cx, cy = centroid
        for i in range(len(nice_clusters)):
            circle = plt.Circle((cx, cy), radii[i], color='r', fill=False, linestyle='--')
            plt.gca().add_artist(circle)
        plt.legend()
        if ring_index is not None:
            plt.title(f"Final estimation of number of nice clusters: {len(nice_clusters)} with indexed rings {ring_index}")
        else:
            plt.title(f"Final estimation of number of nice clusters: {len(nice_clusters)} with no ring found")
        plt.axis("equal")
        fig.savefig(plot)

    def extract_control_points(self, eps_range, threshold, filter, radius_tol, ring_tol, plot=True):
        """
        Extract control points from powder, cluster them into concentric rings and find appropriate ring index
        """
        print("Extracting control points from binarized powder...")
        self.extract(threshold)
        print("Regrouping control points by panel...")
        self.regroup_by_panel()
        q_data = np.array(self.calibrant.get_peaks(unit='q_nm^-1'))
        ratio_q = q_data[:-1] / q_data[1:]
        data = np.array([])
        for k, X in enumerate(self.panels):
            print(f"Processing panel {k}...")
            eps = self.hyperparameter_eps_search(X, eps_range, filter, radius_tol)
            print(f"Best eps for panel {k}: {eps}")
            labels = self.clusterise(X, eps=eps, min_samples=4)
            centers, radii = self.fit_circles_on_clusters(X, labels)
            nice_clusters, centroid = self.find_nice_clusters(centers, radii, filter)
            radii = self.fit_concentric_rings(X, labels, nice_clusters, centroid)
            labels, nice_clusters = self.merge_rings(labels, nice_clusters, radii, radius_tol)
            print(f"Number of nice clusters for panel {k}: {len(nice_clusters)}")
            radii = self.fit_concentric_rings(X, labels, nice_clusters, centroid)
            if len(radii) > 1:
                sorted_radii = np.sort(radii)
                permutation = np.argsort(radii)
                final_clusters = nice_clusters[permutation]
                ratio_radii = sorted_radii[:-1] / sorted_radii[1:]
                ring_index = self.ring_indexing(ratio_q, ratio_radii, final_clusters, ring_tol)
            else:
                final_clusters = nice_clusters
                ring_index = None
            if plot:
                plot_name = f"{self.exp}_r{self.run:04}_{self.det_type}_panel{k}_clustering.png"
                self.plot_final_clustering(X, labels, nice_clusters, centroid, radii, ring_index, plot_name)
            if ring_index is None:
                print(f"No ring index found for panel {k}")
            else:
                for i in range(len(final_clusters)):
                    if ring_index[i] != -1:
                        cluster = labels == final_clusters[i]
                        xy = X[cluster]
                        xy[:, 0] += self.detector.center_modules[k] * self.detector.asics_shape[0] * self.detector.ss_size
                        ring_idx = ring_index[i]
                        if len(data) == 0:
                            data = np.column_stack((xy[:, 0], xy[:, 1], np.full(len(xy), ring_idx)))
                        else:
                            data = np.append(data, np.column_stack((xy[:, 0], xy[:, 1], np.full(len(xy), ring_idx))), axis=0)
        data = np.array(data)
        self.data = data