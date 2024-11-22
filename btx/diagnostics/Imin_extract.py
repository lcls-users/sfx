import numpy as np
from matplotlib import pyplot as plt
from pyFAI.calibrant import CalibrantFactory, CALIBRANT_FACTORY
from btx.interfaces.ipsana import *
from scipy.optimize import least_squares
from sklearn.cluster import DBSCAN

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
        pixel1=0.000176,
        pixel2=0.000176,
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

class IminExtractor():
    """
    Class for extracting minimal intensity from clustering powder diffraction data into concentric rings
    
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
    Imin_range : array
        Range of Imin values to search
    eps_range : array
        Range of eps values to search
    filter : float
        Minimum radius gor a cluster to be considered nice
    radius_tol : float
        Absolute tolerance for merging radii of concentric rings
    """
    def __init__(self, exp, run, det_type, powder, Imin_range, eps_range, filter=50, radius_tol=20):
        self.exp = exp
        self.run = run
        self.det_type = det_type
        self.diagnostics = PsanaInterface(exp, run, det_type)
        self.powder = np.load(powder)
        self.detector = self.get_detector(det_type)
        self.extract_Imin_score(Imin_range=Imin_range, eps_range=eps_range, filter=filter, radius_tol=radius_tol)

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

    def extract(self, Imin):
        """
        Extract control points from powder

        Parameters
        ----------
        threshold : float
            Threshold value for binarization
        """
        threshold = np.percentile(self.powder, Imin)
        binary_powder = np.where(self.powder > threshold, 1, 0)
        X = np.nonzero(binary_powder)
        print(f'Extracted {len(X[0])} control points')
        X = np.array(list(zip(X[0], X[1])))
        N = X.shape[0]
        self.X = X
        print(f"Extracted {N} control points after dilation and thinning")

    def regroup_by_panel(self):
        """
        Regroup points by panel
        """
        self.panels = []
        self.panels_normalized = []
        for module in range(self.detector.n_modules):
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
        scores = []
        for i in nice_clusters:
            cluster = labels == i
            xy = X[cluster]
            x = xy[:, 1]
            y = xy[:, 0]
            result = fit_concentric_circles(cx, cy, x, y)
            r = result.x
            cost = result.cost
            radii.append(r[0])
            scores.append(cost)
        return np.array(radii), np.array(scores)
    
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
            score = 0
            if len(centers) > 0:
                nice_clusters, centroid = self.find_nice_clusters(centers, radii, filter)
                if len(nice_clusters) > 0:
                    radii = self.fit_concentric_rings(X, labels, nice_clusters, centroid)
                    labels, nice_clusters = self.merge_rings(labels, nice_clusters, radii, radius_tol)
                    score = self.score_clutering(X, labels, nice_clusters)
            scores.append(score)
        return eps_range[np.argmax(scores)]

    def extract_Imin_score(self, Imin_range, eps_range, filter, radius_tol):
        """
        Extract control points from powder, cluster them into concentric rings and find appropriate ring index
        """
        scores = {}
        for Imin in Imin_range:
            print(f"Extracting control points from binarized powder with threshold={Imin}...")
            self.extract(Imin)
            print("Regrouping control points by panel...")
            self.regroup_by_panel()
            for k, X in enumerate(self.panels):
                if len(X) == 0:
                    print(f"Skipping panel {k} as no control points were found")
                    continue
                print(f"Processing panel {k}...")
                eps = self.hyperparameter_eps_search(X, eps_range, filter, radius_tol)
                print(f"Best eps for panel {k}: {eps}")
                labels = self.clusterise(X, eps=eps, min_samples=4)
                centers, radii = self.fit_circles_on_clusters(X, labels)
                centroid = []
                if len(centers) == 0:
                    print(f"All data clustered as noise in panel {k}")                
                else:
                    nice_clusters, centroid = self.find_nice_clusters(centers, radii, filter)
                    radii, costs = self.fit_concentric_rings(X, labels, nice_clusters, centroid)
                    labels, nice_clusters = self.merge_rings(labels, nice_clusters, radii, radius_tol)
                    print(f"Number of nice clusters for panel {k}: {len(nice_clusters)}")
                    radii, costs = self.fit_concentric_rings(X, labels, nice_clusters, centroid)
                    scores[Imin][f'panel_{k}'] = np.mean(costs)
            self.scores = scores