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
            center_modules=[2, 6, 10, 14]
        ):
            self.n_modules = n_modules
            self.n_asics = n_asics
            self.asics_shape = asics_shape
            self.ss_size = ss_size
            self.fs_size = fs_size
            self.pixel_size = pixel1
            self.center_modules = center_modules

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
            center_modules=[0, 1, 2, 3]
        ):
            self.n_modules = n_modules
            self.n_asics = n_asics
            self.asics_shape = asics_shape
            self.ss_size = ss_size
            self.fs_size = fs_size
            self.pixel_size = pixel1
            self.center_modules = center_modules

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
        center_modules=[1, 2, 5, 6]
    ):
        self.n_modules = n_modules
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = ss_size
        self.fs_size = fs_size
        self.pixel_size = pixel1
        self.center_modules = center_modules

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
        center_modules=[0]
    ):
        self.n_modules = n_modules
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = ss_size
        self.fs_size = fs_size
        self.pixel_size = pixel1
        self.center_modules = center_modules

class ControlPointExtractor():

    def __init__(self, exp, run, det_type, powder, calibrant):
        self.diagnostics = PsanaInterface(exp, run, det_type)
        self.powder = np.load(powder)
        self.calibrant = CALIBRANT_FACTORY(calibrant)
        wavelength = self.diagnostics.get_wavelength() * 1e-10
        self.calibrant.set_wavelength(wavelength)
        self.detector = self.get_detector(det_type)
        self.extract_control_points(eps_range=np.arange(20, 60, 1), plot=None)

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

    def extract(self):
        """
        Extract control points from powder
        """
        gray_powder = (self.powder - self.powder.min()) / (self.powder.max() - self.powder.min()) * 255
        powder = gray_powder.astype(np.uint8)
        powder_smoothed = gaussian_filter(powder, sigma=1)
        gradx_powder = np.zeros_like(powder)
        grady_powder = np.zeros_like(powder)
        gradx_powder[:-1, :-1] = (powder_smoothed[1:, :-1] - powder_smoothed[:-1, :-1] + powder_smoothed[1:, 1:] - powder_smoothed[:-1, 1:]) / 2 
        grady_powder[:-1, :-1] = (powder_smoothed[:-1, 1:] - powder_smoothed[:-1, :-1] + powder_smoothed[1:, 1:] - powder_smoothed[1:, :-1]) / 2
        mag = np.sqrt(gradx_powder**2 + grady_powder**2)
        binary_powder = (mag > 1).astype(np.uint8)
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
        for i in range(self.detector.n_modules):
            module = self.detector.center_modules[i]
            panel = self.X[(self.X[:, 0] >= module * self.detector.asics_shape[0] * self.detector.ss_size) & (self.X[:, 0] < (module + 1) * self.detector.asics_shape[0] * self.detector.ss_size)]
            self.panels.append(panel)
            panel[:, 0] = panel[:, 0] - module * self.detector.asics_shape[0] * self.detector.ss_size
            self.panels_normalized.append(panel)
            print(f"Panel {module} has {len(panel)} control points")

    def clusterise(self, X, eps, min_samples):
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        return db.labels_

    def fit_circles_on_clusters(self, X, labels):
        """
        Fit circles to clusterised control points
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
    
    def find_nice_clusters(self, centers, radii, X, labels):
        """
        Find nicely clustered control points based on fitted centers
        """
        # Use DBSCAN to cluster the centers
        db = DBSCAN(eps=100, min_samples=2).fit(centers)
        labels_c = db.labels_

        # Nice cluters are the ones with label 0
        true_centers = labels_c == 0
        nice_clusters = np.arange(0, len(true_centers))[true_centers]
        nice_cluster_radii = radii[true_centers]

        # Filter out clusters where the radius is less than 200pix
        filtered_indices = nice_cluster_radii >= 200
        nice_clusters = nice_clusters[filtered_indices]

        # centroid: Mean center of the true centers
        if len(nice_clusters) != 0:
            centroid = np.mean(centers[true_centers][filtered_indices], axis=0)
        else:
            centroid = np.array([0, 0])
        return nice_clusters, centroid

    def fit_concentric_rings(self, X, labels, nice_clusters, centroid):
        """
        Fit concentric rings to nicely clustered control points
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
    
    def merge_rings(self, labels, nice_clusters, radii):
        """
        Merge radii of concentric rings if radii are close to each other
        """
        # Find close clusters based on radii
        diff_matrix = np.abs(radii[:, np.newaxis] - radii)
        close_clusters = np.argwhere((diff_matrix <= 20) & (diff_matrix != 0))

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
        """
        cp = [len(X[labels==i]) for i in nice_clusters]
        score = len(nice_clusters) * np.sum(cp)/len(X)
        return score

    def hyperparameter_eps_search(self, X, eps_range):
        """
        Search for the best eps hyperparameter
        """
        scores = []
        for eps in eps_range:
            labels = self.clusterise(X, eps=eps, min_samples=4)
            centers, radii = self.fit_circles_on_clusters(X, labels)
            nice_clusters, centroid = self.find_nice_clusters(centers, radii, X, labels)
            if len(nice_clusters) > 0:
                radii = self.fit_concentric_rings(X, labels, nice_clusters, centroid)
                labels, nice_clusters = self.merge_rings(labels, nice_clusters, radii)
                score = self.score_clutering(X, labels, nice_clusters)
            else:
                score = 0
            scores.append(score)
        return eps_range[np.argmax(scores)]
    
    def ring_indexing(self, ratio_q, ratio_radii):
        ring_index = []
        for j in range(len(ratio_radii)):
            i = np.argmin(abs(ratio_radii[j]-ratio_q))
            min_val = np.min(abs(ratio_radii[j]-ratio_q))
            if min_val < 0.02:
                print(f'Found match for ratio idx {j} being ratio of rings {i}/{i+1}')
                ring_index.append(i)
                ring_index.append(i+1)
        return np.unique(ring_index)
        
    def extract_control_points(self, eps_range, plot=None):
        """
        Extract control points from powder, cluster them into concentric rings and find appropriate ring index
        """
        print("Extracting control points from binarized powder...")
        self.extract()
        print("Regrouping control points by panel...")
        self.regroup_by_panel()
        q_data = np.array(self.calibrant.get_peaks(unit='q_nm^-1'))
        ratio_q = q_data[:-1] / q_data[1:]
        data = np.array([])
        for k, X in enumerate(self.panels):
            print(f"Processing panel {self.detector.center_modules[k]}...")
            eps = self.hyperparameter_eps_search(X, eps_range)
            print(f"Best eps for panel {self.detector.center_modules[k]}: {eps}")
            labels = self.clusterise(X, eps=eps, min_samples=4)
            centers, radii = self.fit_circles_on_clusters(X, labels)
            nice_clusters, centroid = self.find_nice_clusters(centers, radii, X, labels)
            radii = self.fit_concentric_rings(X, labels, nice_clusters, centroid)
            labels, nice_clusters = self.merge_rings(labels, nice_clusters, radii)
            radii = self.fit_concentric_rings(X, labels, nice_clusters, centroid)
            sorted_radii = np.sort(radii)
            permutation = np.argsort(radii)
            final_clusters = nice_clusters[permutation]
            ratio_radii = sorted_radii[:-1] / sorted_radii[1:]
            ring_index = self.ring_indexing(ratio_q, ratio_radii)
            if len(ring_index) == 0:
                print(f"No ring index found for panel {self.detector.center_modules[k]}")
            else:
                for i in range(len(final_clusters)):
                    cluster = labels == final_clusters[i]
                    xy = X[cluster]
                    xy[:, 0] += self.detector.center_modules[k] * self.detector.asics_shape[0] * self.detector.ss_size
                    ring_idx = ring_index[i]
                    data = np.append(data, np.column_stack((xy[:, 0], xy[:, 1], np.full(len(xy), ring_idx))), axis=0)
        data = np.array(data)
        self.data = data