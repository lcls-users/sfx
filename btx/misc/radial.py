import numpy as np
import lmfit
from math import sqrt
from scipy.signal import sosfiltfilt, butter
from scipy.ndimage import map_coordinates

def get_radius_map(shape, center=None):
    """
    Compute each pixel's radius for an array with input shape and center.
    
    Parameters
    ----------
    shape : tuple, 2d
        dimensions of array
    center : 2d tuple or array                                                                                                                                                                                     
        (cx,cy) detector center in pixels; if None, choose image center     
        
    Returns
    -------
    r : numpy.ndarray, with input shape
        map of pixels' radii
    """
    y, x = np.indices(shape)
    if center is None:
        center = (int(shape[1]/2), int(shape[0]/2))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    return r

def radial_profile(data, center=None, mask=None, 
                   filter=False, filter_order=2, filter_threshold=0.25,
                   threshold=10):
    """
    Compute the radial intensity profile of an image.
    
    Parameters
    ----------
    data : numpy.ndarray, shape (n,m)
        detector image
    center : 2d tuple or array
        (cx,cy) detector center in pixels; if None, choose image center
    mask : numpy.ndarray, shape (n,m)
        detector mask, with zeros corresponding to pixels to mask
    filter : bool
        if True, apply a lowpass Butterworth filter to the radial intensity profile
    filter_order : int
        order of the Butterworth filter
    filter_threshold : float
        critical frequency for the bandpass Butterworth filter
    threshold : float
        below this intensity, set the radial intensity profile to zero
        
    Returns
    -------
    radialprofile : numpy.ndarray, 1d
        radial intensity profile of input image
    """
    if center is None:
        center = (int(data.shape[1]/2), int(data.shape[0]/2))
    r = get_radius_map(data.shape, center=center)
    if mask is not None:
        r = np.where(mask==1, r, 0)
    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr   = np.bincount(r.ravel())
    radialprofile = np.divide(tbin, nr, out=np.zeros(nr.shape[0]), where=nr!=0)
    if filter:
        sos = butter(filter_order, filter_threshold, output='sos')
        radialprofile = sosfiltfilt(sos, radialprofile)
    radialprofile[radialprofile<threshold] = 0
    return radialprofile

def pix2q(npixels, wavelength, distance, pixel_size):
    """
    Convert distance from number of pixels from detector center to q-space.
    
    Parameters
    ----------
    npixels : numpy.ndarray, 1d
        distance in number of pixels from detector center
    wavelength : float
        x-ray wavelength in Angstrom
    distance : float
        detector distance in mm
    pixel_size : float
        detector pixel size in mm
        
    Returns
    -------
    qvals : numpy.ndarray, 1d
        magnitude of q-vector in per Angstrom
    """
    theta = np.arctan(npixels*pixel_size/distance)
    return 2.*np.sin(theta/2.)/wavelength

def q2pix(qvals, wavelength, distance, pixel_size):
    """
    Convert distance from q-space to number of pixels from detector center.
    
    Parameters
    ----------
    qvals : numpy.ndarray, 1d
        magnitude of q-vector in per Angstrom
    wavelength : float
        x-ray wavelength in Angstrom
    distance : float
        detector distance in mm
    pixel_size : float
        detector pixel size in mm
        
    Returns
    -------
    npixels : numpy.ndarray, 1d
        distance in number of pixels from detector center
    """
    pix = np.arcsin(qvals*wavelength/2.)
    return 2*distance*pix/pixel_size




class ConcentricCircles:

    def __init__(self, cx, cy, r, num = 100):
        super().__init__()

        self.cx  = cx                                # Beam center position in pixels along x-axis (axis = 1 in numpy format)
        self.cy  = cy                                # Beam center position in pixels along y-axis
        self.r   = np.array([r]).reshape(-1)         # List of radii for all concentric circles in pixels
        self.num = num                               # Number of pixels sampled from a circle
        self.crds = np.zeros((2, num * len(self.r))) # Coordinates where pixels are sampled from all circles.  Unit is pixel.  2 is the size of (x, y)


    def generate_crds(self):
        """
        Generate coordinates of sample points along each concentric circle
        """
        # Fetching initial values that define concentric circles...
        cx = self.cx
        cy = self.cy
        r  = np.array([self.r]).reshape(-1, 1) # Facilitate broadcasting in calculting crds_x and crds_y

        # Find all theta values for generating coordinates of sample points...
        theta = np.linspace(0.0, 2 * np.pi, self.num)
        if not isinstance(theta, np.ndarray): theta = np.array([theta]).reshape(-1)

        # Generate coordinates...
        crds_x = r * np.cos(theta) + cx
        crds_y = r * np.sin(theta) + cy

        # Reshape crds into one flat array to facilitate optimization routine...
        self.crds[1] = crds_x.reshape(-1)
        self.crds[0] = crds_y.reshape(-1)

        return None


    def get_pixel_values(self, img):
        """
        Get pixel values from all sample points.  If a sample point has 
        subpixel coordinates, interpolation will take place.  

        Parameters
        ----------
        img : numpy.ndarray
            a powder image
        """
        pvals = map_coordinates(img, self.crds)

        return pvals




class OptimizeConcentricCircles(ConcentricCircles):

    def __init__(self, cx, cy, r, num):
        super().__init__(cx, cy, r, num)

        # Provide parameters for optimization...
        self.params = self.init_params()
        self.params.add("cx", value = cx)
        self.params.add("cy", value = cy)

        # Set up radius parameter based on number of circles...
        for i in range(len(r)): self.params.add(f"r{i:d}" , value = r[i] )


    def init_params(self):
        """
        Initialize parameters for optimization.
        """
        return lmfit.Parameters()


    def unpack_params(self, params):
        """
        Unpack all parameters from dictionary.

        Parameters
        ----------
        params : dictionary
            parameters for optimization
        """
        return [ v.value  for _, v in params.items() ]


    def residual_model(self, params, img, **kwargs):
        """
        Calculate the residual for least square optimization.

        Parameters
        ----------
        params : dictionary
            parameters for optimization
        img : numpy.ndarray
            a powder image
        kwargs : dictionary
            additional key-value arguments
        """
        parvals = self.unpack_params(params)
        self.cx, self.cy = parvals[:2]
        self.r = parvals[2:]

        self.generate_crds()

        pvals = self.get_pixel_values(img)

        pvals -= img.max()    # Measure the distance from the peak value

        return pvals


    def fit(self, img, **kwargs):
        """
        Fit the residual model.

        Parameters
        ----------
        img : numpy.ndarray
            a powder image
        """
        print(f"___/ Fitting \___")
        res = lmfit.minimize( self.residual_model,
                              self.params,
                              method     = 'leastsq',
                              nan_policy = 'omit',
                              args       = (img, ),
                              **kwargs )

        return res


    def report_fit(self, res):
        """
        Report details of the optimization.

        Parameters
        ----------
        res : dictionary
            dictionary of optimization details
        """
        lmfit.report_fit(res)
