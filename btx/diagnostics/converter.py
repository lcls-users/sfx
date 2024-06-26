import numpy as np
from pyFAI.detectors import Detector
from PSCalib.UtilsConvert import geometry_to_crystfel, header_crystfel, panel_constants_to_crystfel, SEGNAME_TO_PARS
from PSCalib.UtilsConvertCrystFEL import convert_crystfel_to_geometry
from PSCalib.GlobalUtils import CFRAME_LAB, CFRAME_PSANA
from PSCalib.GeometryAccess import GeometryAccess

class ePix10k2M(Detector):
    """
    PyFAI Detector instance for the ePix10k2M
    """

    def __init__(
        self,
        pixel1=0.0001,
        pixel2=0.0001,
        n_modules=16,
        n_asics=4,
        asics_shape = (2, 2), # (rows, cols) = (ss, fs)
        fs_size=192,
        ss_size=176,
        **kwargs,
    ):
        super().__init__(pixel1=pixel1, pixel2=pixel2, max_shape=(n_modules * asics_shape[1] * ss_size, asics_shape[0] * fs_size), **kwargs)
        self.n_modules = n_modules
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = ss_size
        self.fs_size = fs_size
        self.pixel_size = pixel1

class ePix10kaQuad(Detector):
    """
    PyFAI Detector instance for the ePix100
    """

    def __init__(
        self,
        pixel1=0.0001,
        pixel2=0.0001,
        n_modules=4,
        n_asics=4,
        asics_shape = (2, 2),
        fs_size=192,
        ss_size=176,
        **kwargs,
    ):
        super().__init__(pixel1=pixel1, pixel2=pixel2, max_shape=(n_modules * asics_shape[1] * ss_size, asics_shape[0] * fs_size), **kwargs)
        self.n_modules = n_modules
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = ss_size
        self.fs_size = fs_size
        self.pixel_size = pixel1

class Jungfrau4M(Detector):
    """
    PyFAI Detector instance for the Jungfrau
    """

    def __init__(
        self,
        pixel1=0.075,
        pixel2=0.075,
        n_modules=8,
        n_asics=8,
        asics_shape=(4, 2), # (rows, cols) = (ss, fs)
        fs_size=256,
        ss_size=256,
        **kwargs,
    ):
        super().__init__(pixel1=pixel1, pixel2=pixel2, max_shape=(n_modules * asics_shape[1] * ss_size, asics_shape[0] * fs_size), **kwargs)
        self.n_modules = n_modules
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = ss_size
        self.fs_size = fs_size
        self.pixel_size = pixel1

class Rayonix(Detector):
    """
    PyFAI Detector instance for the Rayonix
    """

    def __init__(
        self,
        pixel1=0.044,
        pixel2=0.044,
        n_modules=1,
        n_asics=1,
        asics_shape=(1, 1),
        fs_size=1920,
        ss_size=1920,
        **kwargs,
    ):
        super().__init__(pixel1=pixel1, pixel2=pixel2, max_shape=(n_modules * ss_size, fs_size), **kwargs)
        self.n_modules = n_modules
        self.ss_size = ss_size
        self.fs_size = fs_size
        self.pixel_size = pixel1

class PsanatoCrystFEL:
    """
    Class to convert psana .data geometry files to CrystFEL .geom geometry files in the desired reference frame
    """

    def __init__(self, psana_file, output_file, det_type, cframe=CFRAME_PSANA, zcorr_um=None):
        if det_type == "Rayonix":
            self.rayonix_geometry_to_crystfel(psana_file, output_file, cframe, zcorr_um)
        else:
            geometry_to_crystfel(psana_file, output_file, cframe, zcorr_um)

    def rayonix_geometry_to_crystfel(self, psana_file, output_file, cframe=CFRAME_PSANA, zcorr_um=None):
        geo = GeometryAccess(psana_file, 0, use_wide_pix_center=False)
        x, y, z = geo.get_pixel_coords(oname=None, oindex=0, do_tilt=True, cframe=cframe)
        geo1 = geo.get_seg_geo() # GeometryObject
        seg = geo1.algo # object of the SegmentGeometry subclass
        segname = geo1.oname
        nsegs = int(x.size/seg.size())
        shape = (nsegs,) + seg.shape() # (nsegs, srows, scols)
        x.shape = shape
        y.shape = shape
        z.shape = shape
        txt = header_crystfel()
        for n in range(nsegs):
            z_um = z[n,:]
            if zcorr_um is not None: z_um -= zcorr_um
            txt += panel_constants_to_crystfel(seg, n, x[n,:], y[n,:], z_um)
        if output_file is not None:
            f = open(output_file,'w')
            f.write(txt)
            f.close()

class CrystFELtoPyFAI:
    """
    Class to convert CrystFEL .geom geometry files from a given reference frame to PyFAI corner arrays
    """

    def __init__(self, geom_file, det_type, cframe=CFRAME_PSANA):
        self.det_type = det_type
        self.geom_file = geom_file
        self.cframe = cframe
        self.detector = self.get_detector(det_type)
        self.panels = self.from_CrystFEL(geom_file)
        self.pix_pos = self.get_pixel_coordinates(self.panels)
        self.corner_array = self.get_corner_array(self.pix_pos, self.panels, cframe)
        self.detector.set_pixel_corners(self.corner_array)

    def get_detector(self, det_type):
        """
        Instantiate a PyFAI Detector object based on the detector type
        """
        if det_type == "epix10k2M":
            return ePix10k2M()
        elif "Epix10kaQuad" in det_type:
            return ePix10kaQuad()
        elif det_type == "jungfrau4M":
            return Jungfrau4M()
        elif det_type == "rayonix" or det_type == "Rayonix":
            return Rayonix()
        else:
            raise ValueError("Detector type not recognized")

    def from_CrystFEL(self, fname: str):
        """
        Parse a CrystFEL geometry file
        Read a text ".geom" file and return the dictionary of geometry components

        Parameters
        ----------
        fname : str
            Path to the CrystFEL geometry file
        """
        detector = {
            "panels": {},
            "rigid_groups": {},
            "rigid_group_collections": {},
        }
        with open(fname, "r") as f:
            lines = f.readlines()
            for line in lines:
                # Remove comments
                if line[0] == ";":
                    continue
                if "=" not in line:
                    continue
                fmt_line = line.strip()
                # CrystFEL fmt: object = val
                obj_value = line.split("=")
                # object fmt: "panel/parameter"
                obj = obj_value[0].split("/")  # May be len 1 or 2
                value = obj_value[1]
                if len(obj) == 1:  # e.g. rigid_group_quad ...
                    if "collection" in obj[0].strip():
                        collection_name = obj[0].strip().split("_")[-1]
                        detector["rigid_group_collections"][
                            collection_name
                        ] = value.strip().split(",")
                    else:
                        group_name = obj[0].strip().split("_")[-1]
                        detector["rigid_groups"][group_name] = value.strip().split(",")
                elif len(obj) == 2:  # e.g. p0a0/fs = ...
                    pname = obj[0].strip()
                    if pname in detector["panels"]:
                        panel = detector["panels"][pname]
                    else:
                        panel = {
                            "fs": (0, 0, 0),
                            "ss": (0, 0, 0),
                            "res": 10000,
                            "corner_x": 100,
                            "corner_y": 100,
                            "coffset": 0.1,
                            "min_fs": 0,
                            "max_fs": 123,
                            "min_ss": 0,
                            "max_ss": 123,
                            "no_index": 0,
                        }
                        detector["panels"][pname] = panel
                    if "fs" in obj[1].strip()[-2:]:
                        if "max" in obj[1]:
                            panel["max_fs"] = int(value)
                        elif "min" in obj[1]:
                            panel["min_fs"] = int(value)
                        else:
                            strcoords = value.split()
                            if "z" in strcoords:
                                # -1x -2y -3z
                                fcoords = (
                                    float(strcoords[0].strip("x")),
                                    float(strcoords[1].strip("y")),
                                    float(strcoords[2].strip("z")),
                                )
                                panel["fs"] = fcoords
                            else:
                                # -1x -2y
                                fcoords = (
                                    float(strcoords[0].strip("x")),
                                    float(strcoords[1].strip("y")),
                                    0.0,
                                )
                                panel["fs"] = fcoords
                    elif "ss" in obj[1].strip()[-2:]:
                        if "max" in obj[1]:
                            panel["max_ss"] = int(value)
                        elif "min" in obj[1]:
                            panel["min_ss"] = int(value)
                        else:
                            strcoords = value.split()
                            if "z" in strcoords:
                                # -1x -2y -3z
                                fcoords = (
                                    float(strcoords[0].strip("x")),
                                    float(strcoords[1].strip("y")),
                                    float(strcoords[2].strip("z")),
                                )
                                panel["ss"] = fcoords
                            else:
                                # -1x -2y
                                fcoords = (
                                    float(strcoords[0].strip("x")),
                                    float(strcoords[1].strip("y")),
                                    0.0,
                                )
                                panel["ss"] = fcoords
                    elif "res" in obj[1].strip():
                        panel["res"] = float(value)
                    elif "corner" in obj[1].strip():
                        if "x" in obj[1]:
                            panel["corner_x"] = float(value)
                        elif "y" in obj[1]:
                            panel["corner_y"] = float(value)
                    elif "no_index" in obj[1]:
                        panel["no_index"] = int(value)
                    elif "coffset" in obj[1]:
                        panel["coffset"] = float(value)
            return detector

    def get_pixel_coordinates(self, panels: dict):
        """
        From a parsed CrystFEL geometry file, calculate Epix10k2M pixel coordinates
        in psana reference frame

        Parameters
        ----------
        panels : dict
            Dictionary of panels from a CrystFEL geometry file
        """
        nmods = self.detector.n_modules
        nasics = self.detector.n_asics
        asics_shape = self.detector.asics_shape
        fs_size = self.detector.fs_size
        ss_size = self.detector.ss_size
        pix_arr = np.zeros([nmods, ss_size * asics_shape[0], fs_size * asics_shape[1], 3])
        mean_z = np.mean([panels["panels"][f"p{p}a{a}"]["coffset"] for p in range(nmods) for a in range(nasics)])
        for p in range(nmods):
            pname = f"p{p}"
            for asic in range(nasics):
                asicname = f"a{asic}"
                full_name = pname + asicname
                arow = asic // (nasics//2)
                acol = asic % (nasics//2)
                ss_portion = slice(arow * ss_size, (arow + 1) * ss_size)
                fs_portion = slice(acol * fs_size, (acol + 1) * fs_size)
                res = panels["panels"][full_name]["res"]
                corner_x = panels["panels"][full_name]["corner_x"] / res
                corner_y = panels["panels"][full_name]["corner_y"] / res
                corner_z = panels["panels"][full_name]["coffset"]-mean_z
                # Get tile vectors for ss and fs directions
                ssx, ssy, ssz = np.array(panels["panels"][full_name]["ss"]) / res
                fsx, fsy, fsz = np.array(panels["panels"][full_name]["fs"]) / res
                coords_ss, coords_fs = np.meshgrid(
                    np.arange(0, ss_size), np.arange(0, fs_size), indexing="ij"
                )
                x = corner_x + ssx * coords_ss + fsx * coords_fs
                y = corner_y + ssy * coords_ss + fsy * coords_fs
                z = corner_z + ssz * coords_ss + fsz * coords_fs
                pix_arr[p, ss_portion, fs_portion, 0] = x
                pix_arr[p, ss_portion, fs_portion, 1] = y
                pix_arr[p, ss_portion, fs_portion, 2] = z
        return pix_arr

    def get_corner_array(self, pix_pos, panels, cframe=CFRAME_PSANA):
        """
        Convert to the corner array needed by PyFAI

        Parameters
        ----------
        pix_pos : np.ndarray
            Pixel positions in .geom reference frame

        panels : dict
            Dictionary of panels from a CrystFEL geometry file

        reference_frame : bool
            If True, convert from CrystFEL reference frame to PyFAI reference frame
            If False, convert from psana reference frame to PyFAI reference frame
        """
        nmods = self.detector.n_modules
        nasics = self.detector.n_asics
        asics_shape = self.detector.asics_shape
        fs_size = self.detector.fs_size
        ss_size = self.detector.ss_size
        pixcorner = pix_pos.reshape(nmods * ss_size * asics_shape[0], fs_size * asics_shape[1], 3)
        cx, cy, cz = np.moveaxis(pixcorner, -1, 0)
        # Flattened ss dim, fs, Num corners, ZYX coord
        pyfai_fmt = np.zeros([nmods * ss_size * asics_shape[0], fs_size * asics_shape[1], 4, 3])
        for p in range(nmods):
            pname = f"p{p}"
            for asic in range(nasics):
                full_name = f"{pname}a{asic}"
                arow = asic // (nasics//2)
                acol = asic % (nasics//2)
                slab_offset = p * asics_shape[1] *ss_size
                ss_portion = slice(
                    arow * ss_size + slab_offset, (arow + 1) * ss_size + slab_offset
                )
                fs_portion = slice(acol * fs_size, (acol + 1) * fs_size)
                # Get tile vectors for ss and fs directions
                res = panels["panels"][full_name]["res"]
                ssx, ssy, ssz = np.array(panels["panels"][full_name]["ss"]) / res
                fsx, fsy, fsz = np.array(panels["panels"][full_name]["fs"]) / res
                c1x = cx[ss_portion, fs_portion]
                c1y = cy[ss_portion, fs_portion]
                c1z = cz[ss_portion, fs_portion]
                ss_units = np.array([0, 1, 1, 0])
                fs_units = np.array([0, 0, 1, 1])
                x = c1x[:, :, np.newaxis] + ss_units * ssx + fs_units * fsx
                y = c1y[:, :, np.newaxis] + ss_units * ssy + fs_units * fsy
                z = c1z[:, :, np.newaxis] + ss_units * ssz + fs_units * fsz
                # Convert to PyFAI format for detector definition
                # 0 = z along beam, 1 = dim1 (Y) fs, 2 = dim2 (X) ss
                if cframe==0:
                    # psana frame to pyFAI frame
                    # 0 = z along beam, 1 = dim1 (Y) fs, 2 = dim2 (X) ss
                    pyfai_fmt[ss_portion, fs_portion, :, 0] = z  # 3: along beam
                    pyfai_fmt[ss_portion, fs_portion, :, 1] = x  # 1: bottom to top
                    pyfai_fmt[ss_portion, fs_portion, :, 2] = y  # 2: left to right
                elif cframe==1:
                    # Lab frame to pyFAI frame
                    # 0 = z along beam, 1 = dim1 (Y) fs, 2 = dim2 (X) ss
                    pyfai_fmt[ss_portion, fs_portion, :, 0] = z  # 3: along beam
                    pyfai_fmt[ss_portion, fs_portion, :, 1] = y  # 1: bottom to top
                    pyfai_fmt[ss_portion, fs_portion, :, 2] = x  # 2: left to right
        return pyfai_fmt

class PyFAItoCrystFEL:
    """
    Class to write CrystFEL .geom geometry files from PyFAI SingleGeometry instance
    """

    def __init__(self, sg, psana_file, output_dir):
        self.sg = sg
        self.psana_file = psana_file
        self.output_dir = output_dir
        self.detector = sg.detector
        self.geom = GeometryAccess(psana_file)
        self.get_pixel_coords()

    def get_pixel_coords(self):
        """
        Get pixel coordinates for a given detector object

        Parameters
        ----------
        oname : str
            Detector object name
        oindex : int
            Detector object index
        do_tilt : bool
            Apply detector tilt
        cframe : int
            Reference frame
        """
        top = self.geom.get_top_geo()
        children = top.get_list_of_children()[0]
        X, Y, Z = self.geom.get_pixel_coords(oname=children.oname, oindex=0, do_tilt=True, cframe=CFRAME_PSANA)
        self.shape = X.shape
        self.X = X
        self.Y = Y
        self.Z = Z

    def rotation(self, X, Y, Z, angle):
        """
        Return the X, Y, Z coordinates rotated by angle
        """
        Xr = X * np.cos(angle) - Y * np.sin(angle)
        Yr = X * np.sin(angle) + Y * np.cos(angle)
        return Xr, Yr, Z
    
    def translation(self, X, Y, Z, dx, dy, dz):
        """
        Return the X, Y, Z coordinates translated by dx, dy, dz
        """
        X += dx
        Y += dy
        Z += dz
        return X, Y, Z
    
    def correct_geom(self, poni1, poni2, dist, rot1, rot2, rot3=0):
        """
        Correct the geometry based on the given parameters
        """
        X, Y, Z = self.X, self.Y, self.Z
        X, Y, Z = self.translation(X, Y, Z, -poni1*1e6, -poni2*1e6, -np.mean(self.Z)-dist*1e6)
        X, Y, Z = self.rotation(Y, Z, X, rot1)
        X, Y, Z = self.rotation(Z, X, Y, rot2)
        X, Y, Z = self.rotation(X, Y, Z, rot3)
        self.X = X
        self.Y = Y
        self.Z = Z
    
    def geometry_to_crystfel(self, output_file, zcorr_um=None):
        """
        From corrected X, Y, Z coordinates, write a CrystFEL .geom file
        """
        X, Y, Z = self.X, self.Y, self.Z
        geom = self.geom
        geom1 = geom.get_seg_geo() # GeometryObject
        seg = geom1.algo # object of the SegmentGeometry subclass

        segname = geom1.oname
        assert segname in SEGNAME_TO_PARS.keys(),\
        'segment name %s is not found in the list of implemented detectors %s'%(segname, str(SEGNAME_TO_PARS.keys()))
        valid_nsegs = SEGNAME_TO_PARS[segname]

        nsegs = int(X.size/seg.size())
        assert nsegs in valid_nsegs, 'number of %s segments %d should be in %s' % (seg.name(), nsegs, str(valid_nsegs))

        shape = (nsegs,) + seg.shape() # (nsegs, srows, scols)

        X.shape = shape
        Y.shape = shape
        Z.shape = shape

        txt = header_crystfel()
        for n in range(nsegs):
            z_um = Z[n,:]
            if zcorr_um is not None: z_um -= zcorr_um
            txt += panel_constants_to_crystfel(seg, n, X[n,:], Y[n,:], z_um)

        if output_file is not None:
            f = open(output_file,'w')
            f.write(txt)
            f.close()

class Args:
    def __init__(self, geom_file, det_type, output_file):
        self.fname = geom_file
        self.dettype = det_type
        self.ofname = output_file

class CrystFELtoPsana:
    """
    Write a psana .data file from a CrystFEL .geom file and a detector
    """
    def __init__(self, geom_file, det_type, output_file):
        args = Args(geom_file=geom_file, det_type=det_type, output_file=output_file)
        convert_crystfel_to_geometry(args)