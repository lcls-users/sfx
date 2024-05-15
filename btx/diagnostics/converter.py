import numpy as np
from pyFAI.detectors import Detector
from PSCalib.UtilsConvert import geometry_to_crystfel, SEGNAME_TO_PARS
from PSCalib.GeometryAccess import GeometryAccess
from PSCalib.GlobalUtils import CFRAME_LAB, CFRAME_PSANA, info_ndarr

class PsanatoCrystFEL:
    """
    Convert a psana .data geometry file to a CrystFEL geometry format.
    """

    def __init__(self, psana_file, detector_name=None):
        self.psana_file = psana_file
        self.detector_name = detector_name

    def psana_to_crystfel(self, psana_file, output_file, cframe=CFRAME_LAB, zcorr_um=None):
        """
        Convert a psana .data geometry file to a CrystFEL geometry format.
        """
        geometry_to_crystfel(psana_file, output_file, cframe, zcorr_um)


class CrystFELtoPyFAI:
    """
    Convert a CrystFEL geometry file to a PyFAI-friendly geometry format.
    """

    def __init__(self, geom_file, detector_name=None, psana=True):
        self.geom_file = geom_file
        self.geom = self.from_CrystFEL(geom_file)
        self.pix_pos = self.get_pixel_coordinates(self.geom)
        self.corner_array = self.get_corner_array(self.pix_pos, self.geom, psana)
        self.detector = Epix10k2M()

    @staticmethod
    def from_CrystFEL(fname: str):
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

    @staticmethod
    def get_pixel_coordinates(panels):
        """
        From a parsed CrystFEL geometry file, calculate Epix10k2M pixel coordinates
        in a given reference frame

        Parameters
        ----------
        panels : dict
            Dictionary of panels from a CrystFEL geometry file
        """
        # 176 ss pixels, 192 fs pixels per asic
        # pixel size is 0.0001
        # asics are aranged 2x2 into quads (modules)
        # modules are arranged
        nmods = 16  # 16 panels
        nasics = 4  # asics per panel
        ss_size = 176
        fs_size = 192
        pix_arr = np.zeros([nmods, ss_size * 2, fs_size * 2, 3])
        for p in range(nmods):
            pname = f"p{p}"
            for asic in range(nasics):
                asicname = f"a{asic}"
                full_name = pname + asicname
                arow = asic // 2
                acol = asic % 2
                ss_portion = slice(arow * ss_size, (arow + 1) * ss_size)
                fs_portion = slice(acol * fs_size, (acol + 1) * fs_size)
                res = panels["panels"][full_name]["res"]
                corner_x = panels["panels"][full_name]["corner_x"] / res
                corner_y = panels["panels"][full_name]["corner_y"] / res
                corner_z = 0.0
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

    @staticmethod
    def get_corner_array(pix_pos, panels, reference_frame=True):
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
        nmods = 16  # 16 panels
        nasics = 4  # asics per panel
        ss_size = 176
        fs_size = 192
        pixcorner = pix_pos.reshape(nmods * ss_size * 2, fs_size * 2, 3)
        cx, cy, cz = np.moveaxis(pixcorner, -1, 0)
        # Flattened SS dim, fs, Num corners, ZYX coord
        pyfai_fmt = np.zeros([nmods * ss_size * 2, fs_size * 2, 4, 3])
        for p in range(nmods):
            pname = f"p{p}"
            for asic in range(nasics):
                full_name = f"{pname}a{asic}"
                arow = asic // 2
                acol = asic % 2
                slab_offset = p * ss_size * 2
                ss_portion = slice(
                    arow * 176 + slab_offset, (arow + 1) * ss_size + slab_offset
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
                if reference_frame:
                    # psana frame to pyFAI frame
                    # x1 <-- -x, x2 <-- -y, x3 <-- -z
                    pyfai_fmt[ss_portion, fs_portion, :, 0] = -z  # 3: along beam
                    pyfai_fmt[ss_portion, fs_portion, :, 1] = -x  # 1 : bottom to top
                    pyfai_fmt[ss_portion, fs_portion, :, 2] = -y  # 2: left to right
                else:
                    # Lab frame to pyFAI frame
                    # x1 <-- y, x2 <-- x, x3 <-- z
                    pyfai_fmt[ss_portion, fs_portion, :, 0] = z  # 3: along beam
                    pyfai_fmt[ss_portion, fs_portion, :, 1] = y  # 1: bottom to top
                    pyfai_fmt[ss_portion, fs_portion, :, 2] = x  # 2: left to right
        return pyfai_fmt


class Epix10k2M(Detector):
    """
    PyFAI Detector instance for the Epix10k2M
    """

    def __init__(
        self,
        pixel1=0.0001,
        pixel2=0.0001,
        n_modules=16,
        ss_size=192,
        fs_size=176,
        **kwargs,
    ):
        super().__init__(pixel1, pixel2, **kwargs)
        self.shape = (n_modules * 2 * fs_size, 2 * ss_size)
