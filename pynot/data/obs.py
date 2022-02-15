
from collections import defaultdict
import numpy as np
import os

from pynot.data import io
from pynot.data import organizer as do

output_base_phot = 'imaging'
output_base_spec = 'spectra'

class OBDatabase:
    def __init__(self, fname):
        if os.path.exists(fname):
            data = np.loadtxt(fname, dtype=str, delimiter=':')
            if len(data.shape) == 1:
                data = np.array([data])
            self.data = {obid.strip(): status.strip() for obid, status in data}
        else:
            self.data = {}
        self.fname = fname

    def update_spectra(self, scifile_dict):
        """Expects a nested dictionary: {target1: {setup1: [], setup2: []}, ...}"""
        for target_name, frames_per_setup in scifile_dict.items():
            for insID, frames in frames_per_setup.items():
                for obnum, sci_img in enumerate(frames, 1):
                    # Create working directory:
                    obID = 'ob%i' % obnum
                    ob_path = os.path.join(output_base_spec, sci_img.target_name, insID, obID)
                    if ob_path not in self.data:
                        self.data[ob_path] = ''
        self.save()

    def update_imaging(self, scifile_dict):
        """Expects a nested dictionary: {target1: {setup1: [], setup2: []}, ...}"""
        for target_name, frames_per_setup in scifile_dict.items():
            for insID, frames in frames_per_setup.items():
                ob_path = os.path.join(output_base_phot, target_name, insID)
                if ob_path not in self.data:
                    self.data[ob_path] = ''
        self.save()

    def save(self):
        with open(self.fname, 'w') as output:
            output.write("# PyNOT OB Database\n#\n")
            for obid, status in self.data.items():
                output.write("%s : %s\n" % (obid, status))

    def update(self, ob_path, status):
        self.data[ob_path] = status
        self.save()


def update_ob_database(dataset_fname):
    obd_fname = os.path.splitext(dataset_fname)[0] + '.obd'
    database = io.load_database(dataset_fname)
    obdb = OBDatabase(obd_fname)
    print(" - Updating OB database: %s" % obd_fname)

    # Check Spectroscopy:
    object_filelist = database['SPEC_OBJECT']
    object_images = list(map(do.RawImage, object_filelist))

    # Organize the science files according to target and instrument setup (insID)
    science_frames = defaultdict(lambda: defaultdict(list))
    for sci_img in object_images:
        filt_name = sci_img.filter
        insID = "%s_%s" % (sci_img.grism, sci_img.slit.replace('_', ''))
        if filt_name.lower() not in ['free', 'open', 'none']:
            insID = "%s_%s" % (insID, filt_name)
        science_frames[sci_img.target_name][insID].append(sci_img)

    for target_name, frames_per_setup in science_frames.items():
        for insID, frames in frames_per_setup.items():
            for obnum, sci_img in enumerate(frames, 1):
                # Create working directory:
                obID = 'ob%i' % obnum
                ob_name = os.path.join(sci_img.target_name, insID, obID)
                flux1d = os.path.join(ob_name, 'FLUX1D_%s.fits' % sci_img.target_name)
                if os.path.exists(flux1d):
                    if ob_name in obdb.data:
                        if obdb.data[ob_name] not in ['REDO', 'SKIP']:
                            obdb.data[ob_name] = 'DONE'
                    else:
                        obdb.data[ob_name] = 'DONE'
                else:
                    obdb.data[ob_name] = ''
    obdb.save()

    # Check Photometry:
    object_filelist = database['IMG_OBJECT']
    object_images = list(map(do.RawImage, object_filelist))

    # Organize the science files according to target and instrument setup (insID)
    science_phot_frames = defaultdict(lambda: defaultdict(list))
    for sci_img in object_images:
        science_phot_frames[sci_img.target_name][sci_img.filter].append(sci_img)

    for target_name, frames_per_setup in science_phot_frames.items():
        for insID, frames in frames_per_setup.items():
            # Create working directory:
            ob_name = os.path.join(output_base_phot, target_name, insID)
            reduced_image = os.path.join(os.path.join(output_base_phot, target_name), '%s_%s.fits' % (target_name, insID))
            if os.path.exists(reduced_image):
                if ob_name in obdb.data:
                    if obdb.data[ob_name] not in ['SKIP', 'REDO']:
                        obdb.data[ob_name] = 'DONE'
                else:
                    obdb.data[ob_name] = 'DONE'
            else:
                obdb.data[ob_name] = ''
    obdb.save()

    print(" - Done!")
