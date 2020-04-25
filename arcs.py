"""
Graphic Interface for spectral line identification
"""

__author__ = "Jens-Kristian Krogager"
__version__ = '1.0'

import os
import sys
import numpy as np

from astropy.io import fits


# -- Function to call from PyNOT.main
def create_pixtable(arc_images, grism_name):
    """
    arc_images : list(class:RawImage)
        List of arc images of class RawImage.

    grism_name : str
        Grism name, ex: grism4
    """
    # -- Find out which files to combine:
    arc_type = dict()
    for img in arc_images:
        if img.filetype not in arc_type.keys():
            arc_type[img.filetype] = list()
        arc_type[img.filetype].append(img)
    if 'ARC_HeNe' in arc_type.keys():
        # Remove pure He and Ne frames:
        if 'ARC_He' in list(arc_type.keys()):
            arc_type.pop('ARC_He')
        if 'ARC_Ne' in list(arc_type.keys()):
            arc_type.pop('ARC_Ne')

    arc_images = sum(arc_type.values(), [])
    # Select images with same image size:
    this_slit = arc_images[0].slit
    images_to_combine = [img for img in arc_images if img.slit == this_slit]
    median_img = np.median([im.data for im in images_to_combine], axis=0)
    if 'vert' in this_slit.lower():
        raise ValueError("Vertical slits are not supported yet! %s " % this_slit)

    arc1d = np.sum(median_img, axis=0)
