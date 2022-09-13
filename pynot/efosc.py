# pynot-instrument-module
"""
Instrument definitions for NTT/EFOSC2
"""

import numpy as np
import os
from os.path import dirname, abspath
from astropy.io import fits
from astropy.table import Table


# Define header keyword to use as object name
# OBJECT is not always reliable, TCS target name is more robust
target_keyword = 'OBJECT'
# or
# target_keyword = 'ESO OBS TARG NAME'

name = 'efosc2'

path = dirname(abspath(__file__))

# path to classification rules:
rulefile = os.path.join(path, 'data/efosc.rules')

# path to extinction table:
extinction_fname = os.path.join(path, 'calib/lasilla.ext')

# path to filter names:
filter_table_fname = os.path.join(path, 'calib/efosc_filters.dat')
# List grisms, slits and filters, and define filter translations if needed.

grism_translate = {'Gr#1': 'ef-gr1',
                   'Gr#2': 'ef-gr2',
                   'Gr#3': 'ef-gr3',
                   'Gr#4': 'ef-gr4',
                   'Gr#5': 'ef-gr5',
                   'Gr#6': 'ef-gr6',
                   'Gr#7': 'ef-gr7',
                   'Gr#8': 'ef-gr8',
                   'Gr#9': 'ef-gr9',
                   'Gr#10': 'ef-gr10',
                   'Gr#11': 'ef-gr11',
                   'Gr#12': 'ef-gr12',
                   'Gr#13': 'ef-gr13',
                   'Gr#14': 'ef-gr14',
                   'Gr#15': 'ef-gr15',
                   'Gr#16': 'ef-gr16',
                   'Gr#17': 'ef-gr17',
                   'Gr#18': 'ef-gr18',
                   'Gr#19': 'ef-gr19',
                   'Gr#20': 'ef-gr20',
                   'Free': 'free'}

slit_translate = {'slit#1.0': 'slit_1.0',
                  'slit#1.2': 'slit_1.2',
                  'slit#1.5': 'slit_1.5',
                  'slit#2.0': 'slit_2.0',
                  'slit#0.7': 'slit_0.7',
                  'slit#0.5': 'slit_0.5',
                  'slit#0.3': 'slit_0.3',
                  'slit#5.0': 'slit_5.0',
                  'slit#10': 'slit_10',
                  'free': 'free',
                  'holes_mask': 'holes_mask',
                  }

# Automatically assign arc-line list:
def auto_linelist(hdr):
    # Load ThAr linelist:
    return os.path.join(path, 'calib/HeAr_linelist.dat')

filter_table = Table.read(filter_table_fname, format='ascii.fixed_width')
filter_translate = {long: short for long, short in filter_table['name', 'short_name']}


def get_header(fname):
    with fits.open(fname) as hdu:
        primhdr = hdu[0].header
    if primhdr['INSTRUME'] != 'EFOSC':
        print("[WARNING] - FITS file not originating from NTT/EFOSC!")
    if primhdr['ESO DPR TECH'] == 'SPECTRUM' and primhdr['ESO DPR CATG'] == 'SCIENCE':
        if 'PYNOTCOR' in primhdr:
            return primhdr
        primhdr.remove('CD1_1')
        primhdr.remove('CD1_2')
        primhdr.remove('CD2_1')
        primhdr.remove('CD2_2')
        xbin = get_binx(primhdr)
        ybin = get_biny(primhdr)
        primhdr['CRVAL1'] = 1.
        primhdr['CRVAL2'] = 1.
        primhdr['CRPIX1'] = 1./xbin
        primhdr['CRPIX2'] = 1./ybin
        primhdr['CDELT1'] = 1.*xbin
        primhdr['CDELT2'] = 1.*ybin
        primhdr['CTYPE1'] = 'PIXEL'
        primhdr['CTYPE2'] = 'PIXEL'
        primhdr['PYNOTCOR'] = 'AUTO CORRECT AXES INFO'
    return primhdr


# Function for data/io.py writing of classification results:
def get_header_info(fname):
    primhdr = fits.getheader(fname)
    if primhdr['INSTRUME'] != 'EFOSC':
        raise ValueError("[WARNING] - FITS file not originating from NTT/EFOSC!")
    object_name = primhdr['OBJECT']
    exptime = "%.1f" % primhdr['EXPTIME']
    grism = get_grism(primhdr)
    slit = get_slit(primhdr)
    filt = get_filter(primhdr)
    shape = "%ix%i" % (primhdr['NAXIS1'], primhdr['NAXIS2'])
    return object_name, exptime, grism, slit, filt, shape


def create_pixel_array(hdr, axis):
    """Load reference array from header using CRVAL, CDELT, CRPIX along dispersion axis"""
    if axis not in [1, 2]:
        raise ValueError("Dispersion Axis must be 1 (X-axis) or 2 (Y-axis)!")
    p = hdr['CRVAL%i' % axis]
    if 'CDELT%i' % axis in hdr:
        s = hdr['CDELT%i' % axis]
    else:
        s = hdr['CD%i_%i' % (axis, axis)]
    r = hdr['CRPIX%i' % axis]
    N = hdr['NAXIS%i' % axis]
    pix_array = p + s*(np.arange(N) - (r - 1))
    return pix_array


def get_binning(fname):
    hdr = fits.getheader(fname)
    ccd_setup = get_binning_from_hdr(hdr)
    return ccd_setup


def get_binning_from_hdr(hdr):
    binx = hdr['ESO DET WIN1 BINX']
    biny = hdr['ESO DET WIN1 BINY']
    read = hdr['ESO DET READ MODE'].strip()
    ccd_setup = "%ix%i_%s" % (binx, biny, read)
    return ccd_setup


def get_filter(hdr):
    raw_filter_name = get_filter_raw(hdr)
    if raw_filter_name in filter_translate:
        return filter_translate.get(raw_filter_name)
    else:
        filter_name = raw_filter_name.strip().replace('#', '_')
        filter_name = filter_name.replace('  ', ' ')
        filter_name = filter_name.replace(' ', '_')
        filter_name = filter_name.replace("'", '-')
        filter_name = filter_name.replace('"', '-')
        return filter_name

def get_filter_raw(hdr):
    return hdr['ESO INS FILT1 NAME']

def get_grism(hdr):
    raw_grism = hdr['ESO INS GRIS1 NAME']
    if raw_grism in grism_translate:
        return grism_translate[raw_grism]
    else:
        return 'Free'

def get_slit(hdr):
    raw_slit_name = hdr['ESO INS SLIT1 NAME'].lower()
    if raw_slit_name in slit_translate:
        return slit_translate[raw_slit_name]
    else:
        return raw_slit_name

def get_airmass(hdr):
    """Return the average airmass at mid-exposure"""
    airm_start = hdr['ESO TEL AIRM START']
    airm_end = hdr['ESO TEL AIRM END']
    airmass = 0.5*(airm_start + airm_end)
    return airmass

def get_date(hdr):
    return hdr['DATE-OBS']

###################################################################

def overscan():
    # Note slight change in definition for x-axis
    prescan_x = 0
    overscan_x = 12
    prescan_y = 0
    overscan_y = 12
    return (prescan_x, overscan_x, prescan_y, overscan_y)

def get_ccd_extent():
    return (2060, 2060)

def get_saturation_level():
    # Return Saturation Level in Electrons
    return 88000

def get_detector_arrays(hdr):
    nx = hdr['NAXIS1']
    ny = hdr['NAXIS2']
    xbin = get_binx(hdr)
    ybin = get_biny(hdr)
    X = np.arange(1*xbin, nx*xbin+1, xbin)
    Y = np.arange(1*ybin, ny*ybin+1, ybin)
    return X, Y

def get_binx(hdr):
    return hdr.get('ESO DET WIN1 BINX')

def get_biny(hdr):
    return hdr.get('ESO DET WIN1 BINY')

def set_binx(hdr, val):
    hdr['ESO DET WIN1 BINX'] = val
    return hdr

def set_biny(hdr, val):
    hdr['ESO DET WIN1 BINY'] = val
    return hdr

def get_exptime(hdr):
    # if EXPTIME is in the header, this should always be used!
    return hdr.get('EXPTIME')

def get_object(hdr):
    return hdr.get('OBJECT')

def get_target_name(hdr):
    return hdr.get('OBJECT')

def get_rotpos(hdr):
    return hdr.get('ESO ADA POSANG', 0)

def get_mjd(hdr):
    return hdr['MJD-OBS']

def get_observing_mode(hdr):
    """Determine the observing mode (either spectroscopy or imaging)"""
    if hdr['ESO DPR TECH'] == 'SPECTRUM':
        return 'SPECTROSCOPY'
    elif hdr['ESO DPR TECH'] == 'IMAGE':
        return 'IMAGING'
    else:
        return None

def get_ob_name(hdr):
    # ob_id = hdr.get('ESO OBS ID')
    # if not ob_id:
    #     date = hdr['DATE-OBS']
    #     date = date.split('.')[0]
    #     ob_id = date.replace(':', '_')
    # else:
    #     ob_id = str(ob_id)
    date = hdr['DATE-OBS']
    date = date.split('.')[0]
    ob_id = date.replace(':', '_')
    return ob_id

def get_dispaxis(hdr):
    return 2

def get_gain(hdr):
    return hdr.get('ESO DET OUT1 GAIN')

def get_readnoise(hdr):
    return hdr.get('ESO DET OUT1 RON')
