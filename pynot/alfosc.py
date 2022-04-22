# pynot-instrument-module
"""
Instrument definitions for NOT/ALFOSC
"""
import numpy as np
import os
import datetime
from os.path import dirname, abspath
from astropy.io import fits
from astropy.table import Table

name = 'alfosc'

# absolute path from code directory [DO NOT CHANGE]
path = dirname(abspath(__file__))

# path to classification rules:
rulefile = os.path.join(path, 'data/alfosc.rules')

# path to extinction table:
extinction_fname = os.path.join(path, 'calib/lapalma.ext')

# path to filter names:
filter_table_fname = os.path.join(path, 'calib/alfosc_filters.dat')

# Automatically assign arc-line list:
def auto_linelist(hdr):
    if hdr['CLAMP2'] == 1 or hdr['CLAMP1'] == 1:
        # Load HeNe linelist
        linelist_fname = os.path.join(path, 'calib/HeNe_linelist.dat')
    elif hdr['CLAMP4'] == 1:
        # Load ThAr linelist:
        linelist_fname = os.path.join(path, 'calib/ThAr_linelist.dat')
    else:
        linelist_fname = ''
    return linelist_fname

grism_translate = {'Grism_#3' : 'al-gr3',
                   'Grism_#4' : 'al-gr4',
                   'Grism_#5' : 'al-gr5',
                   'Grism_#6' : 'al-gr6',
                   'Grism_#7' : 'al-gr7',
                   'Grism_#8' : 'al-gr8',
                   'Grism_#10': 'al-gr10',
                   'Grism_#11': 'al-gr11',
                   'Grism_#12': 'al-gr12',
                   'Grism_#14': 'al-gr14',
                   'Grism_#15': 'al-gr15',
                   'Grism_#16': 'al-gr16',
                   'Grism_#17': 'al-gr17',
                   'Grism_#18': 'al-gr18',
                   'Grism_#19': 'al-gr19',
                   'Grism_#20': 'al-gr20'}


slits = ['ech_0.7', 'ech_0.8', 'ech_1.0', 'ech_1.2', 'ech_1.6',
         'ech_1.8', 'ech_2.2', 'pol_1.0', 'pol_1.4', 'pol_1.8',
         'slit_0.4', 'slit_0.5', 'slit_0.75', 'slit_1.0',
         'slit_1.3', 'slit_1.8', 'slit_2.5', 'slit_10.0',
         'vert_0.5', 'vert_0.75', 'vert_0.9', 'vert_1.3',
         'vert_1.8', 'vert_10.0', 'vertoff_0.5', 'vertoff_0.8',
         'vertoff_1.0', 'vertoff_1.3', 'vertoff_1.9', 'vertoff_8.8']


filter_table = Table.read(filter_table_fname, format='ascii.fixed_width')
filter_translate = {long: short for long, short in filter_table['name', 'short_name']}


# Helper functions
def get_header(fname):
    with fits.open(fname) as hdu:
        primhdr = hdu[0].header
        if len(hdu) > 1:
            imghdr = hdu[1].header
            primhdr.update(imghdr)
    if primhdr['INSTRUME'] != 'ALFOSC_FASU':
        print("[WARNING] - FITS file not originating from NOT/ALFOSC!")
    return primhdr


def create_pixel_array(hdr, axis):
    """Load reference array from header using CRVAL, CDELT, CRPIX along dispersion axis"""
    if axis not in [1, 2]:
        raise ValueError("Axis must be 1 (X-axis) or 2 (Y-axis)!")
    p = hdr['CRVAL%i' % axis]
    s = hdr['CDELT%i' % axis]
    r = hdr['CRPIX%i' % axis]
    N = hdr['NAXIS%i' % axis]
    # -- If data are from NOT then check for binning and rescale CRPIX:

    if axis == 1:
        binning = hdr['DETXBIN']
    else:
        binning = hdr['DETYBIN']
    pix_array = p + s*(np.arange(N) - (r/binning - 1))
    return pix_array


def overscan():
    prescan_x = 50
    overscan_x = 50
    prescan_y = 0
    overscan_y = 50
    return (prescan_x, overscan_x, prescan_y, overscan_y)

def get_ccd_extent():
    return (2148, 2102)

def get_detector_arrays(hdr):
    det_window = hdr['DETWIN1']
    xy_ranges = det_window.replace('[', '').replace(']', '')
    xrange, yrange = [list(map(int, minmax.split(':'))) for minmax in xy_ranges.split(',')]
    xmin, xmax = xrange
    ymin, ymax = yrange
    xbin = get_binx(hdr)
    ybin = get_biny(hdr)
    X = np.arange(xmin, xmax+1, xbin)
    Y = np.arange(ymin, ymax+1, ybin)
    return X, Y


def get_binning(fname):
    hdr = fits.getheader(fname)
    binx = hdr['DETXBIN']
    biny = hdr['DETYBIN']
    read = hdr['FPIX']
    ccd_setup = "%ix%i_%i" % (binx, biny, read)
    return ccd_setup


def get_binning_from_hdr(hdr):
    binx = hdr['DETXBIN']
    biny = hdr['DETYBIN']
    read = hdr['FPIX']
    ccd_setup = "%ix%i_%i" % (binx, biny, read)
    return ccd_setup

def get_saturation_level():
    # Get Saturation Level in Electrons
    return 113500

def get_binx(hdr):
    return hdr.get('DETXBIN')

def get_biny(hdr):
    return hdr.get('DETYBIN')

def set_binx(hdr, val):
    hdr['DETXBIN'] = val
    return hdr

def set_biny(hdr, val):
    hdr['DETYBIN'] = val
    return hdr

def get_filter_raw(hdr):
    filter = 'Open'
    for keyword in ['FAFLTNM', 'FBFLTNM', 'ALFLTNM']:
        if 'open' in hdr[keyword].lower():
            pass
        else:
            filter = hdr[keyword]
            break
    if '  ' in filter:
        filter = filter.replace('  ', ' ')
    return filter

def get_filter(hdr):
    raw_filter_name = get_filter_raw(hdr)
    if raw_filter_name in filter_translate:
        return filter_translate.get(raw_filter_name)
    else:
        return 'Open'

def get_grism(hdr):
    raw_grism = hdr['ALGRNM']
    if raw_grism in grism_translate:
        return grism_translate[raw_grism]
    else:
        return 'Open'

def get_slit(hdr):
    return hdr['ALAPRTNM'].lower()

def get_airmass(hdr):
    """Return the average airmass at mid-exposure"""
    return hdr.get('AIRMASS')

def get_exptime(hdr):
    # if EXPTIME is in the header, this should always be used!
    return hdr.get('EXPTIME')

def get_object(hdr):
    return hdr.get('OBJECT')

def get_target_name(hdr):
    return hdr.get('TCSTGT')

def get_rotpos(hdr):
    return hdr.get('ROTPOS', 0)

def get_date(hdr):
    return hdr['DATE-OBS']

def get_mjd(hdr):
    """Input Date String in ISO format as in ALFOSC header: '2016-08-01T16:03:57.796'"""
    date_str = get_date(hdr)
    date = datetime.datetime.fromisoformat(date_str)
    mjd_0 = datetime.datetime(1858, 11, 17)
    dt = date - mjd_0
    mjd = dt.days + dt.seconds/(24*3600.)
    return mjd

def get_observing_mode(hdr):
    """Determine the observing mode (either spectroscopy or imaging)"""
    if hdr['OBS_MODE'].strip().upper() == 'SPECTROSCOPY':
        return 'SPECTROSCOPY'
    elif hdr['OBS_MODE'].strip().upper() == 'IMAGING':
        return 'IMAGING'
    else:
        return None

def get_ob_name(hdr):
    ob_id = hdr.get('FILENAME')
    if not ob_id:
        date = hdr['DATE-OBS']
        date = date.split('.')[0]
        ob_id = date.replace(':', '_')
    else:
        ob_id = ob_id.split('.')[0]
    return ob_id

def get_dispaxis(hdr):
    slit_name = get_slit(hdr)
    if 'vert' in slit_name:
        return 1
    elif 'slit' in slit_name:
        return 2
    else:
        return None

def get_gain(hdr):
    if 'CCDNAME' in hdr and hdr['CCDNAME'] == 'CCD14':
        hdr['GAIN'] = 0.16
    return hdr.get('GAIN', 0.)

def get_readnoise(hdr):
    return hdr.get('RDNOISE')
