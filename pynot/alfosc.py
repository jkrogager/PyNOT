import glob
import numpy as np
from os.path import basename, dirname, abspath
from astropy.io import fits
from astropy.table import Table

# path = '/Users/krogager/coding/PyNOT'
path = dirname(abspath(__file__))

grism_translate = {'Grism_#3': 'grism3',
                   'Grism_#4': 'grism4',
                   'Grism_#5': 'grism5',
                   'Grism_#6': 'grism6',
                   'Grism_#7': 'grism7',
                   'Grism_#8': 'grism8',
                   'Grism_#10': 'grism10',
                   'Grism_#11': 'grism11',
                   'Grism_#12': 'grism12',
                   'Grism_#14': 'grism14',
                   'Grism_#15': 'grism15',
                   'Grism_#16': 'grism16',
                   'Grism_#17': 'grism17',
                   'Grism_#18': 'grism18',
                   'Grism_#19': 'grism19',
                   'Grism_#20': 'grism20'}


slits = ['Ech_0.7', 'Ech_0.8', 'Ech_1.0', 'Ech_1.2', 'Ech_1.6',
         'Ech_1.8', 'Ech_2.2', 'Pol_1.0', 'Pol_1.4', 'Pol_1.8',
         'Slit_0.4', 'Slit_0.5', 'Slit_0.75', 'Slit_1.0',
         'Slit_1.3', 'Slit_1.8', 'Slit_2.5', 'Slit_10.0',
         'Vert_0.5', 'Vert_0.75', 'Vert_0.9', 'Vert_1.3',
         'Vert_1.8', 'Vert_10.0', 'VertOff_0.5', 'VertOff_0.8',
         'VertOff_1.0', 'VertOff_1.3', 'VertOff_1.9', 'VertOff_8.8']


# --- Data taken from: ftp://ftp.stsci.edu/cdbs/current_calspec/
standard_star_files = glob.glob(path + '/calib/std/*.dat')
standard_star_files = [basename(fname) for fname in standard_star_files]
# List of star names in lowercase:
standard_stars = [fname.strip('.dat') for fname in standard_star_files]

# Look-up table from TCS targetnames -> star names
standard_star_names = {'SP0305+261': 'HD19445',
                       'SP0644+375': 'He3',
                       'SP0946+139': 'HD84937',
                       'SP1036+433': 'Feige34',
                       'SP1045+378': 'HD93521',
                       'SP1446+259': 'BD262606',
                       'SP1550+330': 'BD332642',
                       'SP2032+248': 'Wolf1346',
                       'SP2209+178': 'BD174708',
                       'SP2317-054': 'Feige110',
                       'SP0642+021': 'Hiltner600',
                       'GD71': 'GD71',
                       'GD153': 'GD153'}


filter_table = Table.read(path + '/calib/alfosc_filters.dat', format='ascii.fixed_width')
filter_translate = {long: short for long, short in filter_table['name', 'short_name']}


def get_alfosc_header(fname):
    with fits.open(fname) as hdu:
        primhdr = hdu[0].header
        if len(hdu) > 1:
            imghdr = hdu[1].header
            primhdr.update(imghdr)
    if primhdr['INSTRUME'] != 'ALFOSC_FASU':
        print("[WARNING] - FITS file not originating from NOT/ALFOSC!")
    return primhdr


def create_pixel_array(hdr, dispaxis):
    """Load reference array from header using CRVAL, CDELT, CRPIX along dispersion axis"""
    if dispaxis not in [1, 2]:
        raise ValueError("Dispersion Axis must be 1 (X-axis) or 2 (Y-axis)!")
    p = hdr['CRVAL%i' % dispaxis]
    s = hdr['CDELT%i' % dispaxis]
    r = hdr['CRPIX%i' % dispaxis]
    N = hdr['NAXIS%i' % dispaxis]
    # -- If data are from NOT then check for binning and rescale CRPIX:
    binning = 1
    if 'DETXBIN' in hdr:
        if dispaxis == 1:
            binning = hdr['DETXBIN']
        else:
            binning = hdr['DETYBIN']
    pix_array = p + s*(np.arange(N) - (r/binning - 1))
    return pix_array
