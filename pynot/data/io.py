# -*- coding: UTF-8 -*-

"""
Input / Output functions for the DataSet class.
"""

from astropy.io import fits
import numpy as np

from pynot.data.organizer import TagDatabase, get_filter

veclen = np.vectorize(len)


def _save_database_old(database, output_fname):
    """Save file database to file."""
    collection = database.file_database
    output_strings = ['%s: %s' % item for item in collection.items()]
    # Sort the files based on their classification:
    sorted_output = sorted(output_strings, key=lambda x: x.split(':')[1])
    with open(output_fname, 'w') as output:
        output.write("\n".join(sorted_output))


def get_binning_from_hdr(hdr):
    binx = hdr['DETXBIN']
    biny = hdr['DETYBIN']
    read = hdr['FPIX']
    ccd_setup = "%ix%i_%i" % (binx, biny, read)
    return ccd_setup


def get_header_info(fname):
    primhdr = fits.getheader(fname)
    imhdr = fits.getheader(fname, 1)
    if primhdr['INSTRUME'] != 'ALFOSC_FASU':
        raise ValueError("[WARNING] - FITS file not originating from NOT/ALFOSC!")
    object = primhdr['OBJECT']
    exptime = "%.1f" % primhdr['EXPTIME']
    grism = primhdr['ALGRNM']
    slit = primhdr['ALAPRTNM']
    filter = get_filter(primhdr)
    shape = "%ix%i" % (imhdr['NAXIS1'], imhdr['NAXIS2'])
    return object, exptime, grism, slit, filter, shape


def save_database(database, output_fname):
    """Save file database to file."""
    with open(output_fname, 'w') as output:
        output.write("# PyNOT File Classification Table\n\n")
        for filetype, files in sorted(database.items()):
            output.write("# %s:\n" % filetype)
            file_list = list()
            for fname in sorted(files):
                object, exptime, grism, slit, filter, shape = get_header_info(fname)
                file_list.append((fname, filetype, object, exptime, grism, slit, filter, shape))
            file_list = np.array(file_list, dtype=str)
            header_names = ('FILENAME', 'TYPE', 'OBJECT', 'EXPTIME', 'GRISM', 'SLIT', 'FILTER', 'SHAPE')
            max_len = np.max(veclen(file_list), 0)
            max_len = np.max([max_len, [len(n) for n in header_names]], 0)
            line_fmt = "  ".join(["%-{}s".format(n) for n in max_len])
            header = line_fmt % header_names
            output.write('#' + header + '\n')
            for line in file_list:
                output.write(' ' + line_fmt % tuple(line) + '\n')
            output.write("\n")


def load_database(input_fname):
    """Load file database from file."""
    all_lines = np.loadtxt(input_fname, dtype=str, usecols=(0, 1))
    file_database = {key: val for key, val in all_lines}
    return TagDatabase(file_database)
