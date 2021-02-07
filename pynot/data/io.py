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


def get_header_info(fname):
    primhdr = fits.getheader(fname)
    if primhdr['INSTRUME'] != 'ALFOSC_FASU':
        raise ValueError("[WARNING] - FITS file not originating from NOT/ALFOSC!")
    object = primhdr['OBJECT']
    exptime = "%.1f" % primhdr['EXPTIME']
    grism = primhdr['ALGRNM']
    slit = primhdr['ALAPRTNM']
    filter = get_filter(primhdr)
    return object, exptime, grism, slit, filter


def save_database(database, output_fname):
    """Save file database to file."""
    with open(output_fname, 'w') as output:
        output.write("# PyNOT File Classification Table\n\n")
        for filetype, files in sorted(database.items()):
            output.write("# %s:\n" % filetype)
            file_list = list()
            for fname in sorted(files):
                object, exptime, grism, slit, filter = get_header_info(fname)
                file_list.append((fname, filetype, object, exptime, grism, slit, filter))
            file_list = np.array(file_list, dtype=str)
            max_len = np.max(veclen(file_list), 0)
            line_fmt = "  ".join(["%-{}s".format(n) for n in max_len])
            header = line_fmt % ('FILENAME', 'TYPE', 'OBJECT', 'EXPTIME', 'GRISM', 'SLIT', 'FILTER')
            output.write('#' + header + '\n')
            for line in file_list:
                output.write(' ' + line_fmt % tuple(line) + '\n')
            output.write("\n")


def load_database(input_fname):
    """Load file database from file."""
    all_lines = np.loadtxt(input_fname, dtype=str, usecols=(0, 1))
    file_database = {key: val for key, val in all_lines}
    return TagDatabase(file_database)
