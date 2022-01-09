# -*- coding: UTF-8 -*-

"""
Input / Output functions for the DataSet class.
"""

import numpy as np

from pynot.data.organizer import TagDatabase
from pynot import instrument

veclen = np.vectorize(len)


# Function to save dataset: data/io.py
def get_header_info(fname):
    hdr = instrument.get_header(fname)
    object = instrument.get_object(hdr)
    exptime = "%.1f" % instrument.get_exptime(hdr)
    grism = instrument.get_grism(hdr)
    slit = instrument.get_slit(hdr)
    filter = instrument.get_filter(hdr)
    shape = "%ix%i" % (hdr['NAXIS1'], hdr['NAXIS2'])
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
