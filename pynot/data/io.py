# -*- coding: UTF-8 -*-

"""
Input / Output functions for the DataSet class.
"""

from astropy.io import fits
import numpy as np

from pynot.data.organizer import TagDatabase
from pynot import instrument

veclen = np.vectorize(len)


# Function to save dataset: data/io.py
def get_header_info(fname):
    hdr = fits.getheader(fname)
    if hdr['INSTRUME'] == 'PyNOT':
        object = hdr['OBJECT']
        exptime = hdr['EXPTIME']
        grism = hdr['GRISM']
        slit = hdr['SLIT']
        filter = '...'
        shape = "..."
    else:
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
        output.write("## PyNOT File Classification Table\n\n")
        for filetype, files in sorted(database.items()):
            output.write("## %s:\n" % filetype)
            file_list = list()
            files += database.inactive.get(filetype, [])
            sorted_files = sorted(files, key=lambda x: x[1:] if len(x) > 1 else x)
            for fname in sorted_files:
                try:
                    if fname[0] == '#':
                        object, exptime, grism, slit, filter, shape = get_header_info(fname[1:])
                    else:
                        object, exptime, grism, slit, filter, shape = get_header_info(fname)
                except FileNotFoundError:
                    print("[WARNING] - File not found: %s" % fname)
                    continue
                except Exception:
                    print("[WARNING] - Problem reading header information: %s" % fname)
                    object, exptime, grism, slit, filter, shape = '-', '-', '-', '-', '-', '-'
                file_list.append((fname, filetype, object, exptime, grism, slit, filter, shape))
            file_list = np.array(file_list, dtype=str)
            if len(file_list) == 0:
                continue
            header_names = ('FILENAME', 'TYPE', 'OBJECT', 'EXPTIME', 'GRISM', 'SLIT', 'FILTER', 'SHAPE')
            max_len = np.max(veclen(file_list), 0)
            max_len = np.max([max_len, [len(n) for n in header_names]], 0)
            line_fmt = "  ".join(["%-{}s".format(n) for n in max_len])
            header = line_fmt % header_names
            output.write('## ' + header + '\n')
            for line in file_list:
                if line[0].startswith('#'):
                    output.write(line_fmt % tuple(line) + '\n')
                else:
                    output.write(' ' + line_fmt % tuple(line) + '\n')
            output.write("\n")


def load_database(input_fname):
    """Load file database from file."""
    all_lines = np.loadtxt(input_fname, dtype=str, usecols=(0, 1), comments='##')
    # file_database = {key: val for key, val in all_lines}
    # inactive_files = {key: val for key, val in all_lines}
    file_database = {}
    inactive_files = {}
    for fname, ftype in all_lines:
        if fname[0] == '#':
            inactive_files[fname] = ftype
        else:
            file_database[fname] = ftype
    return TagDatabase(file_database, inactive_files)
