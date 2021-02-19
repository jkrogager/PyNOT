# -*- coding: UTF-8 -*-

import numpy as np
import os
import sys
from glob import glob
from astropy.io import fits

from pynot.alfosc import get_mjd, get_binning_from_hdr, get_filter, get_header, lookup_std_star

# -- use os.path
code_dir = os.path.dirname(os.path.abspath(__file__))
v_file = os.path.join(os.path.split(code_dir)[0], 'VERSION')
with open(v_file) as version_file:
    __version__ = version_file.read().strip()

std_fname = os.path.join(os.path.split(code_dir)[0], 'calib/std/namelist.txt')
calib_namelist = np.loadtxt(std_fname, dtype=str)
calib_names = calib_namelist[:, 1]

alfosc_rulefile = os.path.join(code_dir, 'alfosc.rules')

def occurence(inlist):
    """
    Counts the number of occurences of each number in the input list.

    Returns
    =======
    single_values : list
        List of tuples (v, N(v)), where v is any value in `inlist`
        and N(v) is the number of occurences of this value.
    """
    inlist = np.array(inlist)
    n = np.zeros(len(inlist), dtype=bool)
    single_values = list()
    while np.sum(n) < len(inlist):
        e0 = inlist[~n][0]
        Ne = np.sum(inlist == e0)
        n += inlist == e0
        single_values.append((e0, Ne))
    return single_values


# def get_mjd(date_str):
#     """Input Date String in ISO format as in ALFOSC header: '2016-08-01T16:03:57.796'"""
#     date = datetime.datetime.fromisoformat(date_str)
#     mjd_0 = datetime.datetime(1858, 11, 17)
#     dt = date - mjd_0
#     mjd = dt.days + dt.seconds/(24*3600.)
#     return mjd
#
#
# def get_binning(fname):
#     hdr = fits.getheader(fname)
#     binx = hdr['DETXBIN']
#     biny = hdr['DETYBIN']
#     read = hdr['FPIX']
#     ccd_setup = "%ix%i_%i" % (binx, biny, read)
#     return ccd_setup
#
#
# def get_binning_from_hdr(hdr):
#     binx = hdr['DETXBIN']
#     biny = hdr['DETYBIN']
#     read = hdr['FPIX']
#     ccd_setup = "%ix%i_%i" % (binx, biny, read)
#     return ccd_setup
#
#
# def get_filter(hdr):
#     filter = 'Open'
#     for keyword in ['FAFLTNM', 'FBFLTNM', 'ALFLTNM']:
#         if 'open' in hdr[keyword].lower():
#             pass
#         else:
#             filter = hdr[keyword]
#             break
#     if '  ' in filter:
#         filter = filter.replace('  ', ' ')
#     return filter


def match_date(files, date_mjd):
    """
    Return subset of files which were observed at the given date.
    Input
    =====

    files : list(str)
        List of filenames to be matched to the given date

    date_mjd : float
        Modified Julian date to match, MJD
    """
    matches = list()
    for fname in files:
        hdr = fits.getheader(fname)
        mjd = get_mjd(hdr['DATE-OBS'])
        if int(mjd) == int(date_mjd):
            matches.append(fname)

    return matches


def group_calibs_by_date(file_list, lower=0.01, upper=0.99):
    """ Grouping files observed from the same day. """
    file_list = np.array(file_list, dtype=str)
    date_sorted = dict()
    mjd = np.zeros(len(file_list))
    n = np.zeros(len(file_list), dtype=bool)
    N = len(file_list)
    for i, fname in enumerate(file_list):
        hdr = fits.getheader(fname)
        mjd[i] = get_mjd(hdr['DATE-OBS'])

    while sum(n) < N:
        m0 = mjd[~n][0]
        this_date = (mjd - np.floor(m0) > lower) * (mjd - np.floor(m0) < upper)
        date_string = str(int(m0) - 1)
        date_sorted[date_string] = list(file_list[this_date])
        n = n + this_date

    return date_sorted


def parse_value(val):
    val = val.strip()
    if '.' in val:
        if val.replace('.', '').isnumeric():
            new_val = float(val)
    elif val.isnumeric():
        new_val = int(val)
    else:
        new_val = val
    return new_val


def get_unclassified_files(file_list, database):
    missing_files = list()
    for fname in file_list:
        if fname not in database.file_database:
            missing_files.append(fname)
    return missing_files


def classify_file(fname, rules):
    """
    Classify input FITS file according to the set of `rules`, a list of string conditions
    for header keywords. Each rule correspond to one filetype, e.g. BIAS, OBJECT
    Returns a list of rules that
    """
    msg = list()
    try:
        h = fits.getheader(fname)
    except OSError:
        msg.append("File could not be opened: %s" % fname)
        return [], msg

    fileroot = fname.split('/')[-1]
    if fileroot != h['FILENAME']:
        msg.append("Filename does not match FILENAME in header: %s" % fname)
        return [], msg

    matches = list()
    for rule in rules:
        rule = rule.strip()
        if ('#' in rule) or (len(rule) == 0):
            continue

        ftype = rule.split(':')[0].strip()
        all_conditions = rule.split(':')[1].split(' and ')
        criteria = list()
        for cond in all_conditions:
            if '==' in cond:
                key, val = cond.split('==')
                if key not in h:
                    return [], msg
                if 'open' in val.lower():
                    criteria.append('open' in h[key].lower())
                elif 'closed' in val.lower():
                    criteria.append('closed' in h[key].lower())
                else:
                    val = parse_value(val)
                    criteria.append(val == h[key])

            elif '!=' in cond:
                key, val = cond.split('!=')
                if key not in h:
                    return [], msg
                if 'open' in val.lower():
                    criteria.append('open' not in h[key].lower())
                elif 'closed' in val.lower():
                    criteria.append('closed' not in h[key].lower())
                else:
                    val = parse_value(val)
                    criteria.append(val != h[key])

            elif '>' in cond:
                key, val = cond.split('>')
                if key not in h:
                    return [], msg
                val = parse_value(val)
                criteria.append(h[key] > val)

            elif '<' in cond:
                key, val = cond.split('<')
                if key not in h:
                    return [], msg
                val = parse_value(val)
                criteria.append(h[key] < val)

            else:
                raise ValueError("Invalid condition in rule: %s" % rule)

        if np.all(criteria):
            matches.append(ftype)

    if len(matches) == 1:
        ftype = matches[0]
        if ftype == 'IMG_OBJECT':
            # If no filter is defined and CCD readout is fast, the image is most likely an acquisition image
            img_filter = get_filter(h)
            if (img_filter == '' or img_filter == 'Open'):
                if 'FPIX' in h:
                    if h['FPIX'] > 200:
                        matches = ['ACQ_IMG']

        elif ftype == 'SPEC_OBJECT':
            if 'TCSTGT' in h:
                star_target = h['TCSTGT']
                star_name = lookup_std_star(star_target)
                if star_name:
                    matches = ['SPEC_FLUX-STD']

    elif len(matches) == 0:
        msg.append("No classification matched the file: %s" % fname)
    else:
        msg.append("More than one classification was found for file: %s" % fname)
        msg.append(", ".join(matches))

    return matches, msg


def classify(data_in, rule_file=alfosc_rulefile, progress=True):
    """
    The input can be a single .fits file, a string given the path to a directory,
    a list of .fits files, or a list of directories.
    Classify given input 'files' using the rules defined in 'alfosc.rules'.
    Returns
    'Data_Type' containing the classification for the pipeline
                recipes.
    """
    msg = list()

    # Determine the input type:
    if isinstance(data_in, str):
        if os.path.isfile(data_in):
            files = [data_in]
        elif os.path.isdir(data_in):
            files = glob(data_in+'/*.fits')

    elif isinstance(data_in, list):
        if os.path.isfile(data_in[0]):
            files = data_in
        elif os.path.isdir(data_in[0]):
            files = []
            for path in data_in:
                files += glob(path+'/*.fits')
    else:
        raise ValueError("Input must be a string or a list of strings")

    data_types = dict()
    not_classified_files = list()

    if not os.path.exists(rule_file):
        raise FileNotFoundError("ALFOSC ruleset could not be found: %s" % rule_file)

    with open(rule_file) as rulebook:
        rules = rulebook.readlines()

    if progress:
        print("")
        print(" Classifying files: ")

    for num, fname in enumerate(files):
        matches, output_msg = classify_file(fname, rules)

        if len(matches) == 1:
            ftype = matches[0]
            data_types[fname] = ftype

        elif len(matches) == 0:
            msg.append(" [ERROR]  - " + output_msg[0])
            not_classified_files.append(fname)

        else:
            msg.append(" [ERROR]  - " + output_msg[0])
            msg.append("            " + output_msg[1])

        if progress:
            sys.stdout.write("\r  %6.2f%%" % (100.*(num+1)/len(files)))
            sys.stdout.flush()

    msg.append("")
    msg.append("          - Classification finished.")
    msg.append("          - Successfully classified %i out of %i files." % (len(data_types.keys()), len(files)))
    msg.append("")
    if len(files) != len(data_types.keys()):
        msg.append("[WARNING] - Files not classified:")
        for item in not_classified_files:
            msg.append("          - %s" % item)
    msg.append("")
    output_msg = "\n".join(msg)

    database = TagDatabase(data_types)

    return database, output_msg


def reclassify(data_in, database, **kwargs):
    # Determine the input type:
    if isinstance(data_in, str):
        if os.path.isfile(data_in):
            files = [data_in]
        elif os.path.isdir(data_in):
            files = glob(data_in+'/*.fits')

    elif isinstance(data_in, list):
        if os.path.isfile(data_in[0]):
            files = data_in
        elif os.path.isdir(data_in[0]):
            files = []
            for path in data_in:
                files += glob(path+'/*.fits')
    else:
        raise ValueError("Input must be a string or a list of strings")

    missing_files = get_unclassified_files(files, database)

    if len(missing_files) == 0:
        output_msg = "No new files to classify!"
        new_database = database
    else:
        new_database, output_msg = classify(missing_files, **kwargs)
        new_database = database + new_database

    return new_database, output_msg



def write_report(collection, output=''):
    """
    Write a report of a given data collection from the `classify` function

    If `output` is given, the report will be saved to this filename.
    """
    # Sort the files by file-type:
    tags = TagDatabase(collection)

    report = "# Data Report for ALFOSC\n"
    for tag in sorted(tags.keys()):
        report += "\n - %s:\n" % tag
        for filename in sorted(tags[tag]):
            report += " %s\n" % filename
    if output == '':
        print(report)
    else:
        with open(output, 'w') as output:
            output.write(report)


class RawImage(object):
    def __init__(self, fname, filetype=None):
        self.filename = fname
        self.data = fits.getdata(fname)
        self.shape = self.data.shape
        self.filetype = filetype
        # Merge Primary and Image headers:
        self.header = get_header(fname)
        # primhdr = fits.getheader(fname, 0)
        # imghdr = fits.getheader(fname, 1)
        # primhdr.update(imghdr)
        # self.header = primhdr
        self.binning = get_binning_from_hdr(self.header)
        file_root = fname.split('/')[-1]
        self.dispaxis = None
        if file_root != self.header['FILENAME']:
            raise ValueError("The file doesn't seem to be a raw FITS image.")

        if self.header['OBS_MODE'] == 'SPECTROSCOPY':
            self.mode = 'SPEC'
            cd1 = self.header['CDELT1']
            cd2 = self.header['CDELT2']
            self.CD = np.array([[cd1, 0.],
                                [0., cd2]])
        else:
            self.mode = 'IMG'
            # Image Coordinates and Reference System
            cd11 = self.header['CD1_1']
            cd12 = self.header['CD1_2']
            cd21 = self.header['CD2_1']
            cd22 = self.header['CD2_2']
            self.CD = np.array([[cd11, cd21],
                                [cd12, cd22]])

        self.filter = get_filter(self.header)
        self.slit = self.header['ALAPRTNM']
        self.grism = self.header['ALGRNM']
        if 'Vert' in self.slit:
            self.dispaxis = 1
        elif 'Slit' in self.slit:
            self.dispaxis = 2

        self.exptime = self.header['EXPTIME']
        self.object = self.header['OBJECT']
        self.target_name = self.header['TCSTGT']
        self.ra_deg = self.header['RA']
        self.ra_hr = self.header['OBJRA']
        self.dec_deg = self.header['DEC']

        self.rot_angle = self.header['ROTPOS']
        self.airmass = self.header['AIRMASS']
        self.date = self.header['DATE-OBS']
        self.mjd = get_mjd(self.date)

        self.CRVAL = np.array([self.header['CRVAL1'], self.header['CRVAL2']])
        self.CRPIX = np.array([self.header['CRPIX1'], self.header['CRPIX2']])
        self.data_unit = self.header['BUNIT']
        self.x_unit = self.header['CUNIT1']
        self.y_unit = self.header['CUNIT2']

    def match_files(self, filelist, date=True, binning=True, shape=True, grism=False, slit=False, filter=False, get_closest_time=False):
        """Return list of filenames that match the given criteria"""
        matches = list()
        # sort by:
        all_times = list()
        for fname in filelist:
            this_hdr = fits.getheader(fname, 0)
            criteria = list()
            this_mjd = get_mjd(this_hdr['DATE-OBS'])
            if date:
                # Match files from same night, midnight ± 9hr
                dt = self.mjd - this_mjd
                criteria.append(-0.4 < dt < +0.4)

            if binning:
                # Match files with same binning and readout speed:
                this_binning = get_binning_from_hdr(this_hdr)
                criteria.append(this_binning == self.binning)

            if shape:
                # Match files with same image shape:
                this_shape = fits.getdata(fname).shape
                criteria.append(this_shape == self.shape)

            if grism:
                # Match files with same grism:
                this_grism = this_hdr['ALGRNM']
                criteria.append(this_grism == self.grism)

            if slit:
                # Match files with the same slit-width:
                this_slit = this_hdr['ALAPRTNM']
                criteria.append(this_slit == self.slit)

            if filter:
                # Match files with the same filter:
                this_filter = get_filter(this_hdr)
                criteria.append(this_filter == self.filter)

            if np.all(criteria):
                matches.append(fname)
                all_times.append(this_mjd)

        if get_closest_time:
            index = np.argmin(np.abs(np.array(all_times) - self.mjd))
            matches = matches[index:index+1]

        return matches



class TagDatabase(dict):
    def __init__(self, file_database):
        # Convert file_database with file-classifications
        # to a tag_database containing a list of all files with a given tag:
        tag_database = dict()
        for fname, tag in file_database.items():
            if tag in tag_database.keys():
                tag_database[tag].append(fname)
            else:
                tag_database[tag] = [fname]

        # And make this converted tag_database the basis of the TagDatabase class
        dict.__init__(self, tag_database)
        self.file_database = file_database

    def __add__(self, other):
        new_file_database = other.file_database

        for key in self.file_database.keys():
            new_file_database[key] = self.file_database[key]

        return TagDatabase(new_file_database)

    def __radd__(self, other):
        if other == 0:
            return self

        else:
            return self.__add__(other)

    def has_tag(self, tag):
        if tag in self.keys():
            return True

        return False
