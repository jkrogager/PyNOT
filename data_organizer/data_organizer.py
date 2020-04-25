# -*- coding: UTF-8 -*-

import numpy as np
import os
import sys
import datetime
from glob import glob
from astropy.io import fits


# -- use os.path
code_dir = os.path.dirname(os.path.abspath(__file__))
v_file = os.path.join(os.path.split(code_dir)[0], 'VERSION')
with open(v_file) as version_file:
    __version__ = version_file.read().strip()

std_fname = os.path.join(os.path.split(code_dir)[0], 'calib/std/namelist.txt')
calib_namelist = np.loadtxt(std_fname, dtype=str)
calib_names = calib_namelist[:, 1]

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


def print_credits():
    credit_string = """
      \x1b[1m NOT/ALFOSC Data Organizer \x1b[0m %s

    written by:
      Jens-Kristian Krogager
      Institut d'Astrophysique de Paris
    """
    print(credit_string % __version__)


def get_mjd(date_str):
    """Input Date String in ISO format as in ALFOSC header: '2016-08-01T16:03:57.796'"""
    date = datetime.datetime.fromisoformat(date_str)
    mjd_0 = datetime.datetime(1858, 11, 17)
    dt = date - mjd_0
    mjd = dt.days + dt.seconds/(24*3600.)
    return mjd


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


def get_filter(hdr):
    filter = None
    for keyword in ['FAFLTNM', 'FBFLTNM', 'ALFLTNM']:
        if 'open' in hdr[keyword].lower():
            pass
        else:
            filter = hdr[keyword]
    return filter


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


def classify(data_in, rule_file='/Users/krogager/coding/PyNOT/data_organizer/alfosc.rules', verbose=False, progress=True):
    """
    The input can be a single .fits file, a string given the path to a directory,
    a list of .fits files, or a list of directories.
    Classify given input 'files' using the rules defined in 'xsh.rules'
    Returns
    'Data_Type' containing the classification for the pipeline
                recipes.
    """

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

    data_type = dict()
    not_classified_files = list()

    with open(rule_file) as rulebook:
        rules = rulebook.readlines()

    if progress:
        print("")
        print(" Classifying files: ")

    for num, fname in enumerate(files):
        h = fits.getheader(fname)
        fileroot = fname.split('/')[-1]
        if fileroot != h['FILENAME']:
            continue

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
                    if 'open' in val.lower():
                        criteria.append('open' in h[key].lower())
                    elif 'closed' in val.lower():
                        criteria.append('closed' in h[key].lower())
                    else:
                        val = parse_value(val)
                        criteria.append(val == h[key])

                elif '!=' in cond:
                    key, val = cond.split('!=')
                    if 'open' in val.lower():
                        criteria.append('open' not in h[key].lower())
                    elif 'closed' in val.lower():
                        criteria.append('closed' not in h[key].lower())
                    else:
                        val = parse_value(val)
                        criteria.append(val != h[key])

                elif '>' in cond:
                    key, val = cond.split('>')
                    val = parse_value(val)
                    criteria.append(h[key] > val)

                elif '<' in cond:
                    key, val = cond.split('<')
                    val = parse_value(val)
                    criteria.append(h[key] < val)

            if np.all(criteria):
                data_type[fname] = ftype
                matches.append(ftype)

        if len(matches) == 1:
            ftype = matches[0]
            if matches[0] == 'IMG_OBJECT':
                # If no filter is defined and CCD readout is fast, the image is most likely an acquisition image
                img_filter = get_filter(h)
                if img_filter == '' and h['FPIX'] > 200:
                    ftype = 'ACQ_IMG'

            if matches[0] == 'SPEC_OBJECT':
                if h['TCSTGT'] in calib_names:
                    ftype = 'SPEC_STD'

            data_type[fname] = ftype
        elif len(matches) == 0:
            not_classified_files.append(fname)
        else:
            if verbose:
                print(" - ERROR : More than one filetype was found for file: %s" % fname)
                print(matches)
                print("")

        if progress:
            sys.stdout.write("\r  %6.2f%%" % (100.*(num+1)/len(files)))
            sys.stdout.flush()


    if verbose:
        print("\n")
        print(" Classification finished.")
        print(" Successfully classified %i out of %i files." % (len(data_type.keys()), len(files)))
        print("")
        if len(files) != len(data_type.keys()):
            print(" Files not classified:")
            for item in not_classified_files:
                print("   %s" % item)

    return data_type


def write_report(dt, output=''):
    """
    Write a report of a given data collection from the `classify` function

    If `output` is given, the report will be saved to this filename.
    """
    # Sort the files by file-type:
    tags = TagDatabase(dt)

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
        primhdr = fits.getheader(fname, 0)
        imghdr = fits.getheader(fname, 1)
        primhdr.update(imghdr)
        self.header = primhdr
        self.binning = get_binning_from_hdr(primhdr)
        file_root = fname.split('/')[-1]
        if file_root != primhdr['FILENAME']:
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

        self.CRVAL = np.array([primhdr['CRVAL1'], primhdr['CRVAL2']])
        self.CRPIX = np.array([primhdr['CRPIX1'], primhdr['CRPIX2']])
        self.data_unit = primhdr['BUNIT']
        self.x_unit = primhdr['CUNIT1']
        self.y_unit = primhdr['CUNIT2']

    def match_files(self, filelist, date=True, binning=True, shape=True, grism=False, slit=False, filter=False):
        matches = list()
        # sort by:
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


class DataSet(object):
    """
    This is the main class for identifying datasets
    'data_in' gives path to fold containing data files, or a list of data filenames
    'mode' is one of: 'SLIT_STARE', 'SLIT_NOD', 'SLIT_OFFSET', 'IFU_STAR', 'IFU_OFFSET'
    'silent' turns off text notifications to the terminal
    """
    def __str__(self):
        """ Define a string representation of the DataSet. """
        N_files = len(self.tag_database.file_database.keys())
        str_rep = "X-shooter DataSet (MODE: %s). Number of files in dataset: %i" % (self.mode, N_files)
        return str_rep

    def __init__(self, data_in=None, silent=False):

        self.version = __version__
        self.tag_database = TagDatabase(dict())

        if data_in:
            self.add_files(data_in, silent)
            self.check_science_frames()

    def set_tag_database(self, file_database):
        """
        Set a new tag-database from a file_database.
        This will overwrite the current database!
        """
        self.tag_database = TagDatabase(file_database)

    def add_files(self, data_in, silent=False):
        new_file_database = classify(data_in, silent=silent)
        new_tag_database = TagDatabase(new_file_database)
        self._file_database = new_file_database
        self.tag_database = self.tag_database + new_tag_database

    def check_science_frames(self):
        pass

    def has_tag(self, tag):
        return self.tag_database.has_tag(tag)

    def reset_tag(self, tag):
        if self.has_tag(tag):
            self.tag_database[tag] = list()

    def remove_tag(self, tag):
        if self.has_tag(tag):
            fnames = self.tag_database.pop(tag)
            for fname in fnames:
                self.tag_database.file_database.pop(fname)

    def set_tag(self, tag, files):
        if self.has_tag(tag):
            if isinstance(files, list):
                self.tag_database[tag] = files
            elif isinstance(files, str):
                self.tag_database[tag] = [files]
            else:
                print(" - Invalid argument `files`: %r" % files)
        else:
            print(" - Invalid Type: %r" % tag)

    def append(self, tag, files):
        if self.has_tag(tag):
            if isinstance(files, list):
                self.tag_database[tag] += files
            elif isinstance(files, str):
                self.tag_database[tag].append(files)
            else:
                print(" Invalid argument: %r" % files)
        else:
            print(" - Invalid Type: %r" % tag)

    def all_tags(self):
        return self.tag_database.keys()

    def get_files(self, tag):
        if self.has_tag(tag):
            return self.tag_database[tag]
        else:
            return None

    # -- Match DARK frames based on date and then exptime:
    def match_dark(self, fname):
        """Find the dark frames with matching exposure time"""
        target_hdr = fits.getheader(fname)
        target_exptime = target_hdr['EXPTIME']
        target_date = target_hdr['MJD-OBS']

        date_sorted = group_calibs_by_date(self.get_files('DARK_NIR'))
        all_dates = date_sorted.keys()
        date_diff = [float(mjd) - target_date for mjd in all_dates]
        this_date = all_dates[np.argmin(date_diff)]

        matched_dark_frames = list()
        for dark_frame in date_sorted[this_date]:
            hdr = fits.getheader(dark_frame)
            exptime = hdr['EXPTIME']
            if exptime == target_exptime:
                matched_dark_frames.append(dark_frame)

        if len(matched_dark_frames) < 3:
            print(" [WARNING] - Less than 3 DARK frames were supplied: %i" % len(matched_dark_frames))

        return ('DARK', matched_dark_frames)

    def match_flat(self, tag, fname, date=False):
        """Find the flat frames with matching slit width and binning"""
        target_hdr = fits.getheader(fname)
        target_slit = target_hdr['ALAPRTNM']
        target_ccd = get_binning(fname)
        target_date = get_mjd(target_hdr['DATE-OBS'])
        target_exptime = target_hdr['EXPTIME']

        exptimes = list()
        all_matched_flat_frames = list()
        if date:
            date_sorted = group_calibs_by_date(self.get_files(tag))
            all_dates = date_sorted.keys()
            date_diff = [float(mjd) - target_date for mjd in all_dates]
            this_date = all_dates[np.argmin(np.abs(date_diff))]
            for frame in date_sorted[this_date]:
                hdr = fits.getheader(frame)
                slit = hdr['ALAPRTNM']
                ccd = get_binning(frame)
                if slit == target_slit and ccd == target_ccd:
                    all_matched_flat_frames.append(frame)
                    exptimes.append(hdr['EXPTIME'])
        else:
            for frame in self.get_files(tag):
                hdr = fits.getheader(frame)
                slit = hdr['ALAPRTNM']
                ccd = get_binning(frame)
                if slit == target_slit and ccd == target_ccd:
                    all_matched_flat_frames.append(frame)
                    exptimes.append(hdr['EXPTIME'])

        # find the most frequent exposure time:
        exp_list = occurence(exptimes)
        top_freq = np.argmax(np.argmax([val[1] for val in exp_list]))
        target_exptime = exp_list[top_freq][0]
        # Return only files with same exposure time
        matched_flat_frames = list()
        for fname, exptime in zip(all_matched_flat_frames, exptimes):
            if exptime == target_exptime:
                matched_flat_frames.append(fname)

        if len(matched_flat_frames) % 2 == 0:
            print(" - Even number of flat frames! Should be uneven...")

        return (tag, matched_flat_frames)
