# -*- coding: UTF-8 -*-

from collections import defaultdict
import numpy as np
import os
import sys
from glob import glob
from astropy.io import fits
from astropy.io.fits.file import AstropyUserWarning
from astropy.time import Time
import warnings
import re

from pynot.response import lookup_std_star
from pynot import instrument
from pynot.fitsio import verify_header_key

# -- use os.path
code_dir = os.path.dirname(os.path.abspath(__file__))
v_file = os.path.join(os.path.split(code_dir)[0], 'VERSION')
with open(v_file) as version_file:
    __version__ = version_file.read().strip()

warnings.simplefilter('ignore', category=AstropyUserWarning)

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

def make_filter_function(func, target):
    if target is True or target.lower() == 'any':
        return lambda x: True

    target = target.strip()
    if target.startswith('/') and target.endswith('/'):
        target = target[1:-1]
        pattern = re.compile(target)
        return lambda x: len(pattern.findall(func(x))) > 0
    else:
        return lambda x: func(x) == target

def filter_files(flist, func, target):
    criterion = make_filter_function(func, target)
    return list(filter(criterion, flist))


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
        hdr = instrument.get_header(fname)
        mjd = instrument.get_mjd(hdr)
        if int(mjd) == int(date_mjd):
            matches.append(fname)

    return matches

def match_response(sci_img, file_list, exact_date=False):
    """
    Find response function that matches the grism of the input `sci_img`
    If `exact_date` is True, only return a match if the date matches exactly that of `sci_img`
    Otherwise return the file that is closest in time.
    """
    target_grism = sci_img.grism
    target_mjd = sci_img.mjd
    matches = list()
    date_diff = list()
    for fname in file_list:
        # These are PyNOT FITS files, so follows a slightly different Header Format
        hdr = fits.getheader(fname)
        this_grism = hdr['GRISM']
        this_date = Time(hdr['DATE-OBS'])
        if this_grism == target_grism:
            matches.append(fname)
            date_diff.append(this_date.mjd - target_mjd)

    # Find target with smallest time difference
    idx = np.argmin(np.abs(date_diff))
    match_fname = matches[idx]
    min_dt = date_diff[idx]
    if exact_date and min_dt >= 1.:
        # The exact dates is not matched!
        match_fname = ""
    return match_fname


def match_single_calib(raw_img, database, tag, log, **kwargs):
    calib_list = raw_img.match_files(database[tag], **kwargs)

    if len(calib_list) > 1:
        kwargs['get_closest_time'] = True
        calib_list = raw_img.match_files(database[tag], **kwargs)

    if len(calib_list) != 1:
        log.error("Could not match filetype %s" % tag)
        log.error("Check that the filetype exists in the PFC database")
        raise KeyError("Could not match filetype %s" % tag)

    return calib_list[0]


def sort_spec_flat(file_list, date=False):
    """
    Sort spectroscopic flat frames by grism, slit, filter and image size
    """
    sorted_files = defaultdict(list)
    for fname in file_list:
        hdr = instrument.get_header(fname)
        grism = instrument.get_grism(hdr)
        slit = instrument.get_slit(hdr).replace('_', '')
        filt_name = instrument.get_filter(hdr)
        size = "%ix%i" % (hdr['NAXIS1'], hdr['NAXIS2'])
        if filt_name.lower() in ['free', 'open', 'none']:
            file_id = "%s_%s_%s" % (grism, slit, size)
        else:
            file_id = "%s_%s_%s_%s" % (grism, slit, filt_name, size)
        if date:
            date_str = instrument.get_date(hdr)
            if 'T' in date_str:
                date_str = date_str.split('T')[0]
            file_id += "_%s" % date_str
        sorted_files[file_id].append(fname)
    return sorted_files


def sort_arcs(file_list, date=False, grism_only=False):
    """
    Sort arc lapm frames by grism, slit and image size
    """
    sorted_files = defaultdict(list)
    for fname in file_list:
        hdr = instrument.get_header(fname)
        grism = instrument.get_grism(hdr)
        slit = instrument.get_slit(hdr).replace('_', '')
        size = "%ix%i" % (hdr['NAXIS1'], hdr['NAXIS2'])
        if grism_only:
            file_id = grism
        else:
            file_id = "%s_%s_%s" % (grism, slit, size)
        if date:
            date_str = instrument.get_date(hdr)
            if 'T' in date_str:
                date_str = date_str.split('T')[0]
            file_id += "_%s" % date_str
        sorted_files[file_id].append(fname)
    return sorted_files


def sort_std(file_list, date=False):
    sorted_files = defaultdict(lambda: defaultdict(list))
    for fname in file_list:
        hdr = instrument.get_header(fname)
        target_name = instrument.get_target_name(hdr)
        filt_name = instrument.get_filter(hdr)
        grism = instrument.get_grism(hdr)
        slit = instrument.get_slit(hdr).replace('_', '')
        insID = "%s_%s" % (grism, slit)
        if filt_name.lower() not in ['free', 'open', 'none']:
            insID = "%s_%s" % (insID, filt_name)
        if date:
            date_str = instrument.get_date(hdr)
            if 'T' in date_str:
                date_str = date_str.split('T')[0]
            insID += "_%s" % date_str
        sorted_files[target_name][insID].append(fname)
    return sorted_files


def sort_bias(file_list, date=False):
    """
    Sort spectroscopic bias frames by image size
    """
    sorted_files = defaultdict(list)
    for fname in file_list:
        hdr = instrument.get_header(fname)
        size = "%ix%i" % (hdr['NAXIS1'], hdr['NAXIS2'])
        file_id = size
        if date:
            date_str = instrument.get_date(hdr)
            if 'T' in date_str:
                date_str = date_str.split('T')[0]
            file_id += "_%s" % date_str
        sorted_files[file_id].append(fname)
    return sorted_files


def group_calibs_by_date(file_list, lower=0.01, upper=0.99):
    """ Grouping files observed from the same day. """
    file_list = np.array(file_list, dtype=str)
    date_sorted = dict()
    mjd = np.zeros(len(file_list))
    n = np.zeros(len(file_list), dtype=bool)
    N = len(file_list)
    for i, fname in enumerate(file_list):
        hdr = instrument.get_header(fname)
        mjd[i] = instrument.get_mjd(hdr)

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
    Classify input FITS file according to the set of `rules`

    fname : string

    rules : list[string]
        A list of string conditions for header keywords. Each rule corresponds to one filetype:
        e.g. BIAS, SPEC_FLAT, SPEC_OBJECT.

    Returns
    -------
    ftype : string
        Filetype that matches the given input file

    Raises
    ------
    MultipleFileTypeError : if more than one filetype matches the given file
    NoFileTypeError : if no filetype matches the given file
    RuleFormatError : if one or more criteria in a rule cannot be parsed correctly
    raised by fits.getheader : TypeError, IndexError, OSError, FileNotFoundError

    """
    h = fits.getheader(fname)

    matches = list()
    for linenum, rule in enumerate(rules):
        rule = rule.strip()
        if ('#' in rule) or (len(rule) == 0):
            continue

        ftype = rule.split(':')[0].strip()
        all_conditions = rule.split(':')[1].split(' and ')
        criteria = list()
        for cond in all_conditions:
            if '==' in cond:
                key, val = cond.split('==')
                key = verify_header_key(key)
                if key not in h:
                    raise RuleCriterionError(key, cond)
                if 'open' in val.lower():
                    criteria.append('open' in h[key].lower())
                elif 'closed' in val.lower():
                    criteria.append('closed' in h[key].lower())
                else:
                    val = parse_value(val)
                    criteria.append(val == h[key])

            elif '!=' in cond:
                key, val = cond.split('!=')
                key = verify_header_key(key)
                if key not in h:
                    raise RuleCriterionError(key, cond)
                if 'open' in val.lower():
                    criteria.append('open' not in h[key].lower())
                elif 'closed' in val.lower():
                    criteria.append('closed' not in h[key].lower())
                else:
                    val = parse_value(val)
                    criteria.append(val != h[key])

            elif '>' in cond:
                key, val = cond.split('>')
                key = verify_header_key(key)
                if key not in h:
                    raise RuleCriterionError(key, cond)
                val = parse_value(val)
                criteria.append(h[key] > val)

            elif '<' in cond:
                key, val = cond.split('<')
                key = verify_header_key(key)
                if key not in h:
                    raise RuleCriterionError(key, cond)
                val = parse_value(val)
                criteria.append(h[key] < val)

            elif ' contains ' in cond:
                key, val = cond.split('contains')
                key = verify_header_key(key)
                if key not in h:
                    raise RuleCriterionError(key, cond)
                val = parse_value(val)
                criteria.append(val in h[key])

            elif ' !contains ' in cond:
                key, val = cond.split('!contains')
                key = verify_header_key(key)
                if key not in h:
                    raise RuleCriterionError(key, cond)
                val = parse_value(val)
                criteria.append(val not in h[key])

            else:
                raise RuleFormatError("Invalid condition in rule at line %i:  %s" % (linenum, rule))

        if np.all(criteria):
            matches.append(ftype)

    if len(matches) == 1:
        ftype = matches[0]
        if ftype == 'IMG_OBJECT':
            # Check if target is a flux standard calibrator!
            pass

        elif ftype == 'SPEC_OBJECT':
            star_name = lookup_std_star(h)
            if star_name:
                ftype = 'SPEC_FLUX-STD'
        return ftype

    elif len(matches) == 0:
        raise NoFileTypeError("No classification matched the file: %s" % fname)

    else:
        err_msg = "More than one classification was found for file: %s" % fname
        raise MultipleFileTypeError(err_msg, matches)


class RuleCriterionError(Exception):
    def __init__(self, key, condition):
        self.key = key
        self.condition = condition
        self.message = "Error in condition: %s. No FITS header key: %s" % (self.condition, self.key)
        super().__init__(self.message)

class RuleFormatError(Exception):
    pass

class NoFileTypeError(Exception):
    pass

class MultipleFileTypeError(Exception):
    def __init__(self, message, matches):
        self.message = message
        self.matches = matches
        self.matches_str = ", ".join(matches)
        super().__init__(self.message)


def classify(data_in, rule_file=instrument.rulefile, progress=True):
    """
    The input can be a single .fits file, a string given the path to a directory,
    a list of .fits files, or a list of directories.
    Classify given input files using the rules defined in `rule_file`.

    Returns
    -------
    database : TagDatabase or None
        An instance of TagDatabase containing the file classifications

    output_msg : string
        A string of logging messages
    """
    msg = list()

    # Determine the input type:
    if isinstance(data_in, str):
        if os.path.isfile(data_in):
            files = [data_in]
        elif '*' in data_in or '?' in data_in:
            files = glob(data_in)
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
        raise ValueError("Input must be a string or a list of strings [data.organizer.classify]")

    data_types = dict()
    not_classified_files = list()

    if not os.path.exists(rule_file):
        raise FileNotFoundError("Instrument ruleset could not be found: %s" % rule_file)

    with open(rule_file) as rulebook:
        rules = rulebook.readlines()

    if progress:
        print("")
        print(" Classifying files: ")

    for num, fname in enumerate(files):
        try:
            # if it passes then there's only one filetype
            ftype = classify_file(fname, rules)
            data_types[fname] = ftype

        except NoFileTypeError as e:
            msg.append("[WARNING] - " + str(e))
            not_classified_files.append(fname)

        except MultipleFileTypeError as e:
            msg.append("[WARNING] - " + str(e))
            msg.append("[WARNING] - " + e.matches_str)
            not_classified_files.append(fname)

        except (RuleFormatError, TypeError, IndexError, OSError, FileNotFoundError) as e:
            # error! either in file handling from astropy.io.fits or in rulebook
            msg.append(" [ERROR]  - " + str(e))
            msg.append("")
            return None, "\n".join(msg)

        if progress:
            sys.stdout.write("\r  %6.2f%%" % (100.*(num+1)/len(files)))
            sys.stdout.flush()

    msg.append("")
    msg.append("          - Classification finished.")
    msg.append("  [DONE]  - Successfully classified %i out of %i files." % (len(data_types.keys()), len(files)))
    msg.append("")
    if len(files) != len(data_types.keys()):
        msg.append("[WARNING] - %s warnings were raised!" % len(not_classified_files))
        msg.append("[WARNING] - No classification matched the files:")
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
        elif '*' in data_in or '?' in data_in:
            files = glob(data_in)
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


class UnknownObservingMode(Exception):
    pass


class RawImage(object):
    """
    Create a raw image instance from a FITS image.

    Raises:
    ValueError, UnknownObservingMode, OSError, FileNotFoundError, TypeError, IndexError
    """
    def __init__(self, fname, filetype=None):
        self.filename = fname
        self.data = fits.getdata(fname)
        self.shape = self.data.shape
        self.filetype = filetype
        self.header = instrument.get_header(fname)
        self.binning = instrument.get_binning_from_hdr(self.header)
        # file_root = fname.split('/')[-1]
        self.dispaxis = None
        # if file_root != self.header['FILENAME']:
        #     raise ValueError("The file doesn't seem to be a raw FITS image: %s" % fname)

        self.mode = instrument.get_observing_mode(self.header)
        if self.mode.lower().startswith('spec'):
            cd1 = self.header.get('CDELT1')
            cd2 = self.header.get('CDELT2')
            if cd1 is None:
                cd1 = self.header.get('CD1_1')
            if cd2 is None:
                cd2 = self.header.get('CD2_2')
            self.CD = np.array([[cd1, 0.],
                                [0., cd2]])
            self.dispaxis = instrument.get_dispaxis(self.header)
            if self.dispaxis is None:
                raise ValueError("Invalid dispersion axis of image: %s" % fname)

        elif self.mode.lower().startswith('im'):
            # Image Coordinates and Reference System
            if 'CD1_1' in self.header:
                cd11 = self.header['CD1_1']
                cd12 = self.header['CD1_2']
                cd21 = self.header['CD2_1']
                cd22 = self.header['CD2_2']
            else:
                cd11 = self.header['CDELT1']
                cd22 = self.header['CDELT2']
                cd12 = 0.
                cd21 = 0.
            self.CD = np.array([[cd11, cd21],
                                [cd12, cd22]])
            self.dispaxis = None
        else:
            raise UnknownObservingMode("Unknown observing mode: %r of file: %s" % (self.mode, fname))

        self.filter = instrument.get_filter(self.header)
        self.slit = instrument.get_slit(self.header)
        self.grism = instrument.get_grism(self.header)

        self.exptime = instrument.get_exptime(self.header)
        self.object = instrument.get_object(self.header)
        self.target_name = instrument.get_target_name(self.header)
        self.ra_deg = self.header['RA']
        self.ra_hr = self.ra_deg / 15.
        self.dec_deg = self.header['DEC']
        self.ob_name = instrument.get_ob_name(self.header)

        self.rot_angle = instrument.get_rotpos(self.header)
        self.airmass = instrument.get_airmass(self.header)
        self.date = instrument.get_date(self.header)
        self.mjd = instrument.get_mjd(self.header)

        self.CRVAL = np.array([self.header['CRVAL1'], self.header['CRVAL2']])
        self.CRPIX = np.array([self.header['CRPIX1'], self.header['CRPIX2']])
        if 'BUNIT' not in self.header:
            self.header['BUNIT'] = ""
        self.data_unit = self.header['BUNIT']
        if 'CUNIT1' not in self.header:
            self.header['CUNIT1'] = ""
        if 'CUNIT2' not in self.header:
            self.header['CUNIT2'] = ""
        self.x_unit = self.header['CUNIT1']
        self.y_unit = self.header['CUNIT2']
        self.x_type = self.header['CTYPE1']
        self.y_type = self.header['CTYPE2']

    def set_filetype(self, filetype):
        self.filetype = filetype

    def match_files(self, filelist, date=True, binning=True, shape=True, grism=False, slit=False, filter=False, get_closest_time=False, debug=False):
        """Return list of filenames that match the given criteria"""
        matches = list()
        # sort by:
        all_times = list()
        for fname in filelist:
            this_hdr = fits.getheader(fname, 0)
            criteria = list()
            criteria_name = list()
            this_mjd = instrument.get_mjd(this_hdr)
            if date:
                # Match files from same night, midnight ± 9hr
                dt = self.mjd - this_mjd
                criteria.append(-0.4 < dt < +0.4)
                criteria_name.append('name')

            if binning:
                # Match files with same binning and readout speed:
                this_binning = instrument.get_binning_from_hdr(this_hdr)
                criteria.append(this_binning == self.binning)
                criteria_name.append('binning')

            if shape:
                # Match files with same image shape:
                hdr = instrument.get_header(fname)
                this_shape = (hdr['NAXIS2'], hdr['NAXIS1'])
                if 'OVERSCAN' in hdr:
                    # use original image shape before overscan-sub
                    over_x = hdr['OVERSCAN_X']
                    over_y = hdr['OVERSCAN_Y']
                    this_shape = (this_shape[0]+over_y, this_shape[1]+over_x)

                criteria.append(this_shape == self.shape)
                criteria_name.append('shape')

            if grism:
                # Match files with same grism:
                this_grism = instrument.get_grism(this_hdr)
                criteria.append(this_grism == self.grism)
                criteria_name.append('grism')

            if slit:
                # Match files with the same slit-width:
                this_slit = instrument.get_slit(this_hdr)
                criteria.append(this_slit == self.slit)
                criteria_name.append('slit')

            if filter:
                # Match files with the same filter:
                this_filter = instrument.get_filter(this_hdr)
                criteria.append(this_filter == self.filter)
                criteria_name.append('filter')

            if np.all(criteria):
                matches.append(fname)
                all_times.append(this_mjd)

        if get_closest_time:
            index = np.argmin(np.abs(np.array(all_times) - self.mjd))
            matches = matches[index:index+1]

        return matches


class TagDatabase(dict):
    def __init__(self, file_database, inactive_files=None):
        # Convert file_database with file-classifications
        # to a tag_database containing a list of all files with a given tag:
        if inactive_files is None:
            inactive_files = {}

        self.file_database = file_database
        self.inactive_file_database = inactive_files
        tag_database = dict()
        for fname, tag in file_database.items():
            if tag in tag_database.keys():
                tag_database[tag].append(fname)
            else:
                tag_database[tag] = [fname]

        self.inactive = defaultdict(list)
        for fname, tag in inactive_files.items():
            self.inactive[tag].append(fname)

        # And make this converted tag_database the basis of the TagDatabase class
        dict.__init__(self, tag_database)

    def __add__(self, other):
        all_inactive_fnames = [fname.strip('#') for fname in self.inactive_file_database]
        for fname, tag in other.file_database.items():
            if fname in all_inactive_fnames:
                pass
            else:
                self.file_database[fname] = tag

        for fname, tag in other.inactive_file_database.items():
            self.inactive_file_database[fname] = tag

        return TagDatabase(self.file_database, self.inactive_file_database)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def has_tag(self, tag):
        if tag in self.keys():
            return True
        return False

    def get_files(self, tag, grism=None, slit=None, filter=None, date=None, target=None, **kwargs):
        file_list = self.get(tag, [])
        if grism:
            file_list = filter_files(file_list, get_grism, grism)
        if filter:
            file_list = filter_files(file_list, get_filter, filter)
        if slit:
            file_list = filter_files(file_list, get_slit, slit)
        if date:
            file_list = filter_files(file_list, get_date, date)
        if target:
            file_list = filter_files(file_list, get_target, target)
        return file_list

def get_target(fname):
    hdr = fits.getheader(fname)
    return instrument.get_object(hdr)

def get_grism(fname):
    hdr = fits.getheader(fname)
    return instrument.get_grism(hdr)

def get_slit(fname):
    hdr = fits.getheader(fname)
    return instrument.get_slit(hdr)

def get_filter(fname):
    hdr = fits.getheader(fname)
    return instrument.get_filter(hdr)

def get_date(fname):
    hdr = fits.getheader(fname)
    date = instrument.get_date(hdr)
    if 'T' in date:
        date = date.split('T')[0]
    return date
