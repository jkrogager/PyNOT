"""
Automatically Classify and Reduce a given Data Set
"""

from astropy.io import fits
from argparse import ArgumentParser
from collections import defaultdict
import os
import datetime
import yaml

from . import alfosc
from . import data_organizer as do
from .calibs import combine_bias_frames, combine_flat_frames, normalize_spectral_flat

code_dir = os.path.dirname(os.path.abspath(__file__))
calib_dir = os.path.join(code_dir, '/calib/')
defaults_fname = os.path.join(calib_dir, 'default_options.yml')
v_file = os.path.join(code_dir, 'VERSION')
with open(v_file) as version_file:
    __version__ = version_file.read().strip()


class Report(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.time = datetime.datetime.now()
        self.fname = 'pynot_%s.log' % self.time.strftime('%d%b%Y-%Hh%Mm%S')
        self.remarks = list()
        self.lines = list()
        self.header = """
        #  PyNOT Data Processing Pipeline
        # ================================
        # version %s
        %s

        """ % (__version__, self.time.strftime("%b %d, %Y  %H:%M:%S"))
        self.report = ""

        if self.verbose:
            print(self.header)

    def commit(self, text):
        if self.verbose:
            print(text)
        self.lines.append(text)

    def error(self, text):
        text = ' [ERROR]  - ' + text
        if self.verbose:
            print(text)
        if text[-1] != '\n':
            text += '\n'
        self.lines.append(text)

    def warn(self, text):
        text = '[WARNING] - ' + text
        if self.verbose:
            print(text)
        if text[-1] != '\n':
            text += '\n'
        self.lines.append(text)

    def write(self, text, prefix='          > '):
        text = prefix + text
        if self.verbose:
            print(text)
        if text[-1] != '\n':
            text += '\n'
        self.lines.append(text)

    def add_linebreak(self):
        if self.verbose:
            print("")
        self.lines.append("\n")

    def add_remark(self, text):
        self.remarks.append(text)

    def _make_report(self):
        remark_str = ''.join(self.remarks)
        lines_str = ''.join(self.lines)
        self.report = '\n'.join([self.header, remark_str, lines_str])

    def print_report(self):
        self._make_report()
        print(self.report)

    def save(self):
        self._make_report()
        with open(self.fname, 'w') as output:
            output.write(self.report)

    def fatal_error(self):
        print("!! FATAL ERROR !!")
        print("Consult the log: %s\n" % self.fname)
        self.save()


def get_options(option_fname):
    with open(option_fname) as opt_file:
        options = yaml.full_load(opt_file)
    return options


def main(raw_path, options_fname=None, verbose=True):
    log = Report(verbose)

    # -- Parse Options from YAML
    options = get_options(defaults_fname)
    if options_fname:
        user_options = get_options(options_fname)
        options.update(user_options)

    if not os.path.exists(raw_path):
        log.error("Data path does not exist : %s" % raw_path)
        log.fatal_error()
        return

    dataset = options['dataset']
    if dataset is not None:
        # -- load collection
        # -- reclassify (takes already identified files into account)
        pass

    # Classify files:
    log.write("Classyfying files in folder: %s" % raw_path)
    try:
        database, message = do.classify(raw_path, progress=verbose)
        # -- Save collection
        log.commit(message)
        log.add_linebreak()
    except ValueError as err:
        log.error(str(err))
        print(err)
        log.fatal_error()
        return
    except FileNotFoundError as err:
        log.error(str(err))
        log.fatal_error()
        return

    # -- Organize object files in dataset:
    object_filelist = database['SPEC_OBJECT']
    object_images = list(map(do.RawImage, object_filelist))

    log.add_linebreak()
    log.write(" - The following objects were found in the dataset:", prefix='')
    log.write("      OBJECT          GRISM       SLIT      EXPTIME", prefix='')
    for sci_img in object_images:
        log.write("%20s  %9s  %11s  %.0f" % (sci_img.object, sci_img.grism, sci_img.slit, sci_img.exptime), prefix='')
    log.add_linebreak()

    # get list of unique grisms in dataset:
    grism_list = list()
    for sci_img in object_images:
        grism_name = alfosc.grism_translate[sci_img.grism]
        if grism_name not in grism_list:
            grism_list.append(grism_name)

    # -- Check arc line files:
    arc_images = list()
    for arc_type in ['ARC_He', 'ARC_HeNe', 'ARC_Ne', 'ARC_ThAr']:
        if arc_type in database.keys():
            arc_images += database[arc_type]

    arc_images_for_grism = defaultdict(list)
    for arc_img in arc_images:
        raw_grism = fits.getheader(arc_img)['ALGRNM']
        this_grism = alfosc.grism_translate[raw_grism]
        arc_images_for_grism[this_grism].append(arc_img)

    for grism_name in grism_list:
        if len(arc_images_for_grism[grism_name]) == 0:
            log.error("No arc frames defined for grism: %s" % grism_name)
            log.fatal_error()
            return
        else:
            log.write("%s has necessary arc files." % grism_name)

    log.add_linebreak()
    identify_all = options['identify']['all']
    identify_interactive = options['identify']['interactive']
    if identify_interactive and identify_all:
        log.write("Identify: interactively reidentify arc lines for all objects")
    elif identify_interactive and not identify_all:
        # Make pixeltable for all grisms:
        grisms_to_identify = grism_list
        log.write("Identify: interactively reidentify all grisms in dataset:")
        log.write(", ".join(grisms_to_identify))
    else:
        # Check if pixeltables exist:
        grisms_to_identify = list()
        for grism_name in grism_list:
            pixtab_fname = calib_dir + '/%s_pixeltable.dat' % grism_name
            if not os.path.exists(pixtab_fname):
                grisms_to_identify.append(grism_name)
                log.write("%s : pixel table does not exist. Will identify lines..." % grism_name)
            else:
                log.write("%s : pixel table already exists" % grism_name)
        log.add_linebreak()

    if len(grisms_to_identify) > 0:
        log.write("Starting interactive definition of pixel table...")

    for grism_name in grisms_to_identify:
        create_pixtable(arc_images_for_grism, grism_name)

    for sci_img in object_images:
        output_dir = sci_img.target_name
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        master_bias_fname = os.path.join(output_dir, 'MASTER_BIAS.fits')
        grism = alfosc.grism_translate[sci_img.grism]
        comb_flat_fname = os.path.join(output_dir, 'FLAT_COMBINED_%s_%s.fits' % (grism, sci_img.slit))
        norm_flat_fname = os.path.join(output_dir, 'NORM_FLAT_%s_%s.fits' % (grism, sci_img.slit))
        final_2d_fname = os.path.join(output_dir, 'red2D_%s_%s.fits' % (sci_img.target_name, sci_img.date))

        # Combine Bias Frames matched for CCD setup
        bias_frames = sci_img.match_files(database['BIAS'])
        combine_bias_frames(bias_frames, output=master_bias_fname,
                            kappa=bias_kappa,
                            verbose=verbose)

        # Combine Flat Frames matched for CCD setup, grism, slit and filter
        flat_frames = sci_img.match_files(database['SPEC_FLAT'], grism=True, slit=True, filter=True)
        _ = combine_flat_frames(flat_frames, mbias=master_bias_fname,
                                output=comb_flat_fname,
                                kappa=flat_kappa, verbose=verbose)

        # Normalize the spectral flat field:
        normalize_spectral_flat(comb_flat_fname, output=norm_flat_fname,
                                axis=dispaxis,
                                x1=args.flat_x1, x2=args.flat_x2,
                                order=args.flat_order, sigma=args.flat_sigma,
                                plot=args.plot, show=args.show, ext=args.ext,
                                clobber=False, verbose=args.verbose)

        # Sensitivity Function:
        std_fname, = sci_img.match_files(database['SPEC_FLUX-STD'], grism=True, slit=True, filter=True, get_closest_time=True)
        # -- steps in response function

        # Science Reduction:


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("path", type=str, nargs='?', default='',
                        help="Path to directory containing the raw data")
    parser.add_argument("--options", type=str, default='',
                        help="Filename of options in YAML format")
    args = parser.parse_args()

    if args.path:
        main(args.path, options_fname=args.options)

    else:
        print("\n  Running PyNOT Data Processing Pipeline\n")
        if not os.path.exists('options.yml'):
            copy_cmd = "cp %s options.yml" % defaults_fname
            os.system(copy_cmd)
            print(" - Created the default option file: options.yml")
        message = """
         - Update the file and run PyNOT as:
            %] pynot --options options.yml

        Otherwise provide a path to the raw data:
            %] pynot path/to/data

        """
        print(message)
