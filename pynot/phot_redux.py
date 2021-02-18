"""
Automatically Classify and Reduce a given Data Set
"""

from astropy.io import fits
from collections import defaultdict
import os
import sys
import datetime
import numpy as np

from pynot import alfosc
from pynot.data import io
from pynot.data import organizer as do
from pynot.phot import image_combine, create_fringe_image, source_detection
from pynot.calibs import combine_bias_frames, combine_flat_frames
from pynot.functions import get_options, get_version_number
from pynot.scired import raw_correction, correct_cosmics, trim_filter_edge, detect_filter_edge

from PyQt5.QtWidgets import QApplication

code_dir = os.path.dirname(os.path.abspath(__file__))
calib_dir = os.path.join(code_dir, 'calib/')
defaults_fname = os.path.join(calib_dir, 'default_options_img.yml')
__version__ = get_version_number()


class Report(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.time = datetime.datetime.now()
        self.fname = 'pynot_img_%s.log' % self.time.strftime('%d%b%Y-%Hh%Mm%S')
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

    def clear(self):
        self.lines = list()
        self.remarks = list()

    def set_filename(self, fname):
        self.fname = fname

    def commit(self, text):
        if self.verbose:
            print(text, end='', flush=True)
        self.lines.append(text)

    def error(self, text):
        text = ' [ERROR]  - ' + text
        if self.verbose:
            print(text)
        if text[-1] != '\n':
            text += '\n'
        self.lines.append(text)

    def warn(self, text, force=False):
        text = '[WARNING] - ' + text
        if self.verbose or force:
            print(text)
        if text[-1] != '\n':
            text += '\n'
        self.lines.append(text)

    def write(self, text, prefix='          - '):
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

    def exit(self):
        print(" - Pipeline terminated.")
        print(" Consult the log: %s\n" % self.fname)
        self.save()

    def fatal_error(self):
        print(" !! FATAL ERROR !!")
        print(" Consult the log: %s\n" % self.fname)
        self.save()



def run_pipeline(options_fname, verbose=False):
    log = Report(verbose)

    global app
    app = QApplication(sys.argv)

    # -- Parse Options from YAML
    options = get_options(defaults_fname)

    user_options = get_options(options_fname)
    for section_name, section in user_options.items():
        if isinstance(section, dict):
            options[section_name].update(section)
        else:
            options[section_name] = section

    dataset_fname = options['dataset']
    if dataset_fname and os.path.exists(dataset_fname):
        # -- load collection
        database = io.load_database(dataset_fname)
        log.write("Loaded file classification database: %s" % dataset_fname)
        # -- reclassify (takes already identified files into account)

    else:
        log.error("Dataset does not exist : %s" % dataset_fname)
        log.fatal_error()
        return


    # -- Organize object files in dataset:
    if 'IMG_OBJECT' not in database:
        log.error("No imaging data found in the dataset!")
        log.error("Check the classification table... object type 'IMG_OBJECT' missing")
        log.fatal_error()
        return
    object_filelist = database['IMG_OBJECT']
    raw_image_list = list(map(do.RawImage, object_filelist))

    object_images = defaultdict(lambda: defaultdict(list))
    for sci_img in raw_image_list:
        object_images[sci_img.target_name][sci_img.filter].append(sci_img)

    # get list of unique filters in dataset:
    filter_list = np.unique(sum([list(obj.keys()) for obj in object_images.values()], []))

    # -- Check if flat field frames exist:
    flat_images_for_filter = defaultdict(list)
    if 'IMG_FLAT' not in database:
        log.error("No flat field images found in the dataset!")
        log.error("Check the classification table... object type 'IMG_FLAT' missing")
        log.fatal_error()
        return

    flat_images = database['IMG_FLAT']
    for flat_file in flat_images:
        this_filter = do.get_filter(fits.getheader(flat_file))
        if this_filter in filter_list:
            flat_images_for_filter[this_filter].append(flat_file)

    # All files are put in a folder: imaging/OBJNAME/filter/...
    output_base = 'imaging'
    if not os.path.exists(output_base):
        os.mkdir(output_base)

    # Combine Bias Frames matched for CCD setup:
    master_bias_fname = os.path.join(output_base, 'MASTER_BIAS.fits')
    bias_frames = raw_image_list[0].match_files(database['BIAS'])
    if options['mbias']:
        master_bias_fname = options['mbias']
        log.write("Using static master bias frame: %s" % options['mbias'])
    elif len(bias_frames) < 3:
        log.error("Must have at least 3 bias frames to combine, not %i" % len(bias_frames))
        log.error("otherwise provide a static 'master bias' frame!")
        log.fatal_error()
        return
    else:
        log.write("Running task: Bias Combination")
        try:
            _, bias_msg = combine_bias_frames(bias_frames, output=master_bias_fname, mode='img',
                                              kappa=options['bias']['kappa'], overwrite=True)
            log.commit(bias_msg)
            log.add_linebreak()
        except:
            log.error("Median combination of bias frames failed!")
            log.fatal_error()
            print("Unexpected error:", sys.exc_info()[0])
            raise


    master_flat = {}
    filter_edges = {}
    log.write("Running task: Imaging Flat Combination")
    for filter_raw, flat_frames in flat_images_for_filter.items():
        filter_name = alfosc.filter_translate[filter_raw]
        log.write("Combining images for filter: %s" % filter_name)
        comb_flat_fname = os.path.join(output_base, 'FLAT_%s.fits' % filter_name)
        try:
            _, flat_msg = combine_flat_frames(flat_frames, comb_flat_fname, mbias=master_bias_fname,
                                              kappa=options['flat']['kappa'], overwrite=True,
                                              mode='img')
            log.commit(flat_msg)
            master_flat[filter_raw] = comb_flat_fname
        except:
            log.error("Flat field combination failed for filter: %s" % filter_name)
            log.fatal_error()
            print("Unexpected error:", sys.exc_info()[0])
            raise

        # Detect image region:
        try:
            x1, x2, y1, y2 = detect_filter_edge(comb_flat_fname)
            log.write("Detected edges on X-axis: %i  ;  %i" % (x1, x2))
            log.write("Detected edges on Y-axis: %i  ;  %i" % (y1, y2))
            log.add_linebreak()
            filter_edges[filter_raw] = (x1, x2, y1, y2)
        except:
            log.error("Automatic edge detection failed!")
            log.fatal_error()
            print("Unexpected error:", sys.exc_info()[0])
            raise



    # -----------------------------------------------------------------------------------
    # -- Start Reduction of Individual Frames:
    # -----------------------------------------------------------------------------------

    log.add_linebreak()
    log.write("Starting Imaging Pipeline:")
    log.add_linebreak()
    if options['crr']['niter'] > 0:
        log.write("Cosmic Ray Rejection : True")
        log.write("All individual images will be cleaned")
        log.add_linebreak()
    for target_name, images_per_filter in object_images.items():
        # Create working directory:
        if ' ' in target_name:
            target_name = target_name.replace(' ', '_')
        output_obj_base = os.path.join(output_base, target_name)
        if not os.path.exists(output_obj_base):
            os.mkdir(output_obj_base)

        log.write("Target Name: %s" % target_name, prefix=' [TARGET] - ')

        for filter_raw, image_list in images_per_filter.items():
            # Create working directory:
            filter_name = alfosc.filter_translate[filter_raw]
            output_dir = os.path.join(output_obj_base, filter_name)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            log.write("Filter : %s" % filter_name)
            log.write("Saving images to directory: %s" % output_dir)
            log.write("Number of frames: %i" % len(image_list))
            log.add_linebreak()
            log.write("Running task: bias and flat field correction:")

            corrected_images = list()
            temp_images = list()
            for sci_img in image_list:
                log.write("Filename: %s" % sci_img.filename)
                basename = os.path.basename(sci_img.filename)
                corrected_fname = os.path.join(output_dir, 'proc_'+basename)
                trim_fname = os.path.join(output_dir, 'trim_'+basename)
                crr_fname = os.path.join(output_dir, 'crr_'+basename)
                # Bias correction, Flat correction
                flat_fname = master_flat[sci_img.filter]
                try:
                    _ = raw_correction(sci_img.data, sci_img.header, master_bias_fname, flat_fname,
                                       output=corrected_fname, overwrite=True, overscan=50, mode='img')
                    log.commit("          - bias+flat ")
                    temp_images.append(corrected_fname)
                except:
                    log.error("Bias and flat field correction failed!")
                    log.fatal_error()
                    print("Unexpected error:", sys.exc_info()[0])
                    raise

                # Trim edges:
                image_region = filter_edges[filter_raw]
                try:
                    _ = trim_filter_edge(corrected_fname, *image_region, output=trim_fname)
                    log.commit(" trim ")
                except:
                    log.error("Image trim failed!")
                    log.fatal_error()
                    print("Unexpected error:", sys.exc_info()[0])
                    raise

                # Correct Cosmic Rays Hits:
                if options['crr']['niter'] > 0:
                    try:
                        log.commit(" crr ")
                        _ = correct_cosmics(trim_fname, crr_fname, **options['crr'])
                        log.commit("  [done]")
                        corrected_images.append(crr_fname)
                        temp_images.append(trim_fname)
                    except:
                        log.error("Cosmic ray correction failed!")
                        log.fatal_error()
                        print("Unexpected error:", sys.exc_info()[0])
                        raise
                else:
                    corrected_images.append(trim_fname)
                log.commit("\n")

            log.add_linebreak()


            # Create Fringe image:
            if options['skysub']['defringe']:
                log.write("Running task: Creating Average Fringe Image")
                fringe_fname = os.path.join(output_dir, 'fringe_image.fits')
                fringe_pdf_fname = os.path.join(output_dir, 'fringe_image.pdf')
                try:
                    msg = create_fringe_image(corrected_images, output=fringe_fname, fig_fname=fringe_pdf_fname,
                                              threshold=3)
                    log.commit(msg)
                    log.add_linebreak()
                except:
                    log.error("Image combination failed!")
                    log.fatal_error()
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            else:
                fringe_fname = ''


            # Combine individual images for a given filter:
            if len(image_list) > 50:
                log.warning("Large amounts of memory needed for image combination!", force=True)
                log.warning("A total of %i images will be combined." % len(image_list), force=True)
            log.write("Running task: Image Combination")
            combined_fname = os.path.join(output_obj_base, '%s_%s.fits' % (target_name, filter_name))
            comb_log_name = os.path.join(output_dir, 'filelist_%s.txt' % target_name)
            try:
                output_msg = image_combine(corrected_images, output=combined_fname, log_name=comb_log_name,
                                           fringe_image=fringe_fname, **options['combine'])
                log.commit(output_msg)
                log.add_linebreak()
            except:
                log.error("Image combination failed!")
                log.fatal_error()
                print("Unexpected error:", sys.exc_info()[0])
                raise

            # Calculate Zero Point:
            zp = 0  # use instrument mags for now

            # Automatic Source Detection and Aperture Photometry:
            try:
                log.write("Running task: Source Extraction")
                _, _, output_msg = source_detection(combined_fname, zeropoint=zp,
                                                    kwargs_bg=options['sep-background'],
                                                    kwargs_ext=options['sep-extract'])
                log.commit(output_msg)
                log.add_linebreak()
            except:
                log.error("Image combination failed!")
                log.fatal_error()
                print("Unexpected error:", sys.exc_info()[0])
                raise

            # Clean up temporary files:
            if options['clean']:
                log.write("Cleaning up temporary images:")
                for fname in temp_images:
                    os.system("rm %s" % fname)
                    log.write(fname)


    log.exit()
