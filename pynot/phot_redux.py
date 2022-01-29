"""
Automatically Classify and Reduce a given Data Set
"""

from astropy.io import fits
from collections import defaultdict
import os
import sys
import numpy as np

from pynot import instrument
from pynot.data import io
from pynot.data import organizer as do
from pynot.data import obs
from pynot.phot import image_combine, create_fringe_image, source_detection, flux_calibration_sdss
from pynot.calibs import combine_bias_frames, combine_flat_frames
from pynot.functions import get_options, get_version_number
from pynot.scired import raw_correction, correct_cosmics, trim_filter_edge, detect_filter_edge
from pynot.wcs import correct_wcs
from pynot.logging import Report

from PyQt5.QtWidgets import QApplication

code_dir = os.path.dirname(os.path.abspath(__file__))
calib_dir = os.path.join(code_dir, 'calib/')
defaults_fname = os.path.join(calib_dir, 'default_options_img.yml')
__version__ = get_version_number()



def run_pipeline(options_fname, verbose=False, force_restart=False):
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
    try:
        raw_image_list = list(map(do.RawImage, object_filelist))
    except (ValueError, do.UnknownObservingMode, OSError, FileNotFoundError, TypeError, IndexError) as e:
        log.error(str(e))
        log.fatal_error()
        return

    object_images = defaultdict(lambda: defaultdict(list))
    for sci_img in raw_image_list:
        object_images[sci_img.target_name][sci_img.filter].append(sci_img)

    obd_fname = os.path.splitext(dataset_fname)[0] + '.obd'
    obdb = obs.OBDatabase(obd_fname)
    if os.path.exists(obd_fname):
        log.write("Loaded OB database: %s" % obd_fname)
    else:
        log.write("Initiated OB database: %s" % obd_fname)
    obdb.update_imaging(object_images)
    log.write("Updating OB database")
    log.add_linebreak()

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
        this_filter = instrument.get_filter(fits.getheader(flat_file))
        if this_filter in filter_list:
            flat_images_for_filter[this_filter].append(flat_file)

    # All files are put in a folder: imaging/OBJNAME/filter/...
    output_base = obs.output_base_phot
    if not os.path.exists(output_base):
        os.mkdir(output_base)

    # Combine Bias Frames matched for CCD setup:
    master_bias_fname = os.path.join(output_base, 'MASTER_BIAS.fits')
    bias_frames = raw_image_list[0].match_files(database['BIAS'], date=False)
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
                                              kappa=options['bias']['kappa'], method=options['bias']['method'],
                                              overwrite=True)
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
    for filter_name, flat_frames in flat_images_for_filter.items():
        log.write("Combining images for filter: %s" % filter_name)
        comb_flat_fname = os.path.join(output_base, 'FLAT_%s.fits' % filter_name)
        try:
            _, flat_msg = combine_flat_frames(flat_frames, comb_flat_fname, mbias=master_bias_fname,
                                              kappa=options['flat']['kappa'],
                                              method=options['flat']['method'],
                                              overwrite=True, mode='img')
            log.commit(flat_msg)
            master_flat[filter_name] = comb_flat_fname
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
            filter_edges[filter_name] = (x1, x2, y1, y2)
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

        for filter_name, image_list in images_per_filter.items():
            # Create working directory:
            output_dir = os.path.join(output_obj_base, filter_name)
            if obdb.data[output_dir] in ['DONE', 'SKIP'] and not force_restart:
                log.write("Skipping OB: %s  (status=%s)" % (output_dir, obdb.data[output_dir]))
                log.write("Change OB status to blank in the .obd file if you want to redo the reduction")
                log.write("or run the pipeline with the '-f' option to force re-reduction of all OBs")
                log.add_linebreak()
                continue
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
                                       output=corrected_fname, overwrite=True, mode='img')
                    log.commit("          - bias+flat ")
                    temp_images.append(corrected_fname)
                except:
                    log.error("Bias and flat field correction failed!")
                    log.fatal_error()
                    print("Unexpected error:", sys.exc_info()[0])
                    raise

                # Trim edges:
                image_region = filter_edges[filter_name]
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


            N_images = len(corrected_images)
            # Create Fringe image:
            if options['skysub']['defringe'] and N_images > 3:
                log.write("Running task: Creating Average Fringe Image")
                fringe_fname = os.path.join(output_dir, 'fringe_image_%s.fits' % filter_name)
                fringe_pdf_fname = os.path.join(output_dir, 'fringe_image_%s.pdf' % filter_name)
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
            elif options['skysub']['defringe'] and N_images <= 3:
                log.warn("No fringe image can be created. Need at least 3 images.")
            else:
                fringe_fname = ''


            # Combine individual images for a given filter:
            if len(image_list) > 50:
                log.warn("Large amounts of memory needed for image combination!", force=True)
                log.warn("A total of %i images will be combined." % len(image_list), force=True)

            log.write("Running task: Image Combination")
            comb_log_name = os.path.join(output_dir, 'filelist_%s.txt' % target_name)
            combined_fname = os.path.join(output_obj_base, '%s_%s.fits' % (target_name, filter_name))
            try:
                output_msg = image_combine(corrected_images, output=combined_fname, log_name=comb_log_name,
                                           fringe_image=fringe_fname, **options['combine'])
                log.commit(output_msg)
                log.add_linebreak()
            except (IndexError, FileNotFoundError, OSError) as e:
                log.error("Image combination failed!")
                log.error(str(e))
                log.fatal_error()
                return
            except:
                log.datal_error()
                print("Unexpected error:", sys.exc_info()[0])
                raise


            # Automatic Source Detection and Aperture Photometry:
            try:
                log.write("Running task: Source Extraction")
                sep_fname, _, output_msg = source_detection(combined_fname, zeropoint=0,
                                                            kwargs_bg=options['sep-background'],
                                                            kwargs_ext=options['sep-extract'])
                log.commit(output_msg)
                log.add_linebreak()
            except:
                log.error("Source extraction failed!")
                log.fatal_error()
                print("Unexpected error:", sys.exc_info()[0])
                raise


            # Calibrate WCS:
            try:
                log.write("Running task: WCS calibration")
                output_msg = correct_wcs(combined_fname, sep_fname, **options['wcs'])
                log.commit(output_msg)
                log.add_linebreak()
            except:
                log.error("WCS calibration failed!")
                log.fatal_error()
                print("Unexpected error:", sys.exc_info()[0])
                raise


            # Calculate Zero Point:
            if 'SDSS' in filter_name.upper():
                try:
                    log.write("Running task: Self-calibration of magnitude zero point")
                    output_msg = flux_calibration_sdss(combined_fname, sep_fname, **options['sdss_flux'])
                    log.commit(output_msg)
                    log.add_linebreak()
                except:
                    log.error("Zero point calibration failed!")
                    log.fatal_error()
                    print("Unexpected error:", sys.exc_info()[0])
                    raise

            # Clean up temporary files:
            if options['clean']:
                log.write("Cleaning up temporary images:")
                for fname in temp_images:
                    os.system("rm %s" % fname)
                    log.write(fname)
                log.add_linebreak()

            obdb.update(output_dir, 'DONE')

    log.exit()
