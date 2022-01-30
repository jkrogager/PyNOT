"""
Automatically Classify and Reduce a given Data Set
"""

from astropy.io import fits
from collections import defaultdict
import glob
import numpy as np
import os
import sys

from pynot import instrument
from pynot.data import io
from pynot.data import organizer as do
from pynot.data import obs
from pynot.calibs import combine_bias_frames, combine_flat_frames, normalize_spectral_flat, task_bias
from pynot.extraction import auto_extract
from pynot import extract_gui
from pynot.functions import get_options, get_version_number
from pynot.wavecal import rectify, WavelengthError
from pynot.identify_gui import create_pixtable
from pynot.scired import raw_correction, auto_fit_background, correct_cosmics, correct_raw_file
from pynot.scombine import combine_2d, combine_1d
from pynot.response import calculate_response, flux_calibrate
from pynot.logging import Report
from PyQt5.QtWidgets import QApplication

code_dir = os.path.dirname(os.path.abspath(__file__))
calib_dir = os.path.join(code_dir, 'calib/')
defaults_fname = os.path.join(calib_dir, 'default_options.yml')
__version__ = get_version_number()


class State(dict):
    """A collection of variables for the pipeline, such as arc line ID tables etc."""
    def __init__(self):
        dict.__init__(self, {})
        self.current = ''

    def print_current_state(self):
        print(self.current)

    def set_current_state(self, state):
        self.current = state



class ArgumentDict(dict):
    """Access dictionary keys as attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



def run_pipeline(options_fname, object_id=None, verbose=False, interactive=False, force_restart=False):
    log = Report(verbose)
    status = State()

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

    if object_id is None:
        pass
    elif isinstance(object_id, str):
        object_id = [object_id]
    elif isinstance(object_id, list):
        if isinstance(object_id[0], str):
            pass
        else:
            log.error("Wrong input type for `object id`. Must be list of strings")
            log.error("not list of %r" % type(object_id[0]))
            log.fatal_error()
            return
    else:
        log.error("Wrong input type for `object id`. Must be string or list of strings")
        log.error("not %r" % type(object_id))
        log.fatal_error()
        return

    if interactive:
        # Set all interactive steps to True
        options['identify']['interactive'] = True
        options['identify']['all'] = True
        options['extract']['interactive'] = True
        options['response']['interactive'] = True

    dataset_fname = options['dataset']
    if dataset_fname and os.path.exists(dataset_fname):
        # -- load collection
        database = io.load_database(dataset_fname)
        log.write("Loaded file classification database: %s" % dataset_fname)
    else:
        log.error("Dataset does not exist : %s" % dataset_fname)
        log.fatal_error()
        return

    # -- Organize object files in dataset:
    if 'SPEC_OBJECT' not in database:
        log.error("No spectroscopic data found in the dataset!")
        log.error("Check the classification table... object type 'SPEC_OBJECT' missing")
        log.fatal_error()
        return
    object_filelist = database['SPEC_OBJECT']
    try:
        object_images = list(map(do.RawImage, object_filelist))
    except (ValueError, do.UnknownObservingMode, OSError, FileNotFoundError, TypeError, IndexError) as e:
        log.error(str(e))
        log.fatal_error()
        raise

    log.add_linebreak()
    log.write(" - The following objects were found in the dataset:", prefix='')
    log.write("      OBJECT           GRISM        SLIT      EXPTIME       FILENAME", prefix='')
    for sci_img in object_images:
        output_variables = (sci_img.object, sci_img.grism, sci_img.slit, sci_img.exptime, os.path.basename(sci_img.filename))
        log.write("%20s  %9s  %11s   %5.0f  %s" % output_variables, prefix='')
    log.add_linebreak()


    # Start Calibration Tasks:
    output_base = obs.output_base_spec

    # -- bias
    if not database.has_tag('MBIAS') or force_restart:
        task_args = ArgumentDict(options['bias'])
        task_output, log = task_bias(task_args, database=database, log=log, verbose=verbose, output_dir=output_base)
        for tag, filelist in task_output.items():
            database[tag] = filelist
        io.save_database(database, dataset_fname)

    # -- sflat
    # task_args = ArgumentDict(options['flat'])
    # task_output, log = task_bias(task_args, database=database, log=log, verbose=verbose, output_dir=output_base)
    # for tag, filelist in task_output.items():
    #     database[tag] = filelist
    # io.save_database(database, dataset_fname)

    # -- identify
    # get list of unique grisms in dataset:
    grism_list = list()
    for sci_img in object_images:
        grism_name = sci_img.grism
        if grism_name not in grism_list:
            grism_list.append(grism_name)

    # -- Check arc line files:
    arc_images = list()
    for arc_type in ['ARC', 'ARC_HeNe', 'ARC_ThAr', 'ARC_HeAr']:
        if arc_type in database.keys():
            arc_images += database[arc_type]

    if len(arc_images) == 0:
        log.error("No arc line calibration data found in the dataset!")
        log.error("Check the classification table... object type 'ARC_HeNe', 'ARC_ThAr', 'ARC_HeAr' missing")
        log.fatal_error()
        return

    arc_images_for_grism = defaultdict(list)
    for arc_fname in arc_images:
        this_grism = instrument.get_grism(fits.getheader(arc_fname))
        arc_images_for_grism[this_grism].append(arc_fname)

    for grism_name in grism_list:
        if len(arc_images_for_grism[grism_name]) == 0:
            log.error("No arc frames defined for grism: %s" % grism_name)
            log.fatal_error()
            return

    log.add_linebreak()
    identify_all = options['identify']['all']
    identify_interactive = options['identify']['interactive']
    if identify_interactive and identify_all:
        grisms_to_identify = []
        log.write("Identify: interactively reidentify arc lines for all objects")
        log.add_linebreak()

    elif identify_interactive and not identify_all:
        # Make pixeltable for all grisms:
        grisms_to_identify = grism_list
        log.write("Identify: interactively identify all grisms in dataset:")
        log.write(", ".join(grisms_to_identify))
        log.add_linebreak()

    else:
        # Check if pixeltables exist:
        grisms_to_identify = []
        for grism_name in grism_list:
            pixtab_fname = os.path.join(calib_dir, '%s_pixeltable.dat' % grism_name)
            if not os.path.exists(pixtab_fname):
                log.write("%s : pixel table does not exist. Will identify lines..." % grism_name)
                grisms_to_identify.append(grism_name)
            else:
                log.write("%s : pixel table already exists" % grism_name)
                status['%s_pixtab' % grism_name] = pixtab_fname
        log.add_linebreak()


    # Identify interactively for grisms that are not defined
    # add the new pixel tables to the calib cache for future use
    for grism_name in grisms_to_identify:
        log.write("Starting interactive definition of pixel table for %s" % grism_name)
        try:
            arc_fname = arc_images_for_grism[grism_name][0]
            pixtab_fname = os.path.join(calib_dir, '%s_pixeltable.dat' % grism_name)
            linelist_fname = ''
            log.write("Input arc line frame: %s" % arc_fname)

            arc_base_fname = os.path.basename(arc_fname)
            arc_base, ext = os.path.splitext(arc_base_fname)
            output_pixtable = os.path.join(output_base, "pixtab_%s_%s.tab" % (arc_base, grism_name))
            poly_order, saved_pixtab_fname, msg = create_pixtable(arc_fname, grism_name, output_pixtable,
                                                                  pixtab_fname, linelist_fname,
                                                                  order_wl=options['identify']['order_wl'],
                                                                  app=app)
            status['%s_pixtab' % grism_name] = output_pixtable
            log.commit(msg)
        except:
            log.error("Identification of arc lines failed!")
            log.fatal_error()
            log.save()
            print("Unexpected error:", sys.exc_info()[0])
            raise


    # -- response

    # Save overview log:
    print("")
    print(" - Pipeline setup ended successfully.")
    print("   Consult the overview log: %s\n\n" % log.fname)
    log.save()


    # ------------------------------------------------------------------
    # -- Start Main Reduction:
    if object_id is None:
        # Loop over all:
        objects_to_reduce = object_images
    else:
        objects_to_reduce = list()
        for img in object_images:
            if img.object in object_id:
                objects_to_reduce.append(img)

        if len(objects_to_reduce) == 0:
            log.error("No data matched the given object ID: %r" % object_id)
            log.fatal_error()
            return

    # Organize the science files according to target and instrument setup (insID)
    science_frames = defaultdict(lambda: defaultdict(list))
    for sci_img in objects_to_reduce:
        filt_name = sci_img.filter
        insID = "%s_%s" % (sci_img.grism, sci_img.slit.replace('_', ''))
        if filt_name.lower() not in ['free', 'open', 'none']:
            insID = "%s_%s" % (insID, filt_name)
        science_frames[sci_img.target_name][insID].append(sci_img)

    obd_fname = os.path.splitext(dataset_fname)[0] + '.obd'
    obdb = obs.OBDatabase(obd_fname)
    if os.path.exists(obd_fname):
        log.write("Loaded OB database: %s" % obd_fname)
    else:
        log.write("Initiated OB database: %s" % obd_fname)
    obdb.update_spectra(science_frames)
    log.write("Updating OB database")
    log.add_linebreak()

    for target_name, frames_per_setup in science_frames.items():
        for insID, frames in frames_per_setup.items():
            for obnum, sci_img in enumerate(frames, 1):
                # Create working directory:
                obID = 'ob%i' % obnum
                output_dir = os.path.join(output_base, sci_img.target_name, insID, obID)
                if obdb.data[output_dir] in ['DONE', 'SKIP']:
                    if force_restart and obdb.data[output_dir] == 'DONE':
                        pass
                    else:
                        log.write("Skipping OB: %s  (status=%s)" % (output_dir, obdb.data[output_dir]))
                        log.write("Change OB status to blank in the .obd file if you want to redo the reduction")
                        log.write("or run the pipeline with the '-f' option to force re-reduction of all OBs")
                        log.add_linebreak()
                        continue
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Start new log in working directory:
                log_fname = os.path.join(output_dir, 'pynot.log')
                log.clear()
                log.set_filename(log_fname)
                log.write("------------------------------------------------------------", prefix='')
                log.write("Starting PyNOT Longslit Spectroscopic Reduction")
                log.add_linebreak()
                log.write("Target Name: %s" % sci_img.target_name)
                log.write("Input Filename: %s" % sci_img.filename)
                log.write("Saving output to directory: %s" % output_dir)
                log.add_linebreak()

                # Prepare output filenames:
                grism = sci_img.grism
                # master_bias_fname = os.path.join(output_dir, 'MASTER_BIAS.fits')
                comb_flat_fname = os.path.join(output_dir, 'FLAT_COMBINED_%s_%s.fits' % (grism, sci_img.slit))
                norm_flat_fname = os.path.join(output_dir, 'NORM_FLAT_%s_%s.fits' % (grism, sci_img.slit))
                rect2d_fname = os.path.join(output_dir, 'RECT2D_%s.fits' % (sci_img.target_name))
                bgsub2d_fname = os.path.join(output_dir, 'BGSUB2D_%s.fits' % (sci_img.target_name))
                response_pdf = os.path.join(output_dir, 'plot_response_%s.pdf' % (grism))
                corrected_2d_fname = os.path.join(output_dir, 'CORRECTED2D_%s.fits' % (sci_img.target_name))
                flux2d_fname = os.path.join(output_dir, 'FLUX2D_%s.fits' % (sci_img.target_name))
                flux1d_fname = os.path.join(output_dir, 'FLUX1D_%s.fits' % (sci_img.target_name))
                extract_pdf_fname = os.path.join(output_dir, 'plot_extract1D_details.pdf')

                # Find Bias Frame:
                master_bias = sci_img.match_files(database['MBIAS'], date=False)
                if len(master_bias) > 1:
                    master_bias = sci_img.match_files(database['MBIAS'], date=False, get_closest_time=True)

                if len(master_bias) != 1:
                    log.error("Could not find a matching master bias.")
                    log.error("Check filetype MBIAS in %s" % dataset_fname)
                    log.fatal_error()
                    return
                master_bias_fname = master_bias[0]


                # # Find Flat Frame:
                # master_flat = sci_img.match_files(database['NORM_SFLAT'], date=False,
                #                                   grism=True, slit=True, filter=True)
                # if len(master_flat) > 1:
                #     master_flat = sci_img.match_files(database['NORM_SFLAT'], date=False,
                #                                       grism=True, slit=True, filter=True,
                #                                       get_closest_time=True)
                # if len(master_bias) != 1:
                #     log.error("Could not find a matching nomalized flat.")
                #     log.error("Check filetype NORM_SFLAT in %s" % dataset_fname)
                #     log.fatal_error()
                #     return
                # norm_flat_fname = master_flat[0]


                # Combine Flat Frames matched for CCD setup, grism, slit and filter:
                flat_frames = sci_img.match_files(database['SPEC_FLAT'], date=False, grism=True, slit=True, filter=True)
                perform_flat_comb = True
                if options['mflat']:
                    if options['mflat'] is None:
                        norm_flat_fname = ''
                    elif options['mflat'].lower() in ['none', 'null']:
                        norm_flat_fname = ''
                    else:
                        norm_flat_fname = options['mflat']
                    log.write("Using static master flat frame: %s" % options['mflat'])
                    log.add_linebreak()

                elif os.path.exists(os.path.join(output_base, os.path.basename(norm_flat_fname))):
                    norm_flat_fname = os.path.join(output_base, os.path.basename(norm_flat_fname))
                    # check that image shapes match:
                    flat_hdr = fits.getheader(norm_flat_fname)
                    flat_img = fits.getdata(norm_flat_fname)
                    flat_binning = instrument.get_binning_from_hdr(flat_hdr)
                    # Change this to match the image shape *after* overscan correction
                    if sci_img.binning == flat_binning and sci_img.shape == flat_img.shape:
                        log.write("Using normalized flat frame: %s" % norm_flat_fname)
                        log.add_linebreak()
                        perform_flat_comb = False
                    else:
                        perform_flat_comb = True

                if len(flat_frames) == 0:
                    log.error("No flat frames provided!")
                    log.fatal_error()
                    return
                elif perform_flat_comb:
                    try:
                        log.write("Running task: Spectral Flat Combination")
                        _, flat_msg = combine_flat_frames(flat_frames, comb_flat_fname, mbias=master_bias_fname,
                                                          kappa=options['flat']['kappa'],
                                                          method=options['flat']['method'], overwrite=True,
                                                          mode='spec', dispaxis=sci_img.dispaxis)
                        log.commit(flat_msg)
                        status['flat_combined'] = os.path.join(output_base, os.path.basename(comb_flat_fname))
                        copy_flat = "cp %s %s" % (comb_flat_fname, status['flat_combined'])
                        if not os.path.exists(status['flat_combined']):
                            os.system(copy_flat)
                        log.write("Copied combined Flat Image to base working directory")
                        log.add_linebreak()
                    except ValueError as err:
                        log.commit(str(err)+'\n')
                        log.fatal_error()
                        raise
                    except:
                        log.error("Combination of flat frames failed!")
                        log.fatal_error()
                        print("Unexpected error:", sys.exc_info()[0])
                        raise

                    # Normalize the spectral flat field:
                    try:
                        log.write("Running task: Spectral Flat Normalization")
                        _, norm_msg = normalize_spectral_flat(comb_flat_fname, output=norm_flat_fname,
                                                              fig_dir=output_dir, dispaxis=sci_img.dispaxis,
                                                              **options['flat'])
                        log.commit(norm_msg)
                        status['master_flat'] = os.path.join(output_base, os.path.basename(norm_flat_fname))
                        copy_normflat = "cp %s %s" % (norm_flat_fname, status['master_flat'])
                        if not os.path.exists(status['master_flat']):
                            os.system(copy_normflat)
                        log.write("Copied normalized Flat Image to base working directory")
                        log.add_linebreak()
                    except:
                        log.error("Normalization of flat frames failed!")
                        log.fatal_error()
                        print("Unexpected error:", sys.exc_info()[0])
                        raise


                # Identify lines in arc frame:
                arc_fname, = sci_img.match_files(arc_images, date=False, grism=True, slit=True, filter=True, get_closest_time=True)
                corrected_arc2d_fname = os.path.join(output_dir, 'corr_arc2d.fits')
                log.write("Running task: Bias and Flat Field Correction of Arc Frame")
                try:
                    output_msg = correct_raw_file(arc_fname, bias_fname=master_bias_fname, flat_fname=norm_flat_fname,
                                                  output=corrected_arc2d_fname, overwrite=True)
                    log.commit(output_msg)
                    log.add_linebreak()
                except:
                    log.error("Bias and flat field correction of Arc frame failed!")
                    log.fatal_error()
                    print("Unexpected error:", sys.exc_info()[0])
                    raise

                pixtab_fname = status['%s_pixtab' % grism]
                if identify_interactive and identify_all:
                    log.write("Running task: Arc Line Identification")
                    try:
                        linelist_fname = ''
                        output_pixtable_fname = os.path.join(output_base, '%s_pixtab.dat' % insID)
                        order_wl, pixtable, msg = create_pixtable(corrected_arc2d_fname, grism,
                                                                  output_pixtable_fname,
                                                                  pixtab_fname, linelist_fname,
                                                                  order_wl=options['identify']['order_wl'],
                                                                  app=app)
                        status['%s_pixtab' % grism] = pixtable
                        log.commit(msg)
                        log.add_linebreak()
                    except Exception:
                        log.error("Identification of arc lines failed!")
                        log.fatal_error()
                        print("Unexpected error:", sys.exc_info()[0])
                        raise


                # Response Function:
                if 'SPEC_FLUX-STD' in database:
                    flux_std_files = sci_img.match_files(database['SPEC_FLUX-STD'],
                                                         date=False, grism=True, slit=True, filter=True, get_closest_time=True)
                else:
                    flux_std_files = []

                if len(flux_std_files) == 0:
                    log.warn("No spectroscopic standard star was found in the dataset!")
                    log.warn("The reduced spectra will not be flux calibrated")
                    status['response'] = None

                else:
                    std_fname = flux_std_files[0]
                    response_fname = os.path.join(output_base, 'response_%s.fits' % (grism))
                    if os.path.exists(response_fname) and not options['response']['force']:
                        log.write("Using existing response function: %s" % response_fname)
                        log.add_linebreak()
                        status['%s_response' % grism] = response_fname
                    else:
                        std_fname = flux_std_files[0]
                        log.write("Running task: Calculation of Response Function")
                        log.write("Spectroscopic Flux Standard: %s" % std_fname)
                        try:
                            response_fname, response_msg = calculate_response(std_fname, arc_fname=corrected_arc2d_fname,
                                                                              pixtable_fname=status['%s_pixtab' % grism],
                                                                              bias_fname=master_bias_fname,
                                                                              flat_fname=norm_flat_fname,
                                                                              output=response_fname,
                                                                              output_dir=output_dir, pdf_fname=response_pdf,
                                                                              order=options['response']['order'],
                                                                              interactive=options['response']['interactive'],
                                                                              dispaxis=sci_img.dispaxis,
                                                                              order_bg=options['skysub']['order_bg'],
                                                                              rectify_options=options['rectify'],
                                                                              app=app)
                            status['%s_response' % grism] = response_fname
                            log.commit(response_msg)
                            log.add_linebreak()
                        except Exception:
                            log.error("Calculation of response function failed!")
                            # print("Unexpected error:", sys.exc_info()[0])
                            # raise
                            status['%s_response' % grism] = ''
                            log.warn("No flux calibration will be performed!")
                            log.add_linebreak()


                # Bias correction, Flat correction
                log.write("Running task: Bias and Flat Field Correction")
                try:
                    output_msg = raw_correction(sci_img.data, sci_img.header, master_bias_fname, norm_flat_fname,
                                                output=corrected_2d_fname, overwrite=True)
                    log.commit(output_msg)
                    log.add_linebreak()
                except Exception:
                    log.error("Bias and flat field correction failed!")
                    log.fatal_error()
                    print("Unexpected error:", sys.exc_info()[0])
                    raise


                # Call rectify
                log.write("Running task: 2D Rectification and Wavelength Calibration")
                try:
                    rect_msg = rectify(corrected_2d_fname, corrected_arc2d_fname, status['%s_pixtab' % grism],
                                       output=rect2d_fname, fig_dir=output_dir,
                                       dispaxis=sci_img.dispaxis, **options['rectify'])
                    log.commit(rect_msg)
                    log.add_linebreak()
                except WavelengthError:
                    log.error("2D rectification failed!")
                    log.fatal_error()
                    print("Unexpected error:", sys.exc_info()[0])
                    print("")
                    raise


                # Automatic Background Subtraction:
                if options['skysub']['auto']:
                    bgsub_pdf_name = os.path.join(output_dir, 'plot_skysub2D.pdf')
                    log.write("Running task: Background Subtraction")
                    try:
                        bg_msg = auto_fit_background(rect2d_fname, bgsub2d_fname, dispaxis=1,
                                                     plot_fname=bgsub_pdf_name, **options['skysub'])
                        log.commit(bg_msg)
                        log.write("2D sky model is saved in extension 'SKY' of the file: %s" % bgsub2d_fname)
                        log.add_linebreak()
                    except Exception:
                        log.error("Automatic background subtraction failed!")
                        log.fatal_error()
                        print("Unexpected error:", sys.exc_info()[0])
                        raise
                else:
                    log.warn("No sky-subtraction has been performed on the 2D spectrum!")
                    log.write("Cosmic ray rejection may fail... double check the output or turn off 'crr' by setting niter=0.")
                    log.add_linebreak()
                    bgsub2d_fname = rect2d_fname


                # Correct Cosmic Rays Hits:
                if options['crr']['niter'] > 0:
                    log.write("Running task: Cosmic Ray Rejection")
                    crr_fname = os.path.join(output_dir, 'CRR_BGSUB2D_%s.fits' % (sci_img.target_name))
                    try:
                        crr_msg = correct_cosmics(bgsub2d_fname, crr_fname, **options['crr'])
                        log.commit(crr_msg)
                        log.add_linebreak()
                    except Exception:
                        log.error("Cosmic ray correction failed!")
                        log.fatal_error()
                        print("Unexpected error:", sys.exc_info()[0])
                        raise
                else:
                    crr_fname = bgsub2d_fname


                # Flux Calibration:
                if status['%s_response' % grism]:
                    log.write("Running task: Flux Calibration")
                    response_fname = status['%s_response' % grism]
                    try:
                        flux_msg = flux_calibrate(crr_fname, output=flux2d_fname, response_fname=response_fname)
                        log.commit(flux_msg)
                        log.add_linebreak()
                        status['FLUX2D'] = flux2d_fname
                    except Exception:
                        log.error("Flux calibration failed!")
                        log.fatal_error()
                        print("Unexpected error:", sys.exc_info()[0])
                        raise
                else:
                    status['FLUX2D'] = crr_fname


                # Extract 1D spectrum:
                log.write("Running task: 1D Extraction")
                extract_fname = status['FLUX2D']
                if options['extract']['interactive']:
                    try:
                        log.write("Extraction: Starting Graphical User Interface")
                        extract_gui.run_gui(extract_fname, output_fname=flux1d_fname,
                                            app=app, **options['extract'])
                        log.write("Writing fits table: %s" % flux1d_fname, prefix=" [OUTPUT] - ")
                    except:
                        log.error("Interactive 1D extraction failed!")
                        log.fatal_error()
                        print("Unexpected error:", sys.exc_info()[0])
                        raise
                else:
                    try:
                        ext_msg = auto_extract(extract_fname, flux1d_fname,
                                               dispaxis=1, pdf_fname=extract_pdf_fname,
                                               **options['extract'])
                        log.commit(ext_msg)
                        log.add_linebreak()
                    except np.linalg.LinAlgError:
                        log.warn("Automatic extraction failed. Try manual extraction...")
                    except Exception:
                        log.error("Spectral 1D extraction failed!")
                        log.fatal_error()
                        print("Unexpected error:", sys.exc_info()[0])
                        raise

                obdb.update(output_dir, 'DONE')
                log.exit()



            # -- Combine OBs for same target:

            # Check whether to combine or link OB files:
            pattern = os.path.join(output_base, target_name, insID, '*', 'FLUX2D*.fits')
            files_to_combine = glob.glob(pattern)
            files_to_combine = list(filter(lambda x: obdb.data[os.path.dirname(x)] == 'DONE', files_to_combine))
            if len(files_to_combine) > 1:
                # Combine individual OBs
                comb_basename = '%s_%s_flux2d.fits' % (target_name, insID)
                comb2d_fname = os.path.join(output_base, target_name, comb_basename)
                if not os.path.exists(comb2d_fname) or force_restart:
                    log.write("Running task: Spectral Combination")
                    try:
                        comb_output = combine_2d(files_to_combine, comb2d_fname)
                        final_wl, final_flux, final_err, final_mask, output_msg = comb_output
                        log.commit(output_msg)
                        log.add_linebreak()
                    except Exception:
                        log.warn("Combination of 2D spectra failed... Try again manually")
                        raise

                comb_basename = '%s_%s_flux1d.fits' % (target_name, insID)
                comb1d_fname = os.path.join(output_base, target_name, comb_basename)
                if not os.path.exists(comb1d_fname) or force_restart:
                    log.write("Running task: 1D Extraction")
                    if options['extract']['interactive']:
                        try:
                            log.write("Extraction: Starting Graphical User Interface")
                            extract_gui.run_gui(comb2d_fname, output_fname=comb1d_fname,
                                                app=app, **options['extract'])
                            log.write("Writing fits table: %s" % comb1d_fname, prefix=" [OUTPUT] - ")
                        except:
                            log.error("Interactive 1D extraction failed!")
                            log.fatal_error()
                            print("Unexpected error:", sys.exc_info()[0])
                            raise
                    else:
                        try:
                            pdf_basename = '%s_extract1D_details.pdf' % insID
                            extract_pdf_fname = os.path.join(output_base, target_name, pdf_basename)
                            ext_msg = auto_extract(comb2d_fname, comb1d_fname,
                                                   dispaxis=1, pdf_fname=extract_pdf_fname,
                                                   **options['extract'])
                            log.commit(ext_msg)
                            log.add_linebreak()
                        except Exception:
                            log.warn("Automatic extraction failed. Try manual extraction...")

            elif len(files_to_combine) == 1:
                # Create a hard link to the individual file instead
                comb_basename = '%s_%s_flux2d.fits' % (target_name, insID)
                comb2d_fname = os.path.join(output_base, target_name, comb_basename)
                source_2d = files_to_combine[0]
                if not os.path.exists(comb2d_fname):
                    os.link(source_2d, comb2d_fname)
                    log.write("Created file link:")
                    log.write("%s -> %s" % (source_2d, comb2d_fname), prefix=" [OUTPUT] - ")

                comb_basename = '%s_%s_flux1d.fits' % (target_name, insID)
                comb1d_fname = os.path.join(output_base, target_name, comb_basename)
                source_1d = source_2d.replace('FLUX2D', 'FLUX1D')
                if not os.path.exists(comb1d_fname):
                    os.link(source_1d, comb1d_fname)
                    log.write("Created file link:")
                    log.write("%s -> %s" % (source_1d, comb1d_fname), prefix=" [OUTPUT] - ")
                log.add_linebreak()
