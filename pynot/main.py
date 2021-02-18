"""
  PyNOT Data Reduction Pipeline for NOT/ALFOSC

For available tasks, run:
    %] pynot -h

To create a default parameter file, run:
    %] pynot init
"""

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import copy
import os
import sys
import numpy as np
import warnings

from pynot.functions import get_options, get_option_descr, get_version_number

code_dir = os.path.dirname(os.path.abspath(__file__))
calib_dir = os.path.join(code_dir, 'calib/')
defaults_fname_spec = os.path.join(calib_dir, 'default_options.yml')
defaults_fname_phot = os.path.join(calib_dir, 'default_options_img.yml')
parameters = get_options(defaults_fname_spec)
parameter_descr = get_option_descr(defaults_fname_spec)

img_parameters = get_options(defaults_fname_phot)
img_parameter_descr = get_option_descr(defaults_fname_phot)

__version__ = get_version_number()

def print_credits():
    print("")
    print("  PyNOT Data Processing Pipeline ")
    print(" ================================")
    print("  version %s\n" % __version__)
    print("")


def set_default_pars(parser, *, section, default_type, ignore_pars=[], mode='spec'):
    if mode == 'spec':
        params = parameters[section]
        descriptions = parameter_descr[section]
    else:
        params = img_parameters[section]
        descriptions = img_parameter_descr[section]

    for key, val in params.items():
        if key in descriptions:
            par_descr = descriptions[key]
        else:
            par_descr = ''

        if val is None:
            value_type = default_type
        else:
            value_type = type(val)

        if key not in ignore_pars:
            parser.add_argument("--%s" % key, type=value_type, default=val, help=par_descr)


def set_help_width(max_width=30):
    """Return a wider HelpFormatter, if possible."""
    try:
        # https://stackoverflow.com/a/5464440
        return lambda prog: ArgumentDefaultsHelpFormatter(prog, max_help_position=max_width)
    except TypeError:
        warnings.warn("argparse help formatter failed, falling back.")
        return ArgumentDefaultsHelpFormatter


def initialize(args):
    print_credits()
    from pynot.data import organizer as do
    from pynot.data import io

    # Classify files:
    # args.silent is set to "store_false", so by default it is true!
    verbose = args.silent
    database, message = do.classify(args.path, progress=verbose)
    pfc_fname = args.output
    if pfc_fname[-4:] != '.pfc':
        pfc_fname += '.pfc'
    io.save_database(database, pfc_fname)
    if verbose:
        print(message)
    print(" [OUTPUT] - Saved file classification database: %s" % pfc_fname)

    if args.mode == 'spex':
        defaults_fname = defaults_fname_spec
    else:
        defaults_fname = defaults_fname_phot

    with open(defaults_fname) as opt:
        all_lines = opt.readlines()
    for num, line in enumerate(all_lines):
        if line.find('dataset:') == 0:
            break
    else:
        print(" [ERROR]  - Something went horribly wrong in the parameter file!!")
        print(" [ERROR]    Check the file: %s !" % defaults_fname)
        return

    # Input the PFC filename in the default parameters:
    root, rest = line.split(':')
    empty_str, comment = rest.split('#')
    fmt = "%%%is" % (len(empty_str) - 1)
    dataset_input = "%s: %s #%s" % (root, fmt % pfc_fname, comment)
    all_lines[num] = dataset_input

    pars_fname = args.pars
    if pars_fname[-4:] != '.yml':
        pars_fname += '.yml'

    # Write the new parameter file:
    with open(pars_fname, 'w') as parfile:
        parfile.write("".join(all_lines))

    print(" [OUTPUT] - Initiated new parameter file: %s\n" % pars_fname)
    print("\n You can now start the reduction pipeline by running:")
    print("   ]%% pynot %s  %s\n" % (args.mode, pars_fname))


def main():
    parser = ArgumentParser(prog='pynot')
    tasks = parser.add_subparsers(dest='task')

    p_break1 = tasks.add_parser('', help="")

    parser_init = tasks.add_parser('init', formatter_class=set_help_width(31),
                                   help="Initiate a default parameter file")
    parser_init.add_argument("mode", type=str, choices=['spex', 'phot'],
                             help="Create parameter file for spectroscopy or imaging?")
    parser_init.add_argument("path", type=str, nargs='+',
                             help="Path (or paths) to raw ALFOSC data to be classified")
    parser_init.add_argument('-p', "--pars", type=str, default='options.yml',
                             help="Filename of parameter file, default = options.yml")
    parser_init.add_argument("-o", "--output", type=str, default='dataset.pfc',
                             help="Filename of file classification table (*.pfc)")
    parser_init.add_argument("-s", "--silent", action='store_false',
                             help="Minimze the output to terminal")



    # -- BIAS :: Bias Combination
    parser_bias = tasks.add_parser('bias', formatter_class=set_help_width(31),
                                   help="Combine bias frames")
    parser_bias.add_argument("input", type=str,
                             help="Input file containing list of image filenames to combine")
    parser_bias.add_argument('-o', "--output", type=str, default='MASTER_BIAS.fits',
                             help="Output filename of combined bias frame  (default = MASTER_BIAS.fits)")
    parser_bias.add_argument("--kappa", type=float, default=15,
                             help="Threshold for sigma clipping")

    # -- SFLAT :: Spectral Flat Combination
    parser_sflat = tasks.add_parser('sflat', formatter_class=set_help_width(31),
                                    help="Combine and normalize spectral flat frames")
    parser_sflat.add_argument("input", type=str,
                              help="Input file containing list of image filenames to combine")
    parser_sflat.add_argument("--bias", type=str, required=True,
                              help="Filename of combined bias frame  [REQUIRED]")
    parser_sflat.add_argument("-o", "--output", type=str, required=True,
                              help="Output filename of combined bias frame  [REQUIRED]")
    parser_sflat.add_argument("--axis", type=int, default=2,
                              help="Dispersion axis: 1 horizontal, 2: vertical")
    # Define parameters based on default values:
    set_default_pars(parser_sflat, section='flat', default_type=int)


    # -- IMFLAT :: Imaging Flat Combination
    parser_imflat = tasks.add_parser('imflat', formatter_class=set_help_width(31),
                                     help="Combine imaging flat frames")
    parser_imflat.add_argument("input", type=str,
                               help="Input file containing list of image filenames to combine")
    parser_imflat.add_argument("--bias", type=str, required=True,
                               help="Filename of combined bias frame  [REQUIRED]")
    parser_imflat.add_argument("-o", "--output", type=str, required=True,
                               help="Output filename of combined bias frame  [REQUIRED]")
    parser_imflat.add_argument("--kappa", type=float, default=15,
                               help="Threshold for sigma clipping")

    # -- corr :: Raw Correction
    parser_corr = tasks.add_parser('corr', formatter_class=set_help_width(31),
                                   help="Apply bias subtraction, flat field correction and trimming")
    parser_corr.add_argument("input", type=str,
                             help="List of filenames to correct")
    parser_corr.add_argument("--dir", type=str, default='',
                             help="Output directory")
    parser_corr.add_argument("--bias", type=str, required=True,
                             help="Filename of combined bias frame  [REQUIRED]")
    parser_corr.add_argument("--flat", type=str, default='',
                             help="Filename of combined flat frame")
    parser_corr.add_argument("--img", action='store_true',
                             help="Imaging mode")


    # -- identify :: Identify Arc Lines
    parser_id = tasks.add_parser('identify', formatter_class=set_help_width(31),
                                 help="Interactive identification of arc lines")
    parser_id.add_argument("arc", type=str,
                           help="Input filename of arc line image")
    parser_id.add_argument("--lines", type=str, default='',
                           help="Linelist, automatically loaded if possible")
    parser_id.add_argument("--axis", type=int, default=2,
                           help="Dispersion axis: 1 horizontal, 2: vertical")
    parser_id.add_argument("-o", "--output", type=str, default='',
                           help="Output filename of arc line identification table")


    # -- response :: Calculate Response Function
    parser_resp = tasks.add_parser('response', formatter_class=set_help_width(31),
                                   help="Interactive determination of instrument response function")
    parser_resp.add_argument("input", type=str,
                             help="Input filename of 1D spectrum of flux standard star")
    parser_resp.add_argument("-o", "--output", type=str, default='',
                             help="Output filename of response function")


    # -- wave1d :: Wavelength Calibrate 1D Image
    parser_wave1 = tasks.add_parser('wave1d', formatter_class=set_help_width(31),
                                    help="Apply wavelength calibration to 1D spectra")
    parser_wave1.add_argument("input", type=str,
                              help="Input filename of 1D spectrum of flux standard star")
    parser_wave1.add_argument("--table", type=str, required=True,
                              help="Pixeltable of line identification from 'PyNOT-identify' [REQUIRED]")
    parser_wave1.add_argument("-o", "--output", type=str, required=True,
                              help="Output filename of wavelength calibrated 1D spectrum (FITS table) [REQUIRED]")
    parser_wave1.add_argument("--order_wl", type=int,
                              help="Polynomial order for fitting wavelength solution")
    parser_wave1.add_argument("--log", action='store_true',
                              help="Create logarithmically binned spectrum")
    parser_wave1.add_argument("--npix", type=int, default=None,
                              help="Number of pixels in output spectrum  (default= number of input pixels)")
    parser_wave1.add_argument("--no-int", action='store_false',
                              help="Do not interpolate data onto linearized grid!")


    # -- wave2d :: Rectify and Wavelength Calibrate 2D Image
    parser_wave2 = tasks.add_parser('wave2d', formatter_class=set_help_width(31),
                                    help="Rectify 2D image and apply wavelength calibration")
    parser_wave2.add_argument("input", type=str,
                              help="Input filename of 1D spectrum of flux standard star")
    parser_wave2.add_argument("arc", type=str,
                              help="Input filename of arc line image")
    parser_wave2.add_argument("--table", type=str, required=True,
                              help="Pixeltable of line identification from 'PyNOT-identify' [REQUIRED]")
    parser_wave2.add_argument("-o", "--output", type=str, required=True,
                              help="Output filename of rectified, wavelength calibrated 2D image [REQUIRED]")
    parser_wave2.add_argument("--axis", type=int, default=2,
                              help="Dispersion axis: 1 horizontal, 2: vertical")
    parser_wave2.add_argument("--order_wl", type=int, default=5,
                              help="Order of Chebyshev polynomium for wavelength solution")
    # Define parameters based on default values:
    set_default_pars(parser_wave2, section='rectify', default_type=int)


    # -- skysub :: Sky Subtraction of 2D Image
    parser_sky = tasks.add_parser('skysub', formatter_class=set_help_width(31),
                                  help="Sky subtraction of 2D image")
    parser_sky.add_argument("input", type=str,
                            help="Input filename of 2D frame")
    parser_sky.add_argument("-o", "--output", type=str, required=True,
                            help="Output filename of sky-subtracted 2D image [REQUIRED]")
    parser_sky.add_argument("--axis", type=int, default=2,
                            help="Dispersion axis: 1 horizontal, 2: vertical")
    # Define parameters based on default values:
    set_default_pars(parser_sky, section='skysub', default_type=int)


    # -- crr :: Cosmic Ray Rejection and Correction
    parser_crr = tasks.add_parser('crr', formatter_class=set_help_width(31),
                                  help="Identification and correction of Cosmic Ray Hits")
    parser_crr.add_argument("input", type=str,
                            help="Input filename")
    parser_crr.add_argument("-o", "--output", type=str, required=True,
                            help="Output filename of cleaned image [REQUIRED]")
    parser_crr.add_argument('-n', "--niter", type=int, default=4,
                            help="Number of iterations")
    parser_crr.add_argument("--gain", type=float, default=0.16,
                            help="Detector gain, default for ALFOSC CCD14: 0.16 e-/ADU")
    parser_crr.add_argument("--readnoise", type=float, default=4.3,
                            help="Detector read noise, default for ALFOSC CCD14: 4.3 e-")
    # Define parameters based on default values:
    set_default_pars(parser_crr, section='crr', default_type=int,
                     ignore_pars=['niter', 'gain', 'readnoise'])


    # -- flux1d :: Flux calibration of 1D spectrum
    parser_flux1 = tasks.add_parser('flux1d', formatter_class=set_help_width(31),
                                    help="Flux calibration of 1D spectrum")
    parser_flux1.add_argument("input", type=str,
                              help="Input filename of 1D spectrum")
    parser_flux1.add_argument("response", type=str,
                              help="Reference response function from 'PyNOT-response'")
    parser_flux1.add_argument("-o", "--output", type=str, required=True,
                              help="Output filename of flux-calibrated spectrum [REQUIRED]")


    # -- flux2d :: Flux calibration of 2D spectrum
    parser_flux2 = tasks.add_parser('flux2d', formatter_class=set_help_width(31),
                                    help="Flux calibration of 2D spectrum")
    parser_flux2.add_argument("input", type=str,
                              help="Input filename of 2D spectrum")
    parser_flux2.add_argument("response", type=str,
                              help="Reference response function from 'PyNOT-response'")
    parser_flux2.add_argument("-o", "--output", type=str, required=True,
                              help="Output filename of flux-calibrated 2D spectrum [REQUIRED]")


    # -- extract :: Extraction of 1D spectrum from 2D
    parser_ext = tasks.add_parser('extract', formatter_class=set_help_width(31),
                                  help="Extract 1D spectrum from 2D")
    parser_ext.add_argument("input", type=str,
                            help="Input filename of 2D spectrum")
    parser_ext.add_argument("-o", "--output", type=str, default='',
                            help="Output filename of 1D spectrum (FITS Table)")
    parser_ext.add_argument("--axis", type=int, default=1,
                            help="Dispersion axis: 1 horizontal, 2: vertical")
    parser_ext.add_argument('--auto', action='store_true',
                            help="Use automatic extraction instead of interactive GUI")
    # Define parameters based on default values:
    set_default_pars(parser_ext, section='extract', default_type=int,
                     ignore_pars=['interactive'])

    # Spectral Redux:
    parser_redux = tasks.add_parser('spex', formatter_class=set_help_width(30),
                                    help="Run the full spectroscopic pipeline")
    parser_redux.add_argument("params", type=str,
                              help="Input filename of pipeline configuration in YAML format")
    parser_redux.add_argument('-O', "--object", type=str, nargs='+',
                              help="Object name of targets to reduce. Must match OBJECT keyword in the FITS header")
    parser_redux.add_argument("-s", "--silent", action="store_false",
                              help="Minimze the output to terminal")
    parser_redux.add_argument("-i", "--interactive", action="store_true",
                              help="Use interactive interface throughout")

    parser_break = tasks.add_parser('', help="")

    # Imaging Redux:
    parser_phot = tasks.add_parser('phot', formatter_class=set_help_width(30),
                                   help="Run the full imaging pipeline")
    parser_phot.add_argument("params", type=str,
                             help="Input filename of pipeline configuration in YAML format")
    parser_phot.add_argument("-s", "--silent", action="store_false",
                             help="Minimze the output to terminal")

    parser_imtrim = tasks.add_parser('imtrim', formatter_class=set_help_width(30),
                                     help="Trim images")
    parser_imtrim.add_argument("input", type=str,
                               help="List of filenames to trim")
    parser_imtrim.add_argument("--dir", type=str, default='',
                               help="Output directory")
    parser_imtrim.add_argument("--flat", type=str, default='',
                               help="Flat field image to use for edge detection")
    parser_imtrim.add_argument('-e', "--edges", type=int, nargs=4,
                               help="Trim edges  [left  right  bottom  top]")

    parser_imcomb = tasks.add_parser('imcombine', formatter_class=set_help_width(44),
                                     help="Combine images")
    parser_imcomb.add_argument("input", type=str,
                               help="List of filenames to combine")
    parser_imcomb.add_argument("output", type=str,
                               help="Filename of combined image")
    parser_imcomb.add_argument("--log", type=str, default='',
                               help="Filename of image combination log")
    parser_imcomb.add_argument("--fringe", type=str, default='',
                               help="Filename of normalized fringe image")
    set_default_pars(parser_imcomb, section='combine', default_type=int, mode='img')


    parser_fringe = tasks.add_parser('fringe', formatter_class=set_help_width(30),
                                     help="Create average fringe images")
    parser_fringe.add_argument("input", type=str,
                               help="List of filenames to combine")
    parser_fringe.add_argument("output", type=str,
                               help="Filename of normalized fringe image")
    parser_fringe.add_argument("--fig", type=str, default='',
                               help="Filename of figure showing fringe image")
    parser_fringe.add_argument("--sigma", type=float, default=3,
                               help="Masking threshold  (default = 3.0)")

    parser_sep = tasks.add_parser('sep', formatter_class=set_help_width(40),
                                  help="Perform source extraction using SEP (SExtractor)")
    parser_sep.add_argument("input", type=str,
                            help="Input image to analyse")
    parser_sep.add_argument('-z', "--zero", type=float, default=0.,
                            help="Magnitude zeropoint, default is 0 (instrument mags)")
    set_default_pars(parser_sep, section='sep-background', default_type=int, mode='img')
    set_default_pars(parser_sep, section='sep-extract', default_type=int, mode='img')


    args = parser.parse_args()


    # -- Define Workflow
    task = args.task
    log = ""


    if task == 'init':
        initialize(args)

    elif task == 'spex':
        from pynot.redux import run_pipeline
        print_credits()
        run_pipeline(options_fname=args.params,
                     object_id=args.object,
                     verbose=args.silent,
                     interactive=args.interactive)

    elif task == 'bias':
        from pynot.calibs import combine_bias_frames
        print("Running task: Bias combination")
        input_list = np.loadtxt(args.input, dtype=str, usecols=(0,))
        _, log = combine_bias_frames(input_list, args.output, kappa=args.kappa)

    elif task == 'sflat':
        from pynot.calibs import combine_flat_frames, normalize_spectral_flat
        print("Running task: Spectral flat field combination and normalization")
        input_list = np.loadtxt(args.input, dtype=str, usecols=(0,))
        flatcombine, log = combine_flat_frames(input_list, output='', mbias=args.bias, mode='spec',
                                               dispaxis=args.axis, kappa=args.kappa)

        options = copy(vars(args))
        vars_to_remove = ['task', 'input', 'output', 'axis', 'bias', 'kappa']
        for varname in vars_to_remove:
            options.pop(varname)
        _, log = normalize_spectral_flat(flatcombine, args.output, dispaxis=args.axis, **options)

    elif task == 'corr':
        from pynot.scired import correct_raw_file
        print("Running task: Bias subtraction and flat field correction")
        input_list = np.loadtxt(args.input, dtype=str, usecols=(0,))
        if args.img:
            mode = 'img'
        else:
            mode = 'spec'
        if args.dir != '' and not os.path.exists(args.dir):
            os.mkdir(args.dir)

        for fname in input_list:
            basename = os.path.basename(fname)
            output = 'corr_%s' % basename
            if args.dir != '':
                output = os.path.join(args.dir, output)
            _ = correct_raw_file(fname, output=output, bias_fname=args.bias, flat_fname=args.flat,
                                 overscan=50, overwrite=True, mode=mode)
            print(" - Image: %s  ->  %s" % (fname, output))

    elif task == 'identify':
        from PyQt5 import QtWidgets
        from pynot.identify_gui import GraphicInterface
        # Launch App:
        app = QtWidgets.QApplication(sys.argv)
        gui = GraphicInterface(args.arc,
                               linelist_fname=args.lines,
                               dispaxis=args.axis,
                               output=args.output)
        gui.show()
        app.exit(app.exec_())

    elif task == 'response':
        from PyQt5 import QtWidgets
        from pynot.response_gui import ResponseGUI
        # Launch App:
        app = QtWidgets.QApplication(sys.argv)
        gui = ResponseGUI(args.input, output_fname=args.output)
        gui.show()
        app.exit(app.exec_())

    elif task == 'wave1d':
        from pynot.wavecal import wavecal_1d
        print("Running task: 1D Wavelength Calibration")
        log = wavecal_1d(args.input, args.table, output=args.output, order_wl=args.order_wl,
                         log=args.log, N_out=args.npix, linearize=args.no_int)

    elif task == 'wave2d':
        from pynot.wavecal import rectify
        print("Running task: 2D Rectification and Wavelength Calibration")
        options = copy(vars(args))
        vars_to_remove = ['task', 'input', 'arc', 'table', 'output', 'axis']
        for varname in vars_to_remove:
            options.pop(varname)
        log = rectify(args.input, args.arc, args.table, output=args.output, fig_dir='./',
                      dispaxis=args.axis, **options)

    elif task == 'skysub':
        from pynot.scired import auto_fit_background
        print("Running task: Background Subtraction")
        options = copy(vars(args))
        vars_to_remove = ['task', 'input', 'output', 'axis']
        for varname in vars_to_remove:
            options.pop(varname)
        log = auto_fit_background(args.input, args.output, dispaxis=args.axis,
                                  plot_fname="skysub_diagnostics.pdf",
                                  **options)

    elif task == 'crr':
        from pynot.scired import correct_cosmics
        print("Running task: Cosmic Ray Rejection")
        options = copy(vars(args))
        vars_to_remove = ['task', 'input', 'output']
        for varname in vars_to_remove:
            options.pop(varname)
        log = correct_cosmics(args.input, args.output, **options)

    elif task == 'flux1d':
        from pynot.response import flux_calibrate_1d
        print("Running task: Flux Calibration of 1D Spectrum")
        log = flux_calibrate_1d(args.input, output=args.output, response=args.response)

    elif task == 'flux2d':
        from pynot.response import flux_calibrate
        print("Running task: Flux Calibration of 2D Image")
        log = flux_calibrate(args.input, output=args.output, response=args.response)

    elif task == 'extract':
        from PyQt5 import QtWidgets
        from pynot.extract_gui import ExtractGUI
        options = copy(vars(args))
        vars_to_remove = ['task', 'input', 'output', 'axis']
        for varname in vars_to_remove:
            options.pop(varname)

        if args.auto:
            from pynot.extraction import auto_extract
            print("Running task: 1D Extraction")
            log = auto_extract(args.input, args.output, dispaxis=args.axis,
                               pdf_fname="extract_diagnostics.pdf", **options)
        else:
            app = QtWidgets.QApplication(sys.argv)
            gui = ExtractGUI(args.input, output_fname=args.output, dispaxis=args.axis, **options)
            gui.show()
            app.exit(app.exec_())



    elif task == 'phot':
        from pynot.phot_redux import run_pipeline
        print_credits()
        run_pipeline(options_fname=args.params,
                     verbose=args.silent)

    elif task == 'imflat':
        print("Running task: Combination of Imaging Flat Fields")
        from pynot.calibs import combine_flat_frames
        input_list = np.loadtxt(args.input, dtype=str, usecols=(0,))
        _, log = combine_flat_frames(input_list, output=args.output, mbias=args.bias, mode='img',
                                     kappa=args.kappa)

    elif task == 'imtrim':
        print("Running task: Image Trimming")
        from pynot.scired import detect_filter_edge, trim_filter_edge
        if args.edges is not None:
            image_region = args.edges
        elif args.flat is not None:
            image_region = detect_filter_edge(args.flat)
            print(" Automatically detected image edges:")
            print("  left=%i   right=%i  bottom=%i  top=%i" % image_region)
            print("")
        else:
            print(" Invalid input! Either '--flat' or '--edges' must be set!")
            return
        input_list = np.loadtxt(args.input, dtype=str, usecols=(0,))
        for fname in input_list:
            print("  Trimming image: %s" % fname)
            trim_filter_edge(fname, *image_region, output_dir=args.dir)

    elif task == 'imcombine':
        print("Running task: Image Combination")
        from pynot.phot import image_combine
        input_list = np.loadtxt(args.input, dtype=str, usecols=(0,))
        options = copy(vars(args))
        vars_to_remove = ['task', 'input', 'output', 'log', 'fringe']
        for varname in vars_to_remove:
            options.pop(varname)
        log = image_combine(input_list, output=args.output, log_name=args.log, fringe_image=args.fringe, **options)

    elif task == 'fringe':
        print("Running task: Creating Average Fringe Image")
        from pynot.phot import create_fringe_image
        input_list = np.loadtxt(args.input, dtype=str, usecols=(0,))
        log = create_fringe_image(input_list, output=args.output, fig_fname=args.fig, threshold=args.sigma)

    elif task == 'sep':
        print("Running task: Extracting Sources and Measuring Aperture Fluxes")
        from pynot.phot import source_detection
        options = copy(vars(args))
        vars_to_remove = ['task', 'input', 'zero']
        bg_options = {}
        ext_options = {}
        for varname, value in options.items():
            if varname in vars_to_remove:
                continue
            elif varname in ['bw', 'bh', 'fw', 'fh', 'fthresh']:
                bg_options[varname] = value
            else:
                ext_options[varname] = value
        _, _, log = source_detection(args.input, zeropoint=args.zero,
                                     kwargs_bg=bg_options, kwargs_ext=ext_options)

    if log:
        print(log)


if __name__ == '__main__':
    main()
