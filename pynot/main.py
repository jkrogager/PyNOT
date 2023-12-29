"""
  PyNOT Data Reduction Pipeline
  Originally written for NOT/ALFOSC

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
import distutils

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
        if isinstance(val, bool):
            value_type = lambda x: bool(distutils.util.strtobool(x))
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


def initialize(path, mode, pfc_fname='dataset.pfc', pars_fname='options.yml', verbose=True):
    """
    Initialize new dataset with file classification table and default options.

    path : str
        Path to the folder containing the raw files to classify

    mode : str  ['spex'/'spec' or 'phot']
        The mode for the default options: `spec` for spectroscopy, and `phot` for photometry
        `spex` is allowed for backwards compatibility.

    pfc_fname : str
        The filename of the file classification table

    pars_filename : str
        The filename of the default options

    verbose : bool  [default=True]
        Print logging to terminal
    """
    print_credits()
    from pynot.data import organizer as do
    from pynot.data import io

    # Classify files:
    # args.silent is set to "store_false", so by default it is true!
    database, message = do.classify(path, progress=verbose)
    if database is None:
        print(message)
        return

    if pfc_fname[-4:] != '.pfc':
        pfc_fname += '.pfc'
    io.save_database(database, pfc_fname)
    if verbose:
        print(message)
    print(" [OUTPUT] - Saved file classification database: %s" % pfc_fname)

    if mode.lower() in ['spex', 'spec']:
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
    fmt = "%%-%is" % (len(empty_str) - 2)
    dataset_input = "%s:  %s #%s" % (root, fmt % pfc_fname, comment)
    all_lines[num] = dataset_input

    if pars_fname == 'options.yml':
        pars_fname = 'options_spex.yml' if mode == 'spex' else 'options_phot.yml'

    if pars_fname[-4:] != '.yml':
        pars_fname += '.yml'

    # Write the new parameter file:
    with open(pars_fname, 'w') as parfile:
        parfile.write("".join(all_lines))

    print(" [OUTPUT] - Initiated new parameter file: %s\n" % pars_fname)
    print("\n You can now start the reduction pipeline by running:")
    print("   ]%% pynot %s  %s\n" % (mode, pars_fname))


def main(inspect=False):

    parser = ArgumentParser(prog='pynot')
    tasks = parser.add_subparsers(dest='task')

    p_break1 = tasks.add_parser('', help="")

    parser_init = tasks.add_parser('init', formatter_class=set_help_width(31),
                                   help="Initiate a default parameter file")
    parser_init.add_argument("mode", type=str, choices=['spec', 'spex', 'phot'],
                             help="Create parameter file for spectroscopy or imaging?")
    parser_init.add_argument("path", type=str, nargs='+',
                             help="Path (or paths) to raw data to be classified")
    parser_init.add_argument('-p', "--pars", type=str, default='options.yml',
                             help="Filename of parameter file, default = options.yml")
    parser_init.add_argument("-o", "--output", type=str, default='dataset.pfc',
                             help="Filename of file classification table (*.pfc)")
    parser_init.add_argument("-s", "--silent", action='store_false',
                             help="Minimze the output to terminal")

    parser_org = tasks.add_parser('classify', formatter_class=set_help_width(31),
                                  help="Classify the files in `path`")
    parser_org.add_argument("path", type=str, nargs='+',
                            help="Path (or paths) to raw data to be classified")
    parser_org.add_argument("-o", "--output", type=str, default='dataset.pfc',
                            help="Filename of file classification table (*.pfc)")
    parser_org.add_argument("-f", "--force", action="store_true",
                            help="Force overwrite of the file classification table (*.pfc)")

    parser_obd = tasks.add_parser('update-obs', formatter_class=set_help_width(31),
                                  help="Update the OB database based on a classification table")
    parser_obd.add_argument("pfc", type=str,
                            help="File classification table (*.pfc)")

    # -- BIAS :: Bias Combination
    parser_bias = tasks.add_parser('bias', formatter_class=set_help_width(31),
                                   help="Combine bias frames")
    parser_bias.add_argument("input", type=str,
                             help="Input file containing list of image filenames to combine")
    parser_bias.add_argument('-o', "--output", type=str, default='MASTER_BIAS.fits',
                             help="Output filename of combined bias frame  (default = MASTER_BIAS.fits)")
    parser_bias.add_argument("--kappa", type=float, default=15,
                             help="Threshold for sigma clipping")
    parser_bias.add_argument("--method", type=str, default='mean', choices=['mean', 'median'],
                             help="Method for image combination")

    # -- SFLAT :: Spectral Flat Combination
    parser_sflat = tasks.add_parser('sflat', formatter_class=set_help_width(31),
                                    help="Combine and normalize spectral flat frames")
    parser_sflat.add_argument("input", type=str,
                              help="Input file containing list of image filenames to combine")
    parser_sflat.add_argument("--bias", type=str, required=True,
                              help="Filename of combined bias frame  [REQUIRED]")
    parser_sflat.add_argument("-o", "--output", type=str, default='',
                              help="Output filename of combined and normalized flat frame. Constructed from GRISM and SLIT by default")
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
    parser_imflat.add_argument("--method", type=str, default='mean', choices=['mean', 'median'],
                               help="Method for image combination")

    # -- corr :: Raw Correction
    parser_corr = tasks.add_parser('corr', formatter_class=set_help_width(31),
                                   help="Apply bias subtraction, flat field correction and trimming")
    parser_corr.add_argument("input", type=str,
                             help="List of filenames to correct")
    parser_corr.add_argument("--dir", type=str, default='',
                             help="Output directory")
    parser_corr.add_argument("--bias", type=str, default='',
                             help="Filename of combined bias frame. If not given, a constant of 0 is used")
    parser_corr.add_argument("--flat", type=str, default='',
                             help="Filename of combined flat frame. If not given, a constant of 1 is used")
    parser_corr.add_argument("--img", action='store_true',
                             help="Imaging mode")


    # -- identify :: Identify Arc Lines
    parser_id = tasks.add_parser('identify', formatter_class=set_help_width(31),
                                 help="Interactive identification of arc lines")
    parser_id.add_argument("arc", type=str, default='', nargs='?',
                           help="Input filename of arc line image")
    parser_id.add_argument("--lines", type=str, default='',
                           help="Linelist, automatically loaded if possible")
    parser_id.add_argument("--axis", type=int, default=2,
                           help="Dispersion axis: 1 horizontal, 2: vertical")
    parser_id.add_argument("-o", "--output", type=str, default='',
                           help="Output filename of arc line identification table")
    parser_id.add_argument("--air", action='store_true',
                           help="Use air reference wavelengths")
    parser_id.add_argument("--loc", type=int, default=-1,
                           help="Location along the slit to extract lamp spectrum [pixels].")


    # -- response :: Calculate Response Function
    parser_resp = tasks.add_parser('response', formatter_class=set_help_width(31),
                                   help="Interactive determination of instrument response function")
    parser_resp.add_argument("input", type=str, default='', nargs='?',
                             help="Input filename of 1D spectrum of flux standard star")
    parser_resp.add_argument("-o", "--output", type=str, default='',
                             help="Output filename of response function")


    # -- wave1d :: Wavelength Calibrate 1D Image
    parser_wave1 = tasks.add_parser('wave1d', formatter_class=set_help_width(31),
                                    help="Apply wavelength calibration to 1D spectra")
    parser_wave1.add_argument("input", type=str,
                              help="Input filename of 1D spectrum (FITS Table format)")
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
                              help="Input filename of 2D spectrum")
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
    parser_crr.add_argument('-n', "--niter", type=int, default=2,
                            help="Number of iterations")
    # Define parameters based on default values:
    set_default_pars(parser_crr, section='crr', default_type=int,
                     ignore_pars=['niter'])


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


    # -- scombine :: Spectral combination in 1D or 2D
    parser_scomb = tasks.add_parser('scombine', formatter_class=set_help_width(31),
                                    help="Spectral combination")
    parser_scomb.add_argument("input", type=str, nargs='+',
                              help="Input spectra")
    parser_scomb.add_argument("-o", "--output", type=str, default='',
                              help="Output filename (autogenerated if not given)")
    parser_scomb.add_argument("-m", "--method", type=str,
                              help="Combination method", default='mean',
                              choices=['mean', 'median'])
    parser_scomb.add_argument("-s", "--scale", action="store_true",
                              help="Scale spectra before combining")
    parser_scomb.add_argument("--axis", choices=[1, 2], default=1,
                              help="Dispersion axis. 1: horizontal, 2: vertical (default: 1 for processed spectra)")
    parser_scomb.add_argument("-x", "--extended", action="store_true",
                              help="Set this option for 2D spectra of extended sources to turn off automatic localization")
    parser_scomb.add_argument("--mef", action="store_false",
                              help="Set this option to save output as a multiextension FITS file instead of a FITS table.")


    # -- extract :: Extraction of 1D spectrum from 2D
    parser_ext = tasks.add_parser('extract', formatter_class=set_help_width(31),
                                  help="Extract 1D spectrum from 2D")
    parser_ext.add_argument("input", type=str, default='', nargs='?',
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
    parser_redux.add_argument("-f", "--force", action="store_true",
                              help="Force restart of all OBs!")
    parser_redux.add_argument("--no-int", action="store_true",
                              help="Turn off all interactive interfaces")
    parser_redux.add_argument("--mbias", action="store_true",
                              help="Run bias only")
    parser_redux.add_argument("--mflat", action="store_true",
                              help="Run flat combination and normalization only")
    parser_redux.add_argument("--arcs", action="store_true",
                              help="Process arcs")
    parser_redux.add_argument("--response", action="store_true",
                              help="Recalculate the response functions")
    parser_redux.add_argument("--science", action="store_true",
                              help="Restart the science reduction")
    parser_redux.add_argument("-I", "--identify", action="store_true",
                              help="Re-identify all grisms once")
    parser_redux.add_argument("-C", "--calibs", action="store_true",
                              help="Process only static calibrations: [bias, flats, arcs, response]")

    # Imaging Redux:
    parser_phot = tasks.add_parser('phot', formatter_class=set_help_width(30),
                                   help="Run the full imaging pipeline")
    parser_phot.add_argument("params", type=str,
                             help="Input filename of pipeline configuration in YAML format")
    parser_phot.add_argument("-s", "--silent", action="store_false",
                             help="Minimze the output to terminal")
    parser_phot.add_argument("-f", "--force", action="store_true",
                             help="Force restart of all OBs!")

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


    parser_wcs = tasks.add_parser('wcs', formatter_class=set_help_width(30),
                                  help="Perform WCS calibration")
    parser_wcs.add_argument("input", type=str,
                            help="Input image to analyse")
    parser_wcs.add_argument("table", type=str, default='auto', nargs='?',
                            help="Source identification table from SEP (_phot.fits)")
    parser_wcs.add_argument('-o', "--output", type=str, default='',
                            help="Filename of WCS calibrated image (.fits)")
    parser_wcs.add_argument("--fig", type=str, default='',
                            help="Filename of diagnostic figure (.pdf)")
    parser_wcs.add_argument("-d", "--debug", action='store_true',
                            help="Enable debugging mode with additional diagnostic plots")
    set_default_pars(parser_wcs, section='wcs', default_type=int, mode='img')


    parser_autozp = tasks.add_parser('autozp', formatter_class=set_help_width(30),
                                     help="Perform auto-calibration of magnitude zero point using SDSS data")
    parser_autozp.add_argument("input", type=str,
                               help="Input WCS calibrated image to analyse")
    parser_autozp.add_argument("table", type=str,
                               help="Source identification table from SEP (_phot.fits)")
    parser_autozp.add_argument("--fig", type=str, default='',
                               help="Filename of diagnostic figure (.pdf), autogenerated by default")
    set_default_pars(parser_autozp, section='sdss_flux', default_type=int, mode='img')


    parser_findnew = tasks.add_parser('findnew', formatter_class=set_help_width(30),
                                      help="Identify transient sources compared to Gaia")
    parser_findnew.add_argument("input", type=str,
                                help="Input WCS calibrated image to analyse")
    parser_findnew.add_argument("table", type=str,
                                help="Source identification table from SEP (_phot.fits)")
    parser_findnew.add_argument("--bat", type=float, nargs=3,
                                help="Localisation constraint from SWIFT/BAT  (ra [deg]  dec [deg]  radius [arcmin])")
    parser_findnew.add_argument("--xrt", type=float, nargs=3,
                                help="Localisation constraint from SWIFT/XRT  (ra [deg]  dec [deg]  radius [arcsec])")
    parser_findnew.add_argument("--limit", type=float, default=20.1,
                                help="Magnitude limit (default = 20.1 mag to match Gaia depth)")
    parser_findnew.add_argument('-z', "--zp", type=float,
                                help="Magnitude zero point in case the source catalog has not been flux calibrated")

    parser_setup = tasks.add_parser('setup', formatter_class=set_help_width(30),
                                    help="Install new instrument configuration")
    parser_setup.add_argument('module', type=str,
                              help='Filename of the instrument configuration module (*.py)')
    parser_setup.add_argument('-f', '--filters', type=str, default='',
                              help='Filename of the instrument filter definitions (must have columns: name and short_name)')
    parser_setup.add_argument('-r', '--rules', type=str, default='',
                              help='Filename of the classification ruleset')
    parser_setup.add_argument('--ext', type=str, default='',
                              help='Filename of the observatory extinction data (lapalma, lasilla, paranal, or file path)')
    parser_setup.add_argument('-u', '--use', action="store_true",
                              help='Switch immediately to use the newly installed instrument')

    parser_use = tasks.add_parser('use', formatter_class=set_help_width(30),
                                  help="Set current instrument")
    parser_use.add_argument('name', type=str, default='', nargs='?',
                            help='Name of the instrument to use (see --list)')
    parser_use.add_argument('--list', action="store_true",
                            help='Display a list of the currently installed instruments')

    parser_f2a = tasks.add_parser('fits2ascii', formatter_class=set_help_width(30),
                                  help="Convert FITS spectrum to ASCII table format")
    parser_f2a.add_argument('input', type=str,
                            help='Input filename of FITS spectrum (either table or MEF)')
    parser_f2a.add_argument('output', type=str,
                            help='Output filename of the ASCII table')
    parser_f2a.add_argument('--keys', type=str, nargs='+',
                            help='List of keywords from the FITS header to include (ESO keywords must use . not space delimiter)')

    parser_fapp = tasks.add_parser('append-ext', formatter_class=set_help_width(30),
                                   help="Append new data to FITS file or create error image")
    parser_fapp.add_argument('input', type=str,
                             help='Input filename of FITS image to which the new data will be appended')
    parser_fapp.add_argument('data', type=str,
                             help='Filename of the image to append to the `input` FITS file')
    parser_fapp.add_argument('-n', '--name', type=str, default='',
                             help='Name of the new FITS extension.')
    parser_fapp.add_argument('-x', '--ext', type=int, default=0,
                             help='Extension number of the data file which is appended to the `input` file')

    parser_delext = tasks.add_parser('remove-ext', formatter_class=set_help_width(30),
                                     help="Remove a given extension of a FITS file")
    parser_delext.add_argument('input', type=str,
                               help='Input filename of FITS image to which the new data will be appended')
    parser_delext.add_argument('ext', type=str,
                               help='Extension number or name to remove')

    parser_adderr = tasks.add_parser('add-error', formatter_class=set_help_width(30),
                                     help="Create and append error image for a single HDU FITS file")
    parser_adderr.add_argument('input', type=str,
                               help='Input filename of FITS image to which the new data will be appended')
    parser_adderr.add_argument('-f', '--force', action='store_true',
                               help='Overwrite existing error extension (if name is `ERR`)')

    if inspect:
        return parser

    args = parser.parse_args()


    # -- Define Workflow
    task = args.task
    log = ""


    if task == 'init':
        initialize(args.path, args.mode, pfc_fname=args.output, pars_fname=args.pars, verbose=args.silent)

    elif task == 'update-obs':
        from pynot.data.obs import update_ob_database
        update_ob_database(args.pfc)

    elif task == 'spex':
        from pynot.redux import run_pipeline
        # print_credits()
        run_pipeline(options_fname=args.params,
                     object_id=args.object,
                     verbose=args.silent,
                     interactive=args.interactive,
                     force_restart=args.force,
                     make_bias=args.mbias,
                     make_flat=args.mflat,
                     make_arcs=args.arcs,
                     make_identify=args.identify,
                     make_response=args.response,
                     calibs_only=args.calibs,
                     restart_science=args.science
                     )

    elif task == 'bias':
        from pynot.calibs import combine_bias_frames
        print("Running task: Bias combination")
        input_list = np.loadtxt(args.input, dtype=str, usecols=(0,))
        _, log = combine_bias_frames(input_list, args.output, kappa=args.kappa, method=args.method)

    elif task == 'sflat':
        from pynot.calibs import combine_flat_frames, normalize_spectral_flat
        print("Running task: Spectral flat field combination and normalization")
        input_list = np.loadtxt(args.input, dtype=str, usecols=(0,))
        flatcombine, log = combine_flat_frames(input_list, output='', mbias=args.bias, mode='spec',
                                               dispaxis=args.axis, kappa=args.kappa, method=args.method)

        options = copy(vars(args))
        vars_to_remove = ['task', 'input', 'output', 'axis', 'bias', 'kappa']
        for varname in vars_to_remove:
            options.pop(varname)
        _, log = normalize_spectral_flat(flatcombine, args.output, dispaxis=args.axis, **options)
        # from pynot.calibs import task_sflat
        # task_sflat(args)

    elif task == 'corr':
        from pynot.scired import correct_raw_file
        import glob
        print("Running task: Bias subtraction and flat field correction")
        if '.fits' in args.input:
            # Load image or wildcard list:
            if '?' in args.input or '*' in args.input:
                input_list = glob.glob(args.input)
            else:
                input_list = [args.input]
        else:
            input_list = np.loadtxt(args.input, dtype=str, usecols=(0,))

        # Mode determines the header keywords to update (CDELT or CD-matrix)
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
                                 overwrite=True, mode=mode)
            print(" - Image: %s  ->  %s" % (fname, output))

    elif task == 'identify':
        from PyQt5 import QtWidgets
        from pynot.identify_gui import GraphicInterface
        # Launch App:
        app = QtWidgets.QApplication(sys.argv)
        gui = GraphicInterface(args.arc,
                               linelist_fname=args.lines,
                               dispaxis=args.axis,
                               air=args.air,
                               loc=args.loc,
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
        vars_to_remove = ['task', 'input', 'output', 'axis', 'auto']
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
        log = flux_calibrate_1d(args.input, output=args.output, response_fname=args.response)

    elif task == 'flux2d':
        from pynot.response import flux_calibrate
        print("Running task: Flux Calibration of 2D Image")
        log = flux_calibrate(args.input, output=args.output, response_fname=args.response)

    elif task == 'scombine':
        print("Running task: Spectral Combination")
        from pynot.scombine import combine_1d, combine_2d
        from astropy.io import fits
        from glob import glob

        if len(args.input) == 1:
            input_arg = args.input[0]
            if '*' in input_arg or '?' in input_arg:
                filelist = glob.glob(input_arg)
            else:
                filelist = np.loadtxt(input_arg, usecols=(0,), dtype=str)
        else:
            filelist = args.input

        # Check if data are 1D or 2D:
        try:
            data = fits.getdata(filelist[0])
            if isinstance(data, fits.fitsrec.FITS_rec):
                data_is_1d = True
            elif isinstance(data, np.ndarray) and len(data.shape) == 1:
                data_is_1d = True
            elif isinstance(data, np.ndarray) and len(data.shape) == 2:
                data_is_1d = False
            else:
                print("  [ERROR] - Could not recognize the data type!")

        except OSError:
            data_is_1d = True


        if data_is_1d:
            out_args = combine_1d(filelist, output=args.output, method=args.method,
                                  scale=args.scale, table_output=args.mef)
        else:
            out_args = combine_2d(filelist, output=args.output, method=args.method,
                                  scale=args.scale, extended=args.extended, dispaxis=args.axis)
        log = out_args[-1]


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


    # -- Imaging tasks:
    elif task == 'phot':
        from pynot.phot_redux import run_pipeline
        run_pipeline(options_fname=args.params,
                     verbose=args.silent,
                     force_restart=args.force)

    elif task == 'imflat':
        print("Running task: Combination of Imaging Flat Fields")
        from pynot.calibs import combine_flat_frames
        input_list = np.loadtxt(args.input, dtype=str, usecols=(0,))
        _, log = combine_flat_frames(input_list, output=args.output, mbias=args.bias, mode='img',
                                     kappa=args.kappa, method=args.method)

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
        if args.input.endswith('.fits'):
            input_list = [args.input]
        else:
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

    elif task == 'wcs':
        print("Running task: WCS calibration")
        from pynot.wcs import correct_wcs
        options = copy(vars(args))
        vars_to_remove = ['task', 'input', 'output', 'fig', 'table']
        for varname in vars_to_remove:
            options.pop(varname)
        if args.table == 'auto':
            from pynot.phot import source_detection
            phot_table, _, log = source_detection(args.input)
            print(log)
        else:
            phot_table = args.table
        log = correct_wcs(args.input, phot_table, output=args.output, fig_fname=args.fig, **options)


    elif task == 'autozp':
        print("Running task: Zero point auto-calibration (SDSS)")
        from pynot.phot import flux_calibration_sdss
        options = copy(vars(args))
        vars_to_remove = ['task', 'input', 'table', 'fig']
        for varname in vars_to_remove:
            options.pop(varname)
        log = flux_calibration_sdss(args.input, args.table, fig_fname=args.fig, **options)

    elif task == 'findnew':
        print("Running task: Transient identification")
        from pynot.transients import find_new_sources
        if args.bat is None:
            ra_bat, dec_bat, radius_bat = (0, 0, 0)
        else:
            ra_bat, dec_bat, bat_r = args.bat
            radius_bat = bat_r / 60

        if args.xrt is None:
            ra_xrt, dec_xrt, radius_xrt = (0, 0, 0)
        else:
            ra_xrt, dec_xrt, xrt_r = args.xrt
            radius_xrt = xrt_r / 3600
        new_sources, log = find_new_sources(args.input, args.table,
                                            loc_bat=(ra_bat, dec_bat, radius_bat),
                                            loc_xrt=(ra_xrt, dec_xrt, radius_xrt),
                                            mag_lim=args.limit,
                                            zp=args.zp)

    elif task == 'setup':
        print("\nInstalling new instrument settings...")
        from pynot.insconfig import setup_instrument
        log = setup_instrument(args)

    elif task == 'use':
        from pynot.insconfig import get_installed_instruments, change_instrument

        all_instruments, inst_list = get_installed_instruments()
        if args.list:
            print(inst_list)
            return
        else:
            log = change_instrument(args.name, all_instruments)

    elif task == 'classify':
        print_credits()
        from pynot.data import organizer as do
        from pynot.data import io

        # Classify files:
        database, message = do.classify(args.path)
        print("")
        print(message)
        if database is None:
            return

        if not args.output.endswith('.pfc'):
            pfc_fname = args.output + '.pfc'
        else:
            pfc_fname = args.output

        if os.path.exists(pfc_fname):
            print("          - File classification database already exists")
            if args.force:
                print("          - Overwriting the database! Are you sure?  [Y/n]")
                user_input = input("          > ")
                if user_input.lower() in ['y', 'yes', '']:
                    pass
                else:
                    print("Aborting...")
                    return
            else:
                previous_database = io.load_database(pfc_fname)
                database = previous_database + database
                print("          - Merging the databases")
        io.save_database(database, pfc_fname)
        print(" [OUTPUT] - Saved file classification database: %s" % pfc_fname)

    elif task == 'fits2ascii':
        from pynot.fitsio import fits_to_ascii
        fits_to_ascii(args.input, args.output, args.keys)

    elif task == 'append-ext':
        from pynot.fitsio import append_extension
        log = append_extension(args.input, args.data, name=args.name, data_ext=args.ext)

    elif task == 'remove-ext':
        from pynot.fitsio import remove_extension
        log = remove_extension(args.input, args.ext)
    
    elif task == 'add-error':
        from pynot.fitsio import create_error_image
        log = create_error_image(args.input, overwrite=args.force)
    
    else:
        import pynot
        print("Running PyNOT for instrument: %s\n" % pynot.instrument.name)

    if log:
        print(log)


if __name__ == '__main__':
    main()
