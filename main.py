"""
Automatically Classify and Reduce a given Data Set
"""

from argparse import ArgumentParser
from collections import defaultdict
import os
import sys
import numpy as np

import alfosc
import data_organizer as do
from extraction import auto_extract
import extract_gui
from functions import get_options, get_version_number
from wavecal import rectify
from identify_gui import create_pixtable
from scired import raw_correction, auto_fit_background, correct_cosmics
from response import calculate_response, flux_calibrate


code_dir = os.path.dirname(os.path.abspath(__file__))
calib_dir = os.path.join(code_dir, 'calib/')
defaults_fname = os.path.join(calib_dir, 'default_options.yml')
parameters = get_options(defaults_fname)

__version__ = get_version_number()

# pynot  bias  files.list

def main():
    parser = ArgumentParser()
    recipes = parser.add_subparsers(dest='recipe')

    # -- BIAS :: Bias Combination
    parser_bias = recipes.add_parser('bias',
                                     help="Combine bias frames")
    parser_bias.add_argument("input", type=str,
                             help="Input file containing list of image filenames to combine")
    parser_bias.add_argument("output", type=str,
                             help="Output filename of combined bias frame")
    parser_bias.add_argument("--kappa", type=float, default=15,
                             help="Threshold for sigma clipping")

    # -- SFLAT :: Spectral Flat Combination
    parser_sflat = recipes.add_parser('sflat',
                                      help="Combine spectral flat frames")
    parser_sflat.add_argument("input", type=str,
                              help="Input file containing list of image filenames to combine")
    parser_sflat.add_argument("output", type=str,
                              help="Output filename of combined bias frame")
    # Define based on default options:
    for key, val in parameters['flat'].items():
        parser_sflat.add_argument("--%s" % key, type=type(val), default=val)

    # # -- IMFLAT :: Imaging Flat Combination
    # parser_imflat = recipes.add_parser('imflat',
    #                                    help="Combine imaging flat frames")

    # Spectral Redux:
    parser_redux = recipes.add_parser('spec',
                                      help="Combine imaging flat frames")
    parser_redux.add_argument("options", type=str,
                              help="Input filename of pipeline configuration in YAML format")
    parser_redux.add_argument("-v", "--verbose", action="store_true",
                              help="Print log to terminal")
    parser_redux.add_argument("-i", "--interactive", action="store_true",
                              help="Use interactive interface throughout")

    args = parser.parse_args()

    recipe = args.recipe
    if recipe == 'spec':
        main(options_fname=args.options, verbose=args.verbose, interactive=args.interactive)

    elif recipe == 'bias':
        from calibs import combine_bias_frames
        input_list = np.loadtxt(args.input, dtype=str)
        log = combine_bias_frames(input_list, args.output, kappa=args.kappa)

    elif recipe == 'sflat':
        from calibs import combine_flat_frames, normalize_spectral_flat
        flatcombine, log = combine_flat_frames(args.input, output='', mbias=args.mbias, mode='spec',
                                               dispaxis=args.dispaxis, kappa=args.kappa)

        _, log = normalize_spectral_flat(flatcombine, args.output, dispaxis=args.axis,
                                         lower=args.lower, upper=args.upper, order=args.order,
                                         sigma=args.sigma, show=False)

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
        log = ['']
    output_message = "\n".join(log)
    print(output_message)


if __name__ == '__main__':
    main()
