# -*- coding: UTF-8 -*-
"""
  PyNOT - longslit reduction

  The module handles longslit reduction for the ALFOSC instrument
  at the Nordic Optical Telescope (NOT).
  Written by Jens-Kristian Krogager, Mar 2017

  depends on matplotlib, numpy, scipy, pyfits, argparse, astroscrappy

  .. functions:
   - combine_bias_frames
   - combine_flat_frames
   - normalize_spectral_flat
   - fit_background
   - science_reduction_2d
"""

# -- Notes for trace identification:
# Use scipy.signal.find_peaks to identify number of objects in slit:
# Construct SPSF and determine the background noise: sigma.
#
#  peaks = scipy.signal.find_peaks(SPSF, height=3*sigma, width=3.)
#
#  numbers like this seem to work fairly well, the number of sigmas should be a parameter
#  this required an object to be more than 3 pixels wide and more than 3 sigma above background
# The result is:
#  loc = peaks[0]  # and array of pixel locations of the center of each peak
#  height = peaks[1]['peak_heights']   # an array of peak height
# Then the objects will be sorted according to their 'height' i.e., their flux:
#  sorted_objects = sorted(list(zip(loc, height)), key=lambda x: x[1], reverse=True)
#  trace_centers = np.array(sorted_objects)[:, 0]   # sorted array of central pixels


__author__ = 'Jens-Kristian Krogager'
__email__ = "krogager@iap.fr"
__credits__ = ["Jens-Kristian Krogager"]

import numpy as np
import matplotlib.pyplot as plt
try:
    import pyfits as pf
except:
    import astropy.io.fits as pf
from scipy.ndimage import median_filter
from numpy.polynomial import Chebyshev
import os
from os.path import isfile
from argparse import ArgumentParser
import warnings

try:
    from astroscrappy import detect_cosmics
    astroscrappy_installed = True
except:
    astroscrappy_installed = False

from extraction import extract_and_calibrate

code_dir = os.path.dirname(os.path.abspath(__file__))
v_file = os.path.join(code_dir, 'VERSION')
with open(v_file) as version_file:
    __version__ = version_file.read().strip()


def my_formatter(x, p, scale_pow):
    """Format tick marks to exponential notation"""
    return "%.0f" % (x / (10 ** scale_pow))


def mad(img):
    """Calculate Median Absolute Deviation from the median
    This is a robust variance estimator.
    For a Gaussian distribution:
        sigma ≈ 1.4826 * MAD
    """
    return np.median(np.abs(img - np.median(img)))


def fit_background(data, axis=1, x1=None, x2=None, interact=False, order=3, plot=True):
    """Fit background in 2D spectral data.
    The object trace should be masked out by defining x1 and x2 within
    which the background will not be fitted.
    The mask will automatically be defined if None is given (default).
    If interact is set to True, the mask will be interactively defined
    in a plotting window.
    The background is fitted by a Chebyshev polynomium (3rd order by default).

    Parameters
    ==========

    data : np.array (N, M)
        Spectral 2D data array

    axis : integer [default=1]
        Dispersion axis, 0: horizontal spectra, 1: vertical spectra

    x1 : integer [default=None]
        Mask pixels above this number in the fit to the background

    x2 : integer [default=None]
        Mask pixels below this number in the fit to the background

    interact : boolean [default=False]
        If True, this will open an interactive window to select the
        fitting regions. This is good if there are many objects on
        the slit, or if the trace is not well defined.

    order : integer [default=3]
        Order of the Chebyshev polynomium to fit the background

    Returns
    =======

    bg2D : np.array (N, M)
        Background model of the 2D frame, same shape as input data.
    """
    if axis == 0:
        # transpose the horizontal spectra to make them vertical:
        data = data.T

    x = np.arange(data.shape[1])
    # locate trace:
    if x1 is None or x2 is None:
        SPSF = np.median(data, 0)
        trace_center = np.argmax(SPSF)
        if interact:
            print("INTERACTIVE MODE ON!")
            good = False
            plt.close('all')
            plt.figure()
            while good is False:
                plt.plot(x, SPSF, color='RoyalBlue')
                plt.title("Mark left and right bounds of fitting region")
                print("\nMark left and right bounds of fitting region")
                print("")
                # plt.draw()
                sel = plt.ginput(-1, -1)
                mask = np.zeros(len(x), dtype=bool)
                sel = np.array(sel)
                if len(sel) % 2 == 0:
                    for x1, x2 in np.column_stack([sel[::2, 0], sel[1::2, 0]]):
                        mask += (x >= int(x1)) * (x <= int(x2))
                    masked_SPSF = np.ma.masked_where(~mask, SPSF)
                    plt.clf()
                    plt.plot(x, SPSF, color='0.6', lw=0.5)
                    plt.plot(x, masked_SPSF, color='k')
                    plt.draw()

                    answer = raw_input("Is the mask correct? [Y/n]  ")
                    if answer.lower() in ['n', 'no']:
                        good = False
                    else:
                        good = True

                else:
                    print("\n Something went wrong, the number of points must be even!\n")
                    continue

        else:
            width = 20
            x1 = trace_center - width
            x2 = trace_center + width
            obj = (x >= x1) * (x <= x2)
            mask = (x >= 20) * ~obj

    else:
        obj = (x >= x1) * (x <= x2)
        mask = (x >= 20) * ~obj

    bg2D = np.zeros_like(data)
    for i, row in enumerate(data):
        # Median filter the data to remove outliers:
        med_row = median_filter(row, 15)
        noise = mad(row)*1.4826
        this_mask = mask * (np.abs(row - med_row) < 10*noise)
        bg = Chebyshev.fit(x[this_mask], row[this_mask], order)
        bg2D[i] = bg(x)

    if axis == 0:
        # Rotate the data and the model back to horizontal orientation:
        data = data.T
        bg2D = bg2D.T

    plt.close('all')
    if plot:
        fig2D = plt.figure()
        ax1_2d = fig2D.add_subplot(121)
        ax2_2d = fig2D.add_subplot(122)
        noise = mad(data)
        v1 = np.median(data) - 3*noise
        v2 = np.median(data) + 3*noise
        ax1_2d.imshow(data, origin='lower', vmin=v1, vmax=v2)
        ax1_2d.set_title("Raw Data")
        ax2_2d.imshow(bg2D, origin='lower', vmin=v1, vmax=v2)
        ax2_2d.set_title("Background Model")
        if axis == 1:
            ax1_2d.set_xlabel("Spatial Direction  [pixels along slit]")
            ax2_2d.set_xlabel("Spatial Direction  [pixels along slit]")
            ax1_2d.set_ylabel("Spectral Direction  [pixels along wavelength]")
        else:
            ax1_2d.set_ylabel("Spatial Direction  [pixels along slit]")
            ax1_2d.set_xlabel("Spectral Direction  [pixels along wavelength]")
            ax2_2d.set_xlabel("Spectral Direction  [pixels along wavelength]")

        plt.show()

    return bg2D


def science_reduction_2d(sci_input, arc_frame, output='', bias='', flat='', trimx=[None, None],
                         trimy=[None, None], crr=True, axis=1, bg_x1=None, bg_x2=None,
                         bg_interact=False, bg_order=3, bg_plot=False, opt_ext=True,
                         loc_interact=True, ext_background=True, center_order=3, FWHM0=10,
                         loc_xmin=100, loc_xmax=-100, loc_binsize=20, aper_cen=None,
                         wl_order=4, sensitivity=None,
                         verbose=True, clobber=True, show=False):
    """
    Perform science reduction on a single input frame or a list of files.

    Parameters
    ==========

    sci_input : string or list
        Input science data to reduce.

    arc_frame : string or list
        Input arc spectrum or list of arc spectra for each science frame.

    output : string [default='']
        Output filename for final backgroung subtracted image.
        If not given, the output filename will be determined from
        OBJECT header keyword.

    bias : string
        Master bias to subtract from sci_input

    flat : string
        Normalized master flat

    trimx : list (2,)
        Lower and upper bounds, data outside the range will be trimmed

    trimy : list (2,)
        Lower and upper bounds, data outside the range will be trimmed

    crr : boolean [default=True]
        Perform cosmic ray rejection using IRAF.lacos_spec (van Dokkum 2001)

    axis : integer [default=1]
        Dispersion axis, 0: horizontal spectra, 1: vertical spectra

    bg : boolean [default=True]
        Subtract background model?

    bg_ : parameters passed to 'fit_backgroun()'

    verbose : boolean [default=True]
        If True, print status messages.

    clobber : boolean [default=True]
        Overwrite existing output file if True.

    opt_ext : boolean [default=True]
        Perform optimal extraction of 1D spectrum and automatically localize trace?

    loc_interact : boolean [default=True]
        Use interactive window to select trace center if automatic localization fails?
        If not, the aperture is centered on *aper_cen* which defaults to the center of the CCD.

    ext_background : boolean [default=True]
        Refine baclground subtraction by extracting the 1D background around the aperture?
        This is done by default in an aperture of width = FWHM0 on either side of the trace.

    center_order : integer [defualt=3]
        Polynomial order for the position of the trace along the CCD.
        The fitting is performed using a Chebyshev polynomium.

    FWHM0 : integer [default=10]
        Aperture width if optimal extraction is turned off or if localization fails.

    loc_xmin : integer [default=100]
        Lower boundary for trace fitting.

    loc_xmax : integer [default=-100]
        Upper boundary for trace fitting.

    loc_binsize : integer [default=20]
        The numer of spectral bins over which to average the trace before fitting.
    """
    if hasattr(sci_input, '__iter__'):
        pass
    else:
        sci_input = [sci_input]

    img0 = pf.getdata(sci_input[0])

    if hasattr(arc_frame, '__iter__'):
        arc_list = arc_frame
    else:
        arc_list = len(sci_input) * [arc_frame]

    if isfile(bias):
        mbias = pf.getdata(bias)
    else:
        mbias = np.zeros_like(img0)
        print(" WARNING - No master bias file provided!")

    if isfile(flat):
        mflat = pf.getdata(flat)
    else:
        mflat = np.ones_like(img0)
        print(" WARNING - No master flat file provided!")

    for sci_fname, arc_frame in zip(sci_input, arc_list):
        HDU = pf.open(sci_fname)
        if len(HDU) > 1:
            if verbose:
                print(" Merging extensions:")
                HDU.info()
                print("")
            hdr = HDU[0].header
            for key in HDU[1].header.keys():
                hdr[key] = HDU[1].header[key]
        else:
            hdr = pf.getheader(sci_fname)
        sci_raw = pf.getdata(sci_fname)

        x1, x2 = trimx
        y1, y2 = trimy
        sci = (sci_raw[y1:y2, x1:x2] - mbias[y1:y2, x1:x2])/mflat[y1:y2, x1:x2]

        # Calculate error image:
        if hdr['CCDNAME'] == 'CCD14':
            hdr['GAIN'] = 0.16
        g = hdr['GAIN']
        r = hdr['RDNOISE']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            err = np.sqrt(g*sci + r**2)/g

        # Subtract background:
        bg = fit_background(sci, axis=axis, x1=bg_x1, x2=bg_x2,
                            interact=bg_interact, plot=bg_plot,
                            order=bg_order)
        sci = sci - bg

        # Detect and correct cosmic ray hits:
        if crr and astroscrappy_installed:
            mask, sci = detect_cosmics(sci, gain=hdr['GAIN'], readnoise=hdr['RDNOISE'],
                                       niter=4, verbose=verbose)
            # Add comment to FITS header:
            hdr.add_comment("Cosmic Ray Rejection using Astroscrappy (based on van Dokkum 2001)")
            # expand mask to neighbouring pixels:
            big_mask = np.zeros_like(mask)
            for shift, axis in [(1, 0), (-1, 0), (1, 1), (-1, 1)]:
                big_mask += np.roll(mask, shift, axis)
            mask = 4 * (big_mask > 0)

        else:
            mask = np.zeros_like(sci)

        # bg_err = np.sqrt(g*bg + r**2)/g
        # Fix NaN values from negative pixel values:
        err_NaN = np.isnan(err)
        err[err_NaN] = r/g
        # err[err_NaN] = bg_err[err_NaN]

        sci_ext = pf.PrimaryHDU(sci, header=hdr)
        err_ext = pf.ImageHDU(err, header=hdr, name='ERR')
        mask_ext = pf.ImageHDU(mask, name='MASK')
        bg_ext = pf.ImageHDU(bg, name='SKY')

        output_HDU = pf.HDUList([sci_ext, err_ext, mask_ext, bg_ext])

        if len(output) > 0:
            if output[-5:] == '.fits':
                pass
            else:
                output += '.fits'
        else:
            output = "skysub2D_%s.fits" % hdr['OBJECT']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output_HDU.writeto(output, clobber=clobber)

        # Pass the corrected 2D spectrum to extraction and calibration:
        extract_and_calibrate(output, arc_frame, bin_size=loc_binsize, xmin=loc_xmin,
                              xmax=loc_xmax, do_opt_extract=opt_ext, interact=loc_interact,
                              background=ext_background, center_order=center_order,
                              FWHM0=FWHM0, trimx=trimx, trimy=trimy, wl_order=wl_order,
                              aper_cen=aper_cen, sensitivity=sensitivity, show=show)


if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    plt.interactive(True)
    parser = ArgumentParser()
    parser.add_argument("sci_in", type=str,
                        help="Raw science spectrum or file containing a list")
    parser.add_argument("arc_in", type=str,
                        help="Raw arc spectrum or file containing a list")
    parser.add_argument("--bias", type=str, default='MASTER_BIAS.fits',
                        help="Master BIAS frame")
    parser.add_argument("--flat", type=str, default='',
                        help="Normalized master flat frame")
    parser.add_argument("--no-crr", action="store_false",
                        help="When this option is set, no cosmic ray rejection is performed")
    parser.add_argument("-x", "--xrange", nargs=2, type=int, default=[None, None],
                        help="Give lower and upper limit for the x-range to use")
    parser.add_argument("-y", "--yrange", nargs=2, type=int, default=[None, None],
                        help="Give lower and upper limit for the y-range to use")
    parser.add_argument("--axis", type=int, default=1,
                        help="Dispersion axis, 0: horizontal, 1: vertical")
    parser.add_argument("--bg-x1", type=int, default=1,
                        help="Lower boundary for object mask excluded during fit")
    parser.add_argument("--bg-x2", type=int, default=1,
                        help="Upper boundary for object mask excluded during fit")
    parser.add_argument("--bg-interact", action="store_true",
                        help="Interactively select background region?")
    parser.add_argument("--bg-order", type=int, default=3,
                        help="Order for background polynomium")
    parser.add_argument("--bg-extract", type=str2bool, default=True,
                        help="Extract 1D background and refine subtraction?")
    parser.add_argument("--opt-ext", type=str2bool, default=True,
                        help="Disable optimal extraction")
    parser.add_argument("--loc-interact", action="store_true",
                        help="Interactively select aperture center")
    parser.add_argument("--loc-xmin", type=int, default=100,
                        help="Lower boundary on pixels for localization fitting")
    parser.add_argument("--loc-xmax", type=int, default=-100,
                        help="Upper boundary on pixels for localization fitting")
    parser.add_argument("--loc-binsize", type=int, default=30,
                        help="Number of spectral bins to combine for fitting of trace")
    parser.add_argument("--cen-order", type=int, default=3,
                        help="Order for trace centroid polynomium")
    parser.add_argument("--fwhm", type=int, default=10,
                        help="Default aperture width (+/- 1.5 x FWHM)")
    parser.add_argument("--sens", type=str, default=None,
                        help="Sensitivity function to flux calibrate the spectra")
    parser.add_argument("--aper-cen", type=int, default=None,
                        help="Center of aperture if automatic localization fails, default is center of CCD")
    parser.add_argument("--wl-order", type=int, default=4,
                        help="Order for wavelength solution polynomium")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print status updates")
    parser.add_argument("-s", "--show", action="store_true",
                        help="Show 1D extraction and diagnostics")

    args = parser.parse_args()

    if isfile(args.sci_in) and args.sci_in[-5:] == '.fits':
        sci_input = args.sci_in
    else:
        sci_input = list()
        with open(args.sci_in) as input_file:
            for line in input_file.readlines():
                line = line.strip()
                if ' ' in line:
                    fname = line.split()[0]
                else:
                    fname = line
                if fname.strip() != '':
                    sci_input.append(fname)

    if isfile(args.arc_in) and args.arc_in[-5:] == '.fits':
        arc_frame = args.arc_in
    else:
        arc_frame = list()
        with open(args.arc_in) as input_file:
            for line in input_file.readlines():
                line = line.strip()
                if ' ' in line:
                    fname = line.split()[0]
                else:
                    fname = line
                if fname.strip() != '':
                    arc_frame.append(fname)

    science_reduction_2d(sci_input, arc_frame, bias=args.bias, flat=args.flat, trimx=args.xrange,
                         trimy=args.yrange, crr=args.no_crr, axis=args.axis, bg_x1=args.bg_x1,
                         bg_x2=args.bg_x2,
                         bg_interact=args.bg_interact, bg_order=args.bg_order,
                         opt_ext=args.opt_ext, loc_interact=args.loc_interact,
                         ext_background=args.bg_extract,
                         FWHM0=args.fwhm, aper_cen=args.aper_cen,
                         loc_xmin=args.loc_xmin, loc_xmax=args.loc_xmax,
                         loc_binsize=args.loc_binsize, center_order=args.cen_order,
                         wl_order=args.wl_order, verbose=args.verbose, show=args.show)
