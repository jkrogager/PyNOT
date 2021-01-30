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
import astropy.io.fits as pf
from scipy.ndimage import median_filter
from numpy.polynomial import Chebyshev
import os
from os.path import isfile
from argparse import ArgumentParser
import warnings

from astroscrappy import detect_cosmics

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


def auto_fit_background(data, axis=2, order=3, width=20, plot=True, plot_fname=''):
    """Fit background in 2D spectral data.
    The background is fitted along the spatial columns/rows
    by a Chebyshev polynomium.

    Parameters
    ==========
    data : np.array (N, M)
        Spectral 2D data array

    axis : integer  [default=1]
        Dispersion axis, 1: horizontal spectra, 2: vertical spectra

    order : integer  [default=3]
        Order of the Chebyshev polynomium to fit the background

    width : integer  [default=20]
        Half width in pixels of the trace to mask out

    Returns
    =======
    bg2D : np.array (N, M)
        Background model of the 2D frame, same shape as input data.
    """
    if axis == 1:
        # transpose the horizontal spectra to make them vertical
        # since it's faster to fit rows than columns
        data = data.T

    x = np.arange(data.shape[1])
    SPSF = np.median(data, 0)
    trace_center = np.argmax(SPSF)
    width = 20
    x1 = trace_center - width
    x2 = trace_center + width
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
        ax1_2d.set_xlabel("Spatial Axis  [pixels]")
        ax2_2d.set_xlabel("Spatial Axis  [pixels]")
        ax1_2d.set_ylabel("Dispersion Axis  [pixels]")
        fig2D.savefig(plot_fname)
        fig2D.close()

    if axis == 1:
        # Rotate the model back to horizontal orientation:
        bg2D = bg2D.T

    return bg2D


def raw_correction(sci_raw, hdr, bias_fname, flat_fname, output='', crr=True, niter=4, verbose=True, overwrite=True):
    """
    Perform bias subtraction, flat field correction, and cosmic ray rejection

    Parameters
    ==========

    sci_raw : np.array (M, N)
        Input science image to reduce

    hdr : FITS Header
        Header associated with the science image

    bias_fname : string
        Filename of bias image to subtract from `sci_raw`

    flat_fname : string
        Filename of normalized flat field image

    output : string  [default='']
        Output filename for final backgroung subtracted image.
        If not given, the output filename will be determined from
        OBJECT header keyword.

    crr : boolean  [default=True]
        Perform cosmic ray rejection using astroscrappy (based on van Dokkum 2001)

    niter : integer  [default=4]
        Number of iterations for cosmic ray rejection

    verbose : boolean  [default=True]
        If True, print status messages.

    overwrite : boolean  [default=True]
        Overwrite existing output file if True.

    Returns
    -------

    """
    msg = list
    msg.append("          - Running task: bias and flat field correction")
    mbias = pf.getdata(bias_fname)
    msg.append("          - Loaded BIAS image: %s" % bias_fname)
    mflat = pf.getdata(flat_fname)
    msg.append("          - Loaded FLAT field image: %s" % flat_fname)

    sci = (sci_raw - mbias)/mflat

    # Calculate error image:
    if hdr['CCDNAME'] == 'CCD14':
        hdr['GAIN'] = 0.16
    gain = hdr['GAIN']
    readnoise = hdr['RDNOISE']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        err = np.sqrt(gain*sci + readnoise**2) / gain
    msg.append("          - Created noise image")
    msg.append("          - Gain=%.2f  and Read Noise=%.2f" % (gain, readnoise))

    # Fix NaN values from negative pixel values:
    err_NaN = np.isnan(err)
    err[err_NaN] = readnoise/gain
    msg.append("          - Correcting NaNs in noise image: %i pixel(s)" % np.sum(err_NaN))

    # Detect and correct cosmic ray hits:
    if crr:
        msg.append("")
        msg.append("          - Running task: Cosmic Ray Rejection")
        msg.append("          - Cosmic Ray Rejection using Astroscrappy (based on van Dokkum 2001)")
        mask, sci = detect_cosmics(sci, gain=hdr['GAIN'], readnoise=hdr['RDNOISE'],
                                   niter=niter, verbose=verbose)
        # Add comment to FITS header:
        hdr.add_comment("Cosmic Ray Rejection using Astroscrappy (based on van Dokkum 2001)")
        # expand mask to neighbouring pixels:
        msg.append("          - Number of cosmic ray hits identified: %i" % np.sum(mask > 0))
        msg.append("          - Expanding cosmic hit mask by one pixel")
        big_mask = np.zeros_like(mask)
        for shift, axis in [(1, 0), (-1, 0), (1, 1), (-1, 1)]:
            big_mask += np.roll(mask, shift, axis)
        mask = 4 * (big_mask > 0)

    else:
        mask = np.zeros_like(sci)
        msg.append("          - Empty pixel mask created")
    mask_hdr = pf.Header()
    mask_hdr.add_comment("4 = Cosmic Ray Hit")
    hdr['DATAMIN'] = np.nanmin(sci)
    hdr['DATAMAX'] = np.nanmax(sci)

    sci_ext = pf.PrimaryHDU(sci, header=hdr)
    err_ext = pf.ImageHDU(err, header=hdr, name='ERR')
    mask_ext = pf.ImageHDU(mask, header=mask_hdr, name='MASK')
    output_HDU = pf.HDUList([sci_ext, err_ext, mask_ext])

    if output:
        if output[-5:] != '.fits':
            output += '.fits'
    else:
        output = "skysub2D_%s.fits" % hdr['OBJECT']

    output_HDU.writeto(output, overwrite=overwrite)
    msg.append("          - Successfully corrected the image.")
    msg.append("          - Saving output: %s" % output)
    msg.append("")
    output_msg = "\n".join(msg)

    return output_msg
    # # Pass the corrected 2D spectrum to extraction and calibration:
    # extract_and_calibrate(output, arc_frame, bin_size=loc_binsize, xmin=loc_xmin,
    #                       xmax=loc_xmax, do_opt_extract=opt_ext, interact=loc_interact,
    #                       background=ext_background, center_order=center_order,
    #                       FWHM0=FWHM0, trimx=trimx, trimy=trimy, wl_order=wl_order,
    #                       aper_cen=aper_cen, sensitivity=sensitivity, show=show)
