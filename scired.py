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
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from numpy.polynomial import Chebyshev
import os
import warnings

from astroscrappy import detect_cosmics
from alfosc import create_pixel_array

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
    return np.nanmedian(np.abs(img - np.nanmedian(img)))


def trim_overscan(img, hdr, overscan=50):
    # Trim overscan
    X = create_pixel_array(hdr, 1)
    Y = create_pixel_array(hdr, 2)
    img_region_x = (X >= overscan) & (X <= 2148-overscan)
    xlimits = img_region_x.nonzero()[0]
    img_region_y = (Y <= 2102-overscan)
    ylimits = img_region_y.nonzero()[0]
    x1 = min(xlimits)
    x2 = max(xlimits)+1
    y1 = min(ylimits)
    y2 = max(ylimits)+1
    img_trim = img[y1:y2, x1:x2]
    hdr['CRPIX1'] += x1
    hdr['CRPIX2'] += y1
    return img_trim, hdr


def fit_background_image(data, order_bg=3, xmin=0, xmax=None, kappa=10, fwhm_scale=3):
    """
    Fit background in 2D spectral data. The background is fitted along the spatial rows by a Chebyshev polynomium.

    Parameters
    ==========
    data : np.array(M, N)
        Data array holding the input image

    order_bg : integer  [default=3]
        Order of the Chebyshev polynomium to fit the background

    width : integer  [default=20]
        Half width in pixels of the trace to mask out

    Returns
    =======
    bg2D : np.array(M, N)
        Background model of the 2D frame, same shape as input data.
    """
    x = np.arange(data.shape[1])
    if xmax is None:
        xmax = len(x)
    if xmax < 0:
        xmax = len(x) + xmax
    SPSF = np.nanmedian(data, 0)
    noise = 1.5*mad(SPSF)
    peaks, properties = find_peaks(SPSF, prominence=kappa*noise, width=3)
    mask = (x >= xmin) & (x <= xmax)
    for num, center in enumerate(peaks):
        width = properties['widths'][num]
        x1 = center - width*fwhm_scale
        x2 = center + width*fwhm_scale
        obj = (x >= x1) * (x <= x2)
        mask &= ~obj

    bg2D = np.zeros_like(data)
    for i, row in enumerate(data):
        # Median filter the data to remove outliers:
        med_row = median_filter(row, 15)
        noise = mad(row)*1.4826
        this_mask = mask * (np.abs(row - med_row) < 10*noise)
        if np.sum(this_mask) > order_bg+1:
            bg_model = Chebyshev.fit(x[this_mask], row[this_mask], order_bg, domain=[x.min(), x.max()])
            bg2D[i] = bg_model(x)

    return bg2D


def auto_fit_background(data_fname, output_fname, dispaxis=2, order_bg=3, kappa=10, fwhm_scale=3, xmin=0, xmax=None, plot_fname=''):
    """
    Fit background in 2D spectral data. The background is fitted along the spatial rows by a Chebyshev polynomium.

    Parameters
    ==========
    data_fname : string
        Filename of the FITS image to process

    output_fname : string
        Filename of the output FITS image containing the background subtracted image
        as well as the background model in a separate extension.

    dispaxis : integer  [default=1]
        Dispersion axis, 1: horizontal spectra, 2: vertical spectra
        The function does not rotate the final image, only the intermediate image
        since it's faster to operate on rows than on columns.

    order_bg : integer  [default=3]
        Order of the Chebyshev polynomium to fit the background

    width : integer  [default=20]
        Half width in pixels of the trace to mask out

    plot_fname : string  [default='']
        Filename of diagnostic plots. If nothing is given, do not plot.

    Returns
    =======
    output_fname : string
        Background model of the 2D frame, same shape as input data.

    output_msg : string
        Log of messages from the function call
    """
    msg = list()
    data = fits.getdata(data_fname)
    hdr = fits.getheader(data_fname)
    if 'DISPAXIS' in hdr:
        dispaxis = hdr['DISPAXIS']

    if dispaxis == 1:
        # transpose the horizontal spectra to make them vertical
        # since it's faster to fit rows than columns
        data = data.T
    msg.append("          - Loaded input image: %s" % data_fname)

    msg.append("          - Fitting background along the spatial axis with polynomium of order: %i" % order_bg)
    msg.append("          - Automatic masking of outlying pixels and object trace")
    bg2D = fit_background_image(data, order_bg=order_bg, kappa=kappa, fwhm_scale=fwhm_scale, xmin=xmin, xmax=xmax)

    if plot_fname:
        fig2D = plt.figure()
        ax1_2d = fig2D.add_subplot(121)
        ax2_2d = fig2D.add_subplot(122)
        noise = mad(data)
        v1 = np.median(data) - 5*noise
        v2 = np.median(data) + 5*noise
        ax1_2d.imshow(data, origin='lower', vmin=v1, vmax=v2)
        ax1_2d.set_title("Input Image")
        ax2_2d.imshow(data-bg2D, origin='lower', vmin=v1, vmax=v2)
        ax2_2d.set_title("Background Subtracted")
        ax1_2d.set_xlabel("Spatial Axis  [pixels]")
        ax2_2d.set_xlabel("Spatial Axis  [pixels]")
        ax1_2d.set_ylabel("Dispersion Axis  [pixels]")
        fig2D.tight_layout()
        fig2D.savefig(plot_fname)
        plt.close()
        msg.append(" [OUTPUT] - Saving diagnostic figure: %s" % plot_fname)

    data = data - bg2D
    if dispaxis == 1:
        # Rotate the model and data back to horizontal orientation:
        data = data.T
        bg2D = bg2D.T

    with fits.open(data_fname) as hdu:
        hdu[0].data = data
        sky_hdr = fits.Header()
        sky_hdr['BUNIT'] = 'count'
        copy_keywords = ['CRPIX1', 'CRVAL1', 'CDELT1', 'CTYPE1', 'CUNIT1']
        copy_keywords += ['CRPIX2', 'CRVAL2', 'CDELT2']
        sky_hdr['CTYPE2'] = 'LINEAR'
        sky_hdr['CUNIT2'] = 'Pixel'
        for key in copy_keywords:
            sky_hdr[key] = hdr[key]
        sky_hdr['AUTHOR'] = 'PyNOT version %s' % __version__
        sky_hdr['ORDER'] = (order_bg, "Polynomial order along spatial rows")
        sky_ext = fits.ImageHDU(bg2D, header=sky_hdr, name='SKY')
        hdu.append(sky_ext)
        hdu.writeto(output_fname, overwrite=True)

    msg.append(" [OUTPUT] - Saving background subtracted image: %s" % output_fname)
    msg.append("")
    output_msg = "\n".join(msg)
    return output_msg


def correct_cosmics(input_fname, output_fname, niter=4, gain=None, readnoise=None):
    msg = list()
    msg.append("          - Cosmic Ray Rejection using Astroscrappy (based on van Dokkum 2001)")
    sci = fits.getdata(input_fname)
    hdr = fits.getheader(input_fname)
    msg.append("          - Loaded input image: %s" % input_fname)
    with fits.open(input_fname) as hdu:
        if 'SKY' in hdu:
            sky_level = np.median(hdu['SKY'].data)
            msg.append("          - Image has been sky subtracted. Median sky level: %.1f" % sky_level)
        else:
            sky_level = 0.

        if 'MASK' in hdu:
            mask = hdu['MASK'].data
        else:
            mask = np.zeros_like(sci, dtype=int)

    if not gain:
        gain = hdr['GAIN']
        msg.append("          - Read GAIN from FITS header: %.2f" % gain)
    if not readnoise:
        try:
            readnoise = hdr['RDNOISE']
            msg.append("          - Read RDNOISE from FITS header: %.2f" % readnoise)
        except:
            readnoise = hdr['READNOISE']
            msg.append("          - Read READNOISE from FITS header: %.2f" % readnoise)

    crr_mask, sci = detect_cosmics(sci, gain=gain, readnoise=readnoise, niter=niter, pssl=sky_level)
    # Corrected image is in ELECTRONS!! Convert back to ADUs:
    sci = sci/gain - sky_level

    # Add comment to FITS header:
    hdr.add_comment("Cosmic Ray Rejection using Astroscrappy (based on van Dokkum 2001)")
    # expand mask to neighbouring pixels:
    msg.append("          - Number of cosmic ray hits identified: %i" % np.sum(mask > 0))

    mask = mask + 1*(crr_mask > 0)
    with fits.open(input_fname) as hdu:
        if hdu[0].data is not None:
            hdu[0].data = sci
            hdu[0].header = hdr
        else:
            hdu[1].data = sci
            hdu[1].header = hdr

        if 'MASK' in hdu:
            hdu['MASK'].data = mask
            hdu['MASK'].header.add_comment("Cosmic Ray Rejection using Astroscrappy (based on van Dokkum 2001)")
        else:
            mask_hdr = fits.Header()
            mask_hdr.add_comment("0 = Good Pixels")
            mask_hdr.add_comment("1 = Cosmic Ray Hits")
            mask_hdr.add_comment("Cosmic Ray Rejection using Astroscrappy (based on van Dokkum 2001)")
            mask_ext = fits.ImageHDU(mask, header=mask_hdr, name='MASK')
            hdu.append(mask_ext)
        hdu.writeto(output_fname, overwrite=True)
    msg.append(" [OUTPUT] - Saving cosmic ray corrected image: %s" % output_fname)
    msg.append("")
    output_msg = "\n".join(msg)
    return output_msg


def raw_correction(sci_raw, hdr, bias_fname, flat_fname, output='', overscan=50, overwrite=True):
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

    overscan : int  [default=50]
        Number of pixels in overscan at the edge of the CCD.
        The overscan region will be trimmed.

    overwrite : boolean  [default=True]
        Overwrite existing output file if True.

    Returns
    -------
    output_msg : string
        Log of status messages
    """
    msg = list()
    mbias = fits.getdata(bias_fname)
    msg.append("          - Loaded BIAS image: %s" % bias_fname)
    mflat = fits.getdata(flat_fname)
    mflat[mflat == 0] = 1
    msg.append("          - Loaded FLAT field image: %s" % flat_fname)

    sci = (sci_raw - mbias)/mflat

    # Trim overscan
    sci, hdr = trim_overscan(sci, hdr, overscan=overscan)

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
    hdr['DATAMIN'] = np.nanmin(sci)
    hdr['DATAMAX'] = np.nanmax(sci)
    hdr['EXTNAME'] = 'DATA'

    mask = np.zeros_like(sci, dtype=int)
    msg.append("          - Empty pixel mask created")
    mask_hdr = fits.Header()
    mask_hdr.add_comment("0 = Good Pixels")
    mask_hdr.add_comment("1 = Cosmic Ray Hits")

    sci_ext = fits.PrimaryHDU(sci, header=hdr)
    err_ext = fits.ImageHDU(err, header=hdr, name='ERR')
    mask_ext = fits.ImageHDU(mask, header=mask_hdr, name='MASK')
    output_HDU = fits.HDUList([sci_ext, err_ext, mask_ext])
    output_HDU.writeto(output, overwrite=overwrite)
    msg.append("          - Successfully corrected the image.")
    msg.append(" [OUTPUT] - Saving output: %s" % output)
    msg.append("")
    output_msg = "\n".join(msg)

    return output_msg
    # # Pass the corrected 2D spectrum to extraction and calibration:
    # extract_and_calibrate(output, arc_frame, bin_size=loc_binsize, xmin=loc_xmin,
    #                       xmax=loc_xmax, do_opt_extract=opt_ext, interact=loc_interact,
    #                       background=ext_background, center_order=center_order,
    #                       FWHM0=FWHM0, trimx=trimx, trimy=trimy, wl_order=wl_order,
    #                       aper_cen=aper_cen, sensitivity=sensitivity, show=show)
