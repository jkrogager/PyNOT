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
   - correct_cosmics
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
from astropy.io import fits
import os
import warnings

from astroscrappy import detect_cosmics

from pynot import instrument
from pynot.functions import get_version_number


__version__ = get_version_number()


def trim_overscan(img, hdr):
    """Trim the overscan regions on either side in X and on top in Y"""
    if 'OVERSCAN' in hdr and hdr['OVERSCAN'] == 'TRIMMED':
        # Overscan has already been trimmed:
        return img, hdr

    # Get overscan from instrument:
    pre_x, over_x, pre_y, over_y = instrument.overscan()
    Npix_x, Npix_y = instrument.get_ccd_extent()

    # Get detector pixel arrays from instrument:
    X, Y = instrument.get_detector_arrays(hdr)

    img_region_x = (X >= pre_x) & (X <= Npix_x-over_x)
    xlimits = img_region_x.nonzero()[0]
    img_region_y = (Y >= pre_y) & (Y <= Npix_y-over_y)
    ylimits = img_region_y.nonzero()[0]
    x1 = min(xlimits)
    x2 = max(xlimits)+1
    y1 = min(ylimits)
    y2 = max(ylimits)+1
    img_trim = img[y1:y2, x1:x2]
    x_type = hdr['CTYPE1'].strip()
    if 'RA' in x_type or 'DEC' in x_type:
        # Header contains WCS information!
        # Shift the reference pixel instead of the reference value
        hdr['CRPIX1'] -= x1
        hdr['CRPIX2'] -= y1
    else:
        hdr['CRVAL1'] += x1
        hdr['CRVAL2'] += y1

    hdr['NAXIS1'] = img_trim.shape[1]
    hdr['NAXIS2'] = img_trim.shape[0]
    hdr['OVERSCAN'] = 'TRIMMED'
    hdr['OVERSC_X'] = (np.sum(~img_region_x), "Number of pixels removed along X-axis")
    hdr['OVERSC_Y'] = (np.sum(~img_region_y), "Number of pixels removed along Y-axis")
    return img_trim, hdr


def detect_filter_edge(fname):
    """Automatically detect edges in the normalized flat field"""
    # Get median profile along slit:
    img = fits.getdata(fname)

    # Using the normalized flat field, the values are between 0 and 1.
    # Convert the image to a binary mask image:
    img[img > 0.5] = 1.
    img[img < 0.5] = -1.

    fx = np.median(img, 0) > 0.5
    fy = np.median(img, 1) > 0.5

    # Detect initial edges:
    x1 = np.min(fx.nonzero()[0])
    x2 = np.max(fx.nonzero()[0])

    y1 = np.min(fy.nonzero()[0])
    y2 = np.max(fy.nonzero()[0])

    # If the edge is curved, decrease the trim edges
    # until they are fully inside the image region
    lower = img[y1, x1]
    upper = img[y2, x2]
    while lower < 0 and upper < 0:
        x1 += 1
        y1 += 1
        x2 -= 1
        y2 -= 1
        lower = img[y1, x1]
        upper = img[y2, x2]

    return (x1, x2, y1, y2)


def trim_filter_edge(fname, x1, x2, y1, y2, output='', output_dir=''):
    """Trim image edges"""
    # Get median profile along slit:
    msg = list()
    msg.append("          - Loaded file: %s" % fname)

    if output == '':
        basename = os.path.basename(fname)
        output = 'trim_' + basename

    if output_dir != '':
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output = os.path.join(output_dir, output)

    with fits.open(fname) as hdu_list:
        for hdu in hdu_list:
            if hdu.data is None:
                continue

            data = hdu.data
            hdr = hdu.header
            data_trim = data[y1:y2, x1:x2]
            hdr['CRPIX1'] -= x1
            hdr['NAXIS1'] = data_trim.shape[1]
            hdr['CRPIX2'] -= y1
            hdr['NAXIS2'] = data_trim.shape[0]
            hdr.add_comment("Image trimmed by PyNOT")
            hdu.data = data_trim
            hdu.header = hdr
        hdu_list.writeto(output, overwrite=True)
    msg.append(" [OUTPUT] - Saving trimmed image: %s" % output)
    msg.append("")
    output_msg = "\n".join(msg)

    return output_msg


def correct_cosmics(input_fname, output_fname, niter=4, gain=None, readnoise=None,
                    sigclip=4.5, sigfrac=0.3, objlim=5.0, satlevel=113500.0, cleantype='meanmask'):
    """
    Detect and Correct Cosmic Ray Hits based on the method by van Dokkum (2001)
    The corrected frame is saved to a FITS file.

    Parameters
    ----------
    input_fname : string
        Input filename of 2D image

    output_fname : string
        Filename of corrected 2D image

    For details on other parameters, see `astroscrappy.detect_cosmics`

    Returns
    -------
    output_msg : string
        Log of messages from the function call
    """
    msg = list()
    msg.append("          - Cosmic Ray Rejection using Astroscrappy (based on van Dokkum 2001)")
    sci = fits.getdata(input_fname)
    hdr = fits.getheader(input_fname)
    # hdr['EXTNAME'] = 'DATA'
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
        gain = instrument.get_gain(hdr)
        if gain:
            msg.append("          - Read GAIN from FITS header: %.2f" % gain)
        else:
            user_input = input("          > Please give the detector gain:\n          > ")
            try:
                gain = float(user_input)
            except ValueError:
                msg.append(" [ERROR]  - Invalid gain! Must be a number")
                msg.append("")
                return "\n".join(msg)

    if not readnoise:
        readnoise = instrument.get_readnoise(hdr)
        if readnoise:
            msg.append("          - Read RDNOISE from FITS header: %.2f" % readnoise)
        else:
            user_input = input("          > Please give the detector read noise:\n          > ")
            try:
                readnoise = float(user_input)
            except ValueError:
                msg.append(" [ERROR]  - Invalid read noise! Must be a number")
                msg.append("")
                return "\n".join(msg)

    sci = (sci + sky_level) * gain
    crr_mask, sci = detect_cosmics(sci, gain=gain, readnoise=readnoise, niter=niter,
                                   sigclip=sigclip, sigfrac=sigfrac, objlim=objlim,
                                   satlevel=instrument.get_saturation_level(),
                                   cleantype=cleantype)
    # Corrected image is in ELECTRONS! Convert back to ADUs:
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


def raw_correction(sci_raw, hdr, bias_fname, flat_fname='', output='', overwrite=True, mode='spec'):
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

    flat_fname : string  [default='']
        Filename of normalized flat field image. If none is given, no flat field correction will be applied

    output : string  [default='']
        Output filename

    overwrite : boolean  [default=True]
        Overwrite existing output file if True.

    Returns
    -------
    output_msg : string
        Log of status messages
    """
    msg = list()
    if bias_fname:
        mbias = fits.getdata(bias_fname)
        # bias_hdr = instrument.get_header(bias_fname)
        msg.append("          - Loaded combined bias image: %s" % bias_fname)
    else:
        mbias = 0.
        msg.append("          - No bias image. Using bias level = 0")

    if flat_fname:
        mflat = fits.getdata(flat_fname)
        mflat[mflat == 0] = 1
        msg.append("          - Loaded combined flat field image: %s" % flat_fname)
    else:
        mflat = 1.
        msg.append("          - Not flat field image provided. Using flat level = 1")

    # Trim overscan of raw image:
    # - Trimming again of processed images doesn't change anything,
    # - so do it just in case the input has not been trimmed
    sci_raw, hdr = trim_overscan(sci_raw, hdr)
    # sci_raw, hdr = trim_overscan(sci_raw, hdr, overscan=overscan, mode=mode)

    # Correct image:
    sci = (sci_raw - mbias)/mflat

    # Calculate error image:
    gain = instrument.get_gain(hdr)
    readnoise = instrument.get_readnoise(hdr)
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
    hdr['AUTHOR'] = 'PyNOT version %s' % __version__

    mask = np.zeros_like(sci, dtype=int)
    mask[err_NaN] = 4
    msg.append("          - Pixel mask created")
    mask_hdr = fits.Header()
    mask_hdr.add_comment("0 = Good Pixels")
    mask_hdr.add_comment("1 = Cosmic Ray Hits")
    mask_hdr.add_comment("4 = NaN error from base CDD processing")
    for key in ['CRPIX1', 'CRPIX2', 'CRVAL1', 'CRVAL2', 'CTYPE1', 'CTYPE2', 'CUNIT1', 'CUNIT2']:
        if key in hdr:
            mask_hdr[key] = hdr[key]
        else:
            mask_hdr[key] = ''

    if mode == 'spec':
        cdelt1 = hdr.get('CDELT1')
        if cdelt1 is None:
            cdelt1 = hdr.get('CD1_1')
        cdelt2 = hdr.get('CDELT2')
        if cdelt2 is None:
            cdelt2 = hdr.get('CD2_2')
        mask_hdr['CDELT1'] = cdelt1
        mask_hdr['CDELT2'] = cdelt2
    else:
        if 'CD1_1' in hdr:
            mask_hdr['CD1_1'] = hdr['CD1_1']
            mask_hdr['CD1_2'] = hdr['CD1_2']
            mask_hdr['CD2_1'] = hdr['CD2_1']
            mask_hdr['CD2_2'] = hdr['CD2_2']
        else:
            mask_hdr['CDELT1'] = hdr['CDELT1']
            mask_hdr['CDELT2'] = hdr['CDELT2']

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



def correct_raw_file(input_fname, *, output, bias_fname, flat_fname='', overwrite=True, mode='spec'):
    """
    Wrapper for `raw_correction` using file input instead of image input

    Returns
    -------
    output_msg : string
        Log of status messages
    """
    hdr = instrument.get_header(input_fname)
    sci_raw = fits.getdata(input_fname)
    msg = "          - Loaded input image: %s" % input_fname

    output_msg = raw_correction(sci_raw, hdr, bias_fname, flat_fname, output=output,
                                overwrite=overwrite, mode=mode)
    output_msg = msg + '\n' + output_msg

    return output_msg
