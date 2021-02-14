# -*- coding: UTF-8 -*-
"""
Script to combine bias and spectral flat frames for use in final data reduction.
"""
__author__ = 'Jens-Kristian Krogager'
__email__ = "krogager@iap.fr"
__credits__ = ["Jens-Kristian Krogager"]

from argparse import ArgumentParser
import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import ndimage, signal
from numpy.polynomial import Chebyshev
import os
from os.path import exists, basename

from pynot import alfosc
from pynot.functions import mad, my_formatter, get_version_number
from pynot.scired import trim_overscan


__version__ = get_version_number()


def combine_bias_frames(bias_frames, output='', kappa=15, overwrite=True, overscan=50, mode='spec'):
    """Combine individual bias frames to create a 'master bias' frame.
    The combination is performed using robust sigma-clipping and
    median combination. Bad pixels are subsequently replaced by the
    median value of the final combined image.

    Parameters
    ==========

    bias_frames : list of strings, or other iterable
        List containing file names for the individual bias frames

    output : string [default='']
        Output file name for the final combined image.

    kappa : integer [default=15]
        Number of sigmas above which to reject pixels.

    overwrite : boolean [default=False]
        Overwrite existing output file if True.

    Returns
    =======
    output : string
        Filename of combined bias image

    output_msg: string
        The log of the function steps and errors
    """
    msg = list()

    bias = list()
    for frame in bias_frames:
        msg.append("          - Loaded bias frame: %s" % frame)
        raw_img = pf.getdata(frame)
        bias_hdr = alfosc.get_alfosc_header(frame)
        if mode == 'spec':
            trim_bias, bias_hdr = trim_overscan(raw_img, bias_hdr, overscan)
            msg.append("          - Trimming overscan of bias images: %i!" % overscan)
        else:
            trim_bias = raw_img
        if len(bias) > 1:
            assert trim_bias.shape == bias[0].shape, "Images must have same shape!"
        bias.append(trim_bias)

    mask = np.zeros_like(bias[0], dtype=int)
    median_img0 = np.median(bias, 0)
    sig = mad(median_img0)*1.4826
    masked_bias = list()
    for img in bias:
        this_mask = np.abs(img - median_img0) > kappa*sig
        masked_bias.append(np.ma.masked_where(this_mask, img))
        mask += 1*this_mask
    msg.append("          - Masking outlying pixels: kappa = %f" % kappa)
    msg.append("          - Total number of masked pixels: %i" % np.sum(mask > 0))

    master_bias = np.median(masked_bias, 0)
    Ncomb = len(bias) - mask

    master_bias[Ncomb == 0] = np.median(master_bias[Ncomb != 0])
    msg.append("          - Combined %i files" % len(bias))

    hdr = pf.getheader(bias_frames[0], 0)
    hdr1 = pf.getheader(bias_frames[0], 1)
    for key in hdr1.keys():
        hdr[key] = hdr1[key]
    hdr['NCOMBINE'] = len(bias_frames)
    hdr.add_comment('Median combined Master Bias')
    hdr.add_comment('PyNOT version %s' % __version__)
    if not output:
        output = 'MASTER_BIAS.fits'

    pf.writeto(output, master_bias, header=hdr, overwrite=overwrite)
    msg.append(" [OUTPUT] - Saving combined Bias Image: %s" % output)
    msg.append("")
    output_msg = "\n".join(msg)

    return output, output_msg


def combine_flat_frames(raw_frames, output, mbias='', mode='spec', dispaxis=2,
                        kappa=5, verbose=False, overwrite=True, overscan=50):
    """Combine individual spectral flat frames to create a 'master flat' frame.
    The individual frames are normalized to the mode of the 1D collapsed spectral
    shape. Individual frames are clipped using a kappa-sigma-clipping on the mode
    values to discard outliers.
    The input frames are by default matched to a given slit-width, though this can
    be turned off. The variations from one slit to another is very small.
    The normalized 2D frames are then median combined and the final image is
    multiplied by the median normalization to restore the ADU values of the image.

    Parameters
    ==========

    raw_frames : list of strings, or other iterable
        List containing file names for the individual flat frames

    mbias : string [default='']
        Master bias file name to subtract bias from individual frames.
        If nothing is given, no bias level correction is performed.

    output : string [default='']
        Output file name for the final combined image.

    mode : string ['spec' or 'img']
        Combine spectral flats or imaging flats. Default is 'spec'

    dispaxis : integer  [default=2]
        Dispersion axis. 1: Horizontal spectra, 2: vertical spectra
        For the majority of ALFOSC spectra, the default is 2.

    kappa : integer  [default=5]
        Number of sigmas above which to reject pixels.

    verbose : boolean  [default=False]
        If True, print status messages.

    overwrite : boolean  [default=True]
        Overwrite existing output file if True.

    Returns
    =======
    output : string
        Filename of combined flat field image

    output_msg : string
        The log of the function steps and errors
    """
    msg = list()
    if mbias and exists(mbias):
        bias = pf.getdata(mbias)
        bias_hdr = alfosc.get_alfosc_header(mbias)
        if mode == 'spec':
            bias, bias_hdr = trim_overscan(bias, bias_hdr, overscan)
            msg.append("          - Trimming overscan of bias image: %i!" % overscan)
    else:
        msg.append("[WARNING] - No master bias frame provided!")
        bias = 0.

    flats = list()
    flat_peaks = list()
    for fname in raw_frames:
        hdr = alfosc.get_alfosc_header(fname)
        flat = pf.getdata(fname)
        if mode == 'spec':
            flat, hdr = trim_overscan(flat, hdr, overscan)
            flat = flat - bias
            peak_val = np.max(np.mean(flat, dispaxis-1))
            flats.append(flat/peak_val)
            flat_peaks.append(peak_val)
            msg.append("          - Loaded Spectral Flat file: %s   mode=%.1f" % (fname, peak_val))
            msg.append("          - Trimming overscan of Flat images: %i!" % overscan)

        else:
            flat = flat - bias
            pad = np.max(flat.shape) // 4
            peak_val = np.median(flat[pad:-pad, pad:-pad])
            flats.append(flat/peak_val)
            flat_peaks.append(peak_val)
            msg.append("          - Loaded Imaging Flat file: %s   median=%.1f" % (fname, peak_val))

    mask = np.zeros_like(flats[0], dtype=int)
    median_img0 = np.median(flats, 0)
    sig = mad(median_img0)*1.4826
    masked_flats = list()
    for img in flats:
        this_mask = np.abs(img - median_img0) > kappa*sig
        masked_flats.append(np.ma.masked_where(this_mask, img))
        mask += 1*this_mask
    msg.append("          - Standard deviation of raw median image: %.1f ADUs" % sig)
    msg.append("          - Masking outlying pixels using a threshold of kappa=%.1f" % kappa)
    msg.append("          - Total number of masked pixels: %i" % np.sum(mask > 0))
    msg.append("          - Median value of combined flat: %i" % np.sum(mask > 0))

    # Take the mean of the sigma-clipped images.
    flat_combine = np.mean(masked_flats, 0)
    if mode == 'spec':
        # Scale the image back to the original ADU scale
        flat_combine = flat_combine * np.nanmedian(flat_peaks)

    # Identify gaps in the image where no pixels contribute:
    Ncomb = len(flats) - mask
    flat_combine[Ncomb == 0] = np.median(flat_combine[Ncomb != 0])
    if len(flats) == 1:
        msg.append("          - Combined %i file" % len(flats))
    else:
        msg.append("          - Combined %i files" % len(flats))

    hdr = pf.getheader(raw_frames[0], 0)
    hdr1 = pf.getheader(raw_frames[0], 1)
    for key in hdr1.keys():
        hdr[key] = hdr1[key]
    hdr['NCOMBINE'] = len(flats)
    hdr.add_comment('Median combined Master Spectral Flat')
    hdr.add_comment('PyNOT version %s' % __version__)

    if output == '':
        if mode == 'spec':
            grism = alfosc.grism_translate[hdr['ALGRNM']]
            output = 'flatcombine_%s.fits' % grism
        else:
            filter = 'white'
            for keyword in ['FAFLTNM', 'FBFLTNM', 'ALFLTNM']:
                if 'open' in hdr[keyword].lower():
                    pass
                else:
                    filter = hdr[keyword]
            output = 'flatcombine_%s.fits' % filter

    pf.writeto(output, flat_combine, header=hdr, overwrite=overwrite)
    msg.append(" [OUTPUT] - Saving combined Flat Field Image: %s" % output)
    msg.append("")
    output_msg = "\n".join(msg)
    if verbose:
        print(output_msg)

    return output, output_msg


def detect_flat_edges(img, dispaxis=2, savgol_window=21, threshold=10):
    """Use filtered derivative to detect the edges of the frame."""
    # Get median profile along slit:
    med_row = np.nanmedian(img, 2-dispaxis)

    # Calculate the absolute value of the derivative:
    deriv = np.fabs(signal.savgol_filter(med_row, savgol_window, 1, deriv=1))

    # Find peaks of the derivative:
    noise = 1.48*mad(deriv)
    if noise == 0:
        sig_tmp = np.std(deriv)
        noise = np.std(deriv[deriv < 3*sig_tmp])
    edges, props = signal.find_peaks(deriv, height=threshold*noise)

    return edges



def normalize_spectral_flat(fname, output='', fig_dir='', dispaxis=2, overscan=50, order=24, savgol_window=51,
                            med_window=5, edge_threshold=10, edge_window=21, plot=True, overwrite=True, **kwargs):
    """
    Normalize spectral flat field for long-slit observations. Parameters are optimized
    for NOT/ALFOSC spectra with horizontal slits, i.e., vertical spectra [axis=2],
    and grism #4.
    In order to keep the edges from diverging greatly, the code uses a relatively
    low polynomial order to fit the edges while using smoothing to recover the central
    part of the spectral shape.
    The two parts are then stiched together to create the final 1D profile.

    Parameters
    ==========

    fname : string
        Input FITS file with raw lamp flat data

    output : string  [default='']
        Filename of normalized flat frame, if not given the output is not saved to file

    fig_dir : string  [default='']
        Directory where the diagnostic plot is saved (if `plot=True`)

    dispaxis : integer  [default=2]
        Dispersion axis, 1: horizontal spectra, 2: vertical spectra

    overscan : integer  [default=50]
        Overscan region, default for ALFOSC is 50 pixels on either side and on top

    order : integer  [default=24]
        Order for Chebyshev polynomial to fit to the spatial profile (per row/col)

    savgol_window : integer  [default=51]
        Window width in pixels for Savitzky--Golay filter of spatial profile

    med_window : integer  [default=5]
        Window width of median filter along spectral axis before fitting the spatial profile

    edge_threshold : float  [default=10.]
        The detection threshold for automatic edge detection

    edge_window : integer  [default=21]
        The Savitzky--Golay window used for automatic edge detection

    plot : boolean [default=True]
        Plot the 1d and 2d data for inspection?

    overwrite : boolean [default=False]
        Overwrite existing output file if True.

    Returns
    =======
    output : string
        Filename of normalized flat field image

    output_msg: string
        The log of the function steps and errors

    """
    msg = list()
    flat = pf.getdata(fname)
    hdr = alfosc.get_alfosc_header(fname)

    msg.append("          - Input file: %s" % fname)

    flat, hdr = trim_overscan(flat, hdr, overscan)
    msg.append("          - Trimmed overscan: %i" % overscan)
    # Get raw pixel array of spatial axis
    x = np.arange(flat.shape[dispaxis-1])

    # Detect edges of flat:
    try:
        edges = detect_flat_edges(flat, dispaxis=dispaxis,
                                  savgol_window=edge_window, threshold=edge_threshold)
        msg.append("          - Automatic edge detection found %i edges" % len(edges))
        if len(edges) != 2:
            msg.append("[WARNING] - Automatic edge detection failed. Using full frame!")
            edges = [0, len(x)]
    except:
        msg.append("[WARNING] - Automatic edge detection failed. Using full frame!")
        edges = [0, len(x)]

    x1 = edges[0] + 5
    x2 = edges[1] - 5

    model = flat.copy()
    if dispaxis == 2:
        smoothing_kernel = (med_window, 1)
    else:
        smoothing_kernel = (1, med_window)
    smoothed_flat = ndimage.median_filter(flat, smoothing_kernel)
    if dispaxis == 1:
        # Flip image to iterate over rows:
        smoothed_flat = smoothed_flat.T
        model = model.T
    msg.append("          - Median filtering flat frame along spectral axis to reduce noise")
    msg.append("          - Fitting each spatial row/column using Chebyshev polynomials combined with Savitzky--Golay filtering")
    msg.append("          - Polynomial order: %i" % order)
    msg.append("          - Savitzky--Golay filter width: %i" % savgol_window)
    pad = savgol_window // 2
    x_fit = x[x1:x2]
    for num, row in enumerate(smoothed_flat):
        filtered_row = signal.savgol_filter(row[x1:x2], savgol_window, 1)
        # Exclude filter edges, half filter width:
        x_mask = x_fit[pad:-pad]
        row_mask = filtered_row[pad:-pad]
        fit_row = np.polynomial.Chebyshev.fit(x_mask, row_mask, order, domain=[x_fit.min(), x_fit.max()])
        model1d = fit_row(x_fit)
        # Remove edge effects in fitting, the filtered data are more robust:
        # This stiched approach introduces a tiny discontinuity, but usually << 1%, so not important!
        model1d[:pad] = filtered_row[:pad]
        model1d[-pad:] = filtered_row[-pad:]
        # Insert back into model image:
        model[num][x1:x2] = model1d

    if dispaxis == 1:
        # Flip image the model back to original orientation:
        model = model.T

    flat_norm = flat / model
    hdr['DATAMIN'] = np.min(flat_norm)
    hdr['DATAMAX'] = np.max(flat_norm)
    if dispaxis == 1:
        stat_region = flat_norm[x1:x2, :]
    else:
        stat_region = flat_norm[:, x1:x2]
    noise = np.std(stat_region)
    data_range = (np.min(stat_region), np.max(stat_region), np.median(stat_region))
    msg.append("          - Standard deviation of 1D residuals: %.2f ADUs" % noise)
    msg.append("          - Normalized data range: min=%.2e  max=%.2e  median=%.2e" % data_range)

    if plot:
        plt.close('all')
        fig2D = plt.figure()
        fig1D = plt.figure()

        ax1_2d = fig2D.add_subplot(121)
        ax2_2d = fig2D.add_subplot(122)
        ax1_2d.imshow(flat, origin='lower')
        ax1_2d.set_title("Raw Flat")
        v1 = data_range[2] - 3*noise
        v2 = data_range[2] + 3*noise
        ax2_2d.imshow(flat_norm, origin='lower', vmin=v1, vmax=v2)
        ax2_2d.set_title("Normalized Flat")
        ax2_2d.set_yticklabels("")
        if dispaxis == 2:
            ax1_2d.set_xlabel("Spatial Axis [pixels]", fontsize=11)
            ax2_2d.set_xlabel("Spatial Axis [pixels]", fontsize=11)
            ax1_2d.set_ylabel("Dispersion Axis [pixels]", fontsize=11)
        else:
            ax1_2d.set_ylabel("Spatial Axis [pixels]", fontsize=11)
            ax1_2d.set_xlabel("Dispersion Axis [pixels]", fontsize=11)
            ax2_2d.set_xlabel("Dispersion Axis [pixels]", fontsize=11)

        # Plot 1D cross-section along slit:
        ax1_1d = fig1D.add_subplot(211)
        ax2_1d = fig1D.add_subplot(212)

        flat1D = np.nanmedian(flat, 2-dispaxis)
        f1d = signal.savgol_filter(flat1D[x1:x2], savgol_window, 1)
        x_mask = x_fit[pad:-pad]
        row_mask = f1d[pad:-pad]
        fit_row = np.polynomial.Chebyshev.fit(x_mask, row_mask, order, domain=[x_fit.min(), x_fit.max()])
        flat_model = fit_row(x_fit)
        # Remove edge effects in fitting, the filtered data are more robust:
        # This stiched approach introduces a tiny discontinuity, but usually << 1%, so not important!
        flat_model[:pad] = f1d[:pad]
        flat_model[-pad:] = f1d[-pad:]

        residuals = flat1D[x1:x2] - flat_model
        ax1_1d.plot(x, flat1D, 'k-', lw=0.9)
        ax1_1d.plot(x_fit, flat_model, 'crimson', lw=1.5, alpha=0.8)
        ax2_1d.plot(x_fit, residuals/flat_model, 'k', lw=0.5)
        ax2_1d.axhline(0., ls='--', color='0.3', lw=0.5)
        ax1_1d.axvline(x1, color='b', ls=':', alpha=0.8)
        ax1_1d.axvline(x2, color='b', ls=':', alpha=0.8)

        ax2_1d.set_xlabel("Spatial Axis [pixels]", fontsize=11)

        power = np.floor(np.log10(np.max(flat1D))) - 1
        majFormatter = ticker.FuncFormatter(lambda x, p: my_formatter(x, p, power))
        ax1_1d.get_yaxis().set_major_formatter(majFormatter)
        ax1_1d.set_ylabel('Counts  [$10^{{{0:d}}}$ ADU]'.format(int(power)), fontsize=11)

        ax2_1d.set_ylabel('($F_{\\rm 1D} -$ model) / model', fontsize=11)
        plot_noise = np.nanstd(residuals/flat_model)
        ax2_1d.set_ylim(-5*plot_noise, 5*plot_noise)
        ax2_1d.set_xlim(x.min(), x.max())
        ax1_1d.set_xlim(x.min(), x.max())

        ax1_1d.minorticks_on()
        ax2_1d.minorticks_on()

        if not exists(fig_dir) and fig_dir != '':
            os.mkdir(fig_dir)
        file_base = basename(fname)
        fname_root = file_base.strip('.fits')
        fig1d_fname = os.path.join(fig_dir, "specflat_1d_%s.pdf" % fname_root)
        fig2d_fname = os.path.join(fig_dir, "specflat_2d_%s.pdf" % fname_root)
        fig1D.tight_layout()
        fig2D.tight_layout()
        fig1D.savefig(fig1d_fname)
        fig2D.savefig(fig2d_fname)
        msg.append("          - Saved graphic output for 1D model: %s" % fig1d_fname)
        msg.append("          - Saved graphic output for 2D model: %s" % fig2d_fname)
        plt.close('all')

    hdr['ORDER'] = (order, 'Order used for Chebyshev polynomial fit')
    hdr['NORMRMS'] = (noise, 'RMS noise of normalization [ADUs]')
    hdr.add_comment('Normalized Spectral Flat')
    hdr.add_comment('PyNOT version %s' % __version__)
    if output:
        # save the file:
        if output[-5:] == '.fits':
            pass
        else:
            output += '.fits'
    else:
        grism = alfosc.grism_translate[hdr['ALGRNM']]
        slit_name = hdr['ALAPRTNM']
        output = 'NORM_FLAT_%s_%s.fits' % (grism, slit_name)

    pf.writeto(output, flat_norm, header=hdr, overwrite=overwrite)
    msg.append(" [OUTPUT] - Saving normalized MASTER FLAT: %s" % output)
    msg.append("")
    output_msg = "\n".join(msg)

    return output, output_msg
