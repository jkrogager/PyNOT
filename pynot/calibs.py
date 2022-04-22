# -*- coding: UTF-8 -*-
"""
Script to combine bias and spectral flat frames for use in final data reduction.
"""
__author__ = 'Jens-Kristian Krogager'
__email__ = "krogager@iap.fr"
__credits__ = ["Jens-Kristian Krogager"]

from copy import copy
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import ndimage, signal
import os
from os.path import exists, basename

from pynot.data import organizer as organizer
from pynot.logging import Report
from pynot import instrument
from pynot.functions import mad, my_formatter, get_version_number
from pynot.scired import trim_overscan, correct_raw_file
from pynot import reports

__version__ = get_version_number()


def combine_bias_frames(bias_frames, output='', kappa=15, method='mean', overwrite=True, mode='spec', report_fname=''):
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

    method : string [default='mean']
        Method used for image combination (median/mean)

    report_fname : string  [default='']
        Filename of pdf diagnostic report

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
        raw_img = fits.getdata(frame)
        bias_hdr = instrument.get_header(frame)
        trim_bias, bias_hdr = trim_overscan(raw_img, bias_hdr)
        msg.append("          - Trimming overscan of bias images")
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
    msg.append("          - Masking outlying pixels: kappa = %.2f" % kappa)
    msg.append("          - Total number of masked pixels: %i" % np.sum(mask > 0))

    if method.lower() == 'median':
        master_bias = np.ma.median(masked_bias, 0).data
        Ncomb = len(bias) - mask
        master_bias[Ncomb == 0] = np.median(master_bias[Ncomb != 0])
    else:
        master_bias = np.ma.mean(masked_bias, 0).data
        Ncomb = len(bias) - mask
        master_bias[Ncomb == 0] = np.mean(master_bias[Ncomb != 0])
    msg.append("          - Combination method: %s" % method)
    msg.append("          - Combined %i files" % len(bias))
    msg.append("          - Image Stats:")
    msg.append("          - Median = %.1f,  Std.Dev = %.1f" % (np.median(master_bias), 1.48*mad(master_bias)))

    hdr = bias_hdr
    hdr['NCOMBINE'] = len(bias_frames)
    if method.lower() == 'median':
        hdr.add_comment('Median combined Master Bias')
    else:
        hdr.add_comment('Mean combined Master Bias')
    hdr.add_comment('PyNOT version %s' % __version__)
    if not output:
        output = 'MASTER_BIAS.fits'

    fits.writeto(output, master_bias, header=hdr, overwrite=overwrite)
    msg.append(" [OUTPUT] - Saving combined Bias Image: %s" % output)

    if report_fname:
        reports.check_bias(bias, bias_frames, output, report_fname=report_fname)
        msg.append(" [OUTPUT] - Saving Bias Report: %s" % report_fname)
    msg.append("")
    output_msg = "\n".join(msg)

    return output, output_msg


def combine_flat_frames(raw_frames, output, mbias='', mode='spec', dispaxis=None,
                        kappa=5, verbose=False, overwrite=True, method='mean', report_fname='',
                        **kwargs):
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

    report_fname : string  [default='']
        Filename of pdf diagnostic report

    Returns
    =======
    output : string
        Filename of combined flat field image

    output_msg : string
        The log of the function steps and errors
    """
    msg = list()
    if mbias and exists(mbias):
        bias = fits.getdata(mbias)
        # bias_hdr = instrument.get_header(mbias)

    else:
        msg.append("[WARNING] - No master bias frame provided!")
        bias = 0.

    if dispaxis is None:
        if mode == 'spec':
            hdr = instrument.get_header(raw_frames[0])
            dispaxis = instrument.get_dispaxis(hdr)
            msg.append("          - Getting dispersion axis from file: {!s}".format(dispaxis))

        if dispaxis is None and mode == 'spec':
            raise ValueError("dispaxis must be either 1 or 2 for `mode=spec`")


    elif dispaxis not in [1, 2]:
        raise ValueError("dispaxis must be either 1 or 2, not: %r" % dispaxis)

    flats = list()
    flat_peaks = list()
    for fname in raw_frames:
        hdr = instrument.get_header(fname)
        flat = fits.getdata(fname)
        flat, hdr = trim_overscan(flat, hdr)
        msg.append("          - Trimming overscan of Flat images")

        if mode == 'spec':
            flat = flat - bias
            peak_val = np.max(np.mean(flat, dispaxis-1))
            flats.append(flat/peak_val)
            flat_peaks.append(peak_val)
            msg.append("          - Loaded Spectral Flat file: %s   mode=%.1f" % (fname, peak_val))

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
    if method.lower() == 'median':
        flat_combine = np.ma.median(masked_flats, 0).data
    else:
        flat_combine = np.ma.mean(masked_flats, 0).data
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

    # hdr = instrument.get_header(raw_frames[0])
    hdr['NCOMBINE'] = len(flats)
    if method.lower() == 'median':
        hdr.add_comment('Median combined Flat')
    else:
        hdr.add_comment('Mean combined Flat')
    hdr.add_comment('PyNOT version %s' % __version__)

    if output == '':
        if mode == 'spec':
            grism = instrument.get_grism(hdr)
            slit_name = instrument.get_slit(hdr)
            output = 'flatcombine_%s_%s.fits' % (grism, slit_name)
        else:
            filter = 'white'
            filter = instrument.get_filter(hdr)
            output = 'flatcombine_%s.fits' % filter

    fits.writeto(output, flat_combine, header=hdr, overwrite=overwrite)
    msg.append(" [OUTPUT] - Saving combined Flat Field Image: %s" % output)
    msg.append("")
    output_msg = "\n".join(msg)
    if verbose:
        print(output_msg)

    return output, output_msg


def detect_flat_edges(img, dispaxis=2, savgol_window=21, threshold=10, width=10):
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
    edges, props = signal.find_peaks(deriv, height=threshold*noise, width=width)

    return edges



def normalize_spectral_flat(fname, output='', fig_dir='', dispaxis=None, order=24, savgol_window=51,
                            med_window=5, edge_threshold=10, edge_window=21, edge_width=10, plot=True, overwrite=True, **kwargs):
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
        Filename of normalized flat frame.
        If not given, the output is constructed from the grism and slit IDs

    fig_dir : string  [default='']
        Directory where the diagnostic plot is saved (if `plot=True`)

    dispaxis : integer  [default=2]
        Dispersion axis, 1: horizontal spectra, 2: vertical spectra

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

    edge_width : integer  [default=10]
        The minimum width of a peak in the 1st derivative of the filtered
        illumination profile along the slit in units of pixels.

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
    flat = fits.getdata(fname)
    hdr = instrument.get_header(fname)
    grism = instrument.get_grism(hdr)
    slit_name = instrument.get_slit(hdr)
    msg.append("          - Input file: %s" % fname)
    msg.append("          - Grism Name: %s" % grism)
    msg.append("          - Slit Name: %s" % slit_name)

    if dispaxis is None:
        dispaxis = instrument.get_dispaxis(hdr)
        msg.append("          - Getting dispersion axis from file: %i" % dispaxis)
    elif dispaxis not in [1, 2]:
        raise ValueError("dispaxis must be either 1 or 2, not: %r" % dispaxis)

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
    pad = savgol_window // 2 + 1
    x_fit = x[x1:x2]
    for num, row in enumerate(smoothed_flat):
        filtered_row = signal.savgol_filter(row[x1:x2], savgol_window, 2)
        sig = 1.5*mad(row[x1:x2] - filtered_row)
        # mask = (row[x1:x2] - filtered_row) > -sig
        mask = np.abs(row[x1:x2] - filtered_row) < 2*sig
        # Exclude filter edges, half filter width:
        x_mask = x_fit[mask][pad:-pad]
        row_mask = filtered_row[mask][pad:-pad]
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
    hdr.remove('DATAMIN', ignore_missing=True)
    hdr.remove('DATAMAX', ignore_missing=True)
    if dispaxis == 1:
        stat_region = flat_norm[x1:x2, :]
    else:
        stat_region = flat_norm[:, x1:x2]
    # noise = np.std(stat_region)
    noise = mad(stat_region)*1.48
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
        ax1_2d.set_title("Combined Flat\n(%s - %s)" % (grism, slit_name))
        v1 = data_range[2] - 3*noise
        v2 = data_range[2] + 3*noise
        ax2_2d.imshow(flat_norm, origin='lower', vmin=v1, vmax=v2)
        ax2_2d.set_title("Normalized Flat\n(%s - %s)" % (grism, slit_name))
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
        f1d = signal.savgol_filter(flat1D[x1:x2], savgol_window, 2)
        sig1d = 1.5*mad(flat1D[x1:x2] - f1d)
        mask = np.abs(flat1D[x1:x2] - f1d) < 2*sig1d
        x_mask = x_fit[mask][pad:-pad]
        row_mask = f1d[mask][pad:-pad]
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
        plot_noise = 1.48*mad(residuals/flat_model)
        ax2_1d.set_ylim(-10*plot_noise, 10*plot_noise)
        ax2_1d.set_xlim(x.min(), x.max())
        ax1_1d.set_xlim(x.min(), x.max())
        ax1_1d.set_ylim(flat_model.min()-10*sig1d, flat_model.max()+10*sig1d)

        ax1_1d.minorticks_on()
        ax2_1d.minorticks_on()

        if not exists(fig_dir) and fig_dir != '':
            os.mkdir(fig_dir)
        file_base = basename(fname)
        fname_root = os.path.splitext(file_base)[0]
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
        output = 'NORM_FLAT_%s_%s.fits' % (grism, slit_name)

    fits.writeto(output, flat_norm, header=hdr, overwrite=overwrite)
    msg.append(" [OUTPUT] - Saving normalized MASTER FLAT: %s" % output)
    msg.append("")
    output_msg = "\n".join(msg)

    return output, output_msg


def task_bias(options, database, log=None, verbose=True, output_dir='', report_dir=reports.report_dir, **kwargs):
    """
    Define the entry point for the task pynot:bias. This will automatcally
    """
    if log is None:
        log = Report(verbose)
    log.add_linebreak()
    log.write("Running task: Bias combination")

    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Sort output by date if the option 'date' is in kwargs:
    date = 'date' in kwargs
    bias_files = organizer.sort_bias(database.get_files('BIAS', **kwargs), date=date)

    tag = 'MBIAS'
    task_output = {tag: []}
    for file_id, input_list in bias_files.items():
        output_id = options.pop('output', '')
        if output_id:
            file_id += '_'+output_id
        output_fname = os.path.join(output_dir, '%s_%s.fits' % (tag, file_id))
        report_fname = os.path.join(report_dir, '%s_%s_report.pdf' % (tag, file_id))
        _, output_msg = combine_bias_frames(input_list, output_fname, report_fname=report_fname,
                                            **options)
        task_output[tag].append(output_fname)
        log.commit(output_msg)
        log.add_linebreak()
    log.add_linebreak()
    return task_output, log


def task_sflat(options, database, log=None, verbose=True, output_dir='', report_dir=reports.report_dir, **kwargs):
    """
    Define the entry point for the main task of sflat, to be called by pynot.main

    args : command line arguments from argparse of pynot.main
    """
    if log is None:
        log = Report(verbose)
    log.add_linebreak()
    log.write("Running task: Spectral flat field combination and normalization")

    # Sort output by date if the option 'date' is in kwargs:
    date = 'date' in kwargs
    flat_files = organizer.sort_spec_flat(database.get_files('SPEC_FLAT', **kwargs), date=date)

    tag = 'NORM_SFLAT'
    task_output = {tag: []}
    for file_id, input_list in flat_files.items():
        output_id = options.pop('output', '')
        if output_id:
            file_id += '_'+output_id
        # Match master bias file:
        raw_img = organizer.RawImage(input_list[0])
        master_bias = organizer.match_single_calib(raw_img, database, 'MBIAS', log, date=False)

        output_fname = os.path.join(output_dir, 'FLAT_COMBINED_%s.fits' % file_id)
        flatcombine, msg = combine_flat_frames(input_list, output=output_fname, mbias=master_bias,
                                               mode='spec', **options)
        log.commit(msg)
        log.add_linebreak()

        output_fname = os.path.join(output_dir, '%s_%s.fits' % (tag, file_id))
        _, flat_msg = normalize_spectral_flat(flatcombine, output=output_fname,
                                              fig_dir=report_dir, **options)
        log.commit(flat_msg)
        log.add_linebreak()
        task_output[tag].append(output_fname)
    return task_output, log


def task_prep_arcs(options, database, log=None, verbose=True, output_dir='', report_dir=reports.report_dir, **kwargs):
    """
    Prepare all arc frames for analysis: apply bias and flat field corrections
    """
    if log is None:
        log = Report(verbose)
    log.add_linebreak()
    log.write("Running task: Arc frame preparation")

    arc_filelist = []
    for tag in database.keys():
        if 'ARC' in tag and 'CORR' not in tag:
            arc_filelist += database.get_files(tag, **kwargs)

    if len(arc_filelist) == 0:
        log.error("No arc line calibration data found in the dataset!")
        log.error("Check the classification table... object type 'ARC_HeNe', 'ARC_ThAr', 'ARC_HeAr' missing")
        raise KeyError('No ARC files found in database!')

    # Sort output by date if the option 'date' is in kwargs:
    date = 'date' in kwargs
    arc_files = organizer.sort_arcs(arc_filelist, date=date)
    tag = 'ARC_CORR'
    task_output = {tag: []}
    for file_id, input_list in arc_files.items():
        # find BIAS and FLAT
        raw_img = organizer.RawImage(input_list[0])
        master_bias = organizer.match_single_calib(raw_img, database, 'MBIAS', log, date=False)
        norm_flat = organizer.match_single_calib(raw_img, database, 'NORM_SFLAT', log, date=False,
                                                 grism=True, slit=True, filter=True)
        current_fnames = list()
        for arc_fname in input_list:
            arc_basename = 'CORR_' + os.path.basename(arc_fname)
            corrected_arc2d_fname = os.path.join(output_dir, arc_basename)
            output_msg = correct_raw_file(arc_fname, bias_fname=master_bias, flat_fname=norm_flat,
                                          output=corrected_arc2d_fname, overwrite=True)
            log.commit(output_msg)
            task_output[tag].append(corrected_arc2d_fname)
            current_fnames.append(corrected_arc2d_fname)
        # Create PDF diagnostic report
        report_fname = os.path.join(report_dir, 'ARCS_%s_report.pdf' % file_id)
        reports.check_arcs(current_fnames, report_fname)
        log.add_linebreak()


    return task_output, log
