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
from scipy.ndimage import gaussian_filter1d
from numpy.polynomial import Chebyshev
import os
from os.path import exists, basename

import alfosc

code_dir = os.path.dirname(os.path.abspath(__file__))
calib_dir = os.path.join(code_dir, '/calib/')
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


def combine_bias_frames(bias_frames, output='', kappa=15, overwrite=True):
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
    output_msg: string
        The log of the function steps and errors
    """
    msg = list()
    msg.append("          - Running task: Bias Combination")

    bias = list()
    for frame in bias_frames:
        msg.append("          - Loaded bias frame: %s" % frame)
        bias.append(pf.getdata(frame))

    mask = np.zeros_like(bias[0], dtype=int)
    median_img0 = np.median(bias, 0)
    sig = mad(median_img0)*1.4826
    median = np.median(median_img0)
    masked_bias = list()
    for img in bias:
        this_mask = np.abs(img - median) > kappa*sig
        masked_bias.append(np.ma.masked_where(this_mask, img))
        mask += 1*this_mask
    msg.append("          - Masking outlying pixels: kappa = %f" % kappa)
    msg.append("          - Total number of masked pixels: %i" % np.sum(mask))

    master_bias = np.median(masked_bias, 0)
    Ncomb = len(bias) - mask

    master_bias[Ncomb == 0] = np.median(master_bias[Ncomb != 0])
    msg.append("          - Combined %i files" % len(bias))

    if output:
        hdr = pf.getheader(bias_frames[0], 0)
        hdr1 = pf.getheader(bias_frames[0], 1)
        for key in hdr1.keys():
            hdr[key] = hdr1[key]
        hdr['NCOMBINE'] = len(bias_frames)
        hdr.add_comment('Median combined Master Bias')
        hdr.add_comment('PyNOT version %s' % __version__)

        if output[-5:] == '.fits':
            pass
        else:
            output += '.fits'
    else:
        output = 'MASTER_BIAS.fits'

    pf.writeto(output, master_bias, header=hdr, overwrite=overwrite)
    msg.append("          - Successfully median combined bias frames: %s" % output)
    output_msg = "\n".join(msg)

    return output_msg


def combine_flat_frames(raw_frames, mbias='', output='', match_slit='',
                        kappa=5, verbose=False, overwrite=True):
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

    match_slit : string [default='Slit_1.0']
        Slit name to match, if a flat frame is not taken with the
        given slit, it is not included in the combination.
        This is following NOT/ALFOSC naming from 'ALAPRTNM' in the header.

    kappa : integer [default=5]
        Number of sigmas above which to reject pixels.

    verbose : boolean [default=False]
        If True, print status messages.

    overwrite : boolean [default=False]
        Overwrite existing output file if True.

    Returns
    =======
    output : string
        Filename of combined flat field image

    output_msg : string
        The log of the function steps and errors
    """
    msg = list()
    msg.append("          - Running task: Spectral Flat Combination")
    if mbias and exists(mbias):
        bias = pf.getdata(mbias)
    else:
        msg.append("[WARNING] - No master bias frame provided!")
        bias = 0.

    flats = list()
    flat_peaks = list()
    for fname in raw_frames:
        hdr = pf.getheader(fname)
        if match_slit != '' and match_slit in alfosc.slits:
            if hdr['ALAPRTNM'] == match_slit:
                flat = pf.getdata(fname)
                flat = flat - bias
                peak_val = np.max(np.mean(flat, 1))
                flats.append(flat/peak_val)
                flat_peaks.append(peak_val)
                msg.append("          - Loaded FLAT file: %s   mode=%.1f" % (fname, peak_val))

        elif match_slit == '':
            flat = pf.getdata(fname)
            flat = flat - bias
            peak_val = np.max(np.mean(flat, 1))
            flats.append(flat/peak_val)
            flat_peaks.append(peak_val)
            msg.append("          - Loaded FLAT file: %s   mode=%.1f" % (fname, peak_val))

        else:
            msg.append(" [ERROR]  - Invalid Slit Name given:  %s" % match_slit)
            raise ValueError("\n".join(msg))

    # Perform robust clipping using median absolute deviation
    # Assuming a Gaussian distribution:
    sigma = mad(flat_peaks)*1.4826
    median = np.median(flat_peaks)
    frames_to_remove = list()
    for i, peak_val in enumerate(flat_peaks):
        if np.abs(peak_val - median) > kappa*sigma:
            frames_to_remove.append(i)

    # Sort the indeces and pop them in reverse order:
    for index in sorted(frames_to_remove)[::-1]:
        flats.pop(index)

    flat_combine = np.median(flats, 0) * median
    msg.append("          - Combined %i files" % len(flats))

    hdr = pf.getheader(raw_frames[0], 0)
    hdr1 = pf.getheader(raw_frames[0], 1)
    for key in hdr1.keys():
        hdr[key] = hdr1[key]
    hdr['NCOMBINE'] = len(flats)
    hdr.add_comment('Median combined Master Spectral Flat')
    hdr.add_comment('PyNOT version %s' % __version__)

    if output:
        if output[-5:] == '.fits':
            pass
        else:
            output += '.fits'
    else:
        grism = alfosc.grism_translate[hdr['ALGRNM']]
        slit_name = hdr['ALAPRTNM']
        output = 'FLAT_COMBINED_%s_%s.fits' % (grism, slit_name)

    pf.writeto(output, flat_combine, header=hdr, overwrite=overwrite)
    msg.append("          - Generating combined MASTER FLAT: %s" % output)
    output_msg = "\n".join(msg)
    if verbose:
        print(output_msg)

    return output, output_msg


def normalize_spectral_flat(fname, output='', fig_dir='', axis=2, lower=0, upper=2050, order=24, sigma=5,
                            plot=True, show=True, ext=1, overwrite=True, verbose=False):
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

    output : string [default='']
        Filename of normalized flat frame, if not given the output is not saved to file

    axis : integer [default=2]
        Dispersion axis, 1: horizontal spectra, 2: vertical spectra

    lower : integer [default=0]
        Mask pixels below this number in the fit to the spectral shape

    upper : integer [default=2050]
        Mask pixels above this number in the fit to the spectral shape

    order : integer [default=24]
        Order for Chebyshev polynomial to fit to the spectral shape.

    plot : boolean [default=True]
        Plot the 1d and 2d data for inspection?

    show : boolean [default=True]
        Show the figures directly or just save to file? If False, the figures will only be saved
        as pdf files.

    ext : integer [default=1]
        File extension to open, default is 1 for ALFOSC which has a Primary extension with no data
        and the Image extension containing the raw data.

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

    with pf.open(fname) as HDU:
        if len(HDU)+1 <= ext:
            flat = HDU[ext].data
        else:
            ext = 0
            flat = HDU[0].data

        if ext > 0 and HDU[0].size == 0:
            # No data in first extension, merge headers:
            hdr = HDU[0].header
            for key in HDU[1].header.keys():
                hdr[key] = HDU[1].header[key]

        else:
            hdr = HDU[ext].header

    msg.append("          - Running task: Spectral Flat Normalization")
    msg.append("          - Input file: %s" % fname)

    flat1D = np.mean(flat, axis-1)
    x = np.arange(len(flat1D))
    lower = int(lower)
    if axis == 2:
        upper = int(upper / hdr['DETYBIN'])
    else:
        upper = int(upper / hdr['DETXBIN'])
    fit = Chebyshev.fit(x[lower:upper], flat1D[lower:upper], order)

    flat_model = gaussian_filter1d(flat1D, sigma)
    msg.append("          - Creating combined spectral model using Gaussian smoothing")
    msg.append("          - and Chebyshev polynomium of degree: %i" % order)

    # substitute the fit in the ends to remove convolution effects:
    dx = len(x)-upper
    ycut = len(x) - 2*dx
    flat_model[:3*sigma] = fit(x[:3*sigma])
    flat_model[ycut:] = fit(x[ycut:])

    # make 2D spectral shape:
    if axis == 2:
        model2D = np.resize(flat_model, flat.T.shape)
        model2D = model2D.T
    else:
        model2D = np.resize(flat_model, flat.shape)

    flat_norm = flat / model2D
    hdr['DATAMIN'] = np.min(flat_norm)
    hdr['DATAMAX'] = np.max(flat_norm)
    noise = np.std(flat1D - flat_model)
    data_range = (np.min(flat_norm), np.max(flat_norm), np.median(flat_norm))
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
        std_norm = np.std(flat_norm[lower:upper, :])
        v1 = np.mean(flat_norm[lower:upper, :]) - 3*std_norm
        v2 = np.mean(flat_norm[lower:upper, :]) + 3*std_norm
        ax2_2d.imshow(flat_norm, origin='lower', vmin=v1, vmax=v2)
        ax2_2d.set_title("Normalized Flat")
        if axis == 2:
            ax1_2d.set_xlabel("Spatial Axis [pixels]")
            ax2_2d.set_xlabel("Spatial Axis [pixels]")
            ax1_2d.set_ylabel("Dispersion Axis [pixels]")
        else:
            ax1_2d.set_ylabel("Spatial Axis [pixels]")
            ax1_2d.set_xlabel("Dispersion Axis [pixels]")
            ax2_2d.set_xlabel("Dispersion Axis [pixels]")

        ax1_1d = fig1D.add_subplot(211)
        ax2_1d = fig1D.add_subplot(212)

        residuals = flat1D - flat_model
        ax1_1d.plot(x, flat1D, 'k.')
        ax1_1d.plot(x, flat_model, 'crimson', lw=2, alpha=0.8)
        ax2_1d.plot(x, residuals, 'crimson', lw=2, alpha=0.8)
        ax2_1d.axhline(0., ls='--', color='k', lw=0.5)

        ax2_1d.set_xlabel("Dispersion Axis [pixels]")

        power = np.floor(np.log10(np.max(flat1D))) - 1
        majFormatter = ticker.FuncFormatter(lambda x, p: my_formatter(x, p, power))
        ax1_1d.get_yaxis().set_major_formatter(majFormatter)
        ax1_1d.set_ylabel('Counts  [$10^{{{0:d}}}$ ADU]'.format(int(power)))

        power2 = np.floor(np.log10(np.max(residuals))) - 1
        majFormatter2 = ticker.FuncFormatter(lambda x, p: my_formatter(x, p, power2))
        ax2_1d.get_yaxis().set_major_formatter(majFormatter2)
        ax2_1d.set_ylabel('Residual  [$10^{{{0:d}}}$ ADU]'.format(int(power2)))
        noise = np.std(residuals[lower:upper])
        ax2_1d.set_ylim(-8*noise, 8*noise)

        ax1_1d.minorticks_on()
        ax2_1d.minorticks_on()

        if not exists(fig_dir) and fig_dir != '':
            os.mkdir(fig_dir)
        file_base = basename(fname)
        fname_root = file_base.strip('.fits')
        fig1d_fname = os.path.join(fig_dir, "specflat_1d_%s.pdf" % fname_root)
        fig2d_fname = os.path.join(fig_dir, "specflat_2d_%s.pdf" % fname_root)
        fig1D.savefig(fig1d_fname)
        fig2D.savefig(fig2d_fname)
        msg.append("          - Saved graphic output for 1D model: %s" % fig1d_fname)
        msg.append("          - Saved graphic output for 2D model: %s" % fig2d_fname)
        if show:
            plt.show(block=True)
        else:
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
    msg.append("          - Generating normalized MASTER FLAT: %s" % output)
    output_msg = "\n".join(msg)
    if verbose:
        print(output_msg)

    return output, output_msg


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--bias", type=str, nargs='+',
                        help="Raw Bias frame(s)")
    parser.add_argument("--bias-kappa", type=int, default=15,
                        help="Threshold for sigma-kappa clipping in BIAS combiniation")
    parser.add_argument("--flat", type=str, nargs='+',
                        help="Raw Spectral flat frame(s)")
    parser.add_argument("--flat-kappa", type=int, default=5,
                        help="Threshold for sigma-kappa clipping in FLAT combiniation")
    parser.add_argument("--flat-lower", type=int, default=0,
                        help="Lower boundary on pixels used for spectral shape fitting")
    parser.add_argument("--flat-upper", type=int, default=2050,
                        help="Upper boundary on pixels used for spectral shape fitting")
    parser.add_argument("--flat-slit", type=str, default='',
                        help="Only combine flats taking with the given slit")
    parser.add_argument("--flat-order", type=int, default=24,
                        help="Polynomial order for fit to spectral shape")
    parser.add_argument("--flat-sigma", type=int, default=5,
                        help="Kernel width for Gaussian smoothing")
    parser.add_argument("--flat-axis", type=int, default=2,
                        help="Dispersion axis, 1: horizontal, 2: vertical")
    parser.add_argument("--plot-dir", type=str, default="",
                        help="Directory to save plots")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="Plot diagnostics for spectral flat fielding?")
    parser.add_argument("-s", "--show", action="store_true",
                        help="Show diagnostics for spectral flat fielding?")
    parser.add_argument("-x", "--ext", type=int, default=1,
                        help="Extension number of input data")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print status updates")
    args = parser.parse_args()

    # Check if bias frames are present:
    # ---------------------------------
    if args.bias is not None:
        if len(args.bias) == 1:
            bias_frames = np.loadtxt(args.bias[0], usecols=(0,), dtype=str)

        elif len(args.bias) > 1:
            bias_frames = args.bias

        else:
            raise ValueError("Invalid input for --bias")

        combine_bias_frames(bias_frames, output='MASTER_BIAS.fits',
                            kappa=args.bias_kappa,
                            verbose=args.verbose)

    # Check if flat frames are present:
    # ---------------------------------
    if args.flat is not None:
        if len(args.flat) == 1:
            flat_frames = np.loadtxt(args.flat[0], usecols=(0,), dtype=str)

        elif len(args.flat) > 1:
            flat_frames = args.flat

        else:
            raise ValueError("Invalid input for --flat")

        mflat_fname = combine_flat_frames(flat_frames, mbias='MASTER_BIAS.fits',
                                          match_slit=args.flat_slit,
                                          kappa=args.flat_kappa, verbose=args.verbose)

        normalize_spectral_flat(mflat_fname, axis=args.flat_axis,
                                x1=args.flat_x1, x2=args.flat_x2,
                                order=args.flat_order, sigma=args.flat_sigma,
                                plot=args.plot, show=args.show, ext=args.ext,
                                overwrite=False, verbose=args.verbose)
