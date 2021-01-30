# -*- coding: UTF-8 -*-
import numpy as np
try:
    import pyfits as pf
except:
    import astropy.io.fits as pf
from matplotlib.backends import backend_pdf
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from scipy.ndimage import gaussian_filter1d, median_filter
from numpy.polynomial import Chebyshev
import os
from os.path import exists
import warnings

import alfosc


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def fix_nans(y):
    """Fix NaN values in arrays by interpolating over them.

    Input
    -----
    y : 1d numpy array

    Returns
    -------
    y_fix : corrected input array

    Example:
        >>> y = np.array([1, 2, 3, Nan, Nan, 6])
        >>> y_fix = fix_nans(y)
        y_fix: array([ 1.,  2.,  3.,  4.,  5.,  6.])
    """
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    return y


def mad(img):
    """Calculate Median Absolute Deviation from the median
    This is a robust variance estimator.
    For a Gaussian distribution:
        sigma ≈ 1.4826 * MAD
    """
    return np.median(np.abs(img - np.median(img)))


def NNmoffat(x, alpha, beta, mu, logamp):
    """
    One-dimensional non-negative Moffat profile.

    See:  https://en.wikipedia.org/wiki/Moffat_distribution
    """
    amp = 10**logamp
    return amp*(1. + ((x-mu)**2/alpha**2))**(-beta)


def gaussian(x, mu, sigma, amp):
    """ One-dimensional Gaussian profile."""
    return amp * np.exp(-0.5*(x-mu)**2/sigma**2)


def modNN_gaussian(x, mu, sigma, logamp):
    """ One-dimensional modified non-negative Gaussian profile."""
    amp = 10**logamp
    return amp * np.exp(-0.5*(x-mu)**4/sigma**2)


def moffat_model(pars, x):
    a, b, alpha, beta, mu, amp = pars
    bg = a*x + b
    spsf = NNmoffat(x, alpha, beta, mu, amp)
    return bg + spsf


def gaussian_model(pars, x):
    bg, mu, sigma, logamp = pars
    spsf = modNN_gaussian(x, mu, sigma, logamp)
    return bg + spsf


def residuals(pars, x, y):
    spsf = moffat_model(pars, x)
    return y - spsf


def residuals_gauss(pars, x, y):
    spsf = gaussian_model(pars, x)
    return y - spsf


def extract_and_calibrate(input_fname, arc_frame, bin_size=30, xmin=100, xmax=-100,
                          do_opt_extract=True, interact=True, background=True,
                          center_order=3, FWHM0=10, trimy=[None, None], trimx=[None, None],
                          wl_order=4, aper_cen=None, show=False, sensitivity=None):
    """Perform automatic localization of the trace if possible, otherwise use fixed
    aperture to extract the 1D spectrum. The code optimizes the background subtraction
    by estimating the background in an adjacent aperture.
    The spectrum is then wavelength calibrated using an associated *arc_frame*.
    Lastly, the 1D and 2D spectra are flux calibrated using a fixed sensitivity
    function.
    """
    hdr = pf.getheader(input_fname)
    img2D = pf.getdata(input_fname)
    try:
        err2D = pf.getdata(input_fname, 1)
    except:
        if hdr['CCDNAME'] == 'CCD14':
            hdr['GAIN'] = 0.16
        g = hdr['GAIN']
        r = hdr['RDNOISE']
        err2D = np.sqrt(img2D*g + r**2)/g
    try:
        mask2D = pf.getdata(input_fname, 2)
    except:
        mask2D = np.zeros_like(img2D)

    grism = alfosc.grism_translate[hdr['ALGRNM']]

    # Open PDF file for writing diagnostics:
    if exists("diagnostics") is False:
        os.mkdir("diagnostics")
    pdf_filename = "diagnostics/" + hdr['OBJECT'] + '_details.pdf'
    pdf = backend_pdf.PdfPages(pdf_filename)

    x = np.arange(img2D.shape[1])
    y = np.arange(img2D.shape[0])

    y_points = list()
    p_points = list()
    fwhm_points = list()
    plt.close('all')

    SPSF0 = np.mean(img2D, 0)
    x0 = len(SPSF0)/2                 # Peak location
    peak_height = np.max(SPSF0)       # Peak height
    f0 = np.median(SPSF0)             # Background level
    fit_result = leastsq(residuals, [0., f0, 2., 5., x0, np.log10(peak_height)],
                         args=(x[xmin:xmax], SPSF0[xmin:xmax]), full_output=True)
    popt, pcov, info, _, ier = fit_result

    if ier <= 4 and do_opt_extract is True and pcov is not None:
        # solution was found:
        V = np.var(SPSF0[xmin:xmax] - info['fvec'])
        perr = np.sqrt(pcov.diagonal()*V)
        significance = popt/perr
        if significance[-2] > 10. and significance[-1] > 10.:
            # print " Trace detected!"
            alpha = popt[2]
            beta = popt[3]
            FWHM0 = alpha*2*np.sqrt(2**(1./beta)-1.)
            do_opt_extract = True
        else:
            print("\n [WARNING] - No trace detected!")
            do_opt_extract = False

    else:
        print("\n [WARNING] - No trace detected!")
        do_opt_extract = False

    if do_opt_extract is False:
        plt.close('all')
        plt.plot(SPSF0)
        if interact is True:
            central_marking = plt.ginput(1, -1)
            aper_cen, y0 = central_marking[0]
        else:
            if aper_cen is None:
                aper_cen = len(x)/2
        plt.axvline(aper_cen, color='r')
        plt.axvline(aper_cen + 1.5*FWHM0, color='r', ls=':')
        plt.axvline(aper_cen - 1.5*FWHM0, color='r', ls=':')

    for ymin in np.arange(0, img2D.shape[0], bin_size):
        SPSF = np.median(img2D[ymin:ymin+bin_size, :], 0)
        SPSF = gaussian_filter1d(SPSF, FWHM0/2.35)
        x0 = len(SPSF)/2                     # Peak location
        y0 = np.log10(np.nanmax(SPSF))       # Peak height
        f0 = np.median(SPSF)                 # Background level
        fit_result = leastsq(residuals, [0., f0, 1., 5., x0, y0],
                             args=(x[xmin:xmax], SPSF[xmin:xmax]), full_output=True)
        popt, pcov, info, _, ier = fit_result

        if ier <= 4 and pcov is not None:
            # solution was found:
            V = np.var(SPSF[xmin:xmax] - info['fvec'])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                perr = np.sqrt(pcov.diagonal()*V)
            significance = popt/perr
            # if significance[-2] > 3. and significance[-1] > 4:
            if np.mean(significance) > 3.:
                y_points.append(np.mean(y[ymin:ymin+bin_size]))
                p_points.append(popt)
                alpha = popt[2]
                beta = popt[3]
                FWHM = alpha*2*np.sqrt(2**(1./beta)-1.)
                fwhm_points.append(FWHM)

            else:
                pass
        else:
            pass

    fig_trace = plt.figure()
    ax1 = fig_trace.add_subplot(4, 1, 1)
    ax2 = fig_trace.add_subplot(4, 1, 2)
    ax3 = fig_trace.add_subplot(4, 1, 3)
    ax4 = fig_trace.add_subplot(4, 1, 4)
    for i, p in enumerate(p_points):
        ax1.plot(y_points[i], p[2], 'k.')
        ax2.plot(y_points[i], p[3], 'k.')
        ax3.plot(y_points[i], p[4], 'k.')
        ax4.plot(y_points[i], p[5], 'k.')

    ax1.set_ylabel("Moffat $\\alpha$")
    ax2.set_ylabel("Moffat $\\beta$")
    ax3.set_ylabel("Trace center")
    ax4.set_ylabel("Trace amplitude")

    ax1.set_title("Spectral Trace Localization")
    ax4.set_xlabel("Dispersion Axis  (pixels)")

    y_points = np.array(y_points)

    # --- Sigma-clip *alpha*:
    alphas = np.array([p[2] for p in p_points])
    filter_size = bin_size - 1 + bin_size % 2
    a0 = median_filter(alphas, filter_size)
    sig0 = 1.4826*mad(alphas - a0)
    outliers = (np.abs(alphas - a0) > 4*sig0)
    # Fit the good values with polynomium:
    # fit_alpha = Chebyshev.fit(y_points[~outliers], alphas[~outliers], 0)

    # Use median of the good values:
    def fit_alpha(y):
        return 0.*y + np.median(alphas[~outliers])
    ax1.plot(y, fit_alpha(y), alpha=0.8)
    ax1.plot(y_points[outliers], alphas[outliers], color='crimson', ls='', marker='.')
    ax1.set_ylim(np.median(alphas)-5*sig0, np.median(alphas)+5*sig0)

    # --- Sigma-clip *beta*:
    betas = np.array([p[3] for p in p_points])
    # filter_size = bin_size - 1 + bin_size % 2
    filter_size = 9
    b0 = median_filter(betas, filter_size)
    sig0 = 1.4826*mad(betas - b0)
    outliers = np.abs(betas - b0) > 5*sig0
    # Fit the good values with polynomium:
    # fit_beta = Chebyshev.fit(y_points[~outliers], betas[~outliers], 0)

    # Use median of the good values:
    def fit_beta(y):
        return 0.*y + np.median(betas[~outliers])
    ax2.plot(y, fit_beta(y), alpha=0.8)
    ax2.plot(y_points[outliers], betas[outliers], color='crimson', ls='', marker='.')
    ax2.set_ylim(np.median(betas)-5*sig0, np.median(betas)+5*sig0)

    # --- Sigma-clip the trace centers:  [mu]
    centers = np.array([p[4] for p in p_points])
    # filter_size = bin_size - 1 + bin_size % 2
    filter_size = 9
    cen0 = median_filter(centers, filter_size)
    sig0 = 1.4826*mad(centers - cen0)
    outliers = np.abs(centers - cen0) > 10*sig0
    fit_cen = Chebyshev.fit(y_points[~outliers], centers[~outliers], center_order)
    ax3.plot(y, fit_cen(y), alpha=0.8, label="Chebyshev $\\mathcal{O}=%i$" % center_order)
    ax3.plot(y_points[outliers], centers[outliers], color='crimson', ls='', marker='.')

    # --- Sigma-clip *amplitude*:
    amps = np.array([p[5] for p in p_points])
    # filter_size = bin_size - 1 + bin_size % 2
    filter_size = 9
    amp0 = median_filter(amps, filter_size)
    sig0 = 1.4826*mad(amps - amp0)
    outliers = np.abs(amps - amp0) > 5*sig0
    fit_amp = Chebyshev.fit(y_points[~outliers], amps[~outliers], 3)
    ax4.plot(y, fit_amp(y), alpha=0.8)
    ax4.plot(y_points[outliers], amps[outliers], color='crimson', ls='', marker='.')

    pdf.savefig(fig_trace)

    #
    # --- Make 2D extraction profile and background apertures
    profile_2d = np.zeros_like(img2D)
    bg1 = np.zeros_like(img2D)
    bg2 = np.zeros_like(img2D)
    if do_opt_extract:
        pars_array = zip(fit_alpha(y), fit_beta(y), fit_cen(y), fit_amp(y))
        for i, pars in enumerate(pars_array):
            xlow = pars[2] - 2*FWHM0
            xhigh = pars[2] + 2*FWHM0
            aperture = (x >= xlow) * (x <= xhigh)
            profile_1d = NNmoffat(x, *pars) * aperture
            profile_2d[i] = profile_1d/profile_1d.sum()
            bg1_1d = 1. * ((x >= xlow-FWHM0) * (x <= xlow))
            bg2_1d = 1. * ((x >= xhigh) * (x <= xhigh+FWHM0))
            bg1[i] = bg1_1d/bg1_1d.sum()
            bg2[i] = bg2_1d/bg2_1d.sum()
        # Save trace parameters:
        trace_file_dat = open(hdr['OBJECT']+'_trace.dat', 'w')
        trace_file_dat.write("#pixel  alpha  beta  center  amplitude\n")
        trace_file_fit = open(hdr['OBJECT']+'_trace.fit', 'w')
        trace_file_fit.write("#pixel  alpha  beta  center  amplitude\n")
        trace_data = np.column_stack([y_points, alphas, betas, centers, amps])
        trace_fit = np.column_stack([y, fit_alpha(y), fit_beta(y), fit_cen(y), fit_amp(y)])
        np.savetxt(trace_file_dat, trace_data, fmt="%5i  %.2f  %.2f  %.2f  %.2f")
        np.savetxt(trace_file_fit, trace_fit, fmt="%5i  %.2f  %.2f  %.2f  %.2f")
        trace_file_dat.close()
        trace_file_fit.close()
    else:
        for i in range(len(y)):
            xlow = aper_cen - 1.5*FWHM0
            xhigh = aper_cen + 1.5*FWHM0
            profile_1d = 1. * ((x >= xlow) * (x <= xhigh))
            profile_2d[i] = profile_1d/profile_1d.sum()
            bg1_1d = 1. * ((x >= xlow-FWHM0) * (x <= xlow))
            bg2_1d = 1. * ((x >= xhigh) * (x <= xhigh+FWHM0))
            bg1[i] = bg1_1d/bg1_1d.sum()
            bg2[i] = bg2_1d/bg2_1d.sum()

    #
    # --- Show aperture on top of 2D spectrum
    fig2D = plt.figure()
    ax1_2d = fig2D.add_subplot(1, 2, 1)
    ax1_2d.set_title("2D spectrum")
    img_noise = mad(img2D)
    ax1_2d.imshow(img2D, vmin=-6*img_noise, vmax=6*img_noise, origin='lower')
    if do_opt_extract:
        plt.plot(centers[~outliers], y_points[~outliers],
                 marker='o', alpha=0.5, color='RoyalBlue')
        plt.plot(centers[outliers], y_points[outliers],
                 marker='o', ls='', mec='k', color='none', alpha=0.5)
        plt.plot(fit_cen(y)-2.*FWHM0, y, 'k-', alpha=0.5, lw=1.0)
        plt.plot(fit_cen(y)+2.*FWHM0, y, 'k-', alpha=0.5, lw=1.0)
    else:
        plt.plot(centers, y_points, marker='o', ls='', mec='k', color='none', alpha=0.5)
        lower_aper = y*0 + aper_cen-1.5*FWHM0
        upper_aper = y*0 + aper_cen+1.5*FWHM0
        plt.plot(lower_aper, y, 'k-', alpha=0.5, lw=1.0)
        plt.plot(upper_aper, y, 'k-', alpha=0.5, lw=1.0)

    ax2_2d = fig2D.add_subplot(1, 2, 2)
    ax2_2d.set_title("Extraction Profile")
    ax2_2d.imshow(profile_2d, origin='lower')

    ax2_2d.set_xlabel("Spatial Direction")
    ax1_2d.set_xlabel("Spatial Direction")
    ax1_2d.set_ylabel("Spectral Direction")

    pdf.savefig(fig2D)

    # --- Extract 1D spectrum
    x1, x2 = trimx
    y1, y2 = trimy
    P = profile_2d
    V = err2D**2
    M = np.ones_like(mask2D)
    M[mask2D > 0] == 0
    if do_opt_extract:
        spec1D = np.sum(M*P*img2D/V, 1)/np.sum(M*P**2/V, 1)
        err1D = np.sqrt(np.sum(M*P, 1)/np.sum(M*P**2/V, 1))
    else:
        spec1D = np.sum(M*P*img2D/V, 1)/np.sum(M*P**2/V, 1)
        err1D = np.sqrt(np.sum(M*P, 1)/np.sum(M*P**2/V, 1))

    err1D = fix_nans(err1D)

    # # Extract 1D sky spectrum around the aperture:
    # bg1_1d = np.sum(M*bg1*img2D, 1)/np.sum(M*bg1**2, 1)
    # bg2_1d = np.sum(M*bg2*img2D, 1)/np.sum(M*bg2**2, 1)
    # bg_1d = 0.5*(bg1_1d + bg2_1d)
    #
    # # And refine the background subtraction:
    # if background:
    #     spec1D = spec1D - bg_1d

    #
    # --- Arc1D extraction:
    hdr_arc = pf.getheader(arc_frame)

    # - Check that the science- and arc frames were observed with the same grating:
    error_msg = 'Grisms for arc-frame and science frame do not not match!'
    assert hdr['ALGRNM'] == hdr_arc['ALGRNM'], AssertionError(error_msg)

    arc2D = pf.getdata(arc_frame)
    arc2D = arc2D[y1:y2, x1:x2]
    if do_opt_extract:
        arc1D = np.sum(P*arc2D, 1)/np.sum(P**2, 1)
    else:
        arc1D = np.sum(P*arc2D, 1)

    fig_arc = plt.figure()
    ax_arc = fig_arc.add_subplot(111)
    ax_arc.set_title("1D arc spectrum, filename: %s" % arc_frame)
    ax_arc.plot(y, arc1D, lw=1.)
    ax_arc.set_xlabel("Pixels")
    ax_arc.set_ylabel("Counts")
    ax_arc.set_ylim(ymax=1.15*np.max(arc1D))

    dy = 10
    pix_table = np.loadtxt(alfosc.path + "/calib/%s_pixeltable.dat" % grism)
    ybin = hdr['DETYBIN']
    xbin = hdr['DETXBIN']
    pix_table[:, 0] /= ybin
    wl_vac = list()
    pixels = list()
    pixels_err = list()

    # - Fit Wavelength Solution:
    if y1 is None:
        y1 = 0

    for pix, l_vac in pix_table:
        pix = pix - y1
        ylow = int(pix - dy)
        yhigh = int(pix + dy)
        f0 = np.min(arc1D[ylow:yhigh])
        fmax = np.max(arc1D[ylow:yhigh])
        peak_height = np.log10(fmax - f0)
        mu_init = ylow + np.argmax(arc1D[ylow:yhigh])
        fit_result = leastsq(residuals_gauss, [f0, mu_init, 2., peak_height],
                             args=(y[ylow:yhigh], arc1D[ylow:yhigh]),
                             full_output=True)
        popt, pcov, info, _, ier = fit_result
        if ier <= 4 and pcov is not None:
            pixels.append(popt[1])
            wl_vac.append(l_vac)
            fit_var = np.var(arc1D[ylow:yhigh] - info['fvec'])
            perr = np.sqrt(fit_var * pcov.diagonal())
            pixels_err.append(perr[1])
            ax_arc.axvline(popt[1], 0.9, 0.97, color='r', lw=1.5)
            ax_arc.plot(y[ylow:yhigh], gaussian_model(popt, y[ylow:yhigh]),
                        color='crimson', lw='0.5')

    wl_vac = np.array(wl_vac)
    pixels = np.array(pixels)
    pixel_to_wl = Chebyshev.fit(pixels, wl_vac, wl_order)
    wl_to_pixel = Chebyshev.fit(wl_vac, pixels, wl_order)

    pdf.savefig(fig_arc)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.errorbar(wl_vac, pixels, pixels_err, color='k', ls='', marker='.')
    ax1.plot(pixel_to_wl(y), y, 'crimson',
             label="Chebyshev polynomial, $\\mathcal{O}=%i$" % wl_order)

    wl_rms = np.std(pixels - wl_to_pixel(wl_vac))
    ax2.errorbar(wl_vac, pixels - wl_to_pixel(wl_vac), pixels_err, color='k', ls='', marker='.')
    ax2.axhline(0., ls='--', color='k', lw=0.5)

    ax1.set_xlim(pixel_to_wl(y).min(), pixel_to_wl(y).max())
    ax2.set_xlim(pixel_to_wl(y).min(), pixel_to_wl(y).max())

    ax1.set_title("Wavelength Solution:  RMS = %.2f pixels" % wl_rms)
    ax1.set_ylabel(u"Pixel")
    ax2.set_ylabel(u"Residual")
    ax2.set_xlabel(u"Vacuum Wavelength (Å)")
    ax1.legend()

    pdf.savefig(fig)

    #
    # --- Linearize wavelength solution:
    wl0 = pixel_to_wl(y)
    wl = np.linspace(wl0.min(), wl0.max(), len(y))
    dl = np.diff(wl)[0]
    hdr1d = hdr.copy()
    hdr['CD1_1'] = dl
    hdr['CDELT1'] = dl
    hdr['CRVAL1'] = wl.min()
    hdr1d['CD1_1'] = dl
    hdr1d['CDELT1'] = dl
    hdr1d['CRVAL1'] = wl.min()
    if np.diff(wl0)[0] < 0:
        spec1D = np.interp(wl, wl0[::-1], spec1D[::-1])
        err1D = np.interp(wl, wl0[::-1], err1D[::-1])
    else:
        spec1D = np.interp(wl, wl0, spec1D)
        err1D = np.interp(wl, wl0, err1D)

    #
    # --- Flux calibrate:
    # Load Extinction Table:
    wl_ext, A0 = np.loadtxt(alfosc.path + '/calib/lapalma.ext', unpack=True)
    ext = np.interp(wl, wl_ext, A0)

    # Load Sensitivity Function:
    if sensitivity:
        sens_file = sensitivity
        S = pf.getdata(sens_file)
        S_hdr = pf.getheader(sens_file)
        wl_S = S_hdr['CD1_1']*np.arange(len(S)) + S_hdr['CRVAL1']
        sens_int = np.interp(wl, wl_S, S)
        # - Check that the science frame and sensitivity were observed with the same grating:
        error_msg = 'Grisms for science frame and sensitivity function do not not match!'
        assert hdr['ALGRNM'] == S_hdr['ALGRNM'], AssertionError(error_msg)
    else:
        sens_file = alfosc.path + '/calib/%s_sens.fits' % grism
        S = pf.getdata(sens_file)
        S_hdr = pf.getheader(sens_file)
        wl_S = S_hdr['CD1_1']*np.arange(len(S)) + S_hdr['CRVAL1']
        sens_int = np.interp(wl, wl_S, S)

    airm = hdr['AIRMASS']
    t = hdr['EXPTIME']
    ext_correction = 10**(0.4*airm * ext)

    flux_calibration = ext_correction / 10**(0.4*sens_int)
    flux1D = spec1D / t / dl * flux_calibration
    err1D = err1D / t / dl * flux_calibration

    flux_calib2D = np.resize(flux_calibration, img2D.T.shape)
    flux_calib2D = flux_calib2D.T
    flux2D = img2D[::-1] / t / dl * flux_calib2D
    err2D = err2D[::-1] / t / dl * flux_calib2D

    fig1D = plt.figure()
    ax1D = fig1D.add_subplot(111)
    ax1D.set_title("Final extracted 1D spectrum")
    ax1D.plot(wl, flux1D, lw=1.)
    ax1D.plot(wl, err1D, lw=1.)
    ax1D.set_xlabel(u"Wavelength (Å)")
    ax1D.set_ylabel(r"Flux  (${\rm erg\ cm^{-2}\ s^{-1}\ \AA^{-1}}$)")
    if np.nansum((flux1D/err1D) > 5.) > 0:
        flux_max = np.nanmax(flux1D[(flux1D/err1D) > 5.])
    else:
        flux_max = 10.*np.nanmedian(err1D)
    ax1D.set_ylim(-np.nanmedian(err1D), 1.5*flux_max)
    ax1D.set_xlim(wl.min(), wl.max())
    ax1D.axhline(0., ls=':', color='k', lw=1.0)
    hdr1d['BUNIT'] = 'erg/cm2/s/A'
    hdr1d['CUNIT1'] = 'Angstrom'
    hdr1d['EXTNAME'] = 'FLUX'

    hdr['EXTNAME'] = 'FLUX'
    hdr['BUNIT'] = 'erg/cm2/s/A'
    hdr['CUNIT1'] = 'Angstrom'
    hdr['CUNIT2'] = 'Arcsec'
    hdr['CD2_2'] = 0.19 * xbin
    hdr['CDELT2'] = 0.19 * xbin
    hdr['CRVAL2'] = -img2D.shape[1]/2*(0.19*xbin)

    pdf.savefig(fig1D)
    pdf.close()

    # --- Prepare output HDULists:
    ext0_1d = pf.PrimaryHDU(flux1D, header=hdr1d)
    ext1_1d = pf.ImageHDU(err1D, header=hdr1d, name='err')
    HDU1D = pf.HDUList([ext0_1d, ext1_1d])
    output_fname1D = hdr['OBJECT'] + '_final1D.fits'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        HDU1D.writeto(output_fname1D, clobber=True)
    print("\n  Saved calibrated 1D spectrum to file:  %s" % output_fname1D)

    ext0_2d = pf.PrimaryHDU(flux2D.T, header=hdr)
    ext1_2d = pf.ImageHDU(err2D.T, header=hdr, name='err')
    HDU2D = pf.HDUList([ext0_2d, ext1_2d])
    output_fname2D = hdr['OBJECT'] + '_final2D.fits'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        HDU2D.writeto(output_fname2D, clobber=True)
    print("\n  Saved calibrated 2D spectrum to file:  %s" % output_fname2D)
    print("")

    print("  Details from extraction are saved to file:  %s" % pdf_filename)

    if show:
        plt.show()
    else:
        plt.close('all')


######################################################################################
# --- EXTRACT ONLY ---
######################################################################################


def extract(input_fname, arc_frame, output='', bin_size=30, xmin=100, xmax=-100,
            do_opt_extract=True, interact=True, background=True,
            center_order=3, FWHM0=10, trimy=[None, None], trimx=[None, None],
            wl_order=4, aper_cen=None, show=False):
    """Perform automatic localization of the trace if possible, otherwise use fixed
    aperture to extract the 1D spectrum. The code optimizes the background subtraction
    by estimating the background in an adjacent aperture.
    The spectrum is then wavelength calibrated using an associated *arc_frame*.
    """
    hdr = pf.getheader(input_fname)
    img2D = pf.getdata(input_fname)
    try:
        err2D = pf.getdata(input_fname, 1)
    except:
        if hdr['CCDNAME'] == 'CCD14':
            hdr['GAIN'] = 0.16
        g = hdr['GAIN']
        r = hdr['RDNOISE']
        err2D = np.sqrt(img2D*g + r**2)/g
        err_NaN = np.isnan(err2D)
        err2D[err_NaN] = r/g
    try:
        mask2D = pf.getdata(input_fname, 2)
    except:
        mask2D = np.zeros_like(img2D)

    grism = alfosc.grism_translate[hdr['ALGRNM']]

    # Open PDF file for writing diagnostics:
    if exists("diagnostics") is False:
        os.mkdir("diagnostics")
    pdf_filename = "diagnostics/" + hdr['OBJECT'] + '_details.pdf'
    pdf = backend_pdf.PdfPages(pdf_filename)

    x = np.arange(img2D.shape[1])
    y = np.arange(img2D.shape[0])

    y_points = list()
    p_points = list()
    fwhm_points = list()
    plt.close('all')

    SPSF0 = np.median(img2D, 0)
    x0 = len(SPSF0)/2                 # Peak location
    peak_height = np.max(SPSF0)       # Peak height
    f0 = np.median(SPSF0)             # Background level
    fit_result = leastsq(residuals, [0., f0, 2., 5., x0, np.log10(peak_height)],
                         args=(x[xmin:xmax], SPSF0[xmin:xmax]), full_output=True)
    popt, pcov, info, _, ier = fit_result

    if ier <= 4 and pcov is not None:
        # solution was found:
        V = np.var(SPSF0[xmin:xmax] - info['fvec'])
        perr = np.sqrt(pcov.diagonal()*V)
        significance = popt/perr
        if significance[-2] > 10. and significance[-1] > 10.:
            # print " Trace detected!"
            alpha = popt[2]
            beta = popt[3]
            if do_opt_extract:
                FWHM0 = alpha*2*np.sqrt(2**(1./beta)-1.)
        else:
            print("\n [WARNING] - No trace detected!")
            do_opt_extract = False

    else:
        print("\n [WARNING] - No trace detected!")
        do_opt_extract = False

    if do_opt_extract is False:
        plt.close('all')
        plt.plot(SPSF0)
        if interact is True:
            central_marking = plt.ginput(1, -1)
            aper_cen, y0 = central_marking[0]
        else:
            if aper_cen is None:
                index_max = np.argmax(SPSF0[20:-20])
                aper_cen = 20 + x[index_max]
        plt.axvline(aper_cen, color='r')
        plt.axvline(aper_cen + 1.5*FWHM0, color='r', ls=':')
        plt.axvline(aper_cen - 1.5*FWHM0, color='r', ls=':')

    for ymin in np.arange(0, img2D.shape[0], bin_size):
        SPSF = np.median(img2D[ymin:ymin+bin_size, :], 0)
        SPSF = gaussian_filter1d(SPSF, FWHM0/2.35)
        x0 = len(SPSF)/2                     # Peak location
        y0 = np.log10(np.nanmax(SPSF))       # Peak height
        f0 = np.median(SPSF)                 # Background level
        fit_result = leastsq(residuals, [0., f0, 1., 5., x0, y0],
                             args=(x[xmin:xmax], SPSF[xmin:xmax]), full_output=True)
        popt, pcov, info, _, ier = fit_result

        if ier <= 4 and pcov is not None:
            # solution was found:
            V = np.var(SPSF[xmin:xmax] - info['fvec'])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                perr = np.sqrt(pcov.diagonal()*V)
            significance = popt/perr
            # if significance[-2] > 3. and significance[-1] > 4:
            if np.mean(significance) > 3.:
                y_points.append(np.mean(y[ymin:ymin+bin_size]))
                p_points.append(popt)
                alpha = popt[2]
                beta = popt[3]
                FWHM = alpha*2*np.sqrt(2**(1./beta)-1.)
                fwhm_points.append(FWHM)

            else:
                pass
        else:
            pass

    fig_trace = plt.figure()
    ax1 = fig_trace.add_subplot(4, 1, 1)
    ax2 = fig_trace.add_subplot(4, 1, 2)
    ax3 = fig_trace.add_subplot(4, 1, 3)
    ax4 = fig_trace.add_subplot(4, 1, 4)
    for i, p in enumerate(p_points):
        ax1.plot(y_points[i], p[2], 'k.')
        ax2.plot(y_points[i], p[3], 'k.')
        ax3.plot(y_points[i], p[4], 'k.')
        ax4.plot(y_points[i], p[5], 'k.')

    ax1.set_ylabel("Moffat $\\alpha$")
    ax2.set_ylabel("Moffat $\\beta$")
    ax3.set_ylabel("Trace center")
    ax4.set_ylabel("Trace amplitude")

    ax1.set_title("Spectral Trace Localization")
    ax4.set_xlabel("Dispersion Axis  (pixels)")

    y_points = np.array(y_points)

    # --- Sigma-clip *alpha*:
    alphas = np.array([p[2] for p in p_points])
    filter_size = bin_size - 1 + bin_size % 2
    a0 = median_filter(alphas, filter_size)
    sig0 = 1.4826*mad(alphas - a0)
    outliers = (np.abs(alphas - a0) > 4*sig0)
    # Fit the good values with polynomium:
    # fit_alpha = Chebyshev.fit(y_points[~outliers], alphas[~outliers], 0)

    # Use median of the good values:
    def fit_alpha(y):
        return 0.*y + np.median(alphas[~outliers])
    ax1.plot(y, fit_alpha(y), alpha=0.8)
    ax1.plot(y_points[outliers], alphas[outliers], color='crimson', ls='', marker='.')
    ax1.set_ylim(np.median(alphas)-5*sig0, np.median(alphas)+5*sig0)

    # --- Sigma-clip *beta*:
    betas = np.array([p[3] for p in p_points])
    # filter_size = bin_size - 1 + bin_size % 2
    filter_size = 9
    b0 = median_filter(betas, filter_size)
    sig0 = 1.4826*mad(betas - b0)
    outliers = np.abs(betas - b0) > 5*sig0
    # Fit the good values with polynomium:
    # fit_beta = Chebyshev.fit(y_points[~outliers], betas[~outliers], 0)

    # Use median of the good values:
    def fit_beta(y):
        return 0.*y + np.median(betas[~outliers])
    ax2.plot(y, fit_beta(y), alpha=0.8)
    ax2.plot(y_points[outliers], betas[outliers], color='crimson', ls='', marker='.')
    ax2.set_ylim(np.median(betas)-5*sig0, np.median(betas)+5*sig0)

    # --- Sigma-clip the trace centers:  [mu]
    centers = np.array([p[4] for p in p_points])
    # filter_size = bin_size - 1 + bin_size % 2
    filter_size = 9
    cen0 = median_filter(centers, filter_size)
    sig0 = 1.4826*mad(centers - cen0)
    outliers = np.abs(centers - cen0) > 10*sig0
    fit_cen = Chebyshev.fit(y_points[~outliers], centers[~outliers], center_order)
    ax3.plot(y, fit_cen(y), alpha=0.8, label="Chebyshev $\\mathcal{O}=%i$" % center_order)
    ax3.plot(y_points[outliers], centers[outliers], color='crimson', ls='', marker='.')

    # --- Sigma-clip *amplitude*:
    amps = np.array([p[5] for p in p_points])
    # filter_size = bin_size - 1 + bin_size % 2
    filter_size = 9
    amp0 = median_filter(amps, filter_size)
    sig0 = 1.4826*mad(amps - amp0)
    outliers = np.abs(amps - amp0) > 5*sig0
    fit_amp = Chebyshev.fit(y_points[~outliers], amps[~outliers], 3)
    ax4.plot(y, fit_amp(y), alpha=0.8)
    ax4.plot(y_points[outliers], amps[outliers], color='crimson', ls='', marker='.')

    pdf.savefig(fig_trace)

    #
    # --- Make 2D extraction profile and background apertures
    profile_2d = np.zeros_like(img2D, dtype=float)
    bg1 = np.zeros_like(img2D, dtype=float)
    bg2 = np.zeros_like(img2D, dtype=float)
    if do_opt_extract:
        pars_array = zip(fit_alpha(y), fit_beta(y), fit_cen(y), fit_amp(y))
        for i, pars in enumerate(pars_array):
            xlow = pars[2] - 2*FWHM0
            xhigh = pars[2] + 2*FWHM0
            aperture = (x >= xlow) * (x <= xhigh)
            profile_1d = NNmoffat(x, *pars) * aperture
            profile_2d[i] = profile_1d/profile_1d.sum()
            bg1_1d = 1. * ((x >= xlow-FWHM0) * (x <= xlow))
            bg2_1d = 1. * ((x >= xhigh) * (x <= xhigh+FWHM0))
            bg1[i] = bg1_1d/bg1_1d.sum()
            bg2[i] = bg2_1d/bg2_1d.sum()
        # Save trace parameters:
        trace_file_dat = open(hdr['OBJECT']+'_trace.dat', 'w')
        trace_file_dat.write("#pixel  alpha  beta  center  amplitude\n")
        trace_file_fit = open(hdr['OBJECT']+'_trace.fit', 'w')
        trace_file_fit.write("#pixel  alpha  beta  center  amplitude\n")
        trace_data = np.column_stack([y_points, alphas, betas, centers, amps])
        trace_fit = np.column_stack([y, fit_alpha(y), fit_beta(y), fit_cen(y), fit_amp(y)])
        np.savetxt(trace_file_dat, trace_data, fmt="%5i  %.2f  %.2f  %.2f  %.2f")
        np.savetxt(trace_file_fit, trace_fit, fmt="%5i  %.2f  %.2f  %.2f  %.2f")
        trace_file_dat.close()
        trace_file_fit.close()
    else:
        for i in range(len(y)):
            # aper_cen = fit_cen(y)[i]
            xlow = aper_cen - 1.5*FWHM0
            xhigh = aper_cen + 1.5*FWHM0
            profile_1d = 1. * ((x >= xlow) * (x <= xhigh))
            profile_2d[i] = profile_1d/profile_1d.sum()
            bg1_1d = 1. * ((x >= xlow-FWHM0) * (x <= xlow))
            bg2_1d = 1. * ((x >= xhigh) * (x <= xhigh+FWHM0))
            bg1[i] = bg1_1d/bg1_1d.sum()
            bg2[i] = bg2_1d/bg2_1d.sum()

    #
    # --- Show aperture on top of 2D spectrum
    fig2D = plt.figure()
    ax1_2d = fig2D.add_subplot(1, 2, 1)
    ax1_2d.set_title("2D spectrum")
    img_noise = mad(img2D)
    v0 = np.median(img2D)
    ax1_2d.imshow(img2D, vmin=v0-6*img_noise, vmax=v0+6*img_noise, origin='lower')
    if do_opt_extract:
        plt.plot(centers[~outliers], y_points[~outliers],
                 marker='o', alpha=0.5, color='RoyalBlue')
        plt.plot(centers[outliers], y_points[outliers],
                 marker='o', ls='', mec='k', color='none', alpha=0.5)
        plt.plot(fit_cen(y)-2.*FWHM0, y, 'k-', alpha=0.5, lw=1.0)
        plt.plot(fit_cen(y)+2.*FWHM0, y, 'k-', alpha=0.5, lw=1.0)
    else:
        plt.plot(centers, y_points, marker='o', ls='', mec='k', color='none', alpha=0.5)
        lower_aper = y*0 + aper_cen-1.5*FWHM0
        upper_aper = y*0 + aper_cen+1.5*FWHM0
        plt.plot(lower_aper, y, 'k-', alpha=0.5, lw=1.0)
        plt.plot(upper_aper, y, 'k-', alpha=0.5, lw=1.0)

    ax2_2d = fig2D.add_subplot(1, 2, 2)
    ax2_2d.set_title("Extraction Profile")
    ax2_2d.imshow(profile_2d, origin='lower')

    ax2_2d.set_xlabel("Spatial Direction")
    ax1_2d.set_xlabel("Spatial Direction")
    ax1_2d.set_ylabel("Spectral Direction")

    pdf.savefig(fig2D)

    # Prepare data arrays:
    x1, x2 = trimx
    y1, y2 = trimy
    P = profile_2d
    V = err2D**2
    M = np.ones_like(mask2D)
    M[mask2D > 0] == 0

    # Make refined sky spectrum around the aperture:
    bg1_1d = np.sum(M*bg1*img2D, 1)
    bg2_1d = np.sum(M*bg2*img2D, 1)
    bg_1d = 0.5*(bg1_1d + bg2_1d)
    B = np.resize(bg_1d, img2D.T.shape)
    B = B.T

    # And refine the background subtraction:
    if background:
        data2D = img2D - B
    else:
        data2D = img2D

    # --- Extract 1D spectrum
    spec1D = np.sum(M*P*data2D, 1)/np.sum(M*P**2, 1)
    err1D = np.sqrt(np.sum(M*P, 1)/np.sum(M*P**2/V, 1))
    err1D = fix_nans(err1D)

    #
    # --- Arc1D extraction:
    hdr_arc = pf.getheader(arc_frame)

    # - Check that the science- and arc frames were observed with the same grating:
    error_msg = 'Grisms for arc-frame and science frame do not not match!'
    assert hdr['ALGRNM'] == hdr_arc['ALGRNM'], AssertionError(error_msg)

    arc2D = pf.getdata(arc_frame)
    arc2D = arc2D[y1:y2, x1:x2]
    # if do_opt_extract:
    arc1D = np.sum(P*arc2D, 1)/np.sum(P**2, 1)

    fig_arc = plt.figure()
    ax_arc = fig_arc.add_subplot(111)
    ax_arc.set_title("1D arc spectrum, filename: %s" % arc_frame)
    ax_arc.plot(y, arc1D, lw=1.)
    ax_arc.set_xlabel("Pixels")
    ax_arc.set_ylabel("Counts")
    ax_arc.set_ylim(ymax=1.15*np.max(arc1D))

    dy = 10
    pix_table = np.loadtxt(alfosc.path + "/calib/%s_pixeltable.dat" % grism)
    ybin = hdr['DETYBIN']
    pix_table[:, 0] /= ybin
    wl_vac = list()
    pixels = list()
    pixels_err = list()

    # - Fit Wavelength Solution:
    if y1 is None:
        y1 = 0

    for pix, l_vac in pix_table:
        pix = pix - y1
        ylow = int(pix - dy)
        yhigh = int(pix + dy)
        f0 = np.min(arc1D[ylow:yhigh])
        fmax = np.max(arc1D[ylow:yhigh])
        peak_height = np.log10(fmax - f0)
        mu_init = ylow + np.argmax(arc1D[ylow:yhigh])
        fit_result = leastsq(residuals_gauss, [f0, mu_init, 2., peak_height],
                             args=(y[ylow:yhigh], arc1D[ylow:yhigh]),
                             full_output=True)
        popt, pcov, info, _, ier = fit_result
        if ier <= 4 and pcov is not None:
            pixels.append(popt[1])
            wl_vac.append(l_vac)
            fit_var = np.var(arc1D[ylow:yhigh] - info['fvec'])
            perr = np.sqrt(fit_var * pcov.diagonal())
            pixels_err.append(perr[1])
            ax_arc.axvline(popt[1], 0.9, 0.97, color='r', lw=1.5)
            ax_arc.plot(y[ylow:yhigh], gaussian_model(popt, y[ylow:yhigh]),
                        color='crimson', lw='0.5')

    wl_vac = np.array(wl_vac)
    pixels = np.array(pixels)
    pixel_to_wl = Chebyshev.fit(pixels, wl_vac, wl_order)
    wl_to_pixel = Chebyshev.fit(wl_vac, pixels, wl_order)

    pdf.savefig(fig_arc)

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.errorbar(wl_vac, pixels, pixels_err, color='k', ls='', marker='.')
    ax1.plot(pixel_to_wl(y), y, 'crimson',
             label="Chebyshev polynomial, $\\mathcal{O}=%i$" % wl_order)

    wl_rms = np.std(pixels - wl_to_pixel(wl_vac))
    ax2.errorbar(wl_vac, pixels - wl_to_pixel(wl_vac), pixels_err, color='k', ls='', marker='.')
    ax2.axhline(0., ls='--', color='k', lw=0.5)

    ax1.set_xlim(pixel_to_wl(y).min(), pixel_to_wl(y).max())
    ax2.set_xlim(pixel_to_wl(y).min(), pixel_to_wl(y).max())

    ax1.set_title("Wavelength Solution:  RMS = %.2f pixels" % wl_rms)
    ax1.set_ylabel(u"Pixel")
    ax2.set_ylabel(u"Residual")
    ax2.set_xlabel(u"Vacuum Wavelength (Å)")
    ax1.legend()

    pdf.savefig(fig)

    #
    # --- Linearize wavelength solution:
    wl0 = pixel_to_wl(y)
    wl = np.linspace(wl0.min(), wl0.max(), len(y))
    dl = np.diff(wl)[0]
    hdr1d = hdr.copy()
    hdr['CD1_1'] = dl
    hdr['CDELT1'] = dl
    hdr['CRVAL1'] = wl.min()
    hdr1d['CD1_1'] = dl
    hdr1d['CDELT1'] = dl
    hdr1d['CRVAL1'] = wl.min()
    if np.diff(wl0)[0] < 0:
        spec1D = np.interp(wl, wl0[::-1], spec1D[::-1])
        err1D = np.interp(wl, wl0[::-1], err1D[::-1])
    else:
        spec1D = np.interp(wl, wl0, spec1D)
        err1D = np.interp(wl, wl0, err1D)

    # --- Prepare output HDULists:
    ext0_1d = pf.PrimaryHDU(spec1D, header=hdr1d)
    ext1_1d = pf.ImageHDU(err1D, header=hdr1d, name='err')
    HDU1D = pf.HDUList([ext0_1d, ext1_1d])
    if output:
        if output[-4:] == '.fits':
            output_fname1D = output
        else:
            output_fname1D = output + '.fits'
    else:
        output_fname1D = hdr['OBJECT'] + '_spec1D.fits'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        HDU1D.writeto(output_fname1D, clobber=True)
    print("\n  Saved 1D extracted spectrum to file:  %s" % output_fname1D)

    print("  Details from extraction are saved to file:  %s" % pdf_filename)

    if show:
        plt.show()
    else:
        plt.close('all')
    pdf.close()

    return wl, spec1D, err1D
