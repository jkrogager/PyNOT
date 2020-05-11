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
from scipy.signal import find_peaks
from numpy.polynomial import Chebyshev
import os
from os.path import exists
import warnings

from lmfit import Parameters, minimize

import alfosc


def get_FWHM(y, x=None):
    """
    Measure the FWHM of the profile given as `y`.
    If `x` is given, then report the FWHM in terms of data units
    defined by the `x` array. Otherwise, report pixel units.

    Parameters
    ----------
    y : np.ndarray, shape (N)
        Input profile whose FWHM should be determined.

    x : np.ndarray, shape (N)  [default = None]
        Input data units, must be same shape as `y`.

    Returns
    -------
    fwhm : float
        FWHM of `y` in units of pixels.
        If `x` is given, the FWHM is returned in data units
        corresponding to `x`.
    """
    if x is None:
        x = np.arange(len(y))

    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]

    if np.sum(zero_crossings) > 2:
        raise ValueError('Invalid profile! More than 2 crossings detected.')
    elif np.sum(zero_crossings) < 2:
        raise ValueError('Invalid profile! Less than 2 crossings detected.')
    else:
        pass

    halfmax_x = list()
    for i in zero_crossings_i:
        x_i = x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))
        halfmax_x.append(x_i)

    fwhm = halfmax_x[1] - halfmax_x[0]
    return fwhm


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


def NN_gaussian(x, mu, sigma, logamp):
    """ One-dimensional modified non-negative Gaussian profile."""
    amp = 10**logamp
    return amp * np.exp(-0.5*(x-mu)**2/sigma**2)


def trace_model(pars, x, N):
    model = np.zeros_like(x)
    for i in range(N):
        p = [pars['mu_%i' % i],
             pars['sig_%i' % i],
             pars['logamp_%i' % i]]
        model += NN_gaussian(x, *p)
    model += pars['bg']
    return model


def model_residuals(pars, x, y, N):
    return y - trace_model(pars, x, N)


def prep_parameters(peaks, prominence):
    values = zip(peaks, prominence)
    pars = Parameters()
    pars.add('bg', value=0.)
    for i, (x0, amp) in enumerate(values):
        pars.add('mu_%i' % i, value=x0)
        pars.add('sig_%i' % i, value=2.)
        pars.add('logamp_%i' % i, value=np.log10(amp))
    return pars


def median_filter_data(x, kappa=5., window=51):
    med_x = median_filter(x, window)
    MAD = np.median(np.abs(x - med_x))
    mask = np.abs(x - med_x) < kappa*MAD
    return (med_x, mask)


def extract(img2D, mask=None, dx=5, center_order=5, width_order=5, aper_cen=None, aper_width=10):
    """
    Perform automatic localization of the trace if possible, otherwise use fixed
    aperture to extract the 1D spectrum.
    The spectra are assumed to be horizontal. Check orientation before passing img2D!
    """

    if mask is None:
        mask2D = np.zeros_like(img2D)

    # Open PDF file for writing diagnostics:
    if exists("diagnostics") is False:
        os.mkdir("diagnostics")
    pdf_filename = "diagnostics/extract_details.pdf"
    pdf = backend_pdf.PdfPages(pdf_filename)

    x = np.arange(img2D.shape[1], dtype=np.float64)
    y = np.arange(img2D.shape[0], dtype=np.float64)

    plt.close('all')

    spsf = np.median(img2D, axis=1)
    spsf = spsf - np.median(spsf)
    if aper_cen is None:
        aper_cen = len(spsf)//2

    # Detect peaks:
    kappa = 10.
    noise = mad(spsf)*1.48
    peaks, properties = find_peaks(spsf, prominence=kappa*noise)
    prominences = properties['prominences']
    if len(peaks) == 0:
        print("[ERROR] - No object found in slit!")
        print("          Extracting using default aperture:")
        print("          Width of %i pixels centered at pixel %i" % (aper_width, aper_cen))
        return

    N_obj = len(peaks)
    # min_obj_sep = np.min(np.diff(peaks))
    print(" Found %i objects in slit" % N_obj)

    # Fit trace with N objects:
    pars = prep_parameters(peaks, prominences)
    trace_parameters = list()
    for num in range(0, img2D.shape[1], dx):
        col = np.sum(img2D[:, num:num+dx], axis=1)
        popt = minimize(model_residuals, pars, args=(y, col, N_obj))
        trace_parameters.append(popt.params)

    # trace_solutions = list()
    trace_models_2d = list()
    x_binned = np.arange(0., img2D.shape[1], dx, dtype=np.float64)
    for n in range(N_obj):
        # Median filter
        mu = np.array([p['mu_%i' % n] for p in trace_parameters])
        sig = np.array([p['sig_%i' % n] for p in trace_parameters])
        mu_med, mask_mu = median_filter_data(mu)
        sig_med, mask_sig = median_filter_data(sig)

        # Fit polynomium:
        mu_fit = Chebyshev.fit(x_binned[mask_mu], mu[mask_mu], deg=center_order)
        sig_fit = Chebyshev.fit(x_binned[mask_sig], sig[mask_sig], deg=width_order)

        # trace_solutions.append([mu_fit(x), sig_fit(x)])
        trace2D = np.zeros_like(img2D)
        for num, x_i in enumerate(x):
            P_i = NN_gaussian(y, mu_fit(x_i), sig_fit(x_i), 0.)
            P_i = P_i/np.sum(P_i)
            trace2D[:, num] = P_i
        trace_models_2d.append(trace2D)


    return trace_parameters, trace_models_2d
    #
    #
    # # Prepare data arrays:
    # x1, x2 = trimx
    # y1, y2 = trimy
    # P = profile_2d
    # V = err2D**2
    # M = np.ones_like(mask2D)
    # M[mask2D > 0] == 0
    #
    # # --- Extract 1D spectrum
    # spec1D = np.sum(M*P*img2D, 1)/np.sum(M*P**2, 1)
    # err1D = np.sqrt(np.sum(M*P, 1)/np.sum(M*P**2/V, 1))
    # err1D = fix_nans(err1D)
    #
    # pdf.close()
    #
    # return spec1D, err1D
