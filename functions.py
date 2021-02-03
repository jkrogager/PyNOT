import numpy as np
import yaml
import os


def get_version_number():
    code_dir = os.path.dirname(os.path.abspath(__file__))
    v_file = os.path.join(code_dir, 'VERSION')
    with open(v_file) as version_file:
        __version__ = version_file.read().strip()
    return __version__


def get_options(option_fname):
    """Load options from YAML file"""
    with open(option_fname) as opt_file:
        options = yaml.full_load(opt_file)
    return options


def mad(img):
    """Calculate Median Absolute Deviation from the median. This is a robust variance estimator.
    For a Gaussian distribution: sigma ≈ 1.4826 * MAD
    """
    return np.nanmedian(np.abs(img - np.nanmedian(img)))


def NNmoffat(x, mu, alpha, beta, logamp):
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


def NN_mod_gaussian(x, bg, mu, sigma, logamp):
    """ One-dimensional modified non-negative Gaussian profile."""
    amp = 10**logamp
    return bg + amp * np.exp(-0.5*(x-mu)**4/sigma**2)


def tophat(x, low, high):
    """Tophat profile: 1 within [low: high], 0 outside"""
    mask = (x >= low) & (x <= high)
    profile = mask * 1. / np.sum(1.*mask)
    return profile


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


def my_formatter(x, p, scale_pow):
    """Format tick marks to exponential notation"""
    return "%.0f" % (x / (10 ** scale_pow))
