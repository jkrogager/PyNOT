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


def get_indent(x):
    if not x.startswith(' '):
        return 0
    else:
        return get_indent(x[1:]) + 1

def get_option_descr(opt_fname):
    """Get the parameter descriptions from YAML file"""
    with open(opt_fname) as opt_file:
        opt_lines = opt_file.readlines()

    all_comments = {}
    for num, line in enumerate(opt_lines):
        if line[0] == '#':
            continue

        try:
            key, val = line.split(':')[:2]
        except:
            continue

        base_indent = get_indent(key)
        if base_indent == 0:
            # Section header:
            section = {}
            indent = get_indent(opt_lines[num+1])
            if indent == 0:
                continue
            sub_lines = opt_lines[num+1:]
            i = 0
            while get_indent(sub_lines[i]) == indent:
                sub_line = sub_lines[i]
                par, value = sub_line.split(':')[:2]
                par = par.strip()
                if '#' in value:
                    comment = value.split('#')[1]
                    comment = comment.strip()
                else:
                    comment = ''
                section[par] = comment
                i += 1
                if i >= len(sub_lines):
                    break
            all_comments[key.strip()] = section
    return all_comments



def mad(img):
    """Calculate Median Absolute Deviation from the median. This is a robust variance estimator.
    For a Gaussian distribution: sigma ≈ 1.4826 * MAD
    """
    return np.nanmedian(np.abs(img - np.nanmedian(img)))


def NN_moffat(x, mu, alpha, beta, logamp):
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


def string_to_decimal(ra, dec, delimiter=None):
    """
    Function to convert string representation of celestial coordinates
    into decimal degrees.
    E.g., 00:02:34.52 -05:23:24.5  ->  0.6438333, -5.3901389

    Input:
    ======
    'ra' and 'dec' can both be given as a single string or a list of
    three entries containing [deg, arcmin, arcsec].
    """

    assert type(ra) == type(dec), "ra and dec must be same type!"

    delimiter_types = [' ', ':']
    if isinstance(ra, str):
        if delimiter:
            ra = ra.split(delimiter)
            dec = dec.split(delimiter)
        else:
            for this_del in delimiter_types:
                if this_del in ra:
                    delimiter = this_del

            if delimiter:
                ra = ra.split(delimiter)
                dec = dec.split(delimiter)
            else:
                ra = [ra[:2], ra[2:4], ra[4:]]
                dec = [dec[:2], dec[2:4], dec[4:]]

    arc2deg = np.array([1., 60**-1, 60**-2])

    RAdeg = np.sum(np.array([float(r) for r in ra])*15.*arc2deg)
    dec_arcangle = np.array([float(d) for d in dec])
    dec_sign = np.sign(dec_arcangle[0])
    if dec_sign == 0:
        dec_sign = 1
    DECdeg = dec_sign*np.sum(np.abs(dec_arcangle)*arc2deg)

    return (RAdeg, DECdeg)


def decimal_to_string(ra, dec, delimiter=':'):
    # Convert degrees to sexagesimal:
    hour_angle = ra/15.
    hours = np.floor(hour_angle)
    minutes = np.remainder(hour_angle, 1)*60.
    seconds = np.remainder(minutes, 1)*60.
    hms = ["%02.0f" % hours, "%02.0f" % np.floor(minutes), "%05.2f" % seconds]
    ra_str = delimiter.join(hms)

    sign = np.sign(dec)
    degrees = np.abs(dec)
    arcmin = np.remainder(degrees, 1)*60.
    arcsec = np.remainder(arcmin, 1)*60.
    dms = ["%+03.0f" % (sign*np.floor(degrees)), "%02.0f" % np.floor(arcmin), "%05.2f" % arcsec]
    dec_str = delimiter.join(dms)

    return (ra_str, dec_str)


def get_pixtab_parameters(pixtable_fname):
    """Find the polynomial degree, ref_type, loc used when creating the pixel table"""
    found_order = False
    order_wl = 4
    found_ref = False
    ref_type = 'vacuum'
    found_loc = False
    loc = -1
    with open(pixtable_fname) as tab_file:
        all_lines = tab_file.readlines()

    for line in all_lines:
        if line[0] != '#':
            # Reached the end of the header
            break
        elif 'order' in line:
            order_str = line.split('=')[1]
            order_wl = int(order_str.strip())
            found_order = True
        elif 'ref' in line:
            ref_str = line.split('=')[1]
            ref_type = ref_str.strip()
            found_ref = True
        elif 'loc' in line:
            loc_str = line.split('=')[1]
            loc = int(loc_str.strip())
            found_loc = True

    pars = {'order_wl': order_wl,
            'ref_type': ref_type,
            'loc': loc}
    found_all = found_order & found_loc & found_ref

    return pars, found_all


def air2vac(air):
    # From Donald Morton 1991, ApJS 77,119
    if type(air) == float or type(air) == int:
        air = np.array(air)
    air = np.array(air)
    ij = (np.array(air) >= 2000)
    out = np.array(air).copy()
    sigma2 = (1.e4/air)**2
    # fact = 1.0 + 6.4328e-5 + 2.94981e-2/(146.0 - sigma2) + 2.5540e-4/( 41.0 - sigma2)
    fact = 1.0 + 6.4328e-5 + 2.94981e-2/(146.0 - sigma2) + 2.5540e-4/(41.0 - sigma2)
    out[ij] = air[ij]*fact[ij]
    return out


def vac2air(vac):
    # From Donald Morton 1991, ApJS 77,119
    vac = np.array(vac)
    ij = (np.array(vac) >= 2000)
    air = np.array(vac).copy()
    sigma2 = (1.e4/vac)**2
    # fact = 1.0 + 6.4328e-5 + 2.94981e-2/(146.0 - sigma2) + 2.5540e-4/( 41.0 - sigma2)
    fact = 1.0 + 6.4328e-5 + 2.94981e-2/(146.0 - sigma2) + 2.5540e-4/(41.0 - sigma2)
    air[ij] = vac[ij]/fact[ij]
    return air
