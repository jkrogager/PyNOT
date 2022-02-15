"""
Wavelength Calibration and 2D Transformation
"""

__author__ = "Jens-Kristian Krogager"

import os
from astropy.io import fits
import numpy as np
from numpy.polynomial import Chebyshev
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import warnings

import spectres

from pynot import instrument
from pynot.functions import get_version_number, NN_mod_gaussian, get_pixtab_parameters, mad

__version__ = get_version_number()


class WavelengthError(Exception):
    def __init__(self, message):
        self.message = message


def verify_arc_frame(arc_fname, dispaxis=2):
    """
    Return True if the arc lines fill the full detector plane, else return False.
    The full detector plane is considered the region between 10th and 90th percentile
    of the pixels along the slit.

    `dispaxis` = 2 for vertical spectra, 1 for horizontal spectra.
    """
    arc2D = fits.getdata(arc_fname)
    hdr = instrument.get_header(arc_fname)
    if 'DISPAXIS' in hdr:
        dispaxis = hdr['DISPAXIS']
    elif instrument.get_dispaxis(hdr):
        dispaxis = instrument.get_dispaxis(hdr)

    if dispaxis == 2:
        # Reorient image to have dispersion along x-axis:
        arc2D = arc2D.T

    ilow, ihigh = detect_borders(arc2D)
    imax = arc2D.shape[0]
    result = (ilow < 0.1*imax) & (ihigh > 0.9*imax)
    return result


def fit_gaussian_center(x, y):
    bg = np.nanmedian(y)
    logamp = np.log10(np.nanmax(y)-bg)
    sig = 1.5
    max_index = np.argmax(y)
    mu = x[max_index]
    p0 = np.array([bg, mu, sig, logamp])
    try:
        popt, pcov = curve_fit(NN_mod_gaussian, x, y, p0)
        if pcov is None:
            return popt[1], None
        else:
            return popt[1], pcov[1, 1]

    except RuntimeError:
        return np.nan, np.nan


def fit_lines(x, arc1D, ref_table, dx=20):
    pixels = list()
    for pix, l_vac in ref_table:
        xlow = pix - dx
        xhigh = pix + dx
        cutout = (x > xlow) & (x < xhigh)
        try:
            pix_cen, cov = fit_gaussian_center(x[cutout], arc1D[cutout])
        except:
            print("Something went wrong in Line Fitting. Here's the input:")
            print("pix: %r  wl: %r" % (pix, l_vac))
            print("x:", x[cutout])
            print("y:", arc1D[cutout])
            print("lower: %i  upper:%i" % (xlow, xhigh))
            raise

        if cov is not None:
            pixels.append(pix_cen)
        else:
            pixels.append(np.nan)

    pixels = np.array(pixels)
    return pixels


def median_filter_data(x, kappa=5., window=51):
    """
    Calculate rejection mask using median filtering

    x : array, shape N
        Input array to filter.

    kappa : float   [default=5]
        Reject any pixels deviating more than `kappa` times the noise.
        The noise is estimated using median absolute deviation.

    window : int   [default=51]
        Number of pixels used in the median filtering. This value should
        be an odd number and large enough to encompass any artefacts that
        should be removed.

    Return
    ------
    med_x : array, shape N
        Median filtered array of `x`.

    mask : array, shape N  [bool]
        Boolean array of accepted pixels.
        Any reject pixels will have a value of `False`.
    """
    med_x = median_filter(x, window)
    noise = np.nanmedian(np.abs(x - med_x)) * 1.48
    if noise == 0:
        noise = np.nanstd(x - med_x)
    mask = np.abs(x - med_x) < kappa*noise
    return (med_x, mask)


def subtract_arc_background(arc2D, deg=5):
    """Subtract continuum background in arc line frames"""
    if arc2D.dtype not in [np.float64, np.float32]:
        arc2D = arc2D.astype(np.float64)
    pix = np.arange(arc2D.shape[1])
    bg2d = np.zeros_like(arc2D)
    for i, row in enumerate(arc2D):
        med1d, mask1d = median_filter_data(row)
        cheb_fit = Chebyshev.fit(pix[mask1d], row[mask1d], deg=deg)
        bg1d = cheb_fit(pix)
        bg2d[i] = bg1d
    bg_sub2d = arc2D - bg2d
    return bg_sub2d, bg2d


def detect_borders(arc2D, kappa=20.):
    """Detect edge of arc frame"""
    N_lines = np.zeros(arc2D.shape[0])
    for i, row in enumerate(arc2D):
        mad = np.median(np.abs(row-np.median(row)))
        peaks, _ = find_peaks(row - np.median(row), prominence=kappa*mad)
        N_lines[i] = len(peaks)

    mask = (N_lines >= np.median(N_lines)).nonzero()
    row_min = np.min(mask)
    row_max = np.max(mask) + 1
    return (row_min, row_max)


def create_2d_pixtab(arc2D_sub, pix, ref_table, dx=20):
    """Fit reference lines to each row to obtain 2D pixel table"""

    # Image should already be oriented correctly, i.e., dispersion along x-axis
    # and have background subtracted for optimal results

    pixtab2d = list()
    for row in arc2D_sub:
        line_pos = fit_lines(pix, row, ref_table, dx=dx)
        pixtab2d.append(line_pos)

    pixtab2d = np.array(pixtab2d)
    return pixtab2d


def fit_2dwave_solution(pixtab2d, deg=5):
    # Transpose the input table cause it's easier and faster
    # to iterate over rows than columns:
    tab2d = pixtab2d.T
    fit_table2d = np.zeros_like(tab2d)
    col = np.arange(pixtab2d.shape[0])
    for num, points in enumerate(tab2d):
        # Median filter each column
        med_col, mask = median_filter_data(points)

        # Fit Cheb. poly to each filtered column
        try:
            cheb_polyfit = Chebyshev.fit(col[mask], points[mask], deg=deg, domain=(col.min(), col.max()))
        except:
            print("Something went wrong in Chebyshev polynomial fitting. Here's the input:")
            print("x:", col[mask])
            print("y:", points[mask])
            print("degree:", deg)
            print("domain: %r  %r" % (col.min(), col.max()))
            raise
        finally:
            np.savetxt('pixtable_2d_dump.dat', pixtab2d)

        # Insert back into the fit_table
        fit_table2d[num] = cheb_polyfit(col)

    # Transpose back to original orientation of the pixel table
    return fit_table2d.T


def apply_transform(img2D, pix, fit_table2d, ref_table, err2D=None, mask2D=None, header={},
                    order_wl=4, ref_type='vacuum', log=False, N_out=None, interpolate=True):
    """
    Apply 2D wavelength transformation to the input image

    img2D : array, shape(M, N)
        Input 2D spectrum oriented with dispersion along x-axis!

    pix : array, shape (N)
        Input pixel array along dispersion axis.

    fit_table2d : array, shape(M, L)
        Fitted wavelength solutions from corresponding arc-line frame
        for L fitted reference lines.

    ref_table : array, shape(L, 2)
        Reference table for the given setup with two columns:
            pixel  and  wavelength in Å

    err2D : array, shape(M, N)
        Associated error array, must be same shape as `img2D`

    header : FITS Header or dict
        The FITS header corresponding to `img2D`, or a dictionary

    order_wl : int
        Polynomial order used for wavelength fit as a function of input pixel
        A Chebyshev polynomium is used to fit the solution for each row in img2D.

    log : bool  [default=False]
        Use logarithmic binning in wavelength?

    N_out : int
        Number of output pixels along dispersion axis.
        If `None` is given, the default is to use the same number
        of pixels as in the input image.

    interpolate : bool  [default=True]
        Interpolate the image onto new grid or use sub-pixel shifting

    Returns
    -------
    img2D_tr : array, shape(M, N_out)
        Transformed 2D spectrum with wavelength along x-axis.

    wl : array, shape(N_out)
        Coresponding wavelength array along x-axis.

    hdr_tr : FITS Header or dict
        Updated FITS Header or dictionary with wavelength information
    """
    msg = list()
    # Define wavelength grid at midpoint:
    if N_out is None:
        N_out = img2D.shape[1]
    else:
        if not interpolate:
            N_out = img2D.shape[1]
            msg.append("[WARNING] - Interpolation turned off!")
            msg.append("[WARNING] - N_out was given: %i" % N_out)
            msg.append("[WARNING] - Cannot change sampling without interpolating")
            msg.append("[WARNING] - Going back to default: no interpolation and same dimension as input!")

    pix_in = pix
    cen = fit_table2d.shape[0]//2
    ref_wl = ref_table[:, 1]
    central_solution = Chebyshev.fit(fit_table2d[cen], ref_wl, deg=order_wl, domain=[pix_in.min(), pix_in.max()])
    wl_central = central_solution(pix_in)
    if all(np.diff(wl_central) < 0):
        # Wavelengths are decreasing: Flip arrays
        flip_array = True
    elif all(np.diff(wl_central) > 0):
        # Wavelength are increasing: Nothing to do
        flip_array = False
    elif all(np.diff(wl_central) == 0):
        # Wavelengths do not increase: WHAT?!
        msg.append(" [ERROR]  - Wavelength array does not increase! Something went wrong.")
        msg.append("          - Check the parameters `fit_window` and `order_wl`.")
        exit_msg = "\n".join(msg)
        raise WavelengthError(exit_msg)
    else:
        msg.append(" [ERROR]  - Wavelength array is not monotonic.")
        msg.append("          - Check the parameters `fit_window` and `order_wl`.")
        exit_msg = "\n".join(msg)
        raise WavelengthError(exit_msg)

    wl_residuals = np.std(ref_wl - central_solution(fit_table2d[cen]))
    if wl_residuals > 5.:
        msg.append("[WARNING] - Large residuals of wavelength solution: %.2f Å" % wl_residuals)
        msg.append("[WARNING] - Try changing the fitting window of each arc line (par:`fit_window`)")
    else:
        msg.append("          - Residuals of wavelength solution: %.2f Å" % wl_residuals)

    if ref_type == 'air':
        ctype = 'AWAV'
    else:
        ctype = 'WAVE'

    if log:
        wl = np.logspace(np.log10(wl_central.min()), np.log10(wl_central.max()), N_out)
        hdr_tr = header.copy()
        hdr_tr['CRPIX1'] = 1
        hdr_tr['CDELT1'] = np.diff(np.log10(wl))[0]
        hdr_tr['CRVAL1'] = np.log10(wl[0])
        hdr_tr['CTYPE1'] = ctype+'-LOG'
        hdr_tr['CUNIT1'] = 'Angstrom'
        msg.append("          - Creating logarithmically sampled wavelength grid")
        msg.append("          - Sampling: %.3f  (logÅ/pix)" % np.diff(np.log10(wl))[0])
    else:
        wl = np.linspace(wl_central.min(), wl_central.max(), N_out)
        hdr_tr = header.copy()
        hdr_tr['CRPIX1'] = 1
        hdr_tr['CDELT1'] = np.diff(wl)[0]
        hdr_tr['CRVAL1'] = wl[0]
        hdr_tr['CTYPE1'] = ctype
        hdr_tr['CUNIT1'] = 'Angstrom'
        msg.append("          - Creating linearly sampled wavelength grid")
        msg.append("          - Sampling: %.3f  (Å/pix)" % np.diff(wl)[0])

    # Calculate the maximum curvature of the arc lines:
    max_curvature = table2D_max_curvature(fit_table2d)
    msg.append("          - Maximum curvature of arc lines: %.3f pixels" % max_curvature)
    if max_curvature < 0.1:
        interpolate = False
        msg.append("          - Maximum curvature less than 1/10 pixel. No need to interpolate the data")

    if interpolate:
        img2D_tr = np.zeros((img2D.shape[0], N_out))
        err2D_tr = np.zeros((img2D.shape[0], N_out))
        mask2D_tr = np.zeros((img2D.shape[0], N_out))
        if err2D is None:
            msg.append("[WARNING] - Interpolating data without errors!")
        else:
            msg.append("          - Interpolating data with errors")

        for i, row in enumerate(img2D):
            # - fit the chebyshev polynomium
            solution_row = Chebyshev.fit(fit_table2d[i], ref_wl, deg=order_wl, domain=[pix_in.min(), pix_in.max()])
            wl_row = solution_row(pix_in)
            if flip_array:
                # Wavelengths are decreasing: Flip arrays
                row = row[::-1]
                wl_row = wl_row[::-1]

            # -- interpolate the data onto the fixed wavelength grid
            if err2D is not None:
                err_row = err2D[i]
                if flip_array:
                    err_row = err_row[::-1]

                # interp_row, interp_err = spectres.spectres(wl, wl_row, row, spec_errs=err_row, verbose=False, fill=0.)
                interp_row = np.interp(wl, wl_row, row, left=0., right=0.)
                interp_err = np.interp(wl, wl_row, err_row, left=-1, right=-1)
                err2D_tr[i] = interp_err
                img2D_tr[i] = interp_row
            else:
                # interp_row = spectres.spectres(wl, wl_row, row, verbose=False, fill=0.)
                interp_row = np.interp(wl, wl_row, row, left=0., right=0.)
                img2D_tr[i] = interp_row

            mask_row = mask2D[i]
            mask_int = np.interp(wl, wl_row, mask_row)
            mask2D_tr[i] = np.ceil(mask_int).astype(int)

    else:
        msg.append("          - No interpolation used!")
        img2D_tr = img2D
        err2D_tr = err2D
        mask2D_tr = mask2D
    if np.sum(img2D_tr) == 0:
        msg.append(" [ERROR]  - Something went wrong! All fluxes are 0!")
    output_msg = "\n".join(msg)
    return img2D_tr, err2D_tr, mask2D_tr, wl, hdr_tr, output_msg


def get_order_from_file(pixtable_fname):
    """Find the polynomial degree used when creating the pixel table"""
    order_wl = 4
    found = False
    with open(pixtable_fname) as tab_file:
        while True:
            line = tab_file.readline()
            if line[0] != '#':
                # Reached the end of the header
                break
            elif 'order' in line:
                order_str = line.split('=')[1]
                order_wl = int(order_str.strip())
                found = True
                break
    return order_wl, found


def wavecal_1d(input_fname, pixtable_fname, *, output, order_wl=None, log=False, N_out=None, linearize=True):
    """Apply wavelength calibration to 1D spectrum"""
    msg = list()

    pixtable = np.loadtxt(pixtable_fname)
    msg.append("          - Loaded pixel table: %s" % pixtable_fname)
    pixtab_pars, found_all = get_pixtab_parameters(pixtable_fname)
    if order_wl is None:
        order_wl = pixtab_pars['order_wl']
        if found_all:
            msg.append("          - Loaded polynomial order from file: %i" % order_wl)
        else:
            msg.append("          - Using default polynomial order: %i" % order_wl)
    else:
        msg.append("          - Using polynomial order: %i" % order_wl)

    if pixtab_pars['ref_type'] == 'air':
        ctype = 'AWAV'
    else:
        ctype = 'WAVE'

    if log:
        linearize = True
        ctype = ctype + '-LOG'

    # Load input data:
    hdu_list = fits.open(input_fname)
    msg.append("          - Loaded spectrum: %s" % input_fname)
    output_hdu = fits.HDUList()
    for hdu in hdu_list[1:]:
        wl_unit = hdu.columns['WAVE'].unit
        if wl_unit and wl_unit.lower() in ['angstrom', 'nm', 'a', 'aa']:
            msg.append(" [ERROR]  - Spectrum is already wavelength calibrated.")
            msg.append("")
            output_msg = "\n".join(msg)
            return output_msg
        tab = hdu.data
        hdr = hdu.header
        pix = tab['WAVE']
        flux1d = tab['FLUX']
        err1d = tab['ERR']
        if 'OBJ_POS' in hdr:
            aper_cen = hdr['OBJ_POS']
            msg.append("          - Extraction aperture along slit at pixel no. %i" % aper_cen)
            if pixtab_pars['loc'] != -1:
                aperture_offset = np.abs(aper_cen - pixtab_pars['loc'])
                if aperture_offset > 10:
                    msg.append("[WARNING] - The wavelength solution may not be accurate!")
                    msg.append("[WARNING] - The wavelength solution was calculated at position %i along the slit")
                    msg.append("[WARNING] - but the spectrum was extracted at position %i along the slit.")
        else:
            aper_cen = -1
            msg.append("[WARNING] - Extraction aperture not found.")
            msg.append("[WARNING] - Double check if the wavelength solution is for the correct location along the slit!")

        if N_out is None:
            N_out = len(pix)
        else:
            if N_out != len(pix):
                linearize = True

        # Fit wavelength solution
        solution = Chebyshev.fit(pixtable[:, 0], pixtable[:, 1], deg=order_wl, domain=[pix.min(), pix.max()])
        wl = solution(pix)
        res = np.std(pixtable[:, 1] - solution(pixtable[:, 0]))
        msg.append("          - Fitting wavelength solution with polynomium of order: %i" % order_wl)
        msg.append("          - Standard deviation of wavelength residuals: %.3f Å" % res)
        msg.append("          - Setting header keyword: CTYPE1 = %s" % ctype)
        hdr['CUNIT1'] = 'Angstrom'
        hdr['CTYPE1'] = ctype
        hdr['WAVERES'] = (np.round(res, 2), "RMS of wavelength residuals")
        if linearize:
            if log:
                msg.append("          - Interpolating spectrum onto logarithmic grid")
                wl_new = np.logspace(np.log10(wl.min()), np.log10(wl.max()), N_out)
                dv = np.diff(wl_new)[0] / wl_new[0] * 299792.
                dlog = np.diff(np.log10(wl_new))[0]
                msg.append("          - wavelength step: %.3f  [log(Å)]" % dlog)
                msg.append("          - wavelength step: %.1f  [km/s]" % dv)
            else:
                msg.append("          - Interpolating spectrum onto linear grid")
                wl_new = np.linspace(wl.min(), wl.max(), N_out)
                dl = np.diff(wl_new)[0]
                msg.append("          - wavelength step: %.3f  [Å]" % dl)

            if np.diff(wl)[0] < 0:
                # Wavelengths are decreasing: Flip arrays
                flux1d = flux1d[::-1]
                wl = wl[::-1]
                err1d = err1d[::-1]
            interp_flux, interp_err = spectres.spectres(wl_new, wl, flux1d, spec_errs=err1d, verbose=False, fill=0.)
            final_wl = wl_new
        else:
            msg.append("          - Using raw input grid, no interpolation used.")
            msg.append("[WARNING] - Wavelength steps may not be constant!")
            final_wl = wl
            interp_flux = flux1d
            interp_err = err1d

        col_wl = fits.Column(name='WAVE', array=final_wl, format='D', unit=hdr['CUNIT1'])
        col_flux = fits.Column(name='FLUX', array=interp_flux, format='D', unit=hdr['BUNIT'])
        col_err = fits.Column(name='ERR', array=interp_err, format='D', unit=hdr['BUNIT'])
        output_tab = fits.BinTableHDU.from_columns([col_wl, col_flux, col_err], header=hdr)
        output_hdu.append(output_tab)

    output_hdu.writeto(output, overwrite=True)
    msg.append(" [OUTPUT] - Saving wavelength calibrated 1D spectrum: %s" % output)
    msg.append("")
    output_msg = "\n".join(msg)
    return output_msg

# ============== PLOTTING =====================================================

def plot_2d_pixtable(arc2D_sub, pix, pixtab2d, fit_table2d, arc_fname, filename=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mad = np.nanmedian(np.abs(arc2D_sub - np.nanmedian(arc2D_sub)))
    ax.imshow(arc2D_sub, origin='lower', extent=(pix.min(), pix.max(), 1, arc2D_sub.shape[0]),
              vmin=-3*mad, vmax=15*mad, cmap=plt.cm.gray_r, aspect='auto')
    for i in np.arange(0, pixtab2d.shape[0], 10):
        row = pixtab2d[i]
        ax.plot(row, np.ones_like(row)*(i+1), ls='', color='Blue', marker='.', alpha=0.1)
    for col in fit_table2d.T:
        ax.plot(col, np.arange(arc2D_sub.shape[0]), lw=1, color='r', alpha=0.5)
    ax.set_xlim(pix.min(), pix.max())
    ax.set_ylim(1, arc2D_sub.shape[0])
    ax.set_title("Reference arc frame: %s" % arc_fname, fontsize=10)
    if filename:
        fig.savefig(filename)


# FORMAT RESIDUALS:
def table2D_max_curvature(fit_table2d):
    curvatures = list()
    for col in fit_table2d.T:
        delta_col = np.max(col) - np.min(col)
        curvatures.append(delta_col)
    max_curvature = np.max(curvatures)
    return max_curvature


def format_table2D_residuals(pixtab2d, fit_table2d, ref_table):
    """Calculate the residuals of each arc line position along the spatial curvature"""
    resid_log = list()
    wavelengths = ref_table[:, 1]
    resid2D = pixtab2d - fit_table2d
    for wl, resid_col, col in zip(wavelengths, resid2D.T, fit_table2d.T):
        line_resid = 1.48*mad(resid_col)
        median_pix = np.median(col)
        delta_col = np.max(col) - np.min(col)
        resid_log.append([wl, median_pix, line_resid, delta_col])
    return resid_log

# ============== MAIN ===========================================================

def swap_axes_in_header(hdr):
    cdelt1 = hdr.get('CDELT1')
    cdelt2 = hdr.get('CDELT2')
    cd_11 = hdr.get('CD1_1')
    cd_22 = hdr.get('CD2_2')
    cd_12 = hdr.get('CD1_2')
    cd_21 = hdr.get('CD2_1')
    crval1 = hdr.get('CRVAL1')
    crval2 = hdr.get('CRVAL2')
    crpix1 = hdr.get('CRPIX1')
    crpix2 = hdr.get('CRPIX2')
    ctype1 = hdr.get('CTYPE1')
    ctype2 = hdr.get('CTYPE2')
    cunit1 = hdr.get('CUNIT1')
    cunit2 = hdr.get('CUNIT2')

    if cdelt1:
        hdr['CDELT2'] = cdelt1
        hdr['CDELT1'] = cdelt2
    if cd_11:
        hdr['CD2_2'] = cd_11
        hdr['CD1_1'] = cd_22
    if cd_12:
        hdr['CD1_2'] = cd_21
        hdr['CD2_1'] = cd_12

    hdr['CRVAL2'] = crval1
    hdr['CRVAL1'] = crval2
    hdr['CRPIX2'] = crpix1
    hdr['CRPIX1'] = crpix2
    hdr['CTYPE2'] = ctype1
    hdr['CTYPE1'] = ctype2
    hdr['CUNIT2'] = cunit1
    hdr['CUNIT1'] = cunit2

    binx = instrument.get_binx(hdr)
    biny = instrument.get_biny(hdr)
    hdr = instrument.set_binx(hdr, biny)
    hdr = instrument.set_biny(hdr, binx)

    return hdr

def rectify(img_fname, arc_fname, pixtable_fname, output='', fig_dir='', order_bg=5, order_2d=5,
            order_wl=4, log=False, N_out=None, interpolate=True, dispaxis=2, fit_window=20,
            plot=True, overwrite=True, verbose=False, overscan=50, edge_kappa=10.):

    msg = list()
    arc2D = fits.getdata(arc_fname)
    img2D = fits.getdata(img_fname)
    msg.append("          - Loaded image: %s" % img_fname)
    msg.append("          - Loaded reference arc image: %s" % arc_fname)
    err2D = fits.getdata(img_fname, 'ERR')
    msg.append("          - Loaded error image")

    try:
        mask2D = fits.getdata(img_fname, 'MASK')
        msg.append("          - Loaded mask image")
    except KeyError:
        mask2D = np.zeros_like(img2D)
    hdr = instrument.get_header(img_fname)

    ref_table = np.loadtxt(pixtable_fname)
    pixtab_pars, found_all = get_pixtab_parameters(pixtable_fname)
    msg.append("          - Loaded reference pixel table: %s" % pixtable_fname)
    ref_type = pixtab_pars['ref_type']
    msg.append("          - Wavelength solution is in reference system: %s" % ref_type)
    if found_all:
        order_wl = pixtab_pars['order_wl']
    else:
        msg.append("[WARNING] - Not all parameters were loaded from the pixel table!")
    msg.append("          - Polynomial order for wavelength as function of pixels : %i" % order_wl)

    if 'DISPAXIS' in hdr.keys():
        dispaxis = hdr['DISPAXIS']
    elif instrument.get_dispaxis(hdr):
        dispaxis = instrument.get_dispaxis(hdr)

    if dispaxis == 2:
        msg.append("          - Rotating frame to have dispersion along x-axis")
        # Reorient image to have dispersion along x-axis:
        arc2D = arc2D.T
        img2D = img2D.T
        mask2D = mask2D.T
        err2D = err2D.T
        pix_in = instrument.create_pixel_array(hdr, axis=2)
        hdr = swap_axes_in_header(hdr)
    else:
        pix_in = instrument.create_pixel_array(hdr, axis=1)

    ilow, ihigh = detect_borders(arc2D, kappa=edge_kappa)
    msg.append("          - Image shape: (%i, %i)" % arc2D.shape)
    msg.append("          - Detecting arc line borders: %i -- %i" % (ilow, ihigh))
    hdr['CRPIX2'] += ilow
    # Trim images:
    arc2D = arc2D[ilow:ihigh, :]
    img2D = img2D[ilow:ihigh, :]
    err2D = err2D[ilow:ihigh, :]
    mask2D = mask2D[ilow:ihigh, :]

    msg.append("          - Subtracting arc line continuum background")
    msg.append("          - Polynomial order of 1D background: %i" % order_bg)
    arc2D_sub, _ = subtract_arc_background(arc2D, deg=order_bg)

    msg.append("          - Fitting arc line positions within %i pixels" % fit_window)
    msg.append("          - Number of lines to fit: %i" % ref_table.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pixtab2d = create_2d_pixtab(arc2D_sub, pix_in, ref_table, dx=fit_window)

    msg.append("          - Constructing 2D wavelength grid with polynomial order: %i" % order_2d)
    fit_table2d = fit_2dwave_solution(pixtab2d, deg=order_2d)

    msg.append("          - Residuals of arc line positions relative to fitted 2D grid:")
    fit_residuals = format_table2D_residuals(pixtab2d, fit_table2d, ref_table)

    msg.append("              Wavelength    Mean Position   Arc Residual   Max. Curvature")
    for l0, med_line_pos, line_residual, line_minmax in fit_residuals:
        msg.append("              %10.2f    %-13.2f   %-12.3f   %-14.3f" % (l0, med_line_pos, line_residual, line_minmax))

    if plot:
        plot_fname = os.path.join(fig_dir, 'PixTable2D.pdf')
        plot_2d_pixtable(arc2D_sub, pix_in, pixtab2d, fit_table2d, arc_fname, filename=plot_fname)
        msg.append("          - Plotting fitted arc line positions in 2D frame")
        msg.append(" [OUTPUT] - Saving figure: %s" % plot_fname)

    msg.append("          - Interpolating input image onto rectified wavelength solution")
    try:
        transform_output = apply_transform(img2D, pix_in, fit_table2d, ref_table,
                                           err2D=err2D, mask2D=mask2D, header=hdr,
                                           order_wl=order_wl, ref_type=ref_type,
                                           log=log, N_out=N_out, interpolate=interpolate)
        img2D_corr, err2D_corr, mask2D, wl, hdr_corr, trans_msg = transform_output
        msg.append(trans_msg)

    except WavelengthError as error:
        msg.append(error.message)
        output_str = "\n".join(msg)
        print(output_str)
        return "FATAL ERROR"

    hdr.add_comment('PyNOT version %s' % __version__)
    if output:
        if output[-5:] != '.fits':
            output += '.fits'
    else:
        object_name = instrument.get_object(hdr)
        output = 'RECT2D_%s.fits' % (object_name)

    hdr_corr['DISPAXIS'] = 1
    hdr_corr['EXTNAME'] = 'DATA'
    with fits.open(img_fname) as hdu:
        hdu[0].data = img2D_corr
        hdu[0].header = hdr_corr
        if err2D_corr is not None:
            hdu['ERR'].data = err2D_corr
            err_hdr = hdu['ERR'].header
            copy_keywords = ['CRPIX1', 'CRVAL1', 'CDELT1', 'CTYPE1', 'CUNIT1', 'CD1_1']
            copy_keywords += ['CRPIX2', 'CRVAL2', 'CDELT2', 'CTYPE2', 'CUNIT2', 'CD2_2']
            for key in copy_keywords:
                if key in hdr:
                    err_hdr[key] = hdr[key]
            hdu['ERR'].header = err_hdr
        if 'MASK' in hdu:
            hdu['MASK'].data = mask2D
            mask_hdr = hdu['MASK'].header
            copy_keywords = ['CRPIX1', 'CRVAL1', 'CDELT1', 'CTYPE1', 'CUNIT1', 'CD1_1']
            copy_keywords += ['CRPIX2', 'CRVAL2', 'CDELT2', 'CTYPE2', 'CUNIT2', 'CD2_2']
            for key in copy_keywords:
                if key in hdr:
                    mask_hdr[key] = hdr[key]
            hdu['MASK'].header = mask_hdr
        else:
            mask_hdr = fits.Header()
            mask_hdr.add_comment("2 = Good Pixels")
            mask_hdr.add_comment("1 = Cosmic Ray Hits")
            mask_hdr['AUTHOR'] = 'PyNOT version %s' % __version__
            copy_keywords = ['CRPIX1', 'CRVAL1', 'CDELT1', 'CTYPE1', 'CUNIT1', 'CD1_1']
            copy_keywords += ['CRPIX2', 'CRVAL2', 'CDELT2', 'CTYPE2', 'CUNIT2', 'CD2_2']
            for key in copy_keywords:
                if key in hdr:
                    mask_hdr[key] = hdr[key]
            mask_ext = fits.ImageHDU(mask2D, header=mask_hdr, name='MASK')
            hdu.append(mask_ext)
        hdu.writeto(output, overwrite=overwrite)
    msg.append(" [OUTPUT] - Saving rectified 2D image: %s" % output)
    msg.append("")
    output_str = "\n".join(msg)
    if verbose:
        print(output_str)
    return output_str
