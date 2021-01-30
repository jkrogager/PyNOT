"""
Wavelength Calibration and 2D Transformation
"""

__author__ = "Jens-Kristian Krogager"

import os
import sys
from astropy.io import fits
import numpy as np
from numpy.polynomial import Chebyshev
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import warnings

from PyQt5.QtWidgets import QApplication

from identify_gui import GraphicInterface, create_pixel_array
from alfosc import get_alfosc_header


code_dir = os.path.dirname(os.path.abspath(__file__))
v_file = os.path.join(code_dir, 'VERSION')
with open(v_file) as version_file:
    __version__ = version_file.read().strip()


# -- Function to call from PyNOT.main
def create_pixtable(arc_image, grism_name, pixtable_name, linelist_fname, order_wl=4):
    """
    arc_image : str
        Filename of arc image

    grism_name : str
        Grism name, ex: grism4
    """

    fname = os.path.basename(arc_image)
    base_name, ext = os.path.splitext(fname)
    output_pixtable = "%s_arcID_%s.tab" % (base_name, grism_name)
    # Launch App:
    global qapp
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    gui = GraphicInterface(arc_image,
                           grism_name=grism_name,
                           pixtable=pixtable_name,
                           linelist_fname=linelist_fname,
                           output=output_pixtable,
                           order_wl=order_wl,
                           locked=True)
    gui.show()
    app.exit(app.exec_())

    if os.path.exists(output_pixtable) and gui.message == 'ok':
        # The GUI exited successfully
        order_wl = int(gui.poly_order.text())
        msg = "          - Successfully saved line identifications: %s\n" % output_pixtable

        if not os.path.exists(pixtable_name):
            # move output_pixtable to pixtable_name:
            copy_command = "cp %s %s" % (output_pixtable, pixtable_name)
            # os.system(copy_command)                                            # <-- UPDATE BEFORE RELEASE
            print(copy_command)

    else:
        msg = " [ERROR]  - Something went wrong in line identification of %s\n" % grism_name
        order_wl = None
        output_pixtable = None

    del gui

    return order_wl, output_pixtable, msg


def verify_arc_frame(arc_fname, dispaxis=2):
    """
    Return True if the arc lines fill the full detector plane, else return False.
    The full detector plane is considered the region between 10th and 90th percentile
    of the pixels along the slit.

    `dispaxis` = 2 for vertical spectra, 1 for horizontal spectra.
    """
    arc2D = fits.getdata(arc_fname)
    hdr = get_alfosc_header(arc_fname)
    if 'DISPAXIS' in hdr:
        dispaxis = hdr['DISPAXIS']

    if dispaxis == 2:
        # Reorient image to have dispersion along x-axis:
        arc2D = arc2D.T

    ilow, ihigh = detect_borders(arc2D)
    imax = arc2D.shape[0]
    result = (ilow < 0.1*imax) & (ihigh > 0.9*imax)
    return result


def modNN_gaussian(x, bg, mu, sigma, logamp):
    """ One-dimensional modified non-negative Gaussian profile."""
    amp = 10**logamp
    return bg + amp * np.exp(-0.5*(x-mu)**4/sigma**2)

def fit_gaussian_center(x, y):
    bg = np.nanmedian(y)
    logamp = np.log10(np.nanmax(y)-bg)
    sig = 1.5
    max_index = np.argmax(y)
    mu = x[max_index]
    p0 = np.array([bg, mu, sig, logamp])
    try:
        popt, pcov = curve_fit(modNN_gaussian, x, y, p0)
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
    MAD = np.nanmedian(np.abs(x - med_x))
    mask = np.abs(x - med_x) < kappa*MAD
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
        peaks, _ = find_peaks(row, prominence=kappa*mad)
        N_lines[i] = len(peaks)

    mask = (N_lines >= np.median(N_lines)).nonzero()
    row_min = np.min(mask)
    row_max = np.max(mask)
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

        # Insert back into the fit_table
        fit_col = cheb_polyfit(col)
        fit_table2d[num] = fit_col

    # Transpose back to original orientation of the pixel table
    return fit_table2d.T


def apply_transform(img2D, pix, fit_table2d, ref_table, header={}, wl_order=4, log=False, N_out=None, kind='cubic', fill_value='extrapolate'):
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

    header : FITS Header or dict
        The FITS header corresponding to `img2D`, or a dictionary

    wl_order : int
        Polynomial order used for wavelength fit as a function of input pixel
        A Chebyshev polynomium is used to fit the solution for each row in img2D.

    log : bool  [default=False]
        Use logarithmic binning in wavelength?

    N_out : int
        Number of output pixels along dispersion axis.
        If `None` is given, the default is to use the same number
        of pixels as in the input image.

    kind : string
        Keyword argument of scipy.interpolate.interp1d

    fill_value : string or float
        Keyword argument of scipy.interpolate.interp1d

    Returns
    -------
    img2D_tr : array, shape(M, N_out)
        Transformed 2D spectrum with wavelength along x-axis.

    wl : array, shape(N_out)
        Coresponding wavelength array along x-axis.

    hdr_tr : FITS Header or dict
        Updated FITS Header or dictionary with wavelength information
    """

    # Define wavelength grid at midpoint:
    if N_out is None:
        N_out = img2D.shape[1]

    pix_in = pix
    cen = fit_table2d.shape[0]//2
    ref_wl = ref_table[:, 1]
    central_solution = Chebyshev.fit(fit_table2d[cen], ref_wl, deg=wl_order, domain=[pix_in.min(), pix_in.max()])
    wl_central = central_solution(pix_in)
    if log:
        wl = np.logspace(np.log10(wl_central.min()), np.log10(wl_central.max()), N_out)
        hdr_tr = header.copy()
        hdr_tr['CRPIX1'] = 1
        hdr_tr['CDELT1'] = np.diff(np.log10(wl))[0]
        hdr_tr['CRVAL1'] = np.log10(wl[0])
        hdr_tr['CTYPE1'] = 'LOGLAM  '
        hdr_tr['CUNIT1'] = 'ANGSTROM'
    else:
        wl = np.linspace(wl_central.min(), wl_central.max(), N_out)
        hdr_tr = header.copy()
        hdr_tr['CRPIX1'] = 1
        hdr_tr['CDELT1'] = np.diff(wl)[0]
        hdr_tr['CRVAL1'] = wl[0]
        hdr_tr['CTYPE1'] = 'LINEAR  '
        hdr_tr['CUNIT1'] = 'ANGSTROM'

    img2D_tr = np.zeros((img2D.shape[0], N_out))
    for i, row in enumerate(img2D):
        # - fit the chebyshev polynomium
        solution_row = Chebyshev.fit(fit_table2d[i], ref_wl, deg=wl_order, domain=[pix_in.min(), pix_in.max()])
        wl_row = solution_row(pix_in)

        # - interpolate the data onto the fixed input grid
        interp_row = interp1d(wl_row, row, kind=kind, fill_value=fill_value)
        img2D_tr[i] = interp_row(wl)

    return img2D_tr, wl, hdr_tr


# ============== PLOTTING =====================================================

def plot_2d_pixtable(arc2D_sub, pix, pixtab2d, fit_table2d, filename=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mad = np.nanmedian(np.abs(arc2D_sub - np.nanmedian(arc2D_sub)))
    ax.imshow(arc2D_sub, origin='lower', extent=(pix.min(), pix.max(), 1, arc2D_sub.shape[0]),
              vmin=-3*mad, vmax=10*mad, cmap=plt.cm.gray_r, aspect='auto')
    for i in np.arange(0, pixtab2d.shape[0], 5):
        row = pixtab2d[i]
        ax.plot(row, np.ones_like(row)*(i+1), ls='', color='Blue', marker='.', alpha=0.3)
    for col in fit_table2d.T:
        ax.plot(col, np.arange(arc2D_sub.shape[0]), lw=1, color='r', alpha=0.5)
    ax.set_xlim(pix.min(), pix.max())
    ax.set_ylim(1, arc2D_sub.shape[0])
    if filename:
        fig.savefig(filename)


# FORMAT RESIDUALS:

def format_table2D_residuals(pixtab2d, fit_table2d, ref_table):
    """Calculate the residuals of each arc line position along the spatial curvature"""
    resid_log = list()
    wavelengths = ref_table[:, 1]
    resid2D = pixtab2d - fit_table2d
    for wl, col in zip(wavelengths, resid2D.T):
        line_resid = np.std(col)
        resid_log.append([wl, line_resid])
    return resid_log

# ============== MAIN ===========================================================

def rectify(img_fname, arc_fname, pixtable_fname, output='', fig_dir='', order_bg=5, order_2d=5, order_wl=4, log=False, N_out=None,
            kind='linear', fill_value='extrapolate', binning=1, dispaxis=2, fit_window=20, plot=True, overwrite=True, verbose=False):

    msg = list()
    msg.append("          - Running task: Rectify 2D and Wavelength Calibrate")
    arc2D = fits.getdata(arc_fname)
    img2D = fits.getdata(img_fname)
    hdr = get_alfosc_header(img_fname)
    msg.append("          - Loaded image: %s" % img_fname)
    msg.append("          - Loaded reference arc image: %s" % arc_fname)

    ref_table = np.loadtxt(pixtable_fname)
    msg.append("          - Loaded reference pixel table: %s" % pixtable_fname)
    if 'DISPAXIS' in hdr.keys():
        dispaxis = hdr['DISPAXIS']

    if dispaxis == 2:
        msg.append("          - Rotating frame to have dispersion along x-axis")
        # Reorient image to have dispersion along x-axis:
        img2D = img2D.T
        arc2D = arc2D.T
        pix_in = create_pixel_array(hdr, dispaxis=2)

        if 'DETYBIN' in hdr:
            binning = hdr['DETYBIN']
            hdr['DETYBIN'] = hdr['DETXBIN']
            hdr['DETXBIN'] = binning
        hdr['CDELT2'] = hdr['CDELT1']
        hdr['CRVAL2'] = hdr['CRVAL1']
        hdr['CRPIX2'] = hdr['CRPIX1']
        hdr['CTYPE2'] = 'LINEAR'
        hdr['CUNIT2'] = hdr['CUNIT1']
    else:
        pix_in = create_pixel_array(hdr, dispaxis=1)
        if 'DETXBIN' in hdr:
            binning = hdr['DETXBIN']

    ilow, ihigh = detect_borders(arc2D)
    msg.append("          - Image shape: (%i, %i)" % arc2D.shape)
    msg.append("          - Detecting arc line borders: %i -- %i" % (ilow, ihigh))
    # Trim images:
    arc2D = arc2D[ilow:ihigh, :]
    img2D = img2D[ilow:ihigh, :]

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

    msg.append("  Wavelength    Standars Deviation of Line Positions")
    for l0, line_residual in fit_residuals:
        msg.append("  %10.3f    %.3f" % (l0, line_residual))

    msg.append("          - Interpolating input image onto rectified wavelength solution")
    img2D_corr, wl, hdr_corr = apply_transform(img2D, pix_in, fit_table2d, ref_table,
                                               header=hdr, wl_order=order_wl,
                                               log=log, N_out=N_out, kind=kind,
                                               fill_value=fill_value)

    hdr.add_comment('PyNOT version %s' % __version__)
    if plot:
        plot_fname = os.path.join(fig_dir, 'PixTable2D.pdf')
        plot_2d_pixtable(arc2D_sub, pix_in, pixtab2d, fit_table2d, filename=plot_fname)
        msg.append("          - Plotting fitted arc line positions in 2D frame")
        msg.append("          - Saving figure to file: %s" % plot_fname)

    if output:
        if output[-5:] != '.fits':
            output += '.fits'
    else:
        object_name = hdr['OBJECT']
        output = 'RECT2D_%s.fits' % (object_name)

    hdu = fits.PrimaryHDU(data=img2D_corr, header=hdr_corr)
    hdu.writeto(output, overwrite=overwrite)
    msg.append("          - Saving rectified 2D image to file: %s" % output)
    msg.append("")
    output_str = "\n".join(msg)
    if verbose:
        print(output_str)
    return output_str
