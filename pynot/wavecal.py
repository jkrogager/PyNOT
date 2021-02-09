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

from pynot.alfosc import get_alfosc_header, create_pixel_array
from pynot.functions import get_version_number, NN_mod_gaussian
from pynot.scired import trim_overscan

__version__ = get_version_number()


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
        peaks, _ = find_peaks(row, prominence=kappa*mad)
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


def apply_transform(img2D, pix, fit_table2d, ref_table, err2D=None, mask2D=None, header={}, order_wl=4, log=False, N_out=None, interpolate=True):
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

    pix_in = pix
    cen = fit_table2d.shape[0]//2
    ref_wl = ref_table[:, 1]
    central_solution = Chebyshev.fit(fit_table2d[cen], ref_wl, deg=order_wl, domain=[pix_in.min(), pix_in.max()])
    wl_central = central_solution(pix_in)
    wl_residuals = np.std(ref_wl - central_solution(fit_table2d[cen]))
    msg.append("          - Residuals of wavelength solution: %.2f Å" % wl_residuals)
    if log:
        wl = np.logspace(np.log10(wl_central.min()), np.log10(wl_central.max()), N_out)
        hdr_tr = header.copy()
        hdr_tr['CRPIX1'] = 1
        hdr_tr['CDELT1'] = np.diff(np.log10(wl))[0]
        hdr_tr['CRVAL1'] = np.log10(wl[0])
        hdr_tr['CTYPE1'] = 'LOGLAM  '
        hdr_tr['CUNIT1'] = 'Angstrom'
        msg.append("          - Creating logarithmically sampled wavelength grid")
        msg.append("          - Sampling: %.3f  (logÅ/pix)" % np.diff(np.log10(wl))[0])
    else:
        wl = np.linspace(wl_central.min(), wl_central.max(), N_out)
        hdr_tr = header.copy()
        hdr_tr['CRPIX1'] = 1
        hdr_tr['CDELT1'] = np.diff(wl)[0]
        hdr_tr['CRVAL1'] = wl[0]
        hdr_tr['CTYPE1'] = 'LINEAR  '
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
        for i, row in enumerate(img2D):
            # - fit the chebyshev polynomium
            solution_row = Chebyshev.fit(fit_table2d[i], ref_wl, deg=order_wl, domain=[pix_in.min(), pix_in.max()])
            wl_row = solution_row(pix_in)
            if np.diff(wl_row)[0] < 0:
                # Wavelengths are decreasing: Flip arrays
                row = row[::-1]
                wl_row = wl_row[::-1]
                flip_array = True
            else:
                flip_array = False

            # -- interpolate the data onto the fixed wavelength grid
            if err2D is not None:
                err_row = err2D[i]
                if flip_array:
                    err_row = err_row[::-1]
                interp_row, interp_err = spectres.spectres(wl, wl_row, row, spec_errs=err_row, verbose=False, fill=0.)
                err2D_tr[i] = interp_err
            else:
                interp_row = spectres.spectres(wl, wl_row, row, verbose=False, fill=0.)

            mask_row = mask2D[i]
            mask_int = np.interp(wl, wl_row, mask_row)
            mask2D_tr[i] = np.ceil(mask_int).astype(int)
            img2D_tr[i] = interp_row
    else:
        img2D_tr = img2D
        err2D_tr = err2D
        mask2D_tr = mask2D
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
    if order_wl is None:
        order_wl, found_in_file = get_order_from_file(pixtable_fname)
        if found_in_file:
            msg.append("          - Loaded polynomial order from file: %i" % order_wl)
        else:
            msg.append("          - Using default polynomial order: %i" % order_wl)
    else:
        msg.append("          - Using polynomial order: %i" % order_wl)

    if log:
        linearize = True

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
        hdr['CUNIT1'] = 'Angstrom'
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
        line_resid = np.std(resid_col)
        delta_col = np.max(col) - np.min(col)
        resid_log.append([wl, line_resid, delta_col])
    return resid_log

# ============== MAIN ===========================================================

def rectify(img_fname, arc_fname, pixtable_fname, output='', fig_dir='', order_bg=5, order_2d=5,
            order_wl=4, log=False, N_out=None, interpolate=True, binning=1, dispaxis=2, fit_window=20,
            plot=True, overwrite=True, verbose=False, overscan=50):

    msg = list()
    arc2D = fits.getdata(arc_fname)
    arc_hdr = get_alfosc_header(arc_fname)
    arc2D, arc_hdr = trim_overscan(arc2D, arc_hdr, overscan=overscan)
    img2D = fits.getdata(img_fname)
    msg.append("          - Loaded image: %s" % img_fname)
    msg.append("          - Loaded reference arc image: %s" % arc_fname)
    try:
        err2D = fits.getdata(img_fname, 'ERR')
        msg.append("          - Loaded error image")
    except KeyError:
        err2D = None
    try:
        mask2D = fits.getdata(img_fname, 'MASK')
        msg.append("          - Loaded mask image")
    except KeyError:
        mask2D = np.zeros_like(img2D)
    hdr = fits.getheader(img_fname)

    ref_table = np.loadtxt(pixtable_fname)
    msg.append("          - Loaded reference pixel table: %s" % pixtable_fname)
    if 'DISPAXIS' in hdr.keys():
        dispaxis = hdr['DISPAXIS']

    if dispaxis == 2:
        msg.append("          - Rotating frame to have dispersion along x-axis")
        # Reorient image to have dispersion along x-axis:
        arc2D = arc2D.T
        img2D = img2D.T
        mask2D = mask2D.T
        if err2D is not None:
            err2D = err2D.T
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

    msg.append("              Wavelength    Arc Residual   Max. Curvature")
    for l0, line_residual, line_minmax in fit_residuals:
        msg.append("              %10.3f    %-12.3f   %-14.3f" % (l0, line_residual, line_minmax))

    msg.append("          - Interpolating input image onto rectified wavelength solution")
    transform_output = apply_transform(img2D, pix_in, fit_table2d, ref_table,
                                       err2D=err2D, mask2D=mask2D, header=hdr, order_wl=order_wl,
                                       log=log, N_out=N_out, interpolate=interpolate)
    img2D_corr, err2D_corr, mask2D, wl, hdr_corr, trans_msg = transform_output
    msg.append(trans_msg)
    hdr.add_comment('PyNOT version %s' % __version__)
    if plot:
        plot_fname = os.path.join(fig_dir, 'PixTable2D.pdf')
        plot_2d_pixtable(arc2D_sub, pix_in, pixtab2d, fit_table2d, filename=plot_fname)
        msg.append("          - Plotting fitted arc line positions in 2D frame")
        msg.append(" [OUTPUT] - Saving figure: %s" % plot_fname)

    if output:
        if output[-5:] != '.fits':
            output += '.fits'
    else:
        object_name = hdr['OBJECT']
        output = 'RECT2D_%s.fits' % (object_name)

    hdr_corr['DISPAXIS'] = 1
    hdr_corr['EXTNAME'] = 'DATA'
    with fits.open(img_fname) as hdu:
        hdu[0].data = img2D_corr
        hdu[0].header = hdr_corr
        if err2D_corr is not None:
            hdu['ERR'].data = err2D_corr
            err_hdr = hdu['ERR'].header
            copy_keywords = ['CRPIX1', 'CRVAL1', 'CDELT1', 'CTYPE1', 'CUNIT1']
            copy_keywords += ['CRPIX2', 'CRVAL2', 'CDELT2', 'CTYPE2', 'CUNIT2']
            for key in copy_keywords:
                err_hdr[key] = hdr[key]
            hdu['ERR'].header = err_hdr
        if 'MASK' in hdu:
            hdu['MASK'].data = mask2D
            mask_hdr = hdu['MASK'].header
            copy_keywords = ['CRPIX1', 'CRVAL1', 'CDELT1', 'CTYPE1', 'CUNIT1']
            copy_keywords += ['CRPIX2', 'CRVAL2', 'CDELT2', 'CTYPE2', 'CUNIT2']
            for key in copy_keywords:
                mask_hdr[key] = hdr[key]
            hdu['MASK'].header = mask_hdr
        else:
            mask_hdr = fits.Header()
            mask_hdr.add_comment("2 = Good Pixels")
            mask_hdr.add_comment("1 = Cosmic Ray Hits")
            mask_hdr['AUTHOR'] = 'PyNOT version %s' % __version__
            copy_keywords = ['CRPIX1', 'CRVAL1', 'CDELT1', 'CTYPE1', 'CUNIT1']
            copy_keywords += ['CRPIX2', 'CRVAL2', 'CDELT2', 'CTYPE2', 'CUNIT2']
            for key in copy_keywords:
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
