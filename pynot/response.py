# PyNOT/response
"""
Calculate the instrumental response function

Tabulated data can be found here:
https://ftp.eso.org/pub/usg/standards/

"""
__author__ = 'Jens-Kristian Krogager'
__email__ = "krogager.jk@gmail.com"
__credits__ = ["Jens-Kristian Krogager"]

import numpy as np
from astropy.io import fits
from matplotlib.backends import backend_pdf
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.interpolate import UnivariateSpline
import os
from os.path import basename, dirname, abspath
import sys
import glob

from pynot import instrument
from pynot.data import organizer
from pynot.extraction import auto_extract
from pynot import extract_gui
from pynot.functions import get_version_number, my_formatter, mad
from pynot import response_gui
from pynot.logging import Report
from pynot.scired import auto_fit_background, raw_correction
from pynot.scombine import combine_2d
from pynot.wavecal import rectify


__version__ = get_version_number()


path = dirname(abspath(__file__))
# --- Data taken from: ftp://ftp.stsci.edu/cdbs/current_calspec/
_standard_star_files = glob.glob(path + '/calib/std/*.dat')
_standard_star_files = [basename(fname) for fname in _standard_star_files]
# List of star names in lowercase:
standard_stars = [fname.strip('.dat') for fname in _standard_star_files]

# Look-up table from target-names -> star names
# (mostly used for ALFOSC where TCSTGT is different)
std_fname = os.path.join(path, 'calib/std/tcs_namelist.txt')
calib_names = np.loadtxt(std_fname, dtype=str, delimiter=':')
tcs_standard_stars = {row[1].strip(): row[0].strip() for row in calib_names}


def lookup_std_star(hdr):
    """
    Check if the given header contains an object or target name
    which matches one of the defined standard calibration stars.

    Returns `None` if no match is found.
    """
    object_name = instrument.get_object(hdr).replace(' ', '')
    target_name = instrument.get_target_name(hdr).replace(' ', '')
    if object_name.lower() in standard_stars:
        return object_name.lower()
    elif object_name.upper() in tcs_standard_stars:
        return tcs_standard_stars[object_name.upper()]
    elif target_name.lower() in standard_stars:
        return target_name.lower()
    elif target_name.upper() in tcs_standard_stars:
        return tcs_standard_stars[target_name.upper()]
    else:
        return None


def load_spectrum1d(fname):
    table = fits.getdata(fname)
    wl = table['WAVE']
    flux = table['FLUX']
    return wl, flux


def flux_calibrate(input_fname, *, output, response_fname):
    """Apply response function to flux calibrate the input spectrum"""
    msg = list()
    # Load input data:
    hdr = fits.getheader(input_fname)
    img2D = fits.getdata(input_fname)
    err2D = fits.getdata(input_fname, 'ERR')
    msg.append("          - Loaded image: %s" % input_fname)
    cdelt = hdr['CDELT1']
    crval = hdr['CRVAL1']
    crpix = hdr['CRPIX1']
    wl = (np.arange(hdr['NAXIS1']) - (crpix - 1))*cdelt + crval

    # Load Extinction Table:
    wl_ext, A0 = np.loadtxt(instrument.extinction_fname, unpack=True)
    ext = np.interp(wl, wl_ext, A0)
    msg.append("          - Loaded average extinction table:")
    msg.append("            %s" % instrument.extinction_fname)

    # Load Sensitivity Function:
    resp_tab = fits.getdata(response_fname)
    resp_hdr = fits.getheader(response_fname)
    if 'ALGRNM' in resp_hdr:
        resp_hdr['GRISM'] = resp_hdr['ALGRNM']
    if resp_hdr['GRISM'] != instrument.get_grism(hdr):
        msg.append(" [ERROR]  - Grisms of input spectrum and response function do not match!")
        msg.append("")
        output_msg = "\n".join(msg)
        return output_msg

    resp_int = np.interp(wl, resp_tab['WAVE'], resp_tab['RESPONSE'])
    # Truncate values less than 20:
    resp_int[resp_int < 20] = 20.
    msg.append("          - Loaded response function: %s" % response_fname)

    airmass = instrument.get_airmass(hdr)
    if airmass is None:
        user_input = input("          > Please give the airmass:\n          > ")
        try:
            airmass = float(user_input)
        except (ValueError) as e:
            msg.append(" [ERROR]  - Invalid airmass!")
            msg.append(" [ERROR]  - " + str(e))
            msg.append("")
            return "\n".join(msg)

    t = instrument.get_exptime(hdr)
    if t is None:
        user_input = input("          > Please give the exposure time:\n          > ")
        try:
            t = float(user_input)
        except ValueError:
            msg.append(" [ERROR]  - Invalid exposure time: %r" % user_input)
            msg.append("")
            return "\n".join(msg)

    msg.append("          - exposure time: %.1f" % t)
    msg.append("          - airmass: %.3f" % airmass)
    # ext_correction = 10**(0.4*airm * ext)
    # flux_calibration = ext_correction / 10**(0.4*resp_int)
    flux_calibration = 10**(0.4*(airmass*ext - resp_int))
    flux_calib2D = np.resize(flux_calibration, img2D.shape)
    flux2D = img2D / (t * cdelt) * flux_calib2D
    err2D = err2D / (t * cdelt) * flux_calib2D

    with fits.open(input_fname) as hdu:
        hdu[0].data = flux2D
        hdu[0].header['BUNIT'] = 'erg/s/cm2/A'
        hdu[0].header['RESPONSE'] = response_fname

        hdu['ERR'].data = err2D
        hdu['ERR'].header['BUNIT'] = 'erg/s/cm2/A'

        hdu.writeto(output, overwrite=True)
    msg.append(" [OUTPUT] - Saving flux calibrated 2D image: %s" % output)
    msg.append("")
    output_msg = "\n".join(msg)
    return output_msg


def flux_calibrate_1d(input_fname, *, output, response_fname):
    """Apply response function to flux calibrate the input 1D spectrum"""
    msg = list()

    # Load Extinction Table:
    wl_ext, A0 = np.loadtxt(instrument.extinction_fname, unpack=True)
    msg.append("          - Loaded average extinction table:")
    msg.append("            %s" % instrument.extinction_fname)

    # Load Sensitivity Function:
    resp_tab = fits.getdata(response_fname)
    resp_hdr = fits.getheader(response_fname)
    msg.append("          - Loaded response function: %s" % response_fname)

    # Load input data:
    hdu_list = fits.open(input_fname)
    msg.append("          - Loaded image: %s" % input_fname)
    output_hdu = fits.HDUList()
    for hdu in hdu_list[1:]:
        tab = hdu.data
        hdr = hdu.header
        wl = tab['WAVE']
        spec1d = tab['FLUX']
        err1d = tab['ERR']

        if 'ALGRNM' in resp_hdr:
            resp_hdr['GRISM'] = resp_hdr['ALGRNM']
        if resp_hdr['GRISM'] != instrument.get_grism(hdr):
            msg.append(" [ERROR]  - Grisms of input spectrum and response function do not match!")
            msg.append("")
            output_msg = "\n".join(msg)
            return output_msg

        # Interpolate extinction table and response function:
        ext = np.interp(wl, wl_ext, A0)
        resp_int = np.interp(wl, resp_tab['WAVE'], resp_tab['RESPONSE'])
        # Truncate values less than 20:
        resp_int[resp_int < 20] = 20.

        airmass = instrument.get_airmass(hdr)
        if airmass is None:
            user_input = input("          > Please give the airmass:\n          > ")
            try:
                airmass = float(user_input)
            except (ValueError) as e:
                msg.append(" [ERROR]  - Invalid airmass!")
                msg.append(" [ERROR]  - " + str(e))
                msg.append("")
                return "\n".join(msg)

        t = instrument.get_exptime(hdr)
        if t is None:
            user_input = input("          > Please give the exposure time:\n          > ")
            try:
                t = float(user_input)
            except ValueError:
                msg.append(" [ERROR]  - Invalid exposure time: %r" % user_input)
                msg.append("")
                return "\n".join(msg)
        msg.append("          - exposure time: %.1f" % t)
        msg.append("          - airmass: %.3f" % airmass)

        cdelt = np.mean(np.diff(wl))
        flux_calibration = 10**(0.4*(airmass*ext - resp_int))
        flux1d = spec1d / (t * cdelt) * flux_calibration
        err1d = err1d / (t * cdelt) * flux_calibration

        hdr['BUNIT'] = 'erg/s/cm2/A'
        hdr['RESPONSE'] = response_fname
        msg.append("          - Applied flux calibration to object ID: %r" % hdu.name)

        col_wl = fits.Column(name='WAVE', array=wl, format='D', unit=hdu.columns['WAVE'].unit)
        col_flux = fits.Column(name='FLUX', array=flux1d, format='D', unit=hdr['BUNIT'])
        col_err = fits.Column(name='ERR', array=err1d, format='D', unit=hdr['BUNIT'])
        output_tab = fits.BinTableHDU.from_columns([col_wl, col_flux, col_err], header=hdr)
        output_hdu.append(output_tab)

    output_hdu.writeto(output, overwrite=True)
    msg.append(" [OUTPUT] - Saving flux calibrated 2D image: %s" % output)
    msg.append("")
    output_msg = "\n".join(msg)
    return output_msg


def calculate_response(raw_fname, *, arc_fname, pixtable_fname, bias_fname, flat_fname, output='',
                       output_dir='', pdf_fname='', order=3, smoothing=0.02, interactive=False, dispaxis=2,
                       order_bg=5, rectify_options=None, app=None):
    """
    Extract and wavelength calibrate the standard star spectrum.
    Calculate the instrumental response function and fit the median filtered data points
    with a Chebyshev polynomium of the given *order*.

    Parameters
    ==========
    raw_fname : string
        File name for the standard star frame

    arc_fname : string
        File name for the associated arc frame

    pixtable_fname : string
        Filename of pixel table for the identified lines in the arc frame

    bias_fname : string
        Master bias file name to subtract bias level.
        If nothing is given, no bias level correction is performed.

    flat_fname : string
        Normalized flat file name.
        If nothing is given, no spectral flat field correction is performed.

    output : string  [default='']
        Output filename of the response function FITS Table

    output_dir : string  [default='']
        Output directory for the response function and intermediate files.
        Filenames are autogenerated from OBJECT name and GRISM

    pdf_fname : string  [default='']
        Output filename for diagnostic plots.
        If none, autogenerate from OBJECT name

    order : integer  [default=8]
        Order of the spline interpolation of the response function

    smoothing : float  [default=0.02]
        Smoothing factor for spline interpolation

    interactive : boolean  [default=False]
        Interactively subtract background and extract 1D spectrum
        using a graphical interface. Otherwise, automatically identify
        object, subtract background and extract object.

    dispaxis : integer  [default=2]
        Dispersion axis. 1: horizontal spectra, 2: vertical spectra (default for most ALFOSC grisms)

    rectify_options : dict()  [default={}]
        Dictionary of keyword arguments for `rectify`

    Returns
    =======
    response_output : string
        Filename of resulting response function

    output_msg : string
        Log of the function call
    """
    msg = list()
    if rectify_options is None:
        rectify_options = dict()
    else:
        rectify_options = rectify_options.copy()

    hdr = instrument.get_header(raw_fname)
    raw2D = fits.getdata(raw_fname)
    msg.append("          - Loaded flux standard image: %s" % raw_fname)

    # Setup the filenames:
    grism = instrument.get_grism(hdr)
    star = lookup_std_star(hdr)
    if star is None:
        msg.append("[WARNING] - No reference data found for target: %s" % instrument.get_object(hdr))
        msg.append("[WARNING] - The reduced spectra will not be flux calibrated")
        output_msg = "\n".join(msg)
        return None, output_msg

    response_output = 'response_%s.fits' % grism
    std_tmp_fname = 'std_corr2D_%s.fits' % star
    rect2d_fname = 'std_rect2D_%s.fits' % star
    bgsub2d_fname = 'std_bgsub2D_%s.fits' % star
    ext1d_output = 'std_ext1D_%s.fits' % star
    extract_pdf_fname = 'std_ext1D_diagnostics.pdf'
    if output_dir:
        response_output = os.path.join(output_dir, response_output)
        std_tmp_fname = os.path.join(output_dir, std_tmp_fname)
        rect2d_fname = os.path.join(output_dir, rect2d_fname)
        bgsub2d_fname = os.path.join(output_dir, bgsub2d_fname)
        ext1d_output = os.path.join(output_dir, ext1d_output)
        extract_pdf_fname = os.path.join(output_dir, extract_pdf_fname)

    try:
        output_msg = raw_correction(raw2D, hdr, bias_fname, flat_fname,
                                    output=std_tmp_fname, overwrite=True)
        msg.append(output_msg)
    except:
        msg.append("Unexpected error in raw correction: %r" % sys.exc_info()[0])
        output_msg = "\n".join(msg)
        raise Exception(output_msg)

    # Rectify 2D image and wavelength calibrate:
    try:
        rectify_options['plot'] = False
        rect_msg = rectify(std_tmp_fname, arc_fname, pixtable_fname, output=rect2d_fname,
                           dispaxis=dispaxis, **rectify_options)
        msg.append(rect_msg)
    except:
        msg.append("Unexpected error in rectify: %r" % sys.exc_info()[0])
        output_msg = "\n".join(msg)
        raise Exception(output_msg)
    # After RECTIFY all images are oriented with the dispersion axis horizontally


    # Subtract background:
    try:
        bg_msg = auto_fit_background(rect2d_fname, bgsub2d_fname, dispaxis=1, order_bg=order_bg, plot_fname='',
                                     kappa=100, fwhm_scale=5)
        msg.append(bg_msg)
    except:
        msg.append("Unexpected error in auto sky sub: %r" % sys.exc_info()[0])
        output_msg = "\n".join(msg)
        raise Exception(output_msg)


    # Extract 1-dimensional spectrum:
    if interactive:
        try:
            msg.append("          - Starting Graphical User Interface for Spectral Extraction")
            extract_gui.run_gui(bgsub2d_fname, output_fname=ext1d_output,
                                app=app, order_center=5, order_width=5, smoothing=smoothing, dx=20)
            msg.append(" [OUTPUT] - Writing fits table: %s" % ext1d_output)
        except:
            msg.append("Unexpected error in extract GUI: %r" % sys.exc_info()[0])
            output_msg = "\n".join(msg)
            raise Exception(output_msg)
    else:
        try:
            ext_msg = auto_extract(bgsub2d_fname, ext1d_output, dispaxis=1, N=1, pdf_fname=extract_pdf_fname,
                                   model_name='moffat', dx=20, order_center=4, order_width=5, xmin=20, ymin=5, ymax=-5,
                                   kappa_cen=5., w_cen=15)
            msg.append(ext_msg)
        except:
            msg.append("Unexpected error in auto extract: %r" % sys.exc_info()[0])
            output_msg = "\n".join(msg)
            raise Exception(output_msg)

    # Load the 1D extraction:
    wl, ext1d = load_spectrum1d(ext1d_output)
    cdelt = np.mean(np.diff(wl))

    # Load the spectroscopic standard table:
    # The files are located in 'calib/std/'
    std_tab = np.loadtxt(path+'/calib/std/%s.dat' % star.lower())
    msg.append("          - Loaded reference data for object: %s" % star)

    # Calculate the flux in the pass bands:
    msg.append("          - Calculating flux in reference band passes")
    wl0 = list()
    flux0 = list()
    mag = list()
    for l0, m0, b in std_tab:
        l1 = l0 - b/2.
        l2 = l0 + b/2.
        band = (wl >= l1) * (wl <= l2)
        if np.sum(band) > 3:
            f0 = np.nanmean(ext1d[band])
            if f0 > 0:
                flux0.append(f0)
                wl0.append(l0)
                mag.append(m0)
    wl0 = np.array(wl0)
    flux0 = np.array(flux0)
    mag = np.array(mag)

    # Median filter the points:
    msg.append("          - Median filter and smooth the data points to remove outliers")
    med_flux_tab = median_filter(flux0, 5)
    med_flux_tab = gaussian_filter1d(med_flux_tab, 1)
    noise = mad(flux0 - med_flux_tab)*1.5
    good = np.abs(flux0 - med_flux_tab) < 2*noise
    good[:3] = True
    good[-3:] = True

    # Load Extinction Table:
    wl_ext, A0 = np.loadtxt(instrument.extinction_fname, unpack=True)
    msg.append("          - Loaded average extinction table:")
    msg.append("            %s" % instrument.extinction_fname)
    ext = np.interp(wl0, wl_ext, A0)

    # Load Airmass and Exposure time:
    airmass = instrument.get_airmass(hdr)
    if airmass is None:
        user_input = input("          > Please give the airmass:\n          > ")
        try:
            airmass = float(user_input)
        except (ValueError) as e:
            msg.append(" [ERROR]  - Invalid airmass!")
            msg.append(" [ERROR]  - " + str(e))
            msg.append("")
            return "\n".join(msg)

    exptime = instrument.get_exptime(hdr)
    if exptime is None:
        user_input = input("          > Please give the exposure time:\n          > ")
        try:
            exptime = float(user_input)
        except ValueError:
            msg.append(" [ERROR]  - Invalid exposure time: %r" % user_input)
            msg.append("")
            return "\n".join(msg)
    msg.append("          - AIRMASS: %.2f" % airmass)
    msg.append("          - EXPTIME: %.1f" % exptime)

    # Convert AB magnitudes to fluxes (F-lambda):
    F = 10**(-(mag+2.406)/2.5) / (wl0)**2

    # Calculate Sensitivity:
    C = 2.5*np.log10(flux0 / (exptime * cdelt * F)) + airmass*ext

    if interactive:
        msg.append("          - ")
        msg.append("          - Starting Graphical User Interface...")
        try:
            response = response_gui.run_gui(ext1d_output, response_output,
                                            order=3, smoothing=0.02, app=app)
            msg.append(" [OUTPUT] - Saving the response function as FITS table: %s" % response_output)
        except:
            msg.append("Unexpected error in response GUI: %r" % sys.exc_info()[0])
            output_msg = "\n".join(msg)
            raise Exception(output_msg)
    else:
        # Fit a smooth polynomium to the calculated response:
        msg.append("          - Interpolating the filtered response curve data points")
        msg.append("          - Spline degree: %i" % order)
        msg.append("          - Smoothing factor: %.3f" % smoothing)
        response_fit = UnivariateSpline(wl0[good], C[good], k=order, s=smoothing)
        # response_fit = Chebyshev.fit(wl0[good], C[good], order, domain=[wl.min(), wl.max()])
        response = response_fit(wl)


    # -- Prepare PDF figure:
    if not pdf_fname:
        object_name = instrument.get_object(hdr)
        pdf_fname = 'response_diagnostic_%s.pdf' % object_name
        pdf_fname = os.path.join(output_dir, pdf_fname)
    pdf = backend_pdf.PdfPages(pdf_fname)

    # Plot the extracted spectrum
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.plot(wl, ext1d)
    ax.set_ylim(ymin=0.)
    power = np.floor(np.log10(np.nanmax(ext1d))) - 1
    majFormatter = ticker.FuncFormatter(lambda x, p: my_formatter(x, p, power))
    ax.get_yaxis().set_major_formatter(majFormatter)
    ax.set_ylabel(u'Counts  [$10^{{{0:d}}}$ ADU]'.format(int(power)), fontsize=14)
    ax.set_xlabel(u"Wavelength  [Å]", fontsize=14)
    ax.set_title(u"Filename: %s  ,  Star: %s" % (raw_fname, star.upper()))
    pdf.savefig(fig1)

    # Plot the response function:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(wl0, C, color='RoyalBlue', marker='o', ls='')
    ax2.plot(wl0[~good], C[~good], color='r', marker='o', ls='')
    ax2.set_ylabel(u"Response  ($F_{\\lambda}$)", fontsize=14)
    ax2.set_xlabel(u"Wavelength  (Å)", fontsize=14)
    ax2.set_title(u"Response function, grism: "+grism)
    ax2.plot(wl, response, color='crimson', lw=1)
    pdf.savefig(fig2)
    pdf.close()
    msg.append(" [OUTPUT] - Saving the response function diagnostics:  %s" % pdf_fname)

    if interactive:
        # The GUI saved the output already...
        pass
    else:
        # --- Prepare FITS output:
        resp_hdr = fits.Header()
        resp_hdr['AUTHOR'] = 'PyNOT version %s' % __version__
        resp_hdr['OBJECT'] = instrument.get_object(hdr)
        resp_hdr['DATE-OBS'] = instrument.get_date(hdr)
        resp_hdr['EXPTIME'] = exptime
        resp_hdr['AIRMASS'] = airmass
        resp_hdr['GRISM'] = grism
        resp_hdr['SLIT'] = instrument.get_slit(hdr)
        resp_hdr['RA'] = hdr['RA']
        resp_hdr['DEC'] = hdr['DEC']
        resp_hdr['STD-STAR'] = star
        resp_hdr['COMMENT'] = 'PyNOT response function'
        prim = fits.PrimaryHDU(header=resp_hdr)
        col_wl = fits.Column(name='WAVE', array=wl, format='D', unit='Angstrom')
        col_resp = fits.Column(name='RESPONSE', array=response, format='D', unit='-2.5*log(erg/s/cm2/A)')
        tab = fits.BinTableHDU.from_columns([col_wl, col_resp])
        hdu = fits.HDUList()
        hdu.append(prim)
        hdu.append(tab)
        hdu.writeto(response_output, overwrite=True)
        msg.append(" [OUTPUT] - Saving the response function as FITS table: %s" % response_output)
    msg.append("")
    output_msg = "\n".join(msg)
    return response_output, output_msg



def process_std_flux(raw_fname, *, arc_fname, pixtable_fname, bias_fname, flat_fname,
                     output_dir='', dispaxis=2, order_bg=5, rectify_options=None,
                     log=None, verbose=True):
    """
    Extract and wavelength calibrate the standard star spectrum.
    Calculate the instrumental response function and fit the median filtered data points
    with a Chebyshev polynomium of the given *order*.

    Parameters
    ==========
    raw_fname : string
        File name for the standard star frame to process

    arc_fname : string
        File name for the associated arc frame

    pixtable_fname : string
        Filename of pixel table for the identified lines in the arc frame

    bias_fname : string
        Master bias file name to subtract bias level.
        If nothing is given, no bias level correction is performed.

    flat_fname : string
        Normalized flat file name.
        If nothing is given, no spectral flat field correction is performed.

    output_dir : string  [default='']
        Output directory for the response function and intermediate files.
        Filenames are autogenerated from OBJECT name and GRISM

    dispaxis : integer  [default=2]
        Dispersion axis. 1: horizontal spectra, 2: vertical spectra (default for most ALFOSC grisms)

    order_bg : integer  [default=5]
        Polynomial order for background subtraction

    rectify_options : dict()  [default={}]
        Dictionary of keyword arguments for `rectify`

    log : pynot.reporting.Report  [default=None]
        A log of instance pynot.reporting.Report

    Returns
    -------
    bgsub2d_fname : string
        Filename of the processed frame

    log : pynot.reporting.Report
        The updated log
    """
    if log is None:
        log = Report(verbose)

    if rectify_options is None:
        rectify_options = dict()
    else:
        rectify_options = rectify_options.copy()

    hdr = instrument.get_header(raw_fname)
    raw2D = fits.getdata(raw_fname)
    log.write("Loaded flux standard image: %s" % raw_fname)

    ob_id = instrument.get_date(hdr).split('.')[0]
    corr2d_fname = os.path.join(output_dir, 'std_corr2D_%s.fits' % ob_id)
    rect2d_fname = os.path.join(output_dir, 'std_rect2D_%s.fits' % ob_id)
    bgsub2d_fname = os.path.join(output_dir, 'std_bgsub2D_%s.fits' % ob_id)

    try:
        output_msg = raw_correction(raw2D, hdr, bias_fname, flat_fname,
                                    output=corr2d_fname, overwrite=True)
        log.commit(output_msg)
        log.add_linebreak()
    except:
        log.error("Unexpected error in raw correction: %r" % sys.exc_info()[0])
        raise

    # Rectify 2D image and wavelength calibrate:
    try:
        rectify_options['plot'] = True
        rect_msg = rectify(corr2d_fname, arc_fname, pixtable_fname, output=rect2d_fname,
                           dispaxis=dispaxis, fig_dir=output_dir, **rectify_options)
        log.commit(rect_msg)
        log.add_linebreak()
    except:
        log.error("Unexpected error in rectify: %r" % sys.exc_info()[0])
        raise
    # After RECTIFY all images are oriented with the dispersion axis horizontally

    # Subtract background:
    try:
        bg_msg = auto_fit_background(rect2d_fname, bgsub2d_fname, dispaxis=1, order_bg=order_bg, plot_fname='',
                                     kappa=100, fwhm_scale=5)
        log.commit(bg_msg)
        log.add_linebreak()
    except:
        log.error("Unexpected error in auto sky sub: %r" % sys.exc_info()[0])
        raise

    return bgsub2d_fname, log


def task_response(options, database, status, log=None, verbose=True, app=None, output_dir='', **kwargs):
    """
    Reduce the standard stars and determine the response function
    """
    if log is None:
        log = Report(verbose)
    log.add_linebreak()
    log.write("Running task: Response function")

    if len(database['SPEC_FLUX-STD']) == 0:
        log.warn("No spectroscopic standard star was found in the dataset!")
        log.warn("The reduced spectra will not be flux calibrated")
        return None

    tag = 'RESPONSE'
    task_output = {tag: []}
    input_files = database.get_files('SPEC_FLUX-STD', **kwargs)
    date = 'date' in kwargs
    flux_std_files = organizer.sort_std(input_files, date=date)
    for target_name, files_per_target in flux_std_files.items():
        for insID, input_list in files_per_target.items():
            output_id = options.pop('output', '')
            if output_id:
                insID += '_'+output_id
            raw_img = organizer.RawImage(input_list[0])
            hdr = instrument.get_header(input_list[0])

            # Check if reference data exists:
            grism = raw_img.grism
            star = lookup_std_star(hdr)
            if star is None:
                log.warn("No reference data found for target: %s" % target_name)
                log.warn("The response function cannot be calculated for this target")
                continue

            ob_dir = os.path.join(output_dir, star+'_'+insID)
            if not os.path.exists(ob_dir):
                os.makedirs(ob_dir)

            master_bias = organizer.match_single_calib(raw_img, database, 'MBIAS', log, date=False)
            norm_flat = organizer.match_single_calib(raw_img, database, 'NORM_SFLAT', log, date=False,
                                                     grism=True, slit=True, filter=True)
            arc_fname = organizer.match_single_calib(raw_img, database, 'ARC_CORR', log, date=False,
                                                     grism=True, slit=False, filter=True,
                                                     get_closest_time=True)
            pixtab_fnames = status.find_pixtab(grism)
            pixtable = pixtab_fnames[0]

            ext1d_output = os.path.join(ob_dir, '%s_%s_1d.fits' % (star, insID))
            extract_pdf_fname = os.path.join(ob_dir, '%s_%s_extract.pdf' % (star, insID))
            response_output = os.path.join(output_dir, 'response_%s_%s.fits' % (star, insID))

            files_to_combine = list()
            for std_fname in input_list:
                bgsub2d_fname, log = process_std_flux(std_fname, arc_fname=arc_fname,
                                                      pixtable_fname=pixtable,
                                                      bias_fname=master_bias,
                                                      flat_fname=norm_flat,
                                                      output_dir=ob_dir,
                                                      dispaxis=raw_img.dispaxis,
                                                      order_bg=options['skysub']['order_bg'],
                                                      rectify_options=options['rectify'],
                                                      log=log, verbose=verbose)
                files_to_combine.append(bgsub2d_fname)

            if len(files_to_combine) > 1:
                comb_basename = '%s_%s_comb2d.fits' % (star, insID)
                comb2d_fname = os.path.join(ob_dir, comb_basename)
                log.write("Running task: Spectral Combination")
                try:
                    comb_output = combine_2d(files_to_combine, comb2d_fname)
                    _, _, _, _, output_msg = comb_output
                    log.commit(output_msg)
                    log.add_linebreak()
                except Exception:
                    log.warn("Combination of 2D spectra failed... Try again manually")
                    raise
            else:
                comb_output = files_to_combine[0]

            # Extract 1-dimensional spectrum:
            if options['response']['interactive']:
                try:
                    log.write("Starting Graphical User Interface for Spectral Extraction of Flux Standard Star")
                    extract_gui.run_gui(bgsub2d_fname, output_fname=ext1d_output,
                                        app=app, order_center=5, order_width=5, dx=20)
                    log.write("Writing fits table: %s" % ext1d_output, prefix=" [OUTPUT] - ")
                except:
                    log.error("Unexpected error in extract GUI: %r" % sys.exc_info()[0])
                    raise
            else:
                try:
                    ext_msg = auto_extract(bgsub2d_fname, ext1d_output, dispaxis=1, N=1, pdf_fname=extract_pdf_fname,
                                           model_name='moffat', dx=20, order_center=4, order_width=5, xmin=20, ymin=5, ymax=-5,
                                           kappa_cen=5., w_cen=15)
                    log.commit(ext_msg)
                    log.add_linebreak()
                except:
                    log.error("Unexpected error in auto extract: %r" % sys.exc_info()[0])
                    raise

            # Load the 1D extraction:
            wl, ext1d = load_spectrum1d(ext1d_output)
            cdelt = np.mean(np.diff(wl))

            # Load the spectroscopic standard table:
            # The files are located in 'calib/std/'
            std_tab = np.loadtxt(path+'/calib/std/%s.dat' % star.lower())
            log.write("Loaded reference data for object: %s" % star)

            # Calculate the flux in the pass bands:
            log.write("Calculating flux in reference band passes")
            wl0 = list()
            flux0 = list()
            mag = list()
            for l0, m0, b in std_tab:
                l1 = l0 - b/2.
                l2 = l0 + b/2.
                band = (wl >= l1) * (wl <= l2)
                if np.sum(band) > 3:
                    f0 = np.nanmean(ext1d[band])
                    if f0 > 0:
                        flux0.append(f0)
                        wl0.append(l0)
                        mag.append(m0)
            wl0 = np.array(wl0)
            flux0 = np.array(flux0)
            mag = np.array(mag)

            # Median filter the points:
            log.write("Median filter and smooth the data points to remove outliers")
            med_flux_tab = median_filter(flux0, options['response']['med_filter'])
            med_flux_tab = gaussian_filter1d(med_flux_tab, 1)
            noise = mad(flux0 - med_flux_tab)*1.5
            good = np.abs(flux0 - med_flux_tab) < options['response']['kappa']*noise
            # good[:3] = True
            # good[-3:] = True

            # Load Extinction Table:
            wl_ext, A0 = np.loadtxt(instrument.extinction_fname, unpack=True)
            log.write("Loaded average extinction table:")
            log.write("%s" % instrument.extinction_fname)
            ext = np.interp(wl0, wl_ext, A0)

            # Load Airmass and Exposure time:
            airmass = instrument.get_airmass(hdr)
            if airmass is None:
                user_input = input("          > Please give the airmass:\n          > ")
                try:
                    airmass = float(user_input)
                except (ValueError) as e:
                    log.error("Invalid airmass!")
                    log.error(str(e))
                    log.add_linebreak()
                    log.fatal_error()
                    raise

            exptime = instrument.get_exptime(hdr)
            if exptime is None:
                user_input = input("          > Please give the exposure time:\n          > ")
                try:
                    exptime = float(user_input)
                except ValueError:
                    log.error("Invalid exposure time: %r" % user_input)
                    log.add_linebreak()
                    log.fatal_error()
                    raise
            log.write("AIRMASS: %.2f" % airmass)
            log.write("EXPTIME: %.1f" % exptime)

            # Convert AB magnitudes to fluxes (F-lambda):
            F = 10**(-(mag+2.406)/2.5) / (wl0)**2

            # Calculate Sensitivity:
            C = 2.5*np.log10(flux0 / (exptime * cdelt * F)) + airmass*ext

            if options['response']['interactive']:
                log.write("")
                log.write("Starting Graphical User Interface...")
                try:
                    response = response_gui.run_gui(ext1d_output, response_output,
                                                    order=3, smoothing=0.02, app=app)
                    log.write(" [OUTPUT] - Saving response function: %s" % response_output, prefix='')
                except:
                    log.error("Unexpected error in response GUI: %r" % sys.exc_info()[0])
                    raise
            else:
                # Fit a smooth polynomium to the calculated response:
                log.write("Interpolating the filtered response curve data points")
                log.write("Spline degree: %i" % options['response']['order'])
                log.write("Smoothing factor: %.3f" % options['response']['smoothing'])
                response_fit = UnivariateSpline(wl0[good], C[good],
                                                k=options['response']['order'],
                                                s=options['response']['smoothing'])
                # response_fit = Chebyshev.fit(wl0[good], C[good], order, domain=[wl.min(), wl.max()])
                response = response_fit(wl)

                # --- Prepare FITS output:
                resp_hdr = fits.Header()
                resp_hdr['AUTHOR'] = 'PyNOT version %s' % __version__
                resp_hdr['INSTRUME'] = 'PyNOT'
                resp_hdr['OBJECT'] = instrument.get_object(hdr)
                resp_hdr['DATE-OBS'] = instrument.get_date(hdr)
                resp_hdr['EXPTIME'] = exptime
                resp_hdr['AIRMASS'] = airmass
                resp_hdr['GRISM'] = grism
                resp_hdr['SLIT'] = instrument.get_slit(hdr)
                resp_hdr['RA'] = hdr['RA']
                resp_hdr['DEC'] = hdr['DEC']
                resp_hdr['STD-STAR'] = star
                resp_hdr['COMMENT'] = 'PyNOT response function'
                prim = fits.PrimaryHDU(header=resp_hdr)
                col_wl = fits.Column(name='WAVE', array=wl, format='D', unit='Angstrom')
                col_resp = fits.Column(name='RESPONSE', array=response, format='D', unit='-2.5*log(erg/s/cm2/A)')
                tab = fits.BinTableHDU.from_columns([col_wl, col_resp])
                hdu = fits.HDUList()
                hdu.append(prim)
                hdu.append(tab)
                hdu.writeto(response_output, overwrite=True)
                log.write(" [OUTPUT] - Saving response function: %s" % response_output, prefix='')

            # -- Prepare PDF figure:
            pdf_fname = 'response_diagnostic.pdf'
            pdf_fname = os.path.join(ob_dir, pdf_fname)
            pdf = backend_pdf.PdfPages(pdf_fname)

            # Plot the extracted spectrum
            fig1 = plt.figure()
            ax = fig1.add_subplot(111)
            ax.plot(wl, ext1d)
            ax.set_ylim(ymin=0.)
            power = np.floor(np.log10(np.nanmax(ext1d))) - 1
            majFormatter = ticker.FuncFormatter(lambda x, p: my_formatter(x, p, power))
            ax.get_yaxis().set_major_formatter(majFormatter)
            ax.set_ylabel(u'Counts  [$10^{{{0:d}}}$ ADU]'.format(int(power)), fontsize=14)
            ax.set_xlabel(u"Wavelength  [Å]", fontsize=14)
            ax.set_title(u"Filename: %s  ,  Star: %s" % (ext1d_output, star.upper()))
            pdf.savefig(fig1)

            # Plot the response function:
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            ax2.plot(wl0, C, color='RoyalBlue', marker='o', ls='')
            ax2.plot(wl0[~good], C[~good], color='r', marker='o', ls='')
            ax2.set_ylabel(u"Response  ($F_{\\lambda}$)", fontsize=14)
            ax2.set_xlabel(u"Wavelength  (Å)", fontsize=14)
            ax2.set_title(u"Response function, grism: "+grism)
            ax2.plot(wl, response, color='crimson', lw=1)
            pdf.savefig(fig2)
            pdf.close()
            log.write(" [OUTPUT] - Saving the response function diagnostics:  %s" % pdf_fname, prefix='')
            log.add_linebreak()
            task_output[tag].append(response_output)
    return task_output, log
