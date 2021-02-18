# PyNOT/response
"""
Calculate the instrumental response function
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
from numpy.polynomial import Chebyshev
import os
import sys
import warnings

from pynot import alfosc
from pynot.alfosc import get_alfosc_header
from pynot.extraction import auto_extract
from pynot import extract_gui
from pynot.functions import get_version_number, my_formatter, mad
from pynot import response_gui
from pynot.scired import auto_fit_background, raw_correction
from pynot.wavecal import rectify


__version__ = get_version_number()


def load_spectrum1d(fname):
    table = fits.getdata(fname)
    wl = table['WAVE']
    flux = table['FLUX']
    return wl, flux


def flux_calibrate(input_fname, *, output, response):
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
    wl_ext, A0 = np.loadtxt(alfosc.path + '/calib/lapalma.ext', unpack=True)
    ext = np.interp(wl, wl_ext, A0)
    msg.append("          - Loaded average extinction table for La Palma")

    # Load Sensitivity Function:
    resp_tab = fits.getdata(response)
    resp_hdr = fits.getheader(response)
    if resp_hdr['ALGRNM'] != hdr['ALGRNM']:
        msg.append(" [ERROR]  - Grisms of input spectrum and response function do not match!")
        msg.append("")
        output_msg = "\n".join(msg)
        return output_msg

    resp_int = np.interp(wl, resp_tab['WAVE'], resp_tab['RESPONSE'])
    # Truncate values less than 20:
    resp_int[resp_int < 20] = 20.
    msg.append("          - Loaded response function: %s" % response)

    airm = hdr['AIRMASS']
    t = hdr['EXPTIME']
    # ext_correction = 10**(0.4*airm * ext)
    # flux_calibration = ext_correction / 10**(0.4*resp_int)
    flux_calibration = 10**(0.4*(airm*ext - resp_int))
    flux_calib2D = np.resize(flux_calibration, img2D.shape)
    flux2D = img2D / (t * cdelt) * flux_calib2D
    err2D = err2D / (t * cdelt) * flux_calib2D

    with fits.open(input_fname) as hdu:
        hdu[0].data = flux2D
        hdu[0].header['BUNIT'] = 'erg/s/cm2/A'
        hdu[0].header['RESPONSE'] = response

        hdu['ERR'].data = err2D
        hdu['ERR'].header['BUNIT'] = 'erg/s/cm2/A'

        hdu.writeto(output, overwrite=True)
    msg.append(" [OUTPUT] - Saving flux calibrated 2D image: %s" % output)
    msg.append("")
    output_msg = "\n".join(msg)
    return output_msg


def flux_calibrate_1d(input_fname, *, output, response):
    """Apply response function to flux calibrate the input 1D spectrum"""
    msg = list()

    # Load Extinction Table:
    wl_ext, A0 = np.loadtxt(alfosc.path + '/calib/lapalma.ext', unpack=True)
    msg.append("          - Loaded average extinction table for La Palma")

    # Load Sensitivity Function:
    resp_tab = fits.getdata(response)
    resp_hdr = fits.getheader(response)
    msg.append("          - Loaded response function: %s" % response)

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

        if resp_hdr['ALGRNM'] != hdr['ALGRNM']:
            msg.append(" [ERROR]  - Grisms of input spectrum and response function do not match!")
            msg.append("")
            output_msg = "\n".join(msg)
            return output_msg

        # Interpolate extinction table and response function:
        ext = np.interp(wl, wl_ext, A0)
        resp_int = np.interp(wl, resp_tab['WAVE'], resp_tab['RESPONSE'])
        # Truncate values less than 20:
        resp_int[resp_int < 20] = 20.

        airm = hdr['AIRMASS']
        t = hdr['EXPTIME']
        cdelt = np.mean(np.diff(wl))
        flux_calibration = 10**(0.4*(airm*ext - resp_int))
        flux1d = spec1d / (t * cdelt) * flux_calibration
        err1d = err1d / (t * cdelt) * flux_calibration

        hdr['BUNIT'] = 'erg/s/cm2/A'
        hdr['RESPONSE'] = response
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
                       order_wl=4, order_bg=5, rectify_options={}, app=None):
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

    order_wl : integer  [default=4]
        Polynomial order for wavelength solution as function of pixel value (from `identify`)

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

    hdr = get_alfosc_header(raw_fname)
    raw2D = fits.getdata(raw_fname)
    msg.append("          - Loaded flux standard image: %s" % raw_fname)

    # Setup the filenames:
    grism = alfosc.grism_translate[hdr['ALGRNM']]
    star = hdr['TCSTGT']
    # Check if the star name is in the header:
    star = alfosc.lookup_std_star(star)
    if star is None:
        msg.append("[WARNING] - No reference data found for the star %s (TCS Target Name)" % hdr['TCSTGT'])
        msg.append("[WARNING] - The reduced spectra will not be flux calibrated")
        output_msg = "\n".join(msg)
        return None, output_msg

    response_output = 'response_%s_%s.fits' % (star, grism)
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
                                    output=std_tmp_fname, overwrite=True, overscan=50)
        msg.append(output_msg)
    except:
        msg.append("Unexpected error: %r" % sys.exc_info()[0])
        output_msg = "\n".join(msg)
        raise Exception(output_msg)

    # Rectify 2D image and wavelength calibrate:
    try:
        rectify_options['plot'] = False
        rect_msg = rectify(std_tmp_fname, arc_fname, pixtable_fname, output=rect2d_fname,
                           dispaxis=dispaxis, order_wl=order_wl, **rectify_options)
        msg.append(rect_msg)
    except:
        msg.append("Unexpected error: %r" % sys.exc_info()[0])
        output_msg = "\n".join(msg)
        raise Exception(output_msg)
    # After RECTIFY all images are oriented with the dispersion axis horizontally


    # Subtract background:
    try:
        bg_msg = auto_fit_background(rect2d_fname, bgsub2d_fname, dispaxis=1, order_bg=order_bg, plot_fname='',
                                     kappa=100, fwhm_scale=5)
        msg.append(bg_msg)
    except:
        msg.append("Unexpected error: %r" % sys.exc_info()[0])
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
            msg.append("Unexpected error: %r" % sys.exc_info()[0])
            output_msg = "\n".join(msg)
            raise Exception(output_msg)
    else:
        try:
            ext_msg = auto_extract(bgsub2d_fname, ext1d_output, dispaxis=1, N=1, pdf_fname=extract_pdf_fname,
                                   model_name='moffat', dx=20, order_center=4, order_width=5, xmin=20, ymin=5, ymax=-5,
                                   kappa_cen=5., w_cen=15)
            msg.append(ext_msg)
        except:
            msg.append("Unexpected error: %r" % sys.exc_info()[0])
            output_msg = "\n".join(msg)
            raise Exception(output_msg)

    # Load the 1D extraction:
    wl, ext1d = load_spectrum1d(ext1d_output)
    cdelt = np.mean(np.diff(wl))

    # Load the spectroscopic standard table:
    # The files are located in 'calib/std/'
    star_name = alfosc.standard_star_names[star]
    std_tab = np.loadtxt(alfosc.path+'/calib/std/%s.dat' % star_name.lower())
    msg.append("          - Loaded reference data for object: %s" % star_name)

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

    # Load extinction table:
    msg.append("          - Loaded the average extinction data for La Palma")
    wl_ext, A0 = np.loadtxt(alfosc.path + '/calib/lapalma.ext', unpack=True)
    ext = np.interp(wl0, wl_ext, A0)
    exptime = hdr['EXPTIME']
    airmass = hdr['AIRMASS']
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
            msg.append("Unexpected error: %r" % sys.exc_info()[0])
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
        pdf_fname = 'response_diagnostic_' + hdr['OBJECT'] + '.pdf'
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
    ax2.set_title(u"Response function, grism: "+hdr['ALGRNM'])
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
        resp_hdr['GRISM'] = grism
        resp_hdr['OBJECT'] = hdr['OBJECT']
        resp_hdr['DATE-OBS'] = hdr['DATE-OBS']
        resp_hdr['EXPTIME'] = hdr['EXPTIME']
        resp_hdr['AIRMASS'] = hdr['AIRMASS']
        resp_hdr['ALGRNM'] = hdr['ALGRNM']
        resp_hdr['ALAPRTNM'] = hdr['ALAPRTNM']
        resp_hdr['RA'] = hdr['RA']
        resp_hdr['DEC'] = hdr['DEC']
        resp_hdr['COMMENT'] = 'PyNOT response function'
        resp_hdr['AUTHOR'] = 'PyNOT version %s' % __version__
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


#
# def run_response():
#     parser = ArgumentParser()
#     parser.add_argument("input", type=str,
#                         help="Raw flux standard star frame")
#     parser.add_argument("arc", type=str,
#                         help="Raw arc lamp frame")
#     parser.add_argument("--bias", type=str,
#                         help="Combined bias frame")
#     parser.add_argument("--flat", type=str,
#                         help="Normalized spectral flat frame")
#     parser.add_argument("-o", "--output", type=str, default='',
#                         help="Filename of output response function")
#     parser.add_argument("-d", "--dir", type=str, default='',
#                         help="Output directory, default='./'")
#     parser.add_argument("-O", "--options", type=str, default='',
#                         help="Option file (.yml)")
#     parser.add_argument("-I", "--interactive", action='store_true',
#                         help="Interactive mode")
#
#     args = parser.parse_args()
#
#     from functions import get_options
#     code_dir = os.path.dirname(os.path.abspath(__file__))
#     calib_dir = os.path.join(code_dir, 'calib/')
#     defaults_fname = os.path.join(calib_dir, 'default_options.yml')
#     options = get_options(defaults_fname)
#     if args.options:
#         user_options = get_options(args.options)
#         for section_name, section in user_options.items():
#             if isinstance(section, dict):
#                 options[section_name].update(section)
#             else:
#                 options[section_name] = section
#
#     options['response']['interactive'] = args.interactive
#
#     _, output_msg = calculate_response(args.input, output=args.output, arc_fname=args.arc, pixtable_fname=args.pixtable,
#                                        bias_fname=args.bias, flat_fname=args.flat,
#                                        output_dir=args.dir, order=args.order, smoothing=args.smooth,
#                                        interactive=args.int, dispaxis=args.axis,
#                                        order_wl=args.order_wl, order_bg=5, rectify_options=options['rectify'])
#     print(output_msg)
