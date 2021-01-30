# -*- coding: UTF-8 -*-
"""
Script to combine bias and spectral flat frames for use in final data reduction.
"""
__author__ = 'Jens-Kristian Krogager'
__email__ = "krogager@iap.fr"
__credits__ = ["Jens-Kristian Krogager"]

from argparse import ArgumentParser
import numpy as np
try:
    import pyfits as pf
except:
    import astropy.io.fits as pf
from matplotlib.backends import backend_pdf
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.ndimage import gaussian_filter1d, median_filter
from numpy.polynomial import Chebyshev
import os
from os.path import isfile, exists
import warnings

import PyNOT
import alfosc
from extraction import extract
from alfosc import get_alfosc_header

code_dir = os.path.dirname(os.path.abspath(__file__))
v_file = os.path.join(code_dir, 'VERSION')
with open(v_file) as version_file:
    __version__ = version_file.read().strip()


def sensitivity(raw_frame, arc_frame, bias_fname, flat_fname, output='', pdf_filename='', order=8):
    """
    Extract and wavelength calibrate the standard star spectrum.
    Calculate the sensitivity function and fit the median filtered sensitivity
    with a Chebyshev polynomium of the given *order*.

    Parameters
    ==========
    raw_frame : string
        File name for the standard star frame

    arc_frame : string
        File name for the associated arc frame

    bias_fname : string
        Master bias file name to subtract bias level.
        If nothing is given, no bias level correction is performed.

    flat_fname : string
        Normalized flat file name.
        If nothing is given, no spectral flat field correction is performed.

    output : string  [default='']
        Output file name for the sensitivity function.
        If none, autogenerate from OBJECT name

    pdf_fname : string  [default='']
        Output filename for diagnostic plots.
        If none, autogenerate from OBJECT name

    order : integer  [default=8]
        Order of the Chebyshev polynomium to fit the sensitivity

    Returns
    =======
    output : string
        Filename of resulting sensitivity function

    output_msg : string
        Log of the function call
    """
    msg = list
    msg.append("          - Running task: bias and flat field correction")
    mbias = pf.getdata(bias_fname)
    msg.append("          - Loaded BIAS image: %s" % bias_fname)
    mflat = pf.getdata(flat_fname)
    msg.append("          - Loaded FLAT field image: %s" % flat_fname)

    hdr = get_alfosc_header(raw_frame)
    raw2D = pf.getdata(raw_frame)

    # Update gain for the new CCD, wrong gain written in header from the instrument
    if hdr['CCDNAME'] == 'CCD14':
        hdr['GAIN'] = 0.16
    std = (raw2D - mbias)/mflat

    hdr.add_comment('PyNOT version %s' % __version__)
    hdr.add_comment("BIAS subtract: %s" % bias_fname)
    hdr.add_comment("FLAT correction: %s" % flat_fname)

    if not pdf_filename:
        pdf_filename = 'sens_diagnostic_' + hdr['OBJECT'] + '.pdf'
    pdf = backend_pdf.PdfPages(pdf_filename)

    # Save the updated, bias subtracted and flat-fielded frame
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf.writeto('std_tmp.fits', std, header=hdr, clobber=True)

    # Extract 1-dimensional spectrum and subtract background
    wl, ext1d, err1d = extract('std_tmp.fits', arc_frame, do_opt_extract=False, interact=False,
                               background=True, FWHM0=30)
    os.system("rm std_tmp.fits")

    # Check if the star name is in the header:
    if hdr['TCSTGT'] in alfosc.standard_stars:
        star = hdr['TCSTGT']
    else:
        print("")
        print("  No valid star name found in the header!  TCS Target Name =  %s" % hdr['TCSTGT'])
        print("  Check if the star is listed in the directory 'calib/std'.")
        print("")


    # Plot the extracted spectrum
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.plot(wl, ext1d)
    ax.set_ylim(ymin=0.)
    power = np.floor(np.log10(np.max(ext1d))) - 1
    majFormatter = ticker.FuncFormatter(lambda x, p: PyNOT.my_formatter(x, p, power))
    ax.get_yaxis().set_major_formatter(majFormatter)
    ax.set_ylabel(u'Counts  [$10^{{{0:d}}}$ ADU]'.format(int(power)), fontsize=16)
    ax.set_xlabel(u"Wavelength  [Å]", fontsize=16)
    ax.set_title(u"Filename: %s  ,  Star: %s" % (raw_frame, star.upper()))
    pdf.savefig(fig1)

    # Load the spectroscopic standard table:
    # The files are located in 'calib/std/'
    std_tab = np.loadtxt(alfosc.path+'/calib/std/%s.dat' % star.lower())

    # Calculate the flux in the pass bands:
    wl0 = list()
    flux0 = list()
    mag = list()
    for l0, m0, b in std_tab:
        l1 = l0 - b/2.
        l2 = l0 + b/2.
        band = (wl >= l1) * (wl <= l2)
        f0 = np.sum(ext1d[band])
        if not np.isnan(f0) and f0 > 0.:
            flux0.append(f0/b)
            wl0.append(l0)
            mag.append(m0)
    wl0 = np.array(wl0)
    flux0 = np.array(flux0)
    mag = np.array(mag)

    # Median filter the points:
    med_flux_tab = median_filter(flux0, 3)
    med_flux_tab = gaussian_filter1d(med_flux_tab, 1)
    noise = PyNOT.mad(flux0 - med_flux_tab)*1.5
    good = np.abs(flux0 - med_flux_tab) < 4*noise

    # Load extinction table:
    wl_ext, A0 = np.loadtxt(alfosc.path + '/calib/lapalma.ext', unpack=True)
    ext = np.interp(wl0, wl_ext, A0)
    exptime = hdr['EXPTIME']
    airmass = hdr['AIRMASS']

    # Convert AB magnitudes to fluxes (F-lambda):
    F = 10**(-(mag+2.406)/2.5) / (wl0)**2

    # Calculate Sensitivity:
    C = 2.5*np.log10(flux0 / (exptime * F)) + airmass*ext

    # Fit a smooth polynomium to the calculated sensitivity:
    S = Chebyshev.fit(wl0[good], C[good], order)
    sens = S(wl)

    # Plot the sensitivity function:
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(wl0, C, color='RoyalBlue', marker='o', ls='')
    ax2.plot(wl0[~good], C[~good], color='r', marker='o', ls='')
    ax2.set_ylabel(u"Sensitivity  ($F_{\\lambda}$)", fontsize=16)
    ax2.set_xlabel(u"Wavelength  (Å)", fontsize=16)
    ax2.set_title(u"Sensitivity function, grism: "+hdr['ALGRNM'])
    ax2.plot(wl, sens, color='crimson', lw=1)
    pdf.savefig(fig2)
    pdf.close()
    print("  Details from extraction are saved to file:  %s" % pdf_filename)

    # --- Prepare output HDULists:
    dl = np.diff(wl)[0]
    hdr['CD1_1'] = dl
    hdr['CDELT1'] = dl
    hdr['CRVAL1'] = wl.min()
    ext0 = pf.PrimaryHDU(sens, header=hdr)
    HDU1D = pf.HDUList([ext0])
    if output:
        if output[-4:] == '.fits':
            output_fname1D = output
        else:
            output_fname1D = output + '.fits'
    else:
        grism = alfosc.grism_translate[hdr['ALGRNM']]
        output_fname1D = 'sens_%s.fits' % grism

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        HDU1D.writeto(output_fname1D, clobber=True)
    print("\n  Saved 1D extracted spectrum to file:  %s" % output_fname1D)
    print("")

    return wl, sens


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("input", type=str,
                        help="Raw flux standard frame")
    parser.add_argument("arc", type=str,
                        help="Raw arc lamp frame")
    parser.add_argument("--bias", type=str, default='MASTER_BIAS.fits',
                        help="Master bias frame")
    parser.add_argument("--flat", type=str, default='',
                        help="Normalized spectral flat frame")
    parser.add_argument("-o", "--output", type=str, default='',
                        help="Output filename, autogenerated by default")
    parser.add_argument("--order", type=int, default=24,
                        help="Polynomial order for fit to response function")
    # parser.add_argument("--axis", type=int, default=1,
    #                     help="Dispersion axis, 0: horizontal, 1: vertical")
    parser.add_argument("-x", "--xrange", nargs=2, type=int, default=[None, None],
                        help="Give lower and upper limit for the x-range to use")
    parser.add_argument("-y", "--yrange", nargs=2, type=int, default=[None, None],
                        help="Give lower and upper limit for the y-range to use")
    # parser.add_argument("-v", "--verbose", action="store_true",
    #                     help="Print status updates")
    args = parser.parse_args()

    # --- Generate sensitivity function:
    sensitivity(args.input, args.arc, output=args.output, bias=args.bias, flat=args.flat,
                trimx=args.xrange, trimy=args.yrange, order=args.order)
