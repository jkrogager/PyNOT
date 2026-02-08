import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy import signal
from numpy.polynomial import Chebyshev

from pynot.functions import mad, get_version_number
from pynot import instrument


__version__ = get_version_number()


def detect_objects_in_slit(x, spsf, fwhm_scale=1, obj_kappa=20):
    noise = 1.4826 * mad(spsf)
    peaks, properties = signal.find_peaks(spsf, prominence=obj_kappa*noise, width=3)
    object_mask = np.ones(len(spsf), dtype=bool)
    for num, center in enumerate(peaks):
        width = properties['widths'][num]
        x1 = center - width*fwhm_scale
        x2 = center + width*fwhm_scale
        obj = (x >= x1) * (x <= x2)
        object_mask &= ~obj
    return object_mask


def fit_background_row(x, row, mask=None, order_bg=3, med_kernel=15, kappa=5):
    if mask is None:
        mask = np.ones(len(row), dtype=bool)

    # Median filter the data to remove outliers:
    row = row.astype(np.float64)
    med_row = median_filter(row, med_kernel)
    noise = mad(row) * 1.4826
    this_mask = mask * (np.abs(row - med_row) < kappa*noise)
    if np.sum(this_mask) > order_bg+1:
        best_fit = Chebyshev.fit(x[this_mask], row[this_mask], order_bg, domain=[x.min(), x.max()])
        bg_model = best_fit(x)
    else:
        bg_model = np.zeros_like(row)
    return bg_model, this_mask


def fit_background_image(data, order_bg=3, xmin=0, xmax=None, med_kernel=15, kappa=5, fwhm_scale=1, obj_kappa=20):
    """
    Fit background in 2D spectral data. The background is fitted along the spatial rows by a Chebyshev polynomium.

    Parameters
    ==========
    data : np.array(M, N)
        Data array holding the input image

    order_bg : integer  [default=3]
        Order of the Chebyshev polynomium to fit the background

    xmin, xmax : integer  [default=0, None]
        Mask out pixels below xmin and above xmax

    fwhm_scale : float  [default=3]
        Number of FWHM below and above centroid of auto-detected trace
        that will be masked out during fitting.

    kappa : float  [default=5]
        Threshold for masking out cosmic rays etc.

    Returns
    =======
    bg2D : np.array(M, N)
        Background model of the 2D frame, same shape as input data.
    """
    x = np.arange(data.shape[1])
    if xmax is None:
        xmax = len(x)
    if xmax < 0:
        xmax = len(x) + xmax
    mask = (x >= xmin) & (x <= xmax)
    SPSF = np.nanmedian(data, 0)
    object_mask = detect_objects_in_slit(x, SPSF, fwhm_scale=fwhm_scale, obj_kappa=obj_kappa)
    mask &= object_mask
    N_masked_pixels = np.sum(~mask)

    bg2D = np.zeros_like(data)
    for i, row in enumerate(data):
        bg2D[i], _ = fit_background_row(x, row, mask=mask,
                                        order_bg=order_bg, med_kernel=med_kernel, kappa=kappa)

    return bg2D, N_masked_pixels


def auto_fit_background(data_fname, output_fname, dispaxis=2, order_bg=3, med_kernel=15, kappa=10,
                        obj_kappa=20, fwhm_scale=3, xmin=0, xmax=None, plot_fname='', **kwargs):
    """
    Fit background in 2D spectral data. The background is fitted along the spatial rows by a Chebyshev polynomium.

    Parameters
    ==========
    data_fname : string
        Filename of the FITS image to process

    output_fname : string
        Filename of the output FITS image containing the background subtracted image
        as well as the background model in a separate extension.

    dispaxis : integer  [default=1]
        Dispersion axis, 1: horizontal spectra, 2: vertical spectra
        The function does not rotate the final image, only the intermediate image
        since it's faster to operate on rows than on columns.

    order_bg : integer  [default=3]
        Order of the Chebyshev polynomium to fit the background

    xmin, xmax : integer  [default=0, None]
        Mask out pixels below xmin and above xmax

    obj_kappa : float  [default=20]
        Threshold for automatic object detection to be masked out.

    fwhm_scale : float  [default=3]
        Number of FWHM below and above centroid of auto-detected trace
        that will be masked out during fitting.

    med_kernel : int  [default=15]
        Median filter width for defining masking of cosmic rays, CCD artefacts etc.

    kappa : float  [default=10]
        Threshold for masking out cosmic rays, CCD artefacts etc.

    plot_fname : string  [default='']
        Filename of diagnostic plots. If nothing is given, do not plot.

    kwargs : dict
        This has no effect. It's just to catch the unused options passed from the pipeline.

    Returns
    =======
    output_fname : string
        Background model of the 2D frame, same shape as input data.

    output_msg : string
        Log of messages from the function call
    """
    msg = list()
    data = fits.getdata(data_fname)
    hdr = instrument.get_header(data_fname)
    if 'DISPAXIS' in hdr:
        dispaxis = hdr['DISPAXIS']

    if dispaxis == 1:
        # transpose the horizontal spectra to make them vertical
        # since it's faster to fit rows than columns
        data = data.T
    msg.append("          - Loaded input image: %s" % data_fname)

    msg.append("          - Fitting background along the spatial axis with polynomium of order: %i" % order_bg)
    msg.append("          - Automatic masking of outlying pixels and object trace")
    bg2D, N_masked_pixels = fit_background_image(data, order_bg=order_bg,
                                                 med_kernel=med_kernel, kappa=kappa,
                                                 obj_kappa=obj_kappa, fwhm_scale=fwhm_scale,
                                                 xmin=xmin, xmax=xmax)
    msg.append("          - Number of pixels rejected: %i" % int(N_masked_pixels))

    if plot_fname:
        fig2D = plt.figure()
        ax1_2d = fig2D.add_subplot(121)
        ax2_2d = fig2D.add_subplot(122)
        noise = mad(data)
        v1 = np.median(data) - 5*noise
        v2 = np.median(data) + 5*noise
        ax1_2d.imshow(data, origin='lower', vmin=v1, vmax=v2, aspect='auto')
        ax1_2d.set_title("Input Image")
        ax2_2d.imshow(data-bg2D, origin='lower', vmin=v1, vmax=v2, aspect='auto')
        ax2_2d.set_title("Background Subtracted")
        ax1_2d.set_xlabel("Spatial Axis  [pixels]")
        ax2_2d.set_xlabel("Spatial Axis  [pixels]")
        ax1_2d.set_ylabel("Dispersion Axis  [pixels]")
        fig2D.tight_layout()
        fig2D.savefig(plot_fname)
        plt.close()
        msg.append(" [OUTPUT] - Saving diagnostic figure: %s" % plot_fname)

    data = data - bg2D
    if dispaxis == 1:
        # Rotate the model and data back to horizontal orientation:
        data = data.T
        bg2D = bg2D.T

    with fits.open(data_fname) as hdu:
        hdu[0].data = data
        sky_hdr = fits.Header()
        sky_hdr['BUNIT'] = 'count'
        copy_keywords = ['CRPIX1', 'CRVAL1', 'CDELT1', 'CTYPE1', 'CUNIT1']
        copy_keywords += ['CRPIX2', 'CRVAL2', 'CDELT2', 'CD1_1', 'CD2_2', 'CD1_2', 'CD2_1']
        sky_hdr['CTYPE2'] = 'LINEAR'
        sky_hdr['CUNIT2'] = 'Pixel'
        for key in copy_keywords:
            if key in hdr:
                sky_hdr[key] = hdr[key]
        sky_hdr['AUTHOR'] = 'PyNOT version %s' % __version__
        sky_hdr['ORDER'] = (order_bg, "Polynomial order along spatial rows")
        sky_ext = fits.ImageHDU(bg2D, header=sky_hdr, name='SKY')
        hdu.append(sky_ext)
        hdu.writeto(output_fname, overwrite=True)

    msg.append(" [OUTPUT] - Saving background subtracted image: %s" % output_fname)
    msg.append("")
    output_msg = "\n".join(msg)
    return output_msg
