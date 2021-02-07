# coding/PyNOT/multi_extract.py
import numpy as np
from astropy.io import fits
from matplotlib.backends import backend_pdf
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from numpy.polynomial import Chebyshev
import warnings

from lmfit import Parameters, minimize

from pynot.functions import mad, NN_moffat, NN_gaussian, fix_nans, get_version_number

__version__ = get_version_number()


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



def trace_model(pars, x, N, model_name='moffat'):
    model = np.zeros_like(x)
    if model_name == 'gaussian':
        for i in range(N):
            p = [pars['mu_%i' % i],
                 pars['sig_%i' % i],
                 pars['logamp_%i' % i]]
            model += NN_gaussian(x, *p)
    elif model_name == 'moffat':
        for i in range(N):
            p = [pars['mu_%i' % i],
                 pars['a_%i' % i],
                 pars['b_%i' % i],
                 pars['logamp_%i' % i]]
            model += NN_moffat(x, *p)
    model += pars['bg']
    return model


def model_residuals(pars, x, y, N, model_name='moffat'):
    return y - trace_model(pars, x, N, model_name=model_name)


def prep_parameters(peaks, prominence, size=np.inf, model_name='moffat'):
    values = zip(peaks, prominence)
    pars = Parameters()
    pars.add('bg', value=0.)
    if model_name == 'gaussian':
        for i, (x0, amp) in enumerate(values):
            pars.add('mu_%i' % i, value=float(x0), min=0., max=size)
            pars.add('sig_%i' % i, value=2., min=0., max=20.)
            pars.add('logamp_%i' % i, value=np.log10(amp))
    elif model_name == 'moffat':
        for i, (x0, amp) in enumerate(values):
            pars.add('mu_%i' % i, value=float(x0), min=0., max=size)
            pars.add('a_%i' % i, value=2., min=0., max=20.)
            pars.add('b_%i' % i, value=1., min=0., max=20.)
            pars.add('logamp_%i' % i, value=np.log10(amp))
    return pars


def median_filter_data(x, kappa=5., window=21):
    med_x = median_filter(x, window)
    MAD = 1.5*np.nanmedian(np.abs(x - med_x))
    if MAD == 0:
        MAD = np.nanstd(x - med_x)
    mask = np.abs(x - med_x) < kappa*MAD
    return (med_x, mask)


def fit_trace(img2D, x, y, model_name='moffat', dx=50, ymin=5, ymax=-5, xmin=None, xmax=None):
    """
    Perform automatic localization of the trace if possible, otherwise use fixed
    aperture to extract the 1D spectrum.
    The spectra are assumed to be horizontal. Check orientation before passing img2D!
    When fitting the trace, reject pixels in a column below `ymin` and above `ymax`.
    """
    msg = list()
    if not xmin:
        xmin = 0
    if not xmax:
        xmax = len(x)
    if xmax < 0:
        xmax = len(x) + xmax
    if not ymin:
        ymin = 0
    if not ymax:
        ymax = len(y)
    if ymax < 0:
        ymax = len(y) + ymax

    spsf = np.nanmedian(img2D[:, xmin:xmax], axis=1)
    spsf = spsf - np.nanmedian(spsf)
    spsf[:ymin] = 0.
    spsf[ymax:] = 0.

    # Detect peaks:
    kappa = 10.
    noise = mad(spsf)*1.48
    peaks, properties = find_peaks(spsf, prominence=kappa*noise, width=3)
    prominences = properties['prominences']
    msg.append("          - Automatically identifying objects in the image...")

    N_obj = len(peaks)
    if N_obj == 0:
        raise ValueError(" [ERROR]  - No object found in image!")
    elif N_obj == 1:
        spsf[spsf < 0] = 0.
        fwhm = get_FWHM(spsf)
        msg.append("          - Found %i object in slit" % N_obj)
        msg.append("          - FWHM of spectral trace: %.1f" % fwhm)
    else:
        fwhm = None
        msg.append("          - Found %i objects in slit" % N_obj)

    # Fit trace with N objects:
    msg.append("          - Fitting the spectral trace with a %s profile" % model_name.title())
    trace_parameters = list()
    x_binned = np.arange(0., img2D.shape[1], dx, dtype=np.float64)
    for num in range(0, img2D.shape[1], dx):
        pars = prep_parameters(peaks, prominences, size=img2D.shape[0], model_name=model_name)
        col = np.nanmean(img2D[:, num:num+dx], axis=1)
        col_mask = np.ones_like(col, dtype=bool)
        col_mask[:ymin] = 0.
        col_mask[ymax:] = 0.
        try:
            popt = minimize(model_residuals, pars, args=(y[col_mask], col[col_mask], N_obj),
                            kws={'model_name': model_name})
            for par_val in popt.params.values():
                if par_val.stderr is None:
                    par_val.stderr = 100.
            trace_parameters.append(popt.params)
        except ValueError:
            for par_val in pars.values():
                par_val.stderr = 100.
            trace_parameters.append(pars)
    msg.append("          - Fitted %i points along the spectral trace" % len(trace_parameters))
    output_msg = "\n".join(msg)
    return (x_binned, N_obj, trace_parameters, fwhm, output_msg)


def create_2d_profile(img2D, model_name='moffat', dx=25, width_scale=2,
                      xmin=None, xmax=None, ymin=None, ymax=None, order_center=3, order_width=0,
                      w_cen=15, kappa_cen=3., w_width=21, kappa_width=3.):
    """
    img2D : np.array(M, N)
        Input image with dispersion along x-axis!

    model_name : {'moffat' or 'gaussian' or 'tophat'}
        Model type for the spectral PSF

    dx : int  [default=5]
        Fit the trace for every dx column

    width_scale : int  [default=2]
        The scaling factor of the FWHM used for the width of the tophat profile:
        By default the flux is summed within 2*FWHM on either side of the centroid

    xmin, xmax : int  [default=None]
        Minimum and maximum extent to fit along the dispersion axis

    ymin, ymax : int  [default=None]
        Minimum and maximum extent to fit along the spatial axis

    order_center : int  [default=3]
        Order of Chebyshev polynomium for the trace position

    order_width : int  [default=0]
        Order of Chebyshev polynomium for the trace width

    w_cen : int  [default=15]
        Kernel width of median filter for trace position

    kappa_cen : float  [default=3.0]
        Threshold for median filtering. Reject outliers above: ±`kappa` * sigma,
        where sigma is the robust standard deviation of the data points.

    w_width : int  [default=15]
        Kernel width of median filter for trace width parameters

    kappa_width : float  [default=3.0]
        Threshold for median filtering. Reject outliers above: ±`kappa` * sigma,
        where sigma is the robust standard deviation of the data points.

    Returns
    -------
    trace_models_2d : list(np.array(M, N))
        List of trace models, one for each object identified in the image

    trace_info : list
        List of information dictionary for each trace:
        The fitted position and width as well as the fit mask and the fitted values
    """
    msg = list()
    img2D = img2D.astype(np.float64)
    x = np.arange(img2D.shape[1], dtype=np.float64)
    y = np.arange(img2D.shape[0], dtype=np.float64)

    if not xmin:
        xmin = 0
    if not xmax:
        xmax = len(x)
    if xmax < 0:
        xmax = len(x) + xmax
    if not ymin:
        ymin = 0
    if not ymax:
        ymax = len(y)
    if ymax < 0:
        ymax = len(y) + ymax

    if model_name == 'tophat':
        # Fit the centroid using a Moffat profile, but the discard the with for the profile calculation
        fit_values = fit_trace(img2D, x, y, model_name='moffat', dx=dx, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax)
        fwhm = fit_values[3]
        if fwhm is None:
            raise ValueError("FWHM of the spectral trace could not be determined! Maybe more than one object in slit...")
    else:
        fit_values = fit_trace(img2D, x, y, model_name=model_name, dx=dx, ymin=ymin, ymax=ymax, xmin=xmin, xmax=xmax)
    x_binned, N_obj, trace_parameters, fwhm, fit_msg = fit_values
    msg.append(fit_msg)

    msg.append("          - Creating 2D spectral profile from fitted parameters")
    msg.append("          - Profile type: %s" % model_name)
    msg.append("          - Interpolating centroid using Chebyshev polynomium of degree: %i" % order_center)
    msg.append("          - Interpolating profile width using Chebyshev polynomium of degree: %i" % order_width)
    trace_models_2d = list()
    trace_info = list()
    domain = [0, img2D.shape[1]]
    for n in range(N_obj):
        msg.append("          - Working on profile number %i" % (n+1))
        info_dict = dict()
        info_dict['x_binned'] = x_binned
        # Median filter
        mu = np.array([p['mu_%i' % n] for p in trace_parameters])
        mu_err = np.array([p['mu_%i' % n].stderr for p in trace_parameters])
        mu_err[mu_err == 0] = 100.
        w_mu = 1./mu_err**2
        mu_med, mask_mu = median_filter_data(mu, kappa_cen, w_cen)
        mask_mu &= (x_binned > xmin) & (x_binned < xmax)
        mu_fit = Chebyshev.fit(x_binned[mask_mu], mu[mask_mu], deg=order_center, domain=domain, w=w_mu[mask_mu])
        info_dict['mu'] = mu
        info_dict['mu_err'] = mu_err
        info_dict['mask_mu'] = mask_mu
        info_dict['fit_mu'] = mu_fit(x)

        # Fit polynomium:
        trace2D = np.zeros_like(img2D)
        trace2D = np.zeros_like(img2D)
        if model_name == 'gaussian':
            # Median filter
            sig = np.array([p['sig_%i' % n] for p in trace_parameters])
            sig_err = np.array([p['sig_%i' % n].stderr for p in trace_parameters])
            sig_err[sig_err == 0] = 100.
            w_sig = 1./sig_err**2
            sig_med, mask_sig = median_filter_data(sig, kappa_width, w_width)
            mask_sig &= (x_binned > xmin) & (x_binned < xmax)
            sig_fit = Chebyshev.fit(x_binned[mask_sig], sig[mask_sig], deg=order_width, domain=domain, w=w_sig[mask_sig])
            info_dict['sig'] = sig
            info_dict['sig_err'] = sig_err
            info_dict['mask_sig'] = mask_sig
            info_dict['fit_sig'] = sig_fit(x)

            for num, x_i in enumerate(x):
                P_i = NN_gaussian(y, mu_fit(x_i), sig_fit(x_i), 0.)
                P_i = P_i/np.sum(P_i)
                trace2D[:, num] = P_i
            trace_models_2d.append(trace2D)

        elif model_name == 'moffat':
            # Median filter
            a = np.array([p['a_%i' % n] for p in trace_parameters])
            a_med, mask_a = median_filter_data(a, kappa_width, w_width)
            a_err = np.array([p['a_%i' % n].stderr for p in trace_parameters])
            a_err[a_err == 0] = 100.
            w_a = 1./a_err**2
            b = np.array([p['b_%i' % n] for p in trace_parameters])
            b_med, mask_b = median_filter_data(b, kappa_width, w_width)
            b_err = np.array([p['b_%i' % n].stderr for p in trace_parameters])
            b_err[b_err == 0] = 100.
            w_b = 1./b_err**2
            mask_a &= (x_binned > xmin) & (x_binned < xmax)
            mask_b &= (x_binned > xmin) & (x_binned < xmax)

            a_fit = Chebyshev.fit(x_binned[mask_a], a[mask_a], deg=order_width, domain=domain, w=w_a[mask_a])
            b_fit = Chebyshev.fit(x_binned[mask_b], b[mask_b], deg=order_width, domain=domain, w=w_b[mask_b])
            info_dict['a'] = a
            info_dict['a_err'] = a_err
            info_dict['mask_a'] = mask_a
            info_dict['fit_a'] = a_fit(x)
            info_dict['b'] = b
            info_dict['b_err'] = b_err
            info_dict['mask_b'] = mask_b
            info_dict['fit_b'] = b_fit(x)

            for num, x_i in enumerate(x):
                P_i = NN_moffat(y, mu_fit(x_i), a_fit(x_i), b_fit(x_i), 0.)
                P_i = P_i/np.sum(P_i)
                trace2D[:, num] = P_i
            trace_models_2d.append(trace2D)

        elif model_name == 'tophat':
            for num, x_i in enumerate(x):
                center = mu_fit(x_i)
                lower = int(center - width_scale*fwhm)
                upper = int(center + width_scale*fwhm)
                trace2D[lower:upper+1, num] = 1 / (upper - lower + 1)
            trace_models_2d.append(trace2D)
            info_dict['fwhm'] = fwhm
        trace_info.append(info_dict)

    output_msg = "\n".join(msg)
    return (trace_models_2d, trace_info, output_msg)


def plot_diagnostics(pdf, spec1D, err1D, info_dict, width_scale=2):
    """
    Create a diagnostic plot of the
    """
    figsize = (8.3, 11.7)
    if 'sig' in info_dict:
        pars = ['mu', 'sig']
    elif 'fwhm' in info_dict:
        # TopHat profile:
        pars = ['mu']
    else:
        pars = ['mu', 'a', 'b']

    fig, axes = plt.subplots(nrows=len(pars)+1, ncols=1, figsize=figsize)
    x = np.arange(len(info_dict['fit_mu']))
    for par, ax in zip(pars, axes):
        mask = info_dict['mask_'+par]
        ax.errorbar(info_dict['x_binned'][mask], info_dict[par][mask], info_dict[par+'_err'][mask],
                    marker='s', color='0.2', ls='', markersize=4)
        ax.plot(info_dict['x_binned'][~mask], info_dict[par][~mask], marker='x', color='crimson', ls='')
        ax.plot(x, info_dict['fit_'+par], color='RoyalBlue', lw=1.5, alpha=0.9)
        med = np.nanmedian(info_dict[par][mask])
        std = 1.5*mad(info_dict[par][mask])
        ymin = max(0, med-10*std)
        ymax = med+10*std
        if 'fwhm' in info_dict:
            lower = info_dict['fit_'+par] - width_scale*info_dict['fwhm']
            upper = info_dict['fit_'+par] + width_scale*info_dict['fwhm']
            ax.fill_between(x, lower, upper, color='RoyalBlue', alpha=0.2)
            ymin = np.min(lower) - width_scale*info_dict['fwhm']/2
            ymax = np.max(upper) + width_scale*info_dict['fwhm']/2
        ax.set_ylim(ymin, ymax)

        if par == 'mu':
            ax.set_ylabel("Centroid")
        elif par == 'sig':
            ax.set_ylabel("$\\sigma$")
        elif par == 'a':
            ax.set_ylabel("$\\alpha$")
        elif par == 'b':
            ax.set_ylabel("$\\beta$")

    axes[-1].plot(spec1D, color='k', lw=1.0, alpha=0.9, label='Flux')
    axes[-1].plot(err1D, color='crimson', lw=0.7, alpha=0.8, label='Error')
    ymin = 0.
    good = spec1D > 5*err1D
    if np.sum(good) == 0:
        good = spec1D[100:-100] > 0
    ymax = np.nanmax(spec1D[good])
    axes[-1].set_ylim(ymin, ymax)
    axes[-1].set_ylabel("Flux")
    axes[-1].set_xlabel("Dispersion Axis  [pixels]")
    axes[-1].legend()
    fig.tight_layout()
    pdf.savefig(fig)



def auto_extract_img(img2D, err2D, *, N=None, pdf_fname=None, mask=None, model_name='moffat', dx=50, width_scale=2, xmin=None, xmax=None, ymin=None, ymax=None, order_center=3, order_width=0, w_cen=15, kappa_cen=3., w_width=21, kappa_width=3.):
    assert err2D.shape == img2D.shape, "input image and error image do not match in shape"
    if N == 0:
        raise ValueError("Invalid input: N must be an integer larger than or equal to 1, not %r" % N)

    M = np.ones_like(img2D)
    if mask is not None:
        if mask.shape == img2D.shape:
            M[mask > 0] == 0
        else:
            raise ValueError("The provided mask does not match the input image shape")

    msg = list()
    var2D = err2D**2
    var2D[var2D == 0.] = np.median(var2D)*100
    var2D[np.isnan(var2D)] = np.median(var2D)*100

    # Optimal Extraction:
    profile_values = create_2d_profile(img2D, model_name=model_name, dx=dx, width_scale=width_scale,
                                       xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                       order_center=order_center, order_width=order_width,
                                       w_cen=w_cen, kappa_cen=kappa_cen, w_width=w_width, kappa_width=kappa_width)
    trace_models_2d, trace_info, profile_msg = profile_values
    msg.append(profile_msg)

    msg.append("          - Performing optimal extraction")
    if N is not None:
        N_obj = len(trace_models_2d)
        if N_obj != N:
            if N == 1:
                err_msg = "Expected 1 spectrum but found %i" % N_obj
            else:
                err_msg = "Expected %i spectra but found %i" % (N, N_obj)
            raise ValueError(err_msg)

    if pdf_fname:
        pdf = backend_pdf.PdfPages(pdf_fname)

    spectra = list()
    for P, info_dict in zip(trace_models_2d, trace_info):
        spec1D = np.sum(M*P*img2D/var2D, axis=0) / np.sum(M*P**2/var2D, axis=0)
        var1D = np.sum(M*P, axis=0) / np.sum(M*P**2/var2D, axis=0)
        err1D = np.sqrt(var1D)
        err1D = fix_nans(err1D)
        spectra.append([spec1D, err1D])

        if pdf_fname:
            plot_diagnostics(pdf, spec1D, err1D, info_dict, width_scale)

    if pdf_fname:
        msg.append(" [OUTPUT] - Saving diagnostic figures: %s" % pdf_fname)
        pdf.close()
        plt.close('all')

    output_msg = "\n".join(msg)
    return spectra, output_msg


def auto_extract(fname, output, dispaxis=1, *, N=None, pdf_fname=None, mask=None, model_name='moffat', dx=50, width_scale=2, xmin=None, xmax=None, ymin=None, ymax=None, order_center=3, order_width=1, w_cen=15, kappa_cen=3., w_width=21, kappa_width=3., **kwargs):
    """Automatically extract object spectra in the given file. Dispersion along the x-axis is assumed!"""
    msg = list()
    img2D = fits.getdata(fname)
    hdr = fits.getheader(fname)
    if 'DISPAXIS' in hdr:
        dispaxis = hdr['DISPAXIS']

    msg.append("          - Loaded image data: %s" % fname)
    try:
        err2D = fits.getdata(fname, 'ERR')
        msg.append("          - Loaded error image extension")
    except:
        noise = 1.5*mad(img2D)
        err2D = np.ones_like(img2D) * noise
        msg.append("[WARNING] - No error image detected!")
        msg.append("[WARNING] - Generating one from image statistics:")
        msg.append("[WARNING] - Median=%.2e  Sigma=%.2e" % (np.nanmedian(img2D), noise))


    if dispaxis == 2:
        img2D = img2D.T
        err2D = err2D.T

    spectra, ext_msg = auto_extract_img(img2D, err2D, N=N, pdf_fname=pdf_fname, mask=mask,
                                        model_name=model_name, dx=dx, width_scale=width_scale,
                                        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                        order_center=order_center, order_width=order_width,
                                        w_cen=w_cen, kappa_cen=kappa_cen, w_width=w_width, kappa_width=kappa_width)
    msg.append(ext_msg)

    hdu = fits.HDUList()
    hdr['AUTHOR'] = 'PyNOT version %s' % __version__
    hdr['COMMENT'] = 'PyNOT automatically extracted spectrum'
    hdr['COMMENT'] = 'Each spectrum in its own extension'
    if 'CDELT1' in hdr:
        cdelt = hdr['CDELT1']
        crval = hdr['CRVAL1']
        crpix = hdr['CRPIX1']
        wl = (np.arange(hdr['NAXIS1']) - (crpix - 1))*cdelt + crval
    else:
        wl = np.arange(len(spectra[0][0]))


    keywords_base = ['CDELT%i', 'CRPIX%i', 'CRVAL%i', 'CTYPE%i', 'CUNIT%i']
    keywords_to_remove = sum([[key % num for key in keywords_base] for num in [1, 2]], [])
    keywords_to_remove += ['CD1_1', 'CD2_1', 'CD1_2', 'CD2_2']
    keywords_to_remove += ['BUNIT', 'DATAMIN', 'DATAMAX']
    for num, (flux, err) in enumerate(spectra):
        col_wl = fits.Column(name='WAVE', array=wl, format='D', unit=hdr['CUNIT1'])
        col_flux = fits.Column(name='FLUX', array=flux, format='D', unit=hdr['BUNIT'])
        col_err = fits.Column(name='ERR', array=err, format='D', unit=hdr['BUNIT'])
        for key in keywords_to_remove:
            hdr.remove(key, ignore_missing=True)
        tab = fits.BinTableHDU.from_columns([col_wl, col_flux, col_err], header=hdr)
        tab.name = 'OBJ%i' % (num+1)
        hdu.append(tab)

    hdu.writeto(output, overwrite=True, output_verify='silentfix')
    msg.append(" [OUTPUT] - Writing fits table: %s" % output)
    msg.append("")
    output_msg = "\n".join(msg)
    return output_msg
