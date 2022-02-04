import numpy as np
from scipy.interpolate import interp2d
from astropy.io import fits
from functools import reduce
import warnings

from pynot.functions import get_version_number
from pynot.fitsio import load_fits_spectrum, save_fitstable_spectrum
from pynot.txtio import load_ascii_spectrum


__version__ = get_version_number()


def combine_2d(files, output=None, method='mean', scale=False, extended=False, dispaxis=1):
    """Combine a list of 2d-spectra using either median or mean combination.
    For median combination, only the overlapping parts of the spectra will be
    combined. The mean combination uses a weighted average over the entire
    spectral range covered.
    The spectra are all assumed to have the same dispersion axis.

    Input
    =====
    files : list of strings
        Input list of files to combine

    output : string
        Output filename, if the filename will be autogenerated from the OBJECT keyword in the header

    method : string
        Combination method, either 'median' or 'mean'

    dispaxis : int   [default=1]
        Dispersion axis. 1: horizontal spectra, 2: vertical spectra (default for most ALFOSC grisms)

    overwrite : bool   [default=False]
        Overwrite existing files.

    """
    wl = list()
    space = list()
    flux = list()
    err = list()
    mask = list()
    size = list()
    scales = list()
    centers = list()
    exp_times = list()
    meta_string = list()

    msg = list()
    # print("\n Spectral Combination 2D\n")
    for fnum, fname in enumerate(files):
        msg.append("          - Loading file: %s" % fname)
        hdr = fits.getheader(fname)
        data2D = fits.getdata(fname)

        try:
            err2D = fits.getdata(fname, 'ERR')
            err2D[err2D <= 0.] = np.nanmedian(err2D)*100
            msg.append("          - Loaded error image")
        except KeyError:
            msg.append("[WARNING] - No ERR extension could be found in the FITS file!")
            err2D = np.ones_like(data2D)

        try:
            mask2D = fits.getdata(fname, 'MASK')
            msg.append("          - Loaded mask image")
        except KeyError:
            msg.append("[WARNING] - No MASK extension could be found in the FITS file!")
            mask2D = np.zeros_like(data2D)

        if 'CD1_1' in hdr:
            cdelt1 = hdr['CD1_1']
        elif 'CDELT1' in hdr:
            cdelt1 = hdr['CDELT1']
        else:
            msg.append(" [ERROR]  - The pixel sampling could not be found!")
            msg.append(" [ERROR]  - Neither CD1_1 nor CDELT1 is in the header keywords...")
            raise KeyError('CD1_1 or CDELT1 missing in header')
        crval1 = hdr['CRVAL1']
        crpix1 = hdr['CRPIX1']
        naxis1 = hdr['NAXIS1']

        if 'CD2_2' in hdr:
            cdelt2 = hdr['CD2_2']
        elif 'CDELT2' in hdr:
            cdelt2 = hdr['CDELT2']
        else:
            msg.append(" [ERROR]  - The pixel sampling could not be found!")
            msg.append(" [ERROR]  - Neither CD2_2 nor CDELT2 is in the header keywords...")
            raise KeyError('CD2_2 or CDELT2 missing in header')
        crval2 = hdr['CRVAL2']
        crpix2 = hdr['CRPIX2']
        naxis2 = hdr['NAXIS2']

        if 'EXPTIME' in hdr:
            exptime = hdr['EXPTIME']
            exp_times.append(exptime)
        else:
            msg.append("[WARNING] - No exposure time could be found in the header!")
            exp_times.append(1)

        if dispaxis == 2:
            # Vertical spectra:
            # most useful for raw spectra before processing
            spatial = (np.arange(naxis1) - (crpix1 - 1))*cdelt1 + crval1
            wavelength = (np.arange(naxis2) - (crpix2 - 1))*cdelt2 + crval2
            N_pix = len(spatial)
            lower_bound = int(0.1*N_pix)
            upper_bound = int(0.9*N_pix)
            SPSF = np.nanmedian(data2D[lower_bound:upper_bound, :], axis=0)

        elif dispaxis == 1:
            # Horizontal spectra:
            # default format after processing by PyNOT
            wavelength = (np.arange(naxis1) - (crpix1 - 1))*cdelt1 + crval1
            spatial = (np.arange(naxis2) - (crpix2 - 1))*cdelt2 + crval2
            N_pix = len(spatial)
            lower_bound = int(0.1*N_pix)
            upper_bound = int(0.9*N_pix)
            SPSF = np.nanmedian(data2D[:, lower_bound:upper_bound], axis=1)
        else:
            msg.append(" [ERROR]  - Invalid dispaxis!")
            raise ValueError("`dispaxis` must be either 1 or 2 (horizontal or vertical)")

        wl.append(wavelength)
        flux.append(data2D)
        err.append(err2D)
        mask.append(mask2D)
        size.append(data2D.shape)

        if not extended:
            # --- Measure the center and width of the trace
            SPSF[SPSF < 0] = 0.
            # Trim edges of SPSF:
            SPSF[:3] = 0.
            SPSF[-3:] = 0.
            # First guess of trace region based on pixel with max-value +/- 5 pixels:
            y_pix = np.arange(N_pix)
            y0 = np.argmax(SPSF)
            trace_region = (y_pix > y0-10) * (y_pix < y0+10)

            # centroid:
            trace_cen = np.sum((spatial*SPSF)[trace_region])/np.sum(SPSF[trace_region])

            # width:
            sumY2 = np.sum(((spatial-trace_cen)**2 * SPSF)[trace_region])
            sumP = np.sum(SPSF[trace_region])
            trace_fwhm = np.sqrt(sumY2/sumP)*2.35

            # Recenter trace:
            space.append(spatial - trace_cen)
            centers.append(trace_cen)

        else:
            trace_cen = N_pix // 2
            trace_fwhm = 0

            space.append(spatial)
            centers.append(trace_cen)


        if scale:
            this_scale = np.nanmax(SPSF)
        else:
            this_scale = 1.
        scales.append(this_scale)

        meta_data = (fnum+1, trace_cen, trace_fwhm) + data2D.shape + (this_scale, exptime)
        meta_string.append(7*' ' + '       %3i        %+2.1f"      %.2f"    (%i, %i)   %.2e    %.1f' % meta_data)


    msg.append("\n          - IMAGE STATISTICS:")
    msg.append("          --------------------------------------------------------------")
    msg.append("          Image No    Centroid      FWHM       Shape       Flux Scale    EXPTIME")
    for line in meta_string:
        msg.append(line)

    # Determine new wavelength grid:
    # If differences are smaller than one pixel then do not interpolate
    l_min = np.min(wl, 1)
    l_max = np.max(wl, 1)
    diff_lmin = np.max(np.abs(np.diff(l_min)))
    diff_lmax = np.max(np.abs(np.diff(l_max)))
    pix_size1 = np.mean([np.diff(X)[0] for X in wl])

    cen_min = np.min(centers)
    cen_max = np.max(centers)
    pix_size2 = np.mean([np.diff(Y)[0] for Y in space])

    # Rescale to flux values of the order 10^0:
    rescale = 10**int(np.log10(np.max(SPSF)))
    f0 = np.mean(scales) / rescale
    msg.append("          - Rescaling data by a factor of 10^%i" % (-int(np.log10(np.max(SPSF)))))

    # Check whether to use interpolation or not:
    same_size_x = all(x == size[0][0] for x, y in size)
    same_size_y = all(y == size[0][1] for x, y in size)
    array_shape_x = ((diff_lmax < pix_size1) and (diff_lmin < pix_size1))
    array_centroids = (cen_max - cen_min < pix_size2)
    use_interpolation = not (same_size_x & same_size_y & array_shape_x & array_centroids)

    if not use_interpolation:
        msg.append("          - Combining with no interpolation!")
        final_wl = wl[0]
        final_space = space[0]
        int_flux = list()
        int_var = list()
        int_mask = list()
        weight = list()
        for f, e, s, m in zip(flux, err, scales, mask):
            e = e/s*f0
            f = f/s*f0
            int_flux.append(f)
            int_var.append(e**2)
            int_mask.append(m)
            weight.append(1./e**2)

    else:
        msg.append("          - Interpolating data onto new grid!")
        int_flux = list()
        int_var = list()
        int_mask = list()
        weight = list()
        if method == 'median':
            overlap_wl = reduce(np.intersect1d, wl)
            overlap_space = reduce(np.intersect1d, space)

        elif method == 'mean':
            overlap_wl = reduce(np.union1d, wl)
            overlap_space = reduce(np.union1d, space)

        pix = np.mean([np.diff(X)[0] for X in wl])
        pix_y = np.mean([np.diff(Y)[0] for Y in space])
        final_wl = np.arange(overlap_wl.min(), overlap_wl.max(), pix)
        final_space = np.arange(overlap_space.min(), overlap_space.max(), pix_y)

        msg.append("          - Creating new linear wavelength grid:")
        msg.append("          - %.1f -- %.1f  |  sampling: %.1f" % (final_wl.min(), final_wl.max(), pix))

        # interpolate
        for this_wl, y, f, e, s, m in zip(wl, space, flux, err, scales, mask):
            e = e/s*f0
            f = f/s*f0
            f_i = interp2d(this_wl, y, f)
            e_i = interp2d(this_wl, y, e)
            m_i = interp2d(this_wl, y, m)
            w_i = interp2d(this_wl, y, 1./e**2)
            int_flux.append(f_i(final_wl, final_space))
            int_var.append(e_i(final_wl, final_space)**2)
            int_mask.append(m_i(final_wl, final_space))
            weight.append(w_i(final_wl, final_space))

    # Combine spectra:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if method == 'median':
            msg.append("          - Combination method: median")
            hdr['COMBINE'] = "Median"
            final_flux = np.nanmedian(int_flux, 0)
            final_var = np.nansum(int_var, 0)/len(int_flux)
            final_err = np.sqrt(final_var)
            final_mask = np.nansum(int_mask, 0) > 0

        elif method == 'mean':
            msg.append("          - Combination method: inverse variance weighting")
            hdr['COMBINE'] = "Inverse Variance Weighted"
            int_flux = np.array(int_flux)
            weight = np.array(weight)
            final_flux = np.nansum(int_flux*weight, 0) / np.nansum(weight, 0)
            final_var = 1. / np.nansum(weight, 0)
            final_err = np.sqrt(final_var)
            final_mask = np.nansum(int_mask, 0) > 0


    if dispaxis == 1:
        hdr['CRVAL1'] = np.min(final_wl)
        hdr['CD1_1'] = np.diff(final_wl)[0]
        hdr['CRVAL2'] = np.min(final_space)
        hdr['CD2_2'] = np.diff(final_space)[0]
    elif dispaxis == 2:
        hdr['CRVAL2'] = np.min(final_wl)
        hdr['CD2_2'] = np.diff(final_wl)[0]
        hdr['CRVAL1'] = np.min(final_space)
        hdr['CD1_1'] = np.diff(final_space)[0]

    hdr['NCOMBINE'] = len(int_flux)
    hdr['EXPTIME'] = np.sum(exp_times)
    hdr['DATAMIN'] = np.nanmin(final_flux*rescale)
    hdr['DATAMAX'] = np.nanmax(final_flux*rescale)
    hdr['EXTNAME'] = 'DATA'
    hdr['AUTHOR'] = 'PyNOT version %s' % __version__

    if output is None:
        object_name = hdr['OBJECT']
        output = '%s_combined_2d.fits' % object_name
    else:
        if output[-4:].lower() != 'fits':
            output += '.fits'
    msg.append("          - Bringing data back to original flux scale")

    FLUX = fits.PrimaryHDU(data=final_flux*rescale, header=hdr)
    FLUX.name = 'DATA'
    ERR = fits.ImageHDU(data=final_err*rescale, header=hdr, name='ERR')
    MASK = fits.ImageHDU(data=final_mask.astype(int), header=hdr, name='MASK')
    HDU = fits.HDUList([FLUX, ERR, MASK])
    HDU.writeto(output, overwrite=True)
    msg.append(" [OUTPUT] - Saving the combined spectrum to file: %s" % output)
    output_msg = "\n".join(msg)

    return (final_wl, final_flux, final_err, final_mask, output_msg)



def combine_1d(files, output=None, method='mean', scale=False):
    """Combine a list of 1d-spectra using either median or mean combination.
    For median combination, only the overlapping parts of the spectra will be
    combined. The mean combination uses a weighted average over the entire
    spectral range covered. If the spectra have different wavelength coverage,
    they will be interpolated onto a common grid.

    Input
    =====
    files : list of strings
        Input list of files to combine

    output : string   [default=None]
        Output filename, if None then the filename will be autogenerated

    method : string   [default=mean]
        Combination method, either 'median' or 'mean'

    scale : bool   [default=False]
        Scale the individual spectra to their mean values?

    """

    wl_all = list()
    flux_all = list()
    err_all = list()
    mask_all = list()
    size_all = list()
    scales = list()

    msg = list()
    for fname in files:
        if fname.endswith('.fits') or fname.endswith('.fit'):
            # load FITS file.
            wl, flux, err, mask, hdr, load_msg = load_fits_spectrum(fname)
            if load_msg:
                msg.append(load_msg)
            msg.append("          - Loaded FITS spectrum: %s" % fname)

        else:
            wl, flux, err, mask, _, load_msg = load_ascii_spectrum(fname)
            hdr = fits.Header()
            if load_msg:
                msg.append(load_msg)
            msg.append("          - Loaded ASCII spectrum: %s" % fname)

        wl_all.append(wl)
        flux_all.append(flux)
        err_all.append(err)
        mask_all.append(mask)
        size_all.append(len(wl))
        if scale:
            nonzero = flux.nonzero()[0]
            idx_0 = min(nonzero) + len(nonzero)/2
            x1 = idx_0 - 50
            x2 = idx_0 + 51
            scales.append(np.nanmedian(flux[x1:x2]))
        else:
            scales.append(1.)

    # Determine average reference scale:
    f0 = np.nanmean(scales)

    # -- Determine new wavelength grid:
    # If differences are smaller than one pixel then do not interpolate
    same_size = all(x == size_all[0] for x in size_all)
    wl_min = [min(this_wl) for this_wl in wl_all]
    wl_max = [max(this_wl) for this_wl in wl_all]
    wl_diff_lower = np.max(np.abs(np.diff(wl_min)))
    wl_diff_upper = np.max(np.abs(np.diff(wl_max)))
    avg_pix_size = np.mean([np.mean(np.diff(this_wl)) for this_wl in wl_all])
    if same_size and (wl_diff_lower < avg_pix_size) and (wl_diff_upper < avg_pix_size):
        msg.append("          - Combining with no interpolation!")
        final_wl = wl[0]
        int_flux = list()
        int_var = list()
        int_mask = mask_all
        weight = list()
        for f, e, m, s in zip(flux_all, err_all, mask_all, scales):
            v = (e/s*f0)**2
            f = f/s*f0
            int_flux.append(f)
            int_var.append(v)
            weight.append(1./v)

    else:
        msg.append("          - Interpolating onto common wavelength grid!")
        int_flux = list()
        int_var = list()
        int_mask = list()
        weight = list()
        if method == 'median':
            overlap = reduce(np.intersect1d, wl_all)

        elif method == 'mean':
            overlap = reduce(np.union1d, wl_all)

        pix = np.min([np.min(np.diff(this_wl)) for this_wl in wl_all])
        final_wl = np.arange(overlap.min(), overlap.max(), pix)
        msg.append("          - Creating new linear wavelength grid:")
        msg.append("          - %.3f -- %.3f  |  sampling: %.3f" % (final_wl.min(), final_wl.max(), pix))
        hdr['CRVAL1'] = final_wl.min()
        hdr['CRPIX1'] = 1
        hdr['CDELT1'] = pix
        hdr['CD1_1'] = pix
        for key in ['CD1_2', 'CD2_1', 'CD2_2', 'CDELT2', 'CRVAL2', 'CRPIX2', 'CTYPE2', 'CUNIT2']:
            if key in hdr:
                hdr.remove(key)

        # interpolate
        for this_wl, f, e, m, s in zip(wl_all, flux_all, err_all, mask_all, scales):
            m[e <= 0] = 1
            e[e <= 0] = 1.e4
            v = (e*s/f0)**2
            f = f*s/f0
            f_i = np.interp(final_wl, this_wl, f, left=np.nan, right=np.nan)
            v_i = np.interp(final_wl, this_wl, v, left=np.nan, right=np.nan)
            m_i = np.interp(final_wl, this_wl, m, left=np.nan, right=np.nan) > 0
            w_i = np.interp(final_wl, this_wl, 1./v, left=np.nan, right=np.nan)
            int_flux.append(f_i)
            int_var.append(v_i)
            int_mask.append(m_i)
            weight.append(w_i)

    # Combine spectra:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if method == 'median':
            msg.append("          - Combination method: median")
            final_flux = np.nanmedian(int_flux, axis=0)
            final_var = np.nansum(int_var, axis=0)/len(int_flux)
            final_err = np.sqrt(final_var)
            final_mask = np.nansum(int_mask, axis=0) > 0

        else:
            msg.append("          - Combination method: inverse variance weighting")
            int_flux = np.array(int_flux)
            M = ~np.array(int_mask, dtype=bool)
            weight = np.array(weight)
            final_flux = np.nansum(M*int_flux*weight, 0) / np.nansum(M*weight, 0)
            final_var = 1. / np.nansum(M*weight, 0)
            final_err = np.sqrt(final_var)
            final_mask = np.nansum(int_mask, axis=0) > 0

    if output is None:
        obj_name = hdr.get('OBJECT')
        if not obj_name:
            fname = 'combined_spectrum_1d.fits'
        fname = '%s_combined.fits' % obj_name
    elif not output.endswith('.fits'):
        fname = output + '.fits'
    else:
        fname = output

    save_fitstable_spectrum(fname, final_wl, final_flux, final_err, hdr, mask=final_mask)
    msg.append(" [OUTPUT] - Saving the combined spectrum as a FITS table: %s" % output)

    output_msg = "\n".join(msg)

    return (final_wl, final_flux, final_err, final_mask, output_msg)
