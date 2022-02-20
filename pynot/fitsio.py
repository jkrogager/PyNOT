# pynot/fitsio.py
__author__ = "Jens-Kristian Krogager"

import warnings
from astropy.io import fits
import numpy as np


class MultipleSpectraWarning(Warning):
    """Throw warning when several FITS Table extensions or multiple IRAF objects are present"""
    pass

class WavelengthError(Exception):
    """Raised if the header doesn't contain the proper wavelength solution: CRVAL, CD etc."""
    pass

class FormatError(Exception):
    """Raised when the FITS format is not understood"""
    pass


def save_fits_spectrum(fname, wl, flux, err, hdr, bg=None, aper=None, mask=None):
    """Write spectrum to a FITS file with 3 extensions: FLUX, ERR and SKY"""
    # Check if wavelength array increases linearly (to less than 1%):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        delta_log_wl = np.diff(np.log10(wl))
        delta_wl = np.diff(wl)

    if (np.abs(np.diff(delta_wl)) < delta_wl[0]/100.).all():
        # linear wavelength solution: OK
        hdr['CRVAL1'] = np.min(wl)
        hdr['CDELT1'] = delta_wl[0]
        hdr['CTYPE1'] = 'WAVE-LIN'
    elif (np.abs(np.diff(delta_log_wl)) < delta_log_wl[0]/100.).all():
        # logarithmic wavelength solution: OK
        hdr['CRVAL1'] = np.log10(np.min(wl))
        hdr['CDELT1'] = delta_wl[0]
        hdr['CTYPE1'] = 'WAVE-LOG'
    else:
        return False, "Improper wavelength solution: wavelength should increase linearly or logarithmically!"

    hdr['CRPIX1'] = 1
    hdr['NAXIS1'] = len(wl)
    hdr['NAXIS'] = 1

    hdu1 = fits.PrimaryHDU(data=flux, header=hdr)
    hdu1.name = 'FLUX'
    hdu2 = fits.ImageHDU(data=err, header=hdr, name='ERR')
    hdu = fits.HDUList([hdu1, hdu2])
    if bg is not None:
        hdu3 = fits.ImageHDU(data=bg, header=hdr, name='SKY')
        hdu.append(hdu3)
    if aper is not None:
        aper_hdr = fits.Header()
        aper_hdr['AUTHOR'] = 'PyNOT'
        aper_hdr['COMMENT'] = '2D Extraction Aperture'
        aper_hdu = fits.ImageHDU(data=aper, header=aper_hdr, name='APER')
        hdu.append(aper_hdu)
    if mask is not None:
        mask_hdr = fits.Header()
        mask_hdr['AUTHOR'] = 'PyNOT'
        mask_hdr['COMMENT'] = 'Pixel mask, 0: good, 1: bad'
        mask_hdu = fits.ImageHDU(data=mask, header=mask_hdr, name='MASK')
        hdu.append(mask_hdu)
    hdu.writeto(fname, output_verify='silentfix', overwrite=True)
    return True, "File saved successfully"


def save_fitstable_spectrum(fname, wl, flux, err, hdr, bg=None, aper=None, mask=None):
    """Write spectrum to a FITS Table with 4 columns: Wave, FLUX, ERR and SKY"""
    hdu = fits.HDUList()
    hdr['COMMENT'] = 'PyNOT extracted spectrum'
    if bg is None:
        bg = np.zeros_like(flux)
    if mask is None:
        mask = np.zeros_like(flux, dtype=bool)
    col_wl = fits.Column(name='WAVE', array=wl, format='D')
    col_flux = fits.Column(name='FLUX', array=flux, format='D')
    col_err = fits.Column(name='ERR', array=err, format='D')
    col_sky = fits.Column(name='SKY', array=bg, format='D')
    col_mask = fits.Column(name='MASK', array=mask, format='L')
    tab = fits.BinTableHDU.from_columns([col_wl, col_flux, col_err, col_mask, col_sky], header=hdr)
    tab.name = 'DATA'
    hdu.append(tab)
    if aper is not None:
        aper_hdr = fits.Header()
        aper_hdr['AUTHOR'] = 'PyNOT'
        aper_hdr['COMMENT'] = '2D Extraction Aperture'
        aper_hdu = fits.ImageHDU(data=aper, header=aper_hdr, name='APER')
        hdu.append(aper_hdu)
    hdu.writeto(fname, overwrite=True, output_verify='silentfix')
    return True, "File saved successfully"


def get_wavelength_from_header(hdr):
    """
    Obtain wavelength solution from Header keywords:

        Wavelength_i = CRVAL1 + (PIXEL_i - (CRPIX1-1)) * CDELT1

    CDELT1 can be CD1_1 as well.

    If all these keywords are not present in the header, raise a WavelengthError

    Returns
    -------
    wavelength : np.array (float)
        Numpy array of wavelengths.
    """
    if ('CRVAL1' and 'CRPIX1' in hdr.keys()) and ('CDELT1' in hdr.keys() or 'CD1_1' in hdr.keys()):
        if 'CD1_1' in hdr.keys():
            cdelt = hdr['CD1_1']
        else:
            cdelt = hdr['CDELT1']
        crval = hdr['CRVAL1']
        crpix = hdr['CRPIX1']

        wavelength = (np.arange(hdr['NAXIS1']) - (crpix-1))*cdelt + crval

        return wavelength

    else:
        raise WavelengthError("Not enough information in header to create wavelength array")


# -- These names are used to define proper column names for Wavelength, Flux and Error:
wavelength_column_names = ['wl', 'lam', 'lambda', 'loglam', 'wave', 'wavelength', 'awav']
flux_column_names = ['data', 'spec', 'flux', 'flam', 'fnu', 'flux_density']
error_column_names = ['err', 'sig', 'error', 'ivar', 'sigma', 'var']
mask_column_names = ['mask', 'qual', 'dq', 'qc']

# -- These names are used to define proper ImageHDU names for Flux and Error:
flux_HDU_names = ['FLUX', 'SCI', 'FLAM', 'FNU']
error_HDU_names = ['ERR', 'ERRS', 'SIG', 'SIGMA', 'ERROR', 'ERRORS', 'IVAR', 'VAR']
mask_HDU_names = ['MASK', 'QUAL', 'QC', 'DQ']


def get_spectrum_fits_table(tbdata):
    """
    Scan the TableData for columns containing wavelength, flux, error and mask.
    All arrays of {wavelength, flux and error} must be present.

    The columns are identified by matching predefined column names:
        For wavelength arrays: %(WL_COL_NAMES)r

        For flux arrays: %(FLUX_COL_NAMES)r

        For error arrays: %(ERR_COL_NAMES)r

    Returns
    -------
    wavelength : np.array (float)
        Numpy array of wavelengths.
    data : np.array (float)
        Numpy array of flux density.
    error : np.array (float)
        Numpy array of uncertainties on the flux density
    mask : np.array (bool)
        Numpy boolean array of pixel mask. `True` if the pixel is 'good',
        `False` if the pixel is bad and should not be used.
    """
    table_names = [name.lower() for name in tbdata.names]
    wl_in_table = False
    for colname in wavelength_column_names:
        if colname in table_names:
            wl_in_table = True
            if colname == 'loglam':
                wavelength = 10**tbdata[colname]
            else:
                wavelength = tbdata[colname]
            break

    data_in_table = False
    for colname in flux_column_names:
        if colname in table_names:
            data_in_table = True
            data = tbdata[colname]
            break

    error_in_table = False
    for colname in error_column_names:
        if colname in table_names:
            error_in_table = True
            if colname == 'ivar':
                error = 1./np.sqrt(tbdata[colname])
            elif colname == 'var':
                error = np.sqrt(tbdata[colname])
            else:
                error = tbdata[colname]
            break

    all_arrays_found = wl_in_table and data_in_table and error_in_table
    if not all_arrays_found:
        raise FormatError("Could not find all data columns in the table")

    mask_in_table = False
    for colname in mask_column_names:
        if colname in table_names:
            mask_in_table = True
            mask = tbdata[colname]
            break
    if not mask_in_table:
        mask = np.zeros_like(data, dtype=bool)

    return wavelength.flatten(), data.flatten(), error.flatten(), mask.flatten()

# Hack the doc-string of the function to input the variable names:
output_column_names = {'WL_COL_NAMES': wavelength_column_names,
                       'FLUX_COL_NAMES': flux_column_names,
                       'ERR_COL_NAMES': error_column_names}
get_spectrum_fits_table.__doc__ = get_spectrum_fits_table.__doc__ % output_column_names


def get_spectrum_hdulist(HDUlist):
    """
    Scan the HDUList for names that match one of the defined names for
    flux and flux error. If one is missing, the code will raise a FormatError.

    The ImageHDUs are identified by matching predefined Extension names:

        For flux arrays: %(FLUX_HDU_NAMES)r

        For error arrays: %(ERR_HDU_NAMES)r

    Returns
    -------
    data : np.array (float)
        Numpy array of flux density.
    error : np.array (float)
        Numpy array of uncertainties on the flux density
    mask : np.array (bool)
        Numpy boolean array of pixel mask. `True` if the pixel is 'good',
        `False` if the pixel is bad and should not be used.
    data_hdr : astropy.io.fits.Header
        The FITS Header of the given data extension.
        The wavelength information should be contained in this header.
    """
    data_in_hdu = False
    for extname in flux_HDU_names:
        if extname in HDUlist:
            data = HDUlist[extname].data
            data_hdr = HDUlist[extname].header
            data_in_hdu = True
    if not data_in_hdu:
        raise FormatError("Could not find Flux Array")

    error_in_hdu = False
    for extname in error_HDU_names:
        if extname in HDUlist:
            if extname == 'IVAR':
                error = 1./np.sqrt(HDUlist[extname].data)
            elif extname == 'VAR':
                error = np.sqrt(HDUlist[extname].data)
            else:
                error = HDUlist[extname].data
            error_in_hdu = True
    if not error_in_hdu:
        raise FormatError("Could not find Error Array")

    # Does the spectrum contain a pixel mask?
    extname = 'MASK'
    if extname in HDUlist:
        mask = HDUlist[extname].data
    else:
        mask = np.ones_like(data, dtype=bool)

    return data, error, mask, data_hdr

# Hack the doc-string of the function to input the variable names:
output_hdu_names = {'FLUX_HDU_NAMES': flux_HDU_names,
                    'ERR_HDU_NAMES': error_HDU_names}
get_spectrum_hdulist.__doc__ = get_spectrum_hdulist.__doc__ % output_hdu_names


def load_fits_spectrum(fname, ext=None, iraf_obj=None):
    """
    Flexible inference of spectral data from FITS files.
    The function allows to read a large number of spectral formats including
    FITS tables, multi extension FITS ImageHDUs, or IRAF like arrays

    Parameters
    ----------
    fname : string
        Filename for the FITS file to open
    ext : int or string
        Extension number (int) or Extension Name (string)
    iraf_obj : int
        Index of the IRAF array, e.g. the flux is found at index: [spectral_pixels, iraf_obj, 0]

    Returns
    -------
    wavelength : np.array (float)
        Numpy array of wavelengths.
    data : np.array (float)
        Numpy array of flux density.
    error : np.array (float)
        Numpy array of uncertainties on the flux density.
    mask : np.array (bool)
        Numpy boolean array of pixel mask. `True` if the pixel is 'good',
        `False` if the pixel is bad and should not be used.
    header : fits.Header
        FITS Header of the data extension.
    """
    msg = ""
    with fits.open(fname) as HDUlist:
        primhdr = HDUlist[0].header
        primary_has_data = HDUlist[0].data is not None
        if primary_has_data:
            if primhdr['NAXIS'] == 1:
                if len(HDUlist) == 1:
                    raise FormatError("Only one extension: Could not find both Flux and Error Arrays")

                elif len(HDUlist) == 2:
                    data = HDUlist[0].data
                    data_hdr = HDUlist[0].header
                    error = HDUlist[1].data
                    mask = np.ones_like(data, dtype=bool)

                elif len(HDUlist) > 2:
                    data, error, mask, data_hdr = get_spectrum_hdulist(HDUlist)

                try:
                    wavelength = get_wavelength_from_header(primhdr)
                    return wavelength, data, error, mask, primhdr, msg
                except WavelengthError:
                    wavelength = get_wavelength_from_header(data_hdr)
                    return wavelength, data, error, mask, data_hdr, msg
                else:
                    raise FormatError("Could not find Wavelength Array")

            elif primhdr['NAXIS'] == 2:
                raise FormatError("The data seems to be a 2D image of shape: {}".format(HDUlist[0].data.shape))

            elif primhdr['NAXIS'] == 3:
                # This could either be a data cube (such as SINFONI / MUSE)
                # or IRAF format:
                IRAF_in_hdr = 'IRAF' in primhdr.__repr__()
                has_CRVAL3 = 'CRVAL3' in primhdr.keys()
                if IRAF_in_hdr and not has_CRVAL3:
                    # This is most probably an IRAF spectrum file:
                    #  (N_pixels, N_objs, 4)
                    #  The 4 axes are [flux, flux_noskysub, sky_flux, error]
                    data_array = HDUlist[0].data
                    if iraf_obj is None:
                        # Use the first object by default
                        iraf_obj = 0
                        # If other objects are present, throw a warning:
                        if data_array.shape[1] > 1:
                            msg = "[WARNING] - More than one object detected in the file"

                    data = data_array[0][iraf_obj]
                    error = data_array[3][iraf_obj]
                    mask = np.ones_like(data, dtype=bool)
                    wavelength = get_wavelength_from_header(primhdr)
                    return wavelength, data, error, mask, primhdr, msg
                else:
                    raise FormatError("The data seems to be a 3D cube of shape: {}".format(HDUlist[0].data.shape))

        else:
            is_fits_table = isinstance(HDUlist[1], fits.BinTableHDU) or isinstance(HDUlist[1], fits.TableHDU)
            if is_fits_table:
                if ext:
                    tbdata = HDUlist[ext].data
                    data_hdr = HDUlist[ext].header
                else:
                    tbdata = HDUlist[1].data
                    data_hdr = HDUlist[1].header

                has_multi_extensions = len(HDUlist) > 2
                if has_multi_extensions and (ext is None):
                    msg = "[WARNING] - More than one data extension detected in the file"
                wavelength, data, error, mask = get_spectrum_fits_table(tbdata)
                return wavelength, data, error, mask, data_hdr, msg

            elif len(HDUlist) == 2:
                raise FormatError("Only one data extension: Could not find both Flux and Error Arrays")

            elif len(HDUlist) > 2:
                data, error, mask, data_hdr = get_spectrum_hdulist(HDUlist)
                try:
                    wavelength = get_wavelength_from_header(data_hdr)
                except WavelengthError:
                    wavelength = get_wavelength_from_header(primhdr)
                    data_hdr = primhdr
                else:
                    raise FormatError("Could not find Wavelength Array")
                return wavelength, data, error, mask, data_hdr, msg


def load_fits_image(fname):
    """Load a FITS image with an associated error extension and an optional data quality MASK."""
    with fits.open(fname) as hdu_list:
        image = hdu_list[0].data
        hdr = hdu_list[0].header
        # Loop through HDU list instead to check all
        # for hdu in hdu_list:
        if 'ERR' in hdu_list:
            error = hdu_list['ERR'].data
        elif len(hdu_list) > 1:
            error = hdu_list[1].data
        else:
            raise IndexError("No error image detected")

        if 'MASK' in hdu_list:
            mask = hdu_list['MASK'].data
        else:
            mask = np.zeros_like(image, dtype=bool)
    return image, error, mask, hdr


def verify_header_key(key):
    """If given a string with spaces or dots convert to HIERARCH ESO format"""
    check_ESO = False
    key = key.strip()
    if '.' in key:
        key = key.replace('.', ' ')
        check_ESO = True

    if ' ' in key:
        check_ESO = True

    if check_ESO:
        if not key.startswith('ESO'):
            key = 'ESO %s' % key

    return key


def fits_to_ascii(fname, output, keys=None):
    """
    Convert the input FITS file to an ASCII table.

    keys : list[string]
        List of header keywords to include in the table header
    """
    wl, flux, err, mask, hdr, msg = load_fits_spectrum(fname)
    if not keys:
        keys = []
    data = np.column_stack([wl, flux, err, mask])
    tbl_header = ""
    for key in keys:
        key = verify_header_key(key)
        if key in hdr:
            tbl_header += "# %s : %s \n" % (key, str(hdr[key]))
    tbl_header += "#-----------------------------------------\n"
    tbl_header += "# WAVE  FLUX  ERROR  MASK [0:good / 1:bad]\n"
    with open(output, 'w') as out:
        out.write(tbl_header)
        np.savetxt(out, data, fmt="%.4f  %+.4e  %.4e  %i")
