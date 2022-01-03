import numpy as np
from os.path import splitext
from astropy.table import Table


class DataColumnMissing(Exception):
    pass


def save_ascii_spectrum(fname, wl, flux, err, hdr=None, bg=None, mask=None):
    """Write spectrum to an ascii text file with header saved to separate text file."""
    fmt = "%12.4f  % .3e  %.3e"
    col_names = "# Wavelength       Flux        Error"
    data_cols = [wl, flux, err]
    if bg is not None:
        data_cols.append(bg)
        fmt += "  %.3e"
        col_names += "      Sky"

    if mask is not None:
        data_cols.append(mask)
        fmt += "  %i"
        col_names += "      Mask"

    data_table = np.column_stack(data_cols)
    with open(fname, 'w') as output:
        output.write(col_names + "\n")
        np.savetxt(output, data_table, fmt=fmt)

    if hdr is not None:
        basename, ext = splitext(fname)
        header_fname = basename + '_hdr.txt'
        hdr.tofile(header_fname, sep='\n', endcard=False, padding=False, overwrite=True)

    return True, "File saved successfully"


wavelength_column_names = ['wl', 'lam', 'lambda', 'loglam', 'wave', 'wavelength']
flux_column_names = ['data', 'spec', 'flux', 'flam', 'fnu', 'flux_density']
error_column_names = ['err', 'sig', 'error', 'ivar', 'sigma', 'var']
sky_column_names = ['sky', 'bg', 'background']
mask_column_names = ['mask']


def load_ascii_spectrum(fname):
    output_msg = ""
    data = Table.read(fname, format='ascii')
    wl, flux, err, sky, mask = (None, None, None, None, None)
    if len(data.colnames) < 3:
        err_msg = "Less than 3 columns identified! Must provide wave, flux and err..."
        raise DataColumnMissing(err_msg)

    identify_columns = True
    if 'col1' in data.colnames:
        output_msg = "          - No column names found. Using default order: WAVE, FLUX, ERR[, MASK, SKY]"
        identify_columns = False
        wl = data['col1']
        flux = data['col2']
        err = data['col3']
        if len(data.colnames) == 3:
            pass
        elif len(data.colnames) == 4:
            mask = data['col4']
        else:
            mask = data['col4']
            sky = data['col5']

    if identify_columns:
        for colname in data.colnames:
            if colname.lower() in wavelength_column_names:
                wl = data[colname]

            elif colname.lower() in flux_column_names:
                flux = data[colname]

            elif colname.lower() in error_column_names:
                if colname.lower() == 'ivar':
                    err = 1./np.sqrt(data[colname])
                elif colname.lower() == 'var':
                    err = np.sqrt(data[colname])
                else:
                    err = data[colname]

            elif colname.lower() in sky_column_names:
                sky = data[colname]

            elif colname.lower() in mask_column_names:
                mask = data[colname]

    if wl is None:
        raise DataColumnMissing("Could not identify any wavelength column")

    if flux is None:
        raise DataColumnMissing("Could not identify any flux column")

    if err is None:
        raise DataColumnMissing("Could not identify any error column")

    if mask is None:
        mask = np.zeros_like(flux, dtype=bool)

    if sky is None:
        sky = np.zeros_like(flux)

    output_msg = ''
    return (wl, flux, err, mask, sky, output_msg)
