#!/usr/bin/env python3
"""
    PyNOT -- Extract GUI

Graphical interface to extract 1D spectra

"""

__author__ = "Jens-Kristian Krogager"
__email__ = "krogager.jk@gmail.com"
__credits__ = ["Jens-Kristian Krogager"]

import copy
import os
import re
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.colors as mc
import colorsys
from lmfit import Parameters, minimize
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from numpy.polynomial import Chebyshev
from astropy.io import fits
from itertools import cycle
from PyQt5 import QtCore, QtGui, QtWidgets
import warnings

from pynot.functions import fix_nans, mad, tophat, NN_moffat, NN_gaussian


code_dir = os.path.dirname(os.path.abspath(__file__))
v_file = os.path.join(code_dir, 'VERSION')
with open(v_file) as version_file:
    __version__ = version_file.read().strip()


def run_gui(input_fname, output_fname, app=None, **ext_kwargs):
    # global app
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    gui = ExtractGUI(input_fname, output_fname=output_fname, dispaxis=1, locked=True, **ext_kwargs)
    gui.show()
    app.exit(app.exec_())
    del gui


def save_fits_spectrum(fname, wl, flux, err, hdr, bg=None, aper=None):
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
    hdu.writeto(fname, output_verify='silentfix', overwrite=True)
    return True, "File saved successfully"


def save_fitstable_spectrum(fname, wl, flux, err, hdr, bg=None, aper=None):
    """Write spectrum to a FITS Table with 4 columns: Wave, FLUX, ERR and SKY"""
    hdu = fits.HDUList()
    hdr['COMMENT'] = 'PyNOT extracted spectrum'
    hdr['COMMENT'] = 'Each spectrum in its own extension'
    if bg is None:
        bg = np.zeros_like(flux)
    col_wl = fits.Column(name='WAVE', array=wl, format='D')
    col_flux = fits.Column(name='FLUX', array=flux, format='D')
    col_err = fits.Column(name='ERR', array=err, format='D')
    col_sky = fits.Column(name='SKY', array=bg, format='D')
    tab = fits.BinTableHDU.from_columns([col_wl, col_flux, col_err, col_sky], header=hdr)
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


def save_ascii_spectrum(fname, wl, flux, err, hdr, bg=None):
    """Write spectrum to an ascii text file with header saved to separate text file."""
    if bg is not None:
        data_table = np.column_stack([wl, flux, err, bg])
        fmt = "%12.4f  % .3e  %.3e  %.3e"
        col_names = "# Wavelength  Flux        Flux_err   Sky"
    else:
        data_table = np.column_stack([wl, flux, err])
        fmt = "%12.4f  % .3e  %.3e"
        col_names = "# Wavelength  Flux        Flux_err"

    basename, ext = os.path.splitext(fname)
    header_fname = basename + '_hdr.txt'

    with open(fname, 'w') as output:
        output.write(col_names + "\n")
        np.savetxt(output, data_table, fmt=fmt)
    hdr.tofile(header_fname, sep='\n', endcard=False, padding=False, overwrite=True)
    return True, "File saved successfully"


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
    """
    if x is None:
        x = np.arange(len(y))

    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]

    if np.sum(zero_crossings) != 2:
        msg = "[WARNING] - automatic determination of FWHM failed. Using default of 5 pixels"
        fwhm = 5
        return fwhm, msg

    halfmax_x = list()
    for i in zero_crossings_i:
        x_i = x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))
        halfmax_x.append(x_i)
    fwhm = halfmax_x[1] - halfmax_x[0]
    msg = ''
    return fwhm, msg


def color_shade(color, amount=1.2):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def make_linear_colormap(color, N=256):
    lum = np.linspace(0., 1., 256)
    c = np.array(mc.to_rgb(color))
    col_array = np.outer(lum, c-1.) + 1.
    return mc.ListedColormap(col_array)

color_list = [
    "#3949AB",
    "#009688",
    "#D81B60",
    "#8E24AA",
    "#FDD835",
]
# Create iterative color-cycle:
color_cycle = cycle(color_list)


def median_filter_data(x, kappa=5., window=21, parname=None):
    med_x = median_filter(x, window)
    MAD = np.median(np.abs(x - med_x))*1.48
    if MAD == 0.:
        MAD = np.std(x - med_x)
    mask = np.abs(x - med_x) < kappa*MAD
    return (med_x, mask)


def gui_label(text, color='black'):
    label_string = "%s" % (text)
    return QtWidgets.QLabel(label_string)


def get_wavelength_from_header(hdr, dispaxis=1):
    """Get image axes from FITS header"""
    if 'CD1_1' in hdr.keys():
        # Use CD matrix:
        cdelt1 = hdr['CD%i_%i' % (dispaxis, dispaxis)]
        crval1 = hdr['CRVAL%i' % dispaxis]
        crpix1 = hdr['CRPIX%i' % dispaxis]
        wavelength = (np.arange(hdr['NAXIS%i' % dispaxis]) - (crpix1 - 1))*cdelt1 + crval1

    elif 'CDELT1' in hdr.keys():
        cdelt1 = hdr['CDELT%i' % dispaxis]
        crval1 = hdr['CRVAL%i' % dispaxis]
        crpix1 = hdr['CRPIX%i' % dispaxis]
        wavelength = (np.arange(hdr['NAXIS%i' % dispaxis]) - (crpix1 - 1))*cdelt1 + crval1

    else:
        # No info, just return pixels:
        wavelength = np.arange(hdr['NAXIS%i' % dispaxis]) + 1.

    return wavelength


def trace_function(pars, x, N, model_type='moffat'):
    model = np.zeros_like(x)
    if model_type == 'gaussian':
        for i in range(N):
            p = [pars['mu_%i' % i],
                 pars['sig_%i' % i],
                 pars['logamp_%i' % i]]
            model += NN_gaussian(x, *p)

    elif model_type == 'moffat':
        for i in range(N):
            p = [pars['mu_%i' % i],
                 pars['a_%i' % i],
                 pars['b_%i' % i],
                 pars['logamp_%i' % i]]
            model += NN_moffat(x, *p)
    model += pars['bg']
    return model


def model_residuals(pars, x, y, N, model_type='moffat'):
    return y - trace_function(pars, x, N, model_type=model_type)


def prep_parameters(peaks, prominence, size=np.inf, model_type='moffat', tie_traces=True):
    values = zip(peaks, prominence)
    pars = Parameters()
    pars.add('bg', value=0.)
    if model_type == 'gaussian':
        for i, (x0, amp) in enumerate(values):
            pars.add('mu_%i' % i, value=float(x0), min=0., max=size)
            pars.add('sig_%i' % i, value=2., min=0., max=20.)
            pars.add('logamp_%i' % i, value=np.log10(amp))
    elif model_type == 'moffat':
        for i, (x0, amp) in enumerate(values):
            pars.add('mu_%i' % i, value=float(x0), min=0., max=size)
            pars.add('a_%i' % i, value=2., min=0., max=20.)
            pars.add('b_%i' % i, value=1., min=0., max=20.)
            pars.add('logamp_%i' % i, value=np.log10(amp))
    if len(peaks) > 1 and tie_traces is True:
        # Define constraints such that mu_1 is a constant offset from mu_0
        for num in range(1, len(peaks)):
            dmu = pars['mu_%i' % num].value - pars['mu_0'].value
            pars.add('dmu_%i' % num, value=dmu, min=dmu-2, max=dmu+2)
            pars['mu_%i' % num].expr = 'mu_0 + dmu_%i' % num
    return pars


def auto_localize(img2D, settings):
    img2D = img2D.astype(np.float64)

    spsf = np.median(img2D, axis=1)
    spsf = spsf - np.median(spsf)

    # Detect peaks:
    noise = mad(spsf) * 1.48
    if noise == 0.:
        noise = np.std(spsf)
    peaks, properties = find_peaks(spsf, prominence=settings['LOCALIZE_THRESHOLD']*noise,
                                   width=settings['LOCALIZE_MIN_WIDTH'],
                                   distance=settings['LOCALIZE_MIN_SEP'])
    if len(peaks) > 0:
        prominences = properties['prominences']
    else:
        prominences = []
    return (peaks, prominences)


class BackgroundModel(object):
    def __init__(self, axis, shape, order=3):
        """Must be tied to the `axis` in the GUI which displays the SPSF"""
        self.ranges = list()
        self.order = order
        self.model2d = np.zeros(shape)
        self.x = np.arange(shape[1])
        self.y = np.arange(shape[0])
        self.vlines = list()
        self.patches = list()
        self.axis = axis

    def add_range(self, i_low, i_high):
        for num, (lower, upper) in enumerate(self.ranges):
            if (lower <= i_low <= upper) or (lower <= i_high <= upper):
                # merge ranges:
                i_low = min(i_low, lower)
                i_high = max(i_high, upper)
                self.set_range(num, i_low, i_high)
                return

        self.ranges.append([i_low, i_high])
        v1 = self.axis.axvline(i_low, color='#29b6f6', ls=':', lw=0.8)
        v2 = self.axis.axvline(i_high, color='#29b6f6', ls=':', lw=0.8)
        patch = self.axis.axvspan(i_low, i_high, color='#29b6f6', alpha=0.3, picker=True)
        self.patches.append(patch)
        self.vlines.append([v1, v2])

    def set_range(self, index, i_low, i_high):
        self.patches[index].remove()
        self.patches.pop(index)
        self.ranges[index] = [i_low, i_high]
        self.vlines[index][0].set_xdata(i_low)
        self.vlines[index][1].set_xdata(i_high)
        patch = self.axis.axvspan(i_low, i_high, color='#29b6f6', alpha=0.3, picker=True)
        self.patches.insert(index, patch)

    def remove_range(self, index):
        self.patches[index].remove()
        self.patches.pop(index)
        for vline in self.vlines[index]:
            vline.remove()
        self.vlines.pop(index)
        self.ranges.pop(index)

    def clear(self):
        for num, range in enumerate(self.ranges):
            self.remove_range(num)


class TraceModel(object):
    def __init__(self, cen, amp, axis, shape, fwhm=5, model_type='Moffat', object_name="", xmin=None, xmax=None, color='RoyalBlue'):
        self.model_type = model_type
        self._original_type = model_type
        self.xmin = xmin
        self.xmax = xmax
        self.color = color
        self.cmap = make_linear_colormap(color)
        self.cen = cen
        self.lower = cen - fwhm
        self.upper = cen + fwhm
        self.amp = amp
        self.model2d = np.zeros(shape, dtype=np.float64)
        self.x = np.arange(shape[1], dtype=np.float64)
        self.y = np.arange(shape[0], dtype=np.float64)
        self.axis = axis
        self.fixed = False
        self.object_name = object_name
        self.active = True

        self.x_binned = np.array([])
        self.mask = {'mu': np.array([], dtype=bool), 'sigma': np.array([], dtype=bool),
                     'alpha': np.array([], dtype=bool), 'beta': np.array([], dtype=bool)
                     }
        self.points = {'mu': np.array([]), 'sigma': np.array([]),
                       'alpha': np.array([]), 'beta': np.array([])
                       }
        self.fit = {'mu': np.array([]), 'sigma': np.array([]),
                    'alpha': np.array([]), 'beta': np.array([])
                    }

        # -- Define artists for the axes:
        v_cen = self.axis.axvline(self.cen, color=color, lw=1., ls='-', picker=True, label='center')
        v_lower = self.axis.axvline(self.lower, color=color, lw=0.8, ls='--', picker=True, label='lower')
        v_upper = self.axis.axvline(self.upper, color=color, lw=0.8, ls='--', picker=True, label='upper')
        self.vlines = [v_lower, v_cen, v_upper]

        # Collector for points:
        self.point_lines = {'mu': [], 'sigma': [],
                            'alpha': [], 'beta': []}
        self.fit_lines = {'mu': None, 'sigma': None,
                          'alpha': None, 'beta': None}
        self.model_image = None
        self.plot_1d = None

    def set_object_name(self, name):
        self.object_name = name

    def set_color(self, color):
        self.color = color
        self.cmap = make_linear_colormap(color)
        for vline in self.vlines:
            vline.set_color(color)
        for line_collection in self.point_lines.values():
            if len(line_collection) > 0:
                for line in line_collection:
                    line.set_color(color)
        for line in self.fit_lines.values():
            if line is not None:
                line.set_color(color)
        if self.model_image is not None:
            self.model_image.set_cmap(self.cmap)
        if self.plot_1d is not None:
            for child in self.plot_1d.get_children():
                child.set_color(color)

    def set_data(self, x, mu=None, alpha=None, beta=None, sigma=None):
        self.x_binned = x
        if mu is not None:
            self.points['mu'] = mu
            self.mask['mu'] = np.ones_like(mu, dtype=bool)
        if alpha is not None:
            self.points['alpha'] = alpha
            self.mask['alpha'] = np.ones_like(alpha, dtype=bool)
        if beta is not None:
            self.points['beta'] = beta
            self.mask['beta'] = np.ones_like(beta, dtype=bool)
        if sigma is not None:
            self.points['sigma'] = sigma
            self.mask['sigma'] = np.ones_like(sigma, dtype=bool)

    def get_unicode_name(self, parname):
        unicode_names = {'mu': 'centroid', 'sigma': 'Gaussian σ',
                         'alpha': 'Moffat α', 'beta': 'Moffat β'}
        return unicode_names[parname]

    def get_data(self):
        if self.model_type.lower() == 'moffat':
            return (self.x_binned, self.points['mu'], self.points['alpha'], self.points['beta'])
        elif self.model_type.lower() == 'gaussian':
            return (self.x_binned, self.points['mu'], self.points['sigma'])
        elif self.model_type.lower() == 'tophat':
            return (self.x_binned, self.points['mu'])

    def get_parnames(self):
        if self.model_type.lower() == 'moffat':
            return ['mu', 'alpha', 'beta']
        elif self.model_type.lower() == 'gaussian':
            return ['mu', 'sigma']
        elif self.model_type.lower() == 'tophat':
            return ['mu']

    def set_centroid(self, cen):
        self.cen = cen
        self.vlines[1].set_xdata(cen)

    def set_range(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.vlines[0].set_xdata(lower)
        self.vlines[2].set_xdata(upper)

    def get_range(self):
        return (self.lower, self.upper)

    def get_mask(self, parname):
        if len(self.mask[parname]) == 0:
            return self.mask[parname]

        if self.xmin is None:
            xmin = 0.
        else:
            xmin = self.xmin

        if self.xmax is None:
            xmax = np.max(self.x_binned)
        else:
            xmax = self.xmax
        limit_mask = (self.x_binned >= xmin) & (self.x_binned <= xmax)
        base_mask = self.mask[parname]
        return base_mask & limit_mask

    def clear_plot(self):
        for vline in self.vlines:
            vline.remove()
        for line_collection in self.point_lines.values():
            if len(line_collection) > 0:
                for line in line_collection:
                    line.remove()
        for line in self.fit_lines.values():
            if line is not None:
                line.remove()
        if self.model_image is not None:
            self.model_image.remove()
        if self.plot_1d is not None:
            for child in self.plot_1d.get_children():
                child.remove()

    def deactivate(self):
        self.active = False
        for vline in self.vlines:
            vline.set_visible(False)
        for line_collection in self.point_lines.values():
            if len(line_collection) > 0:
                for line in line_collection:
                    line.set_visible(False)
        for line in self.fit_lines.values():
            if line is not None:
                line.set_visible(False)
        if self.model_image is not None:
            self.model_image.set_visible(False)
        if self.plot_1d is not None:
            for child in self.plot_1d.get_children():
                child.set_visible(False)

    def activate(self):
        self.active = True
        self.vlines[1].set_visible(True)
        for line_collection in self.point_lines.values():
            if len(line_collection) > 0:
                for line in line_collection:
                    line.set_visible(True)
        for line in self.fit_lines.values():
            if line is not None:
                line.set_visible(True)
        if self.model_image is not None:
            self.model_image.set_visible(True)
        if self.plot_1d is not None:
            for child in self.plot_1d.get_children():
                child.set_visible(True)

    def set_visible(self, vis=True):
        self.vlines[0].set_visible(vis)
        self.vlines[2].set_visible(vis)

    def copy(self, offset=20.):
        new_trace_model = TraceModel(self.cen + offset, self.amp, self.axis, model_type=self.model_type,
                                     shape=self.model2d.shape, color=next(color_cycle))
        lower, upper = self.get_range()
        new_trace_model.set_range(lower+offset, upper+offset)
        new_trace_model.x_binned = self.x_binned.copy()
        new_trace_model.fixed = True
        for param in self.points.keys():
            new_trace_model.points[param] = self.points[param].copy()
            new_trace_model.fit[param] = self.fit[param].copy()
            new_trace_model.mask[param] = self.mask[param].copy()
            if param == 'mu':
                new_trace_model.points[param] += offset
                new_trace_model.fit[param] += offset
        return new_trace_model


class ImageData(object):
    def __init__(self, fname, dispaxis=1):
        self.filename = fname
        data_temp = fits.getdata(fname)
        self.data = data_temp.astype(np.float64)
        self.shape = self.data.shape
        try:
            self.error = fits.getdata(fname, 1)
        except:
            # noise = mad(self.data) * 1.48
            # self.error = np.ones_like(self.data) * noise
            noise_tmp = self.data.copy()
            noise_tmp[noise_tmp <= 0.] = 1.e3
            self.error = np.sqrt(noise_tmp)

        with fits.open(fname) as hdu:
            self.header = hdu[0].header
            if len(hdu) > 1:
                imghdr = hdu[1].header
                if hdu[1].name not in ['ERR', 'MASK']:
                    self.header.update(imghdr)

        if 'DISPAXIS' in self.header:
            dispaxis = self.header['DISPAXIS']

        if dispaxis == 2:
            self.wl = get_wavelength_from_header(self.header, dispaxis)
            self.data = self.data.T
            self.error = self.error.T
            self.shape = self.data.shape
            self.wl_unit = self.header['CUNIT2']
        else:
            self.wl = get_wavelength_from_header(self.header, 1)
            self.wl_unit = self.header['CUNIT1']
        self.x = np.arange(self.data.shape[1], dtype=np.float64)
        self.y = np.arange(self.data.shape[0], dtype=np.float64)
        self.flux_unit = self.header['BUNIT']



class Spectrum(object):
    def __init__(self, wl=None, data=None, error=None, mask=None, hdr={}, bg=None, wl_unit='', flux_unit=''):
        self.wl = wl
        self.data = data
        self.error = error
        self.mask = mask
        self.hdr = hdr
        self.background = bg
        self.wl_unit = wl_unit
        self.flux_unit = flux_unit

        self.plot_line = None


default_settings = {'BACKGROUND_POLY_ORDER': 3,
                    'BACKGROUND_MED_KAPPA': 5,
                    'LOCALIZE_THRESHOLD': 10.,
                    'LOCALIZE_MIN_SEP': 5,
                    'LOCALIZE_MIN_WIDTH': 3,
                    }

options_descriptions = {'BACKGROUND_POLY_ORDER': "Polynomial Order of Rows in Background Model",
                        'BACKGROUND_MED_KAPPA': "Median Filter for Rows in Background Model",
                        'LOCALIZE_THRESHOLD': "Significance for Object Detection",
                        'LOCALIZE_MIN_SEP': "Minimum Separation between Objects",
                        'LOCALIZE_MIN_WIDTH': "Minimum FWHM of Object Trace",
                        }


class ExtractGUI(QtWidgets.QMainWindow):
    def __init__(self, fname=None, dispaxis=1, model_name='moffat', dx=25, width_scale=2., xmin=0, xmax=None, ymin=0, ymax=None, order_center=3, order_width=0, parent=None, locked=False, output_fname='', **kwargs):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setWindowTitle('PyNOT: Extract')
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        # Set attributes:
        self.image2d = None
        self.background = None
        self.last_fit = tuple()
        self.filename_2d = fname
        self.output_fname = output_fname
        self.dispaxis = 1
        self.settings = default_settings
        self.first_time_open = True

        self.model_type = 'Moffat'
        self.data1d = list()
        self.background = None
        self.trace_models = list()
        self.delete_picked_object = False
        self.picked_object = None
        self.state = None
        self.bg_value1 = None
        self.xmask = list()

        # SPSF controls:
        self.add_btn = QtWidgets.QPushButton("Add Object")
        self.add_btn.clicked.connect(lambda x: self.set_state('add'))
        self.add_btn.setShortcut("ctrl+A")
        self.add_bg_btn = QtWidgets.QPushButton("Select Background")
        self.add_bg_btn.setShortcut("ctrl+B")
        self.add_bg_btn.clicked.connect(lambda x: self.set_state('bg1'))

        self.remove_btn = QtWidgets.QPushButton("Delete Object")
        self.remove_btn.setShortcut("ctrl+D")
        self.remove_btn.clicked.connect(lambda x: self.set_state('delete'))
        QtWidgets.QShortcut(QtGui.QKeySequence("Escape"), self, activated=self.clear_state)


        # Limits for profile averaging and fitting:
        self.xmin_edit = QtWidgets.QLineEdit("%i" % xmin)
        if xmax is not None:
            self.xmax_edit = QtWidgets.QLineEdit("%i" % xmax)
        else:
            self.xmax_edit = QtWidgets.QLineEdit("")
        self.xmin_edit.setValidator(QtGui.QIntValidator(0, 1000000))
        self.xmax_edit.setValidator(QtGui.QIntValidator(0, 1000000))
        self.xmin_edit.returnPressed.connect(self.limits_updated)
        self.xmax_edit.returnPressed.connect(self.limits_updated)

        self.ymin_edit = QtWidgets.QLineEdit("%i" % ymin)
        if ymax is not None:
            self.ymax_edit = QtWidgets.QLineEdit("%i" % ymax)
        else:
            self.ymax_edit = QtWidgets.QLineEdit("")
        self.ymin_edit.setValidator(QtGui.QIntValidator(0, 1000000))
        self.ymax_edit.setValidator(QtGui.QIntValidator(0, 1000000))
        self.ymin_edit.returnPressed.connect(self.limits_updated)
        self.ymax_edit.returnPressed.connect(self.limits_updated)

        # Fitting Parameters:
        self.model_chooser = QtWidgets.QComboBox()
        self.model_chooser.addItems(["Moffat", "Gaussian", "Tophat"])
        self.model_chooser.setCurrentText(model_name.title())
        # self.model_chooser.currentTextChanged.connect(self.model_change)

        self.bins_edit = QtWidgets.QLineEdit("%i" % dx)
        self.bins_edit.setValidator(QtGui.QIntValidator(0, 9999))
        self.bins_edit.returnPressed.connect(self.fit_trace)

        self.med_kappa_edit = QtWidgets.QLineEdit("3")
        self.med_kappa_edit.setValidator(QtGui.QDoubleValidator())
        self.med_kappa_edit.returnPressed.connect(self.median_filter_points)

        self.med_window_edit = QtWidgets.QLineEdit("11")
        self.med_window_edit.setValidator(QtGui.QIntValidator(3, 1000))
        self.med_window_edit.returnPressed.connect(self.median_filter_points)

        self.median_btn = QtWidgets.QPushButton("Median Filter Points")
        self.median_btn.clicked.connect(self.median_filter_points)

        self.c_order_edit = QtWidgets.QLineEdit("%i" % order_center)
        self.c_order_edit.setValidator(QtGui.QIntValidator(0, 100))
        self.c_order_edit.returnPressed.connect(self.fit_trace)
        self.w_order_edit = QtWidgets.QLineEdit("%i" % order_width)
        self.w_order_edit.setValidator(QtGui.QIntValidator(0, 100))
        self.w_order_edit.returnPressed.connect(self.fit_trace)

        self.extract_btn = QtWidgets.QPushButton("Extract 1D Spectrum")
        self.extract_btn.setShortcut("ctrl+E")
        self.extract_btn.clicked.connect(self.extract)

        self.fit_btn = QtWidgets.QPushButton("Fit Spectral Trace")
        self.fit_btn.setShortcut("ctrl+F")
        self.fit_btn.clicked.connect(self.fit_trace)

        # SPSF Viewer:
        self.fig_spsf = Figure(figsize=(4, 3))
        self.canvas_spsf = FigureCanvas(self.fig_spsf)
        self.canvas_spsf.mpl_connect('key_press_event', self.on_key_press)
        self.canvas_spsf.mpl_connect('pick_event', self.on_pick)
        self.canvas_spsf.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas_spsf.mpl_connect('button_release_event', self.on_release)
        self.canvas_spsf.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas_spsf.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.spsf_mpl_toolbar = NavigationToolbar(self.canvas_spsf, self)
        self.spsf_mpl_toolbar.setFixedHeight(20)
        # self.spsf_mpl_toolbar.setFixedWidth(400)
        self.axis_spsf = self.fig_spsf.add_subplot(111)
        self.axis_spsf.axhline(0., color='k', ls=':', alpha=0.7, lw=0.9)
        self.spsf_data_line = None
        self.spsf_bg_line = None

        # List of Trace Models:
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list_widget.itemChanged.connect(self.toggle_trace_models)
        self.list_widget.customContextMenuRequested.connect(self.listItemRightClicked)
        self.list_widget.itemDoubleClicked.connect(self.listItemDoubleClicked)


        # == Tab Widget =======================================================
        self.tab_widget = QtWidgets.QTabWidget(self._main)
        # -- Tab 1: (2D view)
        self.tab1 = QtWidgets.QWidget()
        self.tab_widget.addTab(self.tab1, "2D View")
        self.figure_2d = Figure(figsize=(8, 6))
        self.canvas_2d = FigureCanvas(self.figure_2d)
        self.canvas_2d.mpl_connect('pick_event', self.pick_points)
        self.fig2d_mpl_toolbar = NavigationToolbar(self.canvas_2d, self.tab1)
        self.fig2d_mpl_toolbar.setFixedHeight(20)
        self.axis_2d = self.figure_2d.add_subplot(211)
        self.axis_2d.set_ylabel("Data")
        self.axis_2d.autoscale_view()
        self.axis_2d_bg = self.figure_2d.add_subplot(212)
        self.axis_2d_bg.set_ylabel("Background")
        self.axis_2d_bg.autoscale_view()
        self.figure_2d.tight_layout()

        self.vmin_edit = QtWidgets.QLineEdit("")
        self.vmin_edit.setAlignment(QtCore.Qt.AlignCenter)
        self.vmax_edit = QtWidgets.QLineEdit("")
        self.vmax_edit.setAlignment(QtCore.Qt.AlignCenter)
        self.vmin_edit.returnPressed.connect(self.update_2d)
        self.vmax_edit.returnPressed.connect(self.update_2d)
        self.vminmax_btn = QtWidgets.QPushButton("Update Plot")
        self.vminmax_btn.clicked.connect(self.update_2d)
        self.vminmax_reset_btn = QtWidgets.QPushButton("Reset ranges")
        self.vminmax_reset_btn.clicked.connect(lambda x: self.update_value_range())
        self.vmin_edit.setValidator(QtGui.QDoubleValidator())
        self.vmax_edit.setValidator(QtGui.QDoubleValidator())
        self.bg_fit_btn = QtWidgets.QPushButton("Fit Background")
        self.bg_fit_btn.setShortcut("ctrl+shift+f")
        self.bg_fit_btn.clicked.connect(self.fit_background)
        row_imvals = QtWidgets.QHBoxLayout()
        row_imvals.addWidget(gui_label("v<sub>min</sub> =", color='#111111'))
        row_imvals.addWidget(self.vmin_edit)
        row_imvals.addWidget(gui_label("v<sub>max</sub> =", color='#111111'))
        row_imvals.addWidget(self.vmax_edit)
        row_imvals.addWidget(self.vminmax_btn)
        row_imvals.addWidget(self.vminmax_reset_btn)
        row_imvals.addStretch(1)
        row_imvals.addWidget(self.bg_fit_btn)

        layout_tab1 = QtWidgets.QVBoxLayout()
        layout_tab1.addWidget(self.canvas_2d)
        layout_tab1.addWidget(self.fig2d_mpl_toolbar)
        self.tab1.setLayout(layout_tab1)
        layout_tab1.addLayout(row_imvals)


        # -- Tab 2: (Points view)
        self.tab2 = QtWidgets.QWidget()
        self.tab_widget.addTab(self.tab2, "Fitting Points")
        self.figure_points = Figure(figsize=(8, 6))
        self.axes_points = self.figure_points.subplots(3, 1)
        self.axes_points[0].set_title("Profile Parameters along Dispersion Axis", fontsize=11)
        self.canvas_points = FigureCanvas(self.figure_points)
        self.canvas_points.mpl_connect('pick_event', self.pick_points)
        self.figp_mpl_toolbar = NavigationToolbar(self.canvas_points, self)
        self.figp_mpl_toolbar.setFixedHeight(20)

        row_median = QtWidgets.QHBoxLayout()
        row_median.addWidget(self.median_btn)
        row_median.addStretch(1)
        row_median.addWidget(QtWidgets.QLabel("Kappa: "))
        row_median.addWidget(self.med_kappa_edit)
        row_median.addWidget(QtWidgets.QLabel("Filter Width: "))
        row_median.addWidget(self.med_window_edit)
        row_median.addStretch(1)

        layout_tab2 = QtWidgets.QVBoxLayout()
        layout_tab2.addWidget(self.canvas_points)
        layout_tab2.addWidget(self.figp_mpl_toolbar)
        layout_tab2.addLayout(row_median)
        self.tab2.setLayout(layout_tab2)

        # -- Tab 3: (1D view)
        self.tab3 = QtWidgets.QWidget()
        self.tab_widget.addTab(self.tab3, "1D View")
        self.figure_1d = Figure(figsize=(8, 6))
        self.axis_1d = self.figure_1d.add_subplot(111)
        self.axis_1d.axhline(0., ls=':', color='black', lw=0.5, alpha=0.5)
        self.axis_1d.set_xlabel("Wavelength", fontsize=11)
        self.axis_1d.set_ylabel("Flux", fontsize=11)
        self.canvas_1d = FigureCanvas(self.figure_1d)
        self.fig1d_mpl_toolbar = NavigationToolbar(self.canvas_1d, self)
        self.fig1d_mpl_toolbar.setFixedHeight(20)
        layout_tab3 = QtWidgets.QVBoxLayout()
        layout_tab3.addWidget(self.canvas_1d)
        layout_tab3.addWidget(self.fig1d_mpl_toolbar)
        self.tab3.setLayout(layout_tab3)

        self.tab_shortcut1 = QtWidgets.QShortcut(self)
        self.tab_shortcut1.setKey(QtGui.QKeySequence("Ctrl+1"))
        self.tab_shortcut1.activated.connect(lambda: self.tab_widget.setCurrentIndex(0))

        self.tab_shortcut2 = QtWidgets.QShortcut(self)
        self.tab_shortcut2.setKey(QtGui.QKeySequence("Ctrl+2"))
        self.tab_shortcut2.activated.connect(lambda: self.tab_widget.setCurrentIndex(1))

        self.tab_shortcut3 = QtWidgets.QShortcut(self)
        self.tab_shortcut3.setKey(QtGui.QKeySequence("Ctrl+3"))
        self.tab_shortcut3.activated.connect(lambda: self.tab_widget.setCurrentIndex(2))

        # == TOP MENU BAR:
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.clicked.connect(self.save_spectrum_1d)
        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.clicked.connect(self.load_spectrum)
        self.options_btn = QtWidgets.QPushButton("Options")
        self.options_btn.clicked.connect(lambda checked: SettingsWindow(self))
        if locked:
            self.close_btn = QtWidgets.QPushButton("Done")
            self.close_btn.clicked.connect(self.done)
            self.load_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.save_btn.setText("")
        else:
            self.close_btn = QtWidgets.QPushButton("Close")
            self.close_btn.clicked.connect(self.close)


        # == Layout ===========================================================
        super_layout = QtWidgets.QVBoxLayout(self._main)

        top_menubar = QtWidgets.QHBoxLayout()
        top_menubar.addWidget(self.close_btn)
        top_menubar.addWidget(self.save_btn)
        top_menubar.addWidget(self.load_btn)
        top_menubar.addWidget(self.options_btn)
        top_menubar.addStretch(1)

        top_menubar.addWidget(self.extract_btn)
        top_menubar.addStretch(1)

        top_menubar.addWidget(self.add_btn)
        top_menubar.addWidget(self.remove_btn)
        top_menubar.addWidget(self.add_bg_btn)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setSpacing(2)
        super_layout.setContentsMargins(2, 2, 2, 2)
        super_layout.setSpacing(0)

        super_layout.addLayout(top_menubar)
        super_layout.addLayout(main_layout)

        # TabWidget Layout:
        main_layout.addWidget(self.tab_widget, 1)

        # Right Panel Layout:
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setContentsMargins(5, 5, 5, 5)
        right_panel.addWidget(self.canvas_spsf)
        right_panel.addWidget(self.spsf_mpl_toolbar)

        separatorLine = QtWidgets.QFrame()
        separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        separatorLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        separatorLine.setMinimumSize(3, 20)
        right_panel.addWidget(separatorLine)

        right_panel.addWidget(gui_label("SPSF Ranges", color='dimgray'))

        row_xr = QtWidgets.QHBoxLayout()
        row_xr.addWidget(QtWidgets.QLabel("X-min: "))
        row_xr.addWidget(self.xmin_edit)
        row_xr.addStretch(1)
        row_xr.addWidget(QtWidgets.QLabel("X-max: "))
        row_xr.addWidget(self.xmax_edit)
        row_xr.addStretch(1)
        right_panel.addLayout(row_xr)

        row_yr = QtWidgets.QHBoxLayout()
        row_yr.addWidget(QtWidgets.QLabel("Y-min: "))
        row_yr.addWidget(self.ymin_edit)
        row_yr.addStretch(1)
        row_yr.addWidget(QtWidgets.QLabel("Y-max: "))
        row_yr.addWidget(self.ymax_edit)
        row_yr.addStretch(1)
        right_panel.addLayout(row_yr)

        separatorLine = QtWidgets.QFrame()
        separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        separatorLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        separatorLine.setMinimumSize(3, 20)
        right_panel.addWidget(separatorLine)

        row_model = QtWidgets.QHBoxLayout()
        row_model.addWidget(QtWidgets.QLabel("SPSF Model: "))
        row_model.addWidget(self.model_chooser)
        row_model.addStretch(1)
        row_model.addWidget(QtWidgets.QLabel("Bin size: "))
        row_model.addWidget(self.bins_edit)
        row_model.addStretch(1)
        right_panel.addLayout(row_model)

        separatorLine = QtWidgets.QFrame()
        separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        separatorLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        separatorLine.setMinimumSize(3, 20)
        right_panel.addWidget(separatorLine)

        right_panel.addWidget(gui_label("Polynomial Orders", color='dimgray'))

        row_orders = QtWidgets.QHBoxLayout()
        row_orders.addWidget(QtWidgets.QLabel("Centroid:"))
        row_orders.addWidget(self.c_order_edit)
        row_orders.addStretch(1)
        row_orders.addWidget(QtWidgets.QLabel("Width:"))
        row_orders.addWidget(self.w_order_edit)
        row_orders.addStretch(1)
        right_panel.addLayout(row_orders)


        row_fit = QtWidgets.QHBoxLayout()
        row_fit.addStretch(1)
        row_fit.addWidget(self.fit_btn)
        row_fit.addStretch(1)
        # row_fit.addWidget(self.extract_btn)
        right_panel.addLayout(row_fit)

        separatorLine = QtWidgets.QFrame()
        separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        separatorLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        separatorLine.setMinimumSize(3, 20)
        right_panel.addWidget(separatorLine)

        right_panel.addWidget(gui_label("List of Extraction Apertures", color='dimgray'))
        right_panel.addWidget(self.list_widget)

        main_layout.addLayout(right_panel)

        self.canvas_2d.setFocus()

        self.create_menu()

        # -- Set Data:
        if fname:
            self.load_spectrum(fname, dispaxis)
            self.filename_2d = fname

    def done(self):
        success = self.save_all_extractions(self.output_fname)
        if success:
            self.close()

    def save_aperture_model(self, index):
        """Save the 2D trace model profile of a given object"""
        current_dir = './' + self.output_fname
        filters = "FITS Files (*.fits *.fit)"
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save 2D Trace Models', current_dir, filters)
        if fname and len(self.trace_models) > 0:
            model = self.trace_models[index]
            hdu = fits.HDUList()
            prim_hdr = fits.Header()
            prim_hdr['AUTHOR'] = 'PyNOT'
            prim_hdr['OBJECT'] = self.image2d.header['OBJECT']
            prim_hdr['DATE-OBS'] = self.image2d.header['DATE-OBS']
            prim_hdr['RA'] = self.image2d.header['RA']
            prim_hdr['DEC'] = self.image2d.header['DEC']
            prim_hdr['COMMENT'] = 'PyNOT extraction aperture'
            prim_hdr['APERTYPE'] = model.model_type
            prim = fits.PrimaryHDU(data=model.model2d, header=prim_hdr)
            hdu.append(prim)
            hdu.writeto(fname, overwrite=True)

    def save_all_aperture_models(self):
        """Save the 2D trace model profile"""
        current_dir = './'
        filters = "FITS Files (*.fits *.fit)"
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save 2D Trace Models', current_dir, filters)
        if fname and len(self.trace_models) > 0:
            hdu = fits.HDUList()
            prim_hdr = fits.Header()
            prim_hdr['AUTHOR'] = 'PyNOT'
            prim_hdr['OBJECT'] = self.image2d.header['OBJECT']
            prim_hdr['DATE-OBS'] = self.image2d.header['DATE-OBS']
            prim_hdr['RA'] = self.image2d.header['RA']
            prim_hdr['DEC'] = self.image2d.header['DEC']
            prim_hdr['COMMENT'] = 'PyNOT extraction aperture'
            prim = fits.PrimaryHDU(header=prim_hdr)
            hdu.append(prim)
            for num, model in enumerate(self.trace_models):
                hdr = fits.Header()
                hdr['AUTHOR'] = 'PyNOT'
                hdr['APERTYPE'] = model.model_type
                ext = fits.ImageHDU(model.model2d, name='OBJ%i' % (num+1), header=hdr)
                hdu.append(ext)
            hdu.writeto(fname, overwrite=True)

    def save_spectrum_2d(self):
        """Save the background subtracted 2D spectrum"""
        current_dir = './skysub_2d.fits'
        filters = "FITS Files (*.fits *.fit)"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save 2D', current_dir, filters)
        if path:
            bg_model = self.background.model2d
            data2d = self.image2d.data - bg_model
            prim_hdr = self.image2d.header
            prim_hdr['AUTHOR'] = 'PyNOT'
            prim_hdr['COMMENT'] = '2D background subtracted spectrum'
            prim_hdr['CHEB_ORD'] = self.settings['BACKGROUND_POLY_ORDER']
            sky_hdr = fits.Header()
            sky_hdr['AUTHOR'] = 'PyNOT'
            sky_hdr['COMMENT'] = '2D background subtracted spectrum'
            sky_hdr['CHEB_ORD'] = self.settings['BACKGROUND_POLY_ORDER']
            prim_HDU = fits.PrimaryHDU(data=data2d, header=prim_hdr)
            err_HDU = fits.ImageHDU(data=self.image2d.error, header=prim_hdr, name='ERR')
            sky_HDU = fits.ImageHDU(data=bg_model, header=sky_hdr, name='SKY')
            HDU_list = fits.HDUList([prim_HDU, err_HDU, sky_HDU])
            HDU_list.writeto(path, overwrite=True, output_verify='silentfix')

    def save_spectrum_bg(self):
        """Save the fitted 2D background spectrum"""
        current_dir = './skymodel_2d.fits'
        filters = "FITS Files (*.fits *.fit)"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save 2D', current_dir, filters)
        if path:
            bg_model = self.background.model2d
            prim_hdr = fits.Header()
            prim_hdr['AUTHOR'] = 'PyNOT'
            prim_hdr['COMMENT'] = '2D background model spectrum'
            prim_hdr['CHEB_ORD'] = self.settings['BACKGROUND_POLY_ORDER']
            hdu = fits.PrimaryHDU(data=bg_model, header=prim_hdr)
            hdu.writeto(path, overwrite=True, output_verify='silentfix')

    def save_spectrum_1d(self, index=0):
        if len(self.data1d) == 0:
            msg = "No 1D spectra have been extracted. Nothing to save..."
            QtWidgets.QMessageBox.critical(None, "Save Error", msg)
            return

        SaveWindow(parent=self, index=index)

    def save_all_extractions(self, fname=''):
        if len(self.data1d) == 0:
            msg = "No 1D spectra have been extracted. Nothing to save..."
            QtWidgets.QMessageBox.critical(None, "Save Error", msg)
            return False

        if not fname:
            current_dir = './'
            basename = os.path.join(current_dir, self.image2d.header['OBJECT'] + '_ext.fits')
            filters = "FITS Files (*.fits *.fit)"
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save All Extractions', basename, filters)

        if fname:
            hdu = fits.HDUList()
            prim_hdr = self.image2d.header
            prim_hdr['AUTHOR'] = 'PyNOT'
            prim_hdr['COMMENT'] = 'PyNOT extracted spectra'
            prim_hdr['COMMENT'] = 'Each spectrum in its own extension'
            prim = fits.PrimaryHDU(header=prim_hdr)
            hdu.append(prim)

            keywords_base = ['CDELT%i', 'CRPIX%i', 'CRVAL%i', 'CTYPE%i', 'CUNIT%i']
            keywords_to_remove = sum([[key % num for key in keywords_base] for num in [1, 2]], [])
            keywords_to_remove += ['CD1_1', 'CD2_1', 'CD1_2', 'CD2_2']
            keywords_to_remove += ['BUNIT', 'DATAMIN', 'DATAMAX']
            for num, spectrum in enumerate(self.data1d):
                col_wl = fits.Column(name='WAVE', array=spectrum.wl, format='D', unit=spectrum.wl_unit)
                col_flux = fits.Column(name='FLUX', array=spectrum.data, format='D', unit=spectrum.flux_unit)
                col_err = fits.Column(name='ERR', array=spectrum.error, format='D', unit=spectrum.flux_unit)
                col_sky = fits.Column(name='SKY', array=spectrum.background, format='D', unit=spectrum.flux_unit)
                tab_hdr = spectrum.hdr.copy()
                for key in keywords_to_remove:
                    tab_hdr.remove(key, ignore_missing=True)
                tab = fits.BinTableHDU.from_columns([col_wl, col_flux, col_err, col_sky], header=tab_hdr)
                tab.name = 'OBJ%i' % (num+1)
                hdu.append(tab)
            hdu.writeto(fname, overwrite=True, output_verify='silentfix')
            return True
        else:
            return False

    def load_spectrum(self, fname=None, dispaxis=1):
        if fname is False:
            current_dir = './'
            filters = "FITS files (*.fits | *.fit)"
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open 2D Spectrum', current_dir, filters)
            fname = str(fname)
            if self.first_time_open:
                print(" [INFO] - Don't worry about the warning above. It's an OS warning that can not be suppressed.")
                print("          Everything works as it should")
                self.first_time_open = False

        if not os.path.exists(fname):
            return

        # Clear all models:
        N_models = len(self.trace_models)
        for index in range(N_models)[::-1]:
            self.remove_trace(index)
        if self.background is not None:
            self.background.clear()
        if len(self.data1d) > 0:
            for spec in self.data1d:
                del spec
        self.axis_spsf.clear()
        self.axis_spsf.axhline(0., color='k', ls=':', alpha=0.7, lw=0.9)
        self.spsf_data_line = None
        self.spsf_bg_line = None
        self.axis_2d.clear()
        self.axis_2d_bg.clear()
        self.axis_1d.clear()
        self.image2d = ImageData(fname, dispaxis)
        self.data1d = list()
        self.filename_2d = fname
        self.last_fit = tuple()
        self.background = BackgroundModel(self.axis_spsf, self.image2d.data.shape)
        self.background.model2d += np.median(self.image2d.data)
        self.xmax_edit.setText("%i" % self.image2d.data.shape[1])
        self.ymax_edit.setText("%i" % self.image2d.data.shape[0])
        self.update_2d()
        self.update_value_range()
        self.update_spsf()
        self.localize_trace()
        self.axis_1d.set_xlabel("Wavelength  [%s]" % self.image2d.wl_unit)
        self.axis_1d.set_ylabel("Flux  [%s]" % self.image2d.flux_unit)

    def rotate_image(self):
        if len(self.trace_models) > 0:
            msg = "This action will clear all data that has been defined so far."
            msg += "\nAre you sure you want to continue?"
            answer = QtWidgets.QMessageBox.question(None, "Warning", msg)

            if answer == QtWidgets.QMessageBox.No:
                return

        if self.dispaxis == 1:
            self.load_spectrum(self.filename_2d, dispaxis=2)
            self.dispaxis = 2
        else:
            self.load_spectrum(self.filename_2d, dispaxis=1)
            self.dispaxis = 1

    def clear_state(self):
        self.state = None
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.axis_spsf.set_title("SPSF view", fontsize=10)
        self.canvas_spsf.draw()

    def add_new_object(self, center):
        """Add new trace object at position `center`"""
        if self.image2d is None:
            msg = "Load data before defining an object trace"
            QtWidgets.QMessageBox.critical(None, 'No data loaded', msg)
            return

        x_data, SPSF = self.spsf_data_line.get_data()
        imin = np.argmin(np.abs(x_data - center))
        height = SPSF[imin]
        trace_model = TraceModel(center, height, self.axis_spsf, shape=self.image2d.data.shape,
                                 color=next(color_cycle))
        self.add_trace(trace_model)

    def add_bg_range(self, x1, x2):
        """
        Add new background range between points `x1` and `x2`.
        The points are automatically sorted before adding range.
        """
        if self.image2d is None:
            msg = "Load data before defining background ranges"
            QtWidgets.QMessageBox.critical(None, 'No data loaded', msg)
            return
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        self.background.add_range(xmin, xmax)

    def fit_background(self):
        if self.background is None:
            return
        elif len(self.background.ranges) == 0:
            self.background.model2d *= 0.
            self.background.model2d += np.median(self.image2d.data)
            msg = "No background regions defined"
            info = "Please mark these in the top right figure by pressing 'B' or the 'Mark Background' button."
            WarningDialog(self, msg, info)

        else:
            bg_order = self.settings['BACKGROUND_POLY_ORDER']
            bg_kappa = self.settings['BACKGROUND_MED_KAPPA']
            y = self.image2d.y
            mask = np.zeros_like(y, dtype=bool)
            msg = "Fitting background spectrum..."
            self.progress = QtWidgets.QProgressDialog(msg, "Cancel", 0, len(self.image2d.x), self._main)
            self.progress.setWindowModality(QtCore.Qt.WindowModal)
            self.progress.show()
            for y1, y2 in self.background.ranges:
                mask += (y >= y1) & (y <= y2)
            for i, column in enumerate(self.image2d.data.T):
                if self.progress.wasCanceled():
                    self.background.model2d *= 0.
                    self.background.model2d += np.median(self.image2d.data)
                    return
                # Median filter the data to remove outliers:
                med_column = median_filter(column, 15)
                noise = mad(column)*1.4826
                filtering_mask = (np.abs(column - med_column) < bg_kappa*noise)
                if np.sum(mask & filtering_mask) < bg_order:
                    this_mask = mask
                else:
                    this_mask = mask & filtering_mask
                # Fit Chebyshev polynomial model:
                bg_model = Chebyshev.fit(y[this_mask], column[this_mask], bg_order, domain=(y.min(), y.max()))
                self.background.model2d[:, i] = bg_model(y)
                self.progress.setValue(i+1)

            # Fit SPSF and plot line:
            x_data, SPSF = self.spsf_data_line.get_data()
            mask = np.zeros_like(SPSF, dtype=bool)
            for y1, y2 in self.background.ranges:
                mask += (x_data >= y1) & (x_data <= y2)
            spsf_bg_fit = Chebyshev.fit(x_data[mask], SPSF[mask], bg_order, domain=(x_data.min(), x_data.max()))
            spsf_bg_model = spsf_bg_fit(x_data)
            if self.spsf_bg_line is not None:
                self.spsf_bg_line.set_data(x_data, spsf_bg_model)
            else:
                self.spsf_bg_line, = self.axis_spsf.plot(x_data, spsf_bg_model, color='Blue', alpha=0.7, lw=1.5, ls='--')
            self.canvas_spsf.draw()

        self.update_2d()

    def on_pick(self, event):
        artist = event.artist
        if isinstance(artist, matplotlib.patches.Polygon):
            # -- Delete Background Patch
            if self.delete_picked_object:
                index = self.background.patches.index(artist)
                self.background.remove_range(index)
                self.delete_picked_object = False
                self.clear_state()

        else:
            for num, model in enumerate(self.trace_models):
                if artist in model.vlines:
                    if self.delete_picked_object:
                        self.remove_trace(num)
                        self.delete_picked_object = False
                        self.clear_state()
                    else:
                        old_centroid = copy.copy(model.cen)
                        self.picked_object = (num, artist, model, old_centroid)

    def on_motion(self, event):
        if self.picked_object is None:
            return
        elif not event.inaxes:
            return

        num, artist, trace_model, old_centroid = self.picked_object
        new_position = event.xdata
        if artist.get_label() == 'center':
            offset = new_position - trace_model.cen
            low, high = trace_model.get_range()
            trace_model.set_range(low + offset, high + offset)
            trace_model.set_centroid(new_position)

        elif artist.get_label() == 'lower':
            trace_model.lower = new_position
            artist.set_xdata(new_position)

        elif artist.get_label() == 'upper':
            trace_model.upper = new_position
            artist.set_xdata(new_position)
        self.canvas_spsf.draw()

    def on_release(self, event):
        if self.picked_object is None:
            return
        num, artist, trace_model, old_centroid = self.picked_object
        if artist.get_label() == 'lower':
            if trace_model.lower > trace_model.cen - 1:
                trace_model.lower = trace_model.cen - 1
                artist.set_xdata(trace_model.cen - 1)
                self.canvas_spsf.draw()
        elif artist.get_label() == 'upper':
            if trace_model.upper < trace_model.cen + 1:
                trace_model.upper = trace_model.cen + 1
                artist.set_xdata(trace_model.cen + 1)
                self.canvas_spsf.draw()
        centroid_shift = trace_model.cen - old_centroid
        if np.abs(centroid_shift) > 0:
            trace_model.points['mu'] += centroid_shift
            trace_model.fit['mu'] += centroid_shift
            self.plot_fitted_points()
        self.create_model_trace()
        self.plot_trace_2d()
        self.picked_object = None

    def on_key_press(self, event):
        if event.key == 'b':
            self.set_state('bg1')

        elif event.key == 'a':
            self.set_state('add')

        elif event.key == 'd':
            self.set_state('delete')


    def set_state(self, state):
        if state == 'add':
            self.axis_spsf.set_title("Add New Object: Click on Trace Center", fontsize=10)
            self.canvas_spsf.draw()
            self.state = state
            self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        elif state == 'bg1':
            self.axis_spsf.set_title("Mark Background Range: Click on First Limit", fontsize=10)
            self.canvas_spsf.draw()
            self.state = state
            self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        elif state == 'bg2':
            self.axis_spsf.set_title("Click on Second Limit", fontsize=10)
            self.canvas_spsf.draw()
            self.state = state
            self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        elif state == 'delete':
            self.axis_spsf.set_title("Pick object or background range to delete", fontsize=10)
            self.canvas_spsf.draw()
            self.delete_picked_object = True
            self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def on_mouse_press(self, event):
        if self.state is None:
            pass

        elif self.state == 'add':
            self.add_new_object(event.xdata)
            self.clear_state()

        elif self.state == 'bg1':
            self.bg_value1 = event.xdata
            self.set_state('bg2')

        elif self.state == 'bg2':
            self.add_bg_range(self.bg_value1, event.xdata)
            self.clear_state()
            self.bg_value1 = None

        elif self.state == 'delete':
            msg = "No object selected"
            info = "Please pick an object in the SPSF plot: either a background range or a trace object"
            WarningDialog(self, msg, info)

    def remove_trace(self, index):
        trace_model = self.trace_models[index]
        trace_model.clear_plot()
        self.canvas_spsf.draw()
        self.canvas_2d.draw()
        self.canvas_1d.draw()
        self.canvas_points.draw()
        self.list_widget.takeItem(index)
        self.trace_models.pop(index)

    def add_trace(self, model):
        self.trace_models.append(model)
        N = self.list_widget.count() + 1
        object_name = self.image2d.header['OBJECT'] + '_%i' % N
        model.set_object_name(object_name)
        if model.fixed:
            object_name = "[ COPY ] " + object_name
        item = QtWidgets.QListWidgetItem(object_name)
        # item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        item.setCheckState(QtCore.Qt.Checked)
        item.setForeground(QtGui.QColor(model.color))
        self.list_widget.addItem(item)
        self.canvas_spsf.draw()

    def limits_updated(self):
        if self.image2d is None:
            return

        xmax = int(self.xmax_edit.text())
        ymax = int(self.ymax_edit.text())
        if xmax > self.image2d.data.shape[1]:
            xmax = self.image2d.data.shape[1]
            self.xmax_edit.setText("%i" % xmax)

        if ymax > self.image2d.data.shape[0]:
            ymax = self.image2d.data.shape[0]
            self.ymax_edit.setText("%i" % ymax)

        self.update_spsf()

        xmin, xmax, ymin, ymax = self.get_limits()
        for trace_model in self.trace_models:
            trace_model.xmin = xmin
            trace_model.xmax = xmax

        if len(self.trace_models) > 0 and len(self.trace_models[0].x_binned) > 0:
            self.plot_fitted_points(update_only=True)

    def get_limits(self):
        xmin = int(self.xmin_edit.text())
        xmax = int(self.xmax_edit.text())
        ymin = int(self.ymin_edit.text())
        ymax = int(self.ymax_edit.text())
        return (xmin, xmax, ymin, ymax)

    def update_spsf(self):
        xmin, xmax, ymin, ymax = self.get_limits()
        SPSF = np.nanmedian(self.image2d.data[ymin:ymax, xmin:xmax], axis=1)
        if self.spsf_data_line is None:
            self.spsf_data_line, = self.axis_spsf.plot(self.image2d.y[ymin:ymax], SPSF, color='k', lw=0.7)
        else:
            self.spsf_data_line.set_data(self.image2d.y[ymin:ymax], SPSF)
        vmin = np.median(SPSF) - 10.*mad(SPSF)
        vmax = np.max(SPSF) + 10.*mad(SPSF)
        self.axis_spsf.set_ylim(vmin, vmax)
        self.axis_spsf.set_yticklabels("")
        self.axis_spsf.set_xlim(np.min(self.image2d.y)-2, np.max(self.image2d.y)+2)
        self.axis_spsf.tick_params(axis='x', which='major', labelsize=8)
        self.axis_spsf.set_title("SPSF view", fontsize=10)
        self.canvas_spsf.figure.tight_layout()
        self.canvas_spsf.draw()

    def update_2d(self):
        vmin = self.vmin_edit.text()
        vmax = self.vmax_edit.text()
        if vmin == '':
            vmin = None
        else:
            vmin = float(vmin)

        if vmax == '':
            vmax = None
        else:
            vmax = float(vmax)

        bg = self.background.model2d
        if len(self.axis_2d.images) == 0:
            self.axis_2d.imshow(self.image2d.data - bg, cmap=plt.cm.gray_r, aspect='auto', origin='lower')
            self.axis_2d_bg.imshow(bg, aspect='auto', origin='lower')
        else:
            self.axis_2d.images[0].set_data(self.image2d.data - bg)
            self.axis_2d_bg.images[0].set_data(bg)
            N_Y, N_X = self.image2d.shape
            extent = (-0.5, N_X-0.5, -0.5, N_Y-0.5)
            self.axis_2d.images[0].set_extent(extent)
            self.axis_2d_bg.images[0].set_extent(extent)
        self.update_value_range(vmin, vmax)
        self.canvas_2d.draw()

    def plot_trace_2d(self):
        active_models = list()
        for model in self.trace_models:
            if np.sum(model.model2d) > 0:
                active_models.append(model)
        if len(active_models) == 0:
            msg = "No aperture models defined"
            info = "Fit the aperture model before extracting."
            WarningDialog(self, msg, info)
            return

        for num, model in enumerate(self.trace_models):
            trace_model_2d = model.model2d.copy()
            if np.max(trace_model_2d) != 0.:
                trace_model_2d /= np.max(trace_model_2d)
            alpha_array = 2 * trace_model_2d.copy()
            alpha_array[alpha_array > 0.1] += 0.3
            alpha_array[alpha_array > 0.3] = 0.6
            if model.model_image is None:
                model.model_image = self.axis_2d.imshow(trace_model_2d, vmin=0., vmax=0.5,
                                                        cmap=model.cmap, aspect='auto',
                                                        origin='lower', alpha=alpha_array)
            else:
                model.model_image.set_data(trace_model_2d)
                model.model_image.set_alpha(alpha_array)

            listItem = self.list_widget.item(num)
            if listItem.checkState() == 2:
                model.model_image.set_visible(True)
            else:
                model.model_image.set_visible(False)
        self.canvas_2d.draw()

    def update_value_range(self, vmin=None, vmax=None):
        if len(self.axis_2d.images) == 0:
            return

        if vmin is None and vmax is None:
            bg = self.background.model2d
            noise = mad(self.image2d.data - bg)
            vmin = np.median(self.image2d.data - bg) - 2*noise
            vmax = np.median(self.image2d.data - bg) + 10*noise
            self.vmin_edit.setText("%.2e" % vmin)
            self.vmax_edit.setText("%.2e" % vmax)

        if vmin is not None:
            self.axis_2d.images[0].set_clim(vmin=vmin)
            self.axis_2d_bg.images[0].set_clim(vmin=vmin)

        if vmax is not None:
            self.axis_2d.images[0].set_clim(vmax=vmax)
            self.axis_2d_bg.images[0].set_clim(vmax=vmax)
        self.canvas_2d.draw()

    def localize_trace(self):
        if self.image2d is not None:
            peaks, prominences = auto_localize(self.image2d.data, self.settings)
            if len(peaks) == 0:
                msg = "Automatic trace detection failed!"
                QtWidgets.QMessageBox.critical(None, 'No trace detected', msg)

            else:
                for center, height in zip(peaks, prominences):
                    trace_model = TraceModel(center, height, self.axis_spsf, shape=self.image2d.data.shape,
                                             color=next(color_cycle))
                    self.add_trace(trace_model)

    def model_change(self, text):
        # Changed to always show aperture limits...
        # if text.lower() == 'tophat':
        #     for num, model in enumerate(self.trace_models):
        #         listItem = self.list_widget.item(num)
        #         if listItem.checkState() == 2:
        #             model.set_visible()
        #         else:
        #             model.set_visible(False)
        #     self.canvas_spsf.draw()
        #
        # elif text.lower() == 'gaussian' or text.lower() == 'moffat':
        #     for model in self.trace_models:
        #         model.set_visible(False)
        #     self.canvas_spsf.draw()
        pass

    def update_xmask_in_points(self):
        # Clear old shapes:
        for item in self.xmask:
            item.remove()
        self.xmask = list()
        xmin, xmax, ymin, ymax = self.get_limits()
        if xmin > 0 or xmax < self.image2d.data.shape[1]+1:
            # Define new shaded areas:
            for ax in self.axes_points:
                xlims = ax.get_xlim()
                vspan1 = ax.axvspan(xlims[0], xmin, color='0.3', alpha=0.15)
                vspan2 = ax.axvspan(xmax, xlims[1], color='0.3', alpha=0.15)
                ax.set_xlim(*xlims)
                self.xmask.append(vspan1)
                self.xmask.append(vspan2)
            self.canvas_points.draw()

    def fit_trace(self):
        if self.image2d is None:
            return
        elif len(self.trace_models) == 0:
            msg = "No objects defined                      "
            info = "Please mark the centroid(s) by pressing the key 'A' or the 'Add Object' button."
            WarningDialog(self, msg, info)
            return

        # Fit trace with N objects:
        img2d = self.image2d.data - self.background.model2d
        dx = int(self.bins_edit.text())
        model_type = self.model_chooser.currentText().lower()
        original_model = model_type
        if model_type == 'tophat':
            original_model = 'tophat'
            model_type = 'moffat'
        xmin, xmax, ymin, ymax = self.get_limits()
        x_binned = np.arange(0., img2d.shape[1], dx, dtype=np.float64)
        peaks = list()
        prominences = list()
        for trace_model in self.trace_models:
            if not trace_model.fixed:
                peaks.append(trace_model.cen)
                prominences.append(trace_model.amp)
        this_fit = (dx, original_model, ymin, ymax, self.trace_models)

        if self.last_fit != this_fit:
            # Update the fitted points only if the parameters have changed:
            N_obj = len(peaks)
            trace_parameters = list()
            msg = "Fitting the morphology of the 2D spectrum..."
            self.progress = QtWidgets.QProgressDialog(msg, "Cancel", 0, len(x_binned), self._main)
            self.progress.setWindowModality(QtCore.Qt.WindowModal)
            self.progress.show()
            counter = 0
            pars = prep_parameters(peaks, prominences, size=img2d.shape[0], model_type=model_type, tie_traces=True)
            for num in range(0, img2d.shape[1], dx):
                if self.progress.wasCanceled():
                    return
                # pars = prep_parameters(peaks, prominences, size=img2d.shape[0], model_type=model_type)
                col = np.nansum(img2d[:, num:num+dx], axis=1)
                col_mask = np.ones_like(col, dtype=bool)
                col_mask[:ymin] = False
                col_mask[ymax:] = False
                try:
                    popt = minimize(model_residuals, pars, args=(self.image2d.y[col_mask], col[col_mask], N_obj),
                                    kws={'model_type': model_type}, factor=1.)
                    trace_parameters.append(popt.params)
                except ValueError:
                    trace_parameters.append(pars)
                counter += 1
                self.progress.setValue(counter)

            # Update model with fitted points:
            for num, trace_model in enumerate(self.trace_models):
                if not trace_model.fixed:
                    trace_model.xmin = xmin
                    trace_model.xmax = xmax
                    trace_model.model_type = original_model
                    trace_model._original_type = original_model
                    mu = np.array([par['mu_%i' % num] for par in trace_parameters])
                    if model_type == 'moffat':
                        alpha = np.array([par['a_%i' % num] for par in trace_parameters])
                        beta = np.array([par['b_%i' % num] for par in trace_parameters])
                        trace_model.set_data(x_binned, mu, alpha=alpha, beta=beta)
                        # Set extraction limits to ±2xFWHM:
                        profile = NN_moffat(trace_model.y, np.median(mu), np.median(alpha), np.median(beta), 0.)
                        fwhm, msg = get_FWHM(profile)
                        trace_model.set_centroid(np.median(mu))
                        lower = trace_model.cen - 2*fwhm
                        upper = trace_model.cen + 2*fwhm
                        trace_model.set_range(lower, upper)

                    elif model_type == 'gaussian':
                        sig = np.array([par['sig_%i' % num] for par in trace_parameters])
                        trace_model.set_data(x_binned, mu, sigma=sig)
                        # Set extraction limits to ±2xFWHM:
                        profile = NN_gaussian(trace_model.y, np.median(mu), np.median(sig), 0.)
                        fwhm, msg = get_FWHM(profile)
                        trace_model.set_centroid(np.median(mu))
                        lower = trace_model.cen - 2*fwhm
                        upper = trace_model.cen + 2*fwhm
                        trace_model.set_range(lower, upper)
            self.last_fit = this_fit
            self.canvas_spsf.draw()
            self.create_model_trace()
            self.plot_fitted_points()
        else:
            self.create_model_trace()
            self.plot_fitted_points(update_only=True)
        self.plot_trace_2d()
        self.activate_all_listItems()
        self.tab_widget.setCurrentIndex(1)

    def plot_fitted_points(self, update_only=False):
        if update_only:
            for model in self.trace_models:
                parameters = model.get_parnames()
                for ax, parname in zip(self.axes_points, parameters):
                    mask = model.get_mask(parname)
                    for line in model.point_lines[parname]:
                        line.remove()
                    l1, = ax.plot(model.x_binned[mask], model.points[parname][mask],
                                  color=model.color, marker='o', ls='', mec=color_shade(model.color), picker=True, pickradius=6)
                    l2, = ax.plot(model.x_binned[~mask], model.points[parname][~mask],
                                  color=model.color, marker='o', ls='', alpha=0.3, picker=True, pickradius=6)
                    l3, = ax.plot(model.x_binned[~mask], model.points[parname][~mask],
                                  color='k', marker='x', ls='')
                    model.point_lines[parname] = [l1, l2, l3]
                    if parname == 'mu':
                        l4, = self.axis_2d.plot(model.x_binned[mask], model.points[parname][mask],
                                                color=model.color, marker='o', ls='', alpha=0.5, picker=True, pickradius=6)
                        model.point_lines[parname].append(l4)

                    if not model.active:
                        l1.set_visible(False)
                        l1.set_picker(False)
                        l2.set_visible(False)
                        l2.set_picker(False)
                        l3.set_visible(False)
                        if parname == 'mu':
                            l4.set_visible(False)
                            l4.set_picker(False)
                    else:
                        l1.set_picker(True)
                        l2.set_picker(True)
                        if parname == 'mu':
                            l4.set_picker(True)

                    # -- Plot fit to points:
                    if len(model.fit['mu']) > 0:
                        lf = model.fit_lines[parname]
                        lf.set_data(model.x, model.fit[parname])
                        if not model.active:
                            lf.set_visible(False)

        else:
            for model in self.trace_models:
                for line in model.point_lines['mu']:
                    line.remove()
            self.figure_points.clear()
            fit_model_type = self.model_chooser.currentText().lower()
            if fit_model_type == 'moffat':
                self.axes_points = self.figure_points.subplots(3, 1)
            elif fit_model_type == 'gaussian':
                self.axes_points = self.figure_points.subplots(2, 1)
            else:
                self.axes_points = [self.figure_points.subplots(1, 1)]
            self.axes_points[0].set_title("Profile Parameters along Dispersion Axis", fontsize=11)

            for model in self.trace_models:
                parameters = model.get_parnames()
                for ax, parname in zip(self.axes_points, parameters):
                    ax.tick_params(axis='both', which='major', labelsize=10)
                    mask = model.get_mask(parname)
                    # -- Plot traced points:
                    l1, = ax.plot(model.x_binned[mask], model.points[parname][mask],
                                  color=model.color, marker='o', ls='', mec=color_shade(model.color), picker=True, pickradius=6)
                    l2, = ax.plot(model.x_binned[~mask], model.points[parname][~mask],
                                  color=model.color, marker='o', ls='', alpha=0.3, picker=True, pickradius=6)
                    l3, = ax.plot(model.x_binned[~mask], model.points[parname][~mask],
                                  color='k', marker='x', ls='')
                    model.point_lines[parname] = [l1, l2, l3]
                    if parname == 'mu':
                        l4, = self.axis_2d.plot(model.x_binned[mask], model.points[parname][mask],
                                                color=model.color, marker='o', ls='', alpha=0.3, picker=True, pickradius=6)
                        model.point_lines[parname].append(l4)
                    # -- Plot fit to points:
                    if len(model.fit['mu']) > 0:
                        lf, = ax.plot(model.x, model.fit[parname],
                                      color=model.color, ls='--', lw=1.0)
                        model.fit_lines[parname] = lf
                    if not model.fixed:
                        ax.set_ylabel("%s" % model.get_unicode_name(parname))
                    if not ax.is_last_row():
                        ax.set_xticklabels("")
        self.canvas_points.figure.tight_layout()
        self.update_xmask_in_points()
        self.canvas_points.draw()
        self.canvas_2d.draw()

    def create_model_trace(self, plot=False):
        center_order, width_order = self.get_trace_orders()
        for model in self.trace_models:
            if len(model.x_binned) == 0:
                # Force model_type to be 'tophat':
                model.model_type = 'tophat'
                model._original_type = 'tophat'
                # Make flat box aperture:
                lower, upper = model.get_range()
                lower_array = lower * np.ones_like(model.x)
                upper_array = upper * np.ones_like(model.x)
                pars_table = np.column_stack([lower_array, upper_array])
                for num, pars in enumerate(pars_table):
                    P_i = tophat(model.y, *pars)
                    if np.sum(P_i) == 0:
                        model.model2d[:, num] = P_i
                    else:
                        model.model2d[:, num] = P_i / np.sum(P_i)

            else:
                domain = (0., np.max(model.x))
                x_binned = model.x_binned
                # Fit the centroid `mu`:
                mask_mu = model.get_mask('mu')
                mu = model.points['mu']
                mu_fit = Chebyshev.fit(x_binned[mask_mu], mu[mask_mu], deg=center_order, domain=domain)
                model.fit['mu'] = mu_fit(model.x)

                # Fit the remaining parameters:
                parameters = model.get_parnames()
                # Remove 'mu' from the list, at index 0:
                parameters.pop(0)
                for parname in parameters:
                    mask = model.get_mask(parname)
                    par = model.points[parname]
                    cheb_fit = Chebyshev.fit(x_binned[mask], par[mask], deg=width_order, domain=domain)
                    model.fit[parname] = cheb_fit(model.x)

                # Create 2D model:
                if model.model_type == 'moffat':
                    model_function = NN_moffat
                    pars_table = np.column_stack([model.fit['mu'], model.fit['alpha'], model.fit['beta'], np.zeros_like(model.x)])
                elif model.model_type == 'gaussian':
                    model_function = NN_gaussian
                    pars_table = np.column_stack([model.fit['mu'], model.fit['sigma'], np.zeros_like(model.x)])
                elif model.model_type == 'tophat':
                    model_function = tophat
                    lower, upper = model.get_range()
                    cen = model.cen
                    delta_lower = np.abs(cen - lower)
                    delta_upper = np.abs(upper - cen)
                    mu = model.fit['mu']
                    lower_array = np.round(mu - delta_lower, 0)
                    upper_array = np.round(mu + delta_upper, 0)
                    pars_table = np.column_stack([lower_array, upper_array])

                # lower, upper = model.get_range()
                # dlow = np.abs(model.cen - lower)
                # dhigh = np.abs(model.cen - upper)
                for num, pars in enumerate(pars_table):
                    P_i = model_function(model.y, *pars)
                    # # if model.model_type != '':
                    # if model.model_type != 'tophat':
                    #     il = int(pars[0] - dlow)
                    #     ih = int(pars[0] + dhigh)
                    #     P_i[:il] = 0.
                    #     P_i[ih:] = 0.
                    if np.sum(P_i) == 0:
                        model.model2d[:, num] = P_i
                    else:
                        model.model2d[:, num] = P_i / np.sum(P_i)

        if plot is True:
            self.plot_trace_2d()

    def get_trace_orders(self):
        c_order = int(self.c_order_edit.text())
        w_order = int(self.w_order_edit.text())
        return (c_order, w_order)

    def pick_points(self, event):
        artist = event.artist
        if isinstance(artist, matplotlib.lines.Line2D):
            for model in self.trace_models:
                if not model.active:
                    continue
                for parname in ['mu', 'alpha', 'beta', 'sigma']:
                    if artist in model.point_lines[parname]:
                        x_picked = event.mouseevent.xdata
                        y_picked = event.mouseevent.ydata
                        X = model.x_binned
                        Y = model.points[parname]
                        if x_picked is None:
                            x_picked = 0.
                        if y_picked is None:
                            y_picked = 0.
                        dist = np.sqrt((X - x_picked)**2 + (Y - y_picked)**2)
                        idx = np.argmin(dist)
                        val = model.mask[parname][idx]
                        if event.canvas == self.canvas_2d:
                            for par in model.get_parnames():
                                model.mask[par][idx] = ~val
                        else:
                            model.mask[parname][idx] = ~val
                        self.plot_fitted_points(update_only=True)

    def median_filter_points(self):
        kappa = float(self.med_kappa_edit.text())
        window = int(self.med_window_edit.text())
        for model in self.trace_models:
            parameters = model.get_parnames()
            for parname in parameters:
                filtered_data, mask = median_filter_data(model.points[parname], kappa=kappa, window=window)
                if np.sum(mask) == 0:
                    msg = "No points were included after filtering the variable %s.\n" % model.get_unicode_name(parname)
                    msg += "Something must be wrong; Try a larger kappa value..."
                    QtWidgets.QMessageBox.critical(None, "Masking error", msg)
                    continue
                model.mask[parname] = mask
        self.create_model_trace()
        self.plot_fitted_points(update_only=True)

    def extract(self):
        active_models = list()
        for model in self.trace_models:
            if np.sum(model.model2d) > 0:
                active_models.append(model)
        if len(active_models) == 0:
            msg = "No aperture models defined"
            info = "Fit the aperture model or create a box aperture before extracting."
            WarningDialog(self, msg, info)
            return

        for spec1d in self.data1d:
            del spec1d
        data1d_list = []
        for num, model in enumerate(self.trace_models):
            P = model.model2d
            bg2d = self.background.model2d
            img2d = self.image2d.data - bg2d
            V = self.image2d.error**2
            M = np.ones_like(img2d)

            with warnings.catch_warnings():
                P = P / np.sum(P, axis=0)
                warnings.simplefilter('ignore')
                if model.model_type == 'tophat':
                    data1d = np.sum((P > 0)*img2d, axis=0)
                    err1d = np.sqrt(np.sum(V*(P > 0), axis=0))
                else:
                    data1d = np.sum(M*P*img2d/V, axis=0) / np.sum(M*P**2/V, axis=0)
                    err1d = np.sqrt(np.sum(M*P, axis=0) / np.sum(M*P**2/V, axis=0))
                err1d = fix_nans(err1d)
                bg1d = np.sum(M*P*bg2d, axis=0) / np.sum(M*P**2, axis=0)

            wl = self.image2d.wl
            spec1d = Spectrum(wl=wl, data=data1d, error=err1d, hdr=self.image2d.header,
                              wl_unit=self.image2d.wl_unit, flux_unit=self.image2d.flux_unit)
            spec1d.background = bg1d
            data1d_list.append(spec1d)
        self.data1d = data1d_list
        self.plot_data1d()

    def plot_data1d(self):
        if len(self.data1d) == 0:
            return
        ylims = list()
        for num, model in enumerate(self.trace_models):
            spec1d = self.data1d[num]
            if model.plot_1d is not None:
                for child in model.plot_1d.get_children():
                    child.remove()
                    # -- find a way to update the data instead...
            model.plot_1d = self.axis_1d.errorbar(spec1d.wl, spec1d.data, spec1d.error,
                                                  color=model.color, lw=1., elinewidth=0.5)
            good_pixels = spec1d.data > 5*spec1d.error
            if np.sum(good_pixels) < 3:
                data_min = 0.
                data_max = 2*np.nanmean(spec1d.data[50:-50])
            else:
                data_min = -3*np.nanmean(spec1d.error[good_pixels])
                data_max = 1.2*np.nanmax(spec1d.data[good_pixels])
            ylims.append([data_min, data_max])
            listItem = self.list_widget.item(num)
            if listItem.checkState() == 2:
                for child in model.plot_1d.get_children():
                    child.set_visible(True)
            else:
                for child in model.plot_1d.get_children():
                    child.set_visible(False)
        ymin = np.min(ylims)
        ymax = np.max(ylims)
        self.axis_1d.set_ylim(ymin, ymax)
        self.canvas_1d.figure.tight_layout()
        self.canvas_1d.draw()
        self.tab_widget.setCurrentIndex(2)

    def toggle_trace_models(self, listItem):
        index = self.list_widget.row(listItem)
        if listItem.checkState() == 2:
            # Active:
            self.trace_models[index].activate()
            self.trace_models[index].set_visible(True)
        else:
            # Inactive:
            self.trace_models[index].deactivate()
        self.plot_trace_2d()
        self.canvas_spsf.draw()
        self.canvas_points.draw()
        self.canvas_1d.draw()

    def activate_all_listItems(self):
        N_rows = self.list_widget.count()
        for num in range(N_rows):
            item = self.list_widget.item(num)
            item.setCheckState(2)

    def update_settings(self):
        pass

    def create_menu(self):
        load_file_action = QtWidgets.QAction("Load Spectrum", self)
        load_file_action.setShortcut("ctrl+O")
        load_file_action.triggered.connect(self.load_spectrum)

        save_bg_action = QtWidgets.QAction("Save 2D Background", self)
        save_bg_action.triggered.connect(self.save_spectrum_bg)

        save_2d_action = QtWidgets.QAction("Save 2D Subtracted Spectrum", self)
        save_2d_action.triggered.connect(self.save_spectrum_2d)

        save_1d_action = QtWidgets.QAction("Save 1D Spectrum", self)
        save_1d_action.setShortcut("ctrl+S")
        save_1d_action.triggered.connect(self.save_spectrum_1d)

        save_all_1d_action = QtWidgets.QAction("Save All Extractions", self)
        save_all_1d_action.setShortcut("ctrl+shift+S")
        save_all_1d_action.triggered.connect(self.save_all_extractions)

        save_all_trace_action = QtWidgets.QAction("Save All Aperture Models", self)
        save_all_trace_action.triggered.connect(self.save_all_aperture_models)

        save_trace_action = QtWidgets.QAction("Save Aperture Model", self)
        save_trace_action.triggered.connect(self.save_aperture_model)

        rotate_action = QtWidgets.QAction("Flip 2D Image Axes", self)
        rotate_action.setShortcut("ctrl+R")
        rotate_action.triggered.connect(self.rotate_image)

        make2dmodel_action = QtWidgets.QAction("Make 2D Extraction Profile", self)
        make2dmodel_action.setShortcut("ctrl+M")
        make2dmodel_action.setChecked(True)
        make2dmodel_action.triggered.connect(lambda x: self.create_model_trace(plot=True))

        settings_action = QtWidgets.QAction("Settings", self)
        settings_action.setShortcut("ctrl+,")
        settings_action.triggered.connect(lambda checked: SettingsWindow(self))

        view_hdr_action = QtWidgets.QAction("Display Header", self)
        view_hdr_action.setShortcut("ctrl+shift+H")
        view_hdr_action.triggered.connect(self.display_header)

        main_menu = self.menuBar()
        file_menu = main_menu.addMenu("File")
        file_menu.addAction(load_file_action)
        file_menu.addSeparator()

        file_menu.addAction(save_1d_action)
        file_menu.addAction(save_trace_action)

        file_menu.addAction(save_all_1d_action)
        file_menu.addAction(save_all_trace_action)
        file_menu.addSeparator()
        file_menu.addAction(save_2d_action)
        file_menu.addAction(save_bg_action)

        edit_menu = main_menu.addMenu("Edit")
        edit_menu.addAction(rotate_action)
        edit_menu.addSeparator()
        edit_menu.addAction(make2dmodel_action)
        edit_menu.addSeparator()
        edit_menu.addAction(settings_action)

        view_menu = main_menu.addMenu("View")
        view_menu.addAction(view_hdr_action)

    def listItemDoubleClicked(self, item):
        index = self.list_widget.currentIndex().row()
        if index < 0 or len(self.trace_models) == 0:
            return
        current_model = self.trace_models[index]
        if np.sum(current_model.model2d) == 0 or len(current_model.fit['mu']) == 0:
            current_model.model_type = 'tophat'
        ModelPropertiesWindow(current_model, index, self)

    def listItemRightClicked(self, QPos):
        index = self.list_widget.currentIndex().row()
        if index < 0 or len(self.trace_models) == 0:
            return
        current_model = self.trace_models[index]
        self.listMenu = QtWidgets.QMenu()
        remove_menu_item = self.listMenu.addAction("Delete Aperture")
        remove_menu_item.triggered.connect(lambda x: self.remove_trace(index))
        edit_menu_item = self.listMenu.addAction("Edit Properties")
        edit_menu_item.triggered.connect(lambda x: ModelPropertiesWindow(current_model, index, self))
        comp_menu_item = self.listMenu.addAction("Copy Aperture")
        comp_menu_item.triggered.connect(lambda x: self.copy_trace(current_model))
        if current_model.plot_1d is not None:
            self.listMenu.addSeparator()
            save_menu_item = self.listMenu.addAction("Save Spectrum")
            save_menu_item.triggered.connect(lambda x: self.save_spectrum_1d(index=index))
            save_item_model = self.listMenu.addAction("Save Aperture")
            save_item_model.triggered.connect(lambda x: self.save_aperture_model(index))
        parentPosition = self.list_widget.mapToGlobal(QtCore.QPoint(0, 0))
        self.listMenu.move(parentPosition + QPos)
        self.listMenu.show()

    def copy_trace(self, trace_model):
        if len(trace_model.fit['mu']) == 0:
            msg = "Aperture not defined"
            info = "Fit the trace parameters before copying."
            WarningDialog(self, msg, info)
            return

        new_trace_model = trace_model.copy()
        if new_trace_model.model_type == 'tophat':
            new_trace_model.set_visible(True)
        self.add_trace(new_trace_model)
        if len(new_trace_model.points['mu']) > 0:
            self.create_model_trace()
            self.plot_fitted_points()
            self.plot_trace_2d()

    def display_header(self):
        if self.image2d is not None:
            HeaderViewer(self.image2d.header, parent=self)
        else:
            msg = "No Data Loaded"
            info = "Load a 2D spectrum first"
            WarningDialog(self, msg, info)


class SettingsWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(SettingsWindow, self).__init__(parent)
        self.setWindowTitle("Edit Settings")
        self.save_button = QtWidgets.QPushButton("Update")
        self.save_button.clicked.connect(self.save_settings)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_all)
        self.quit_button = QtWidgets.QPushButton("Close")
        self.quit_button.clicked.connect(self.close)
        self.parent = parent
        self.current_settings = dict()

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.save_button)
        hbox.addWidget(self.reset_button)
        hbox.addStretch(1)
        hbox.addWidget(self.quit_button)

        vbox = QtWidgets.QVBoxLayout()
        self.show_options(vbox)
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

        self.show()

    def save_settings(self):
        for option, editor in self.current_settings.items():
            old_value = self.parent.settings[option]
            value_type = type(old_value)
            text = editor.text()
            self.parent.settings[option] = value_type(text)
        self.parent.update_settings()
        self.close()

    def show_options(self, layout):
        for option, value in list(self.parent.settings.items()):
            label = QtWidgets.QLabel("%s:" % options_descriptions[option])
            editor = QtWidgets.QLineEdit("%r" % value)
            if isinstance(value, int):
                editor.setValidator(QtGui.QIntValidator(0, 1000000))
            elif isinstance(value, float):
                editor.setValidator(QtGui.QDoubleValidator())
            # editor.returnPressed.connect(self.save_settings)
            layout.addWidget(label)
            layout.addWidget(editor)
            self.current_settings[option] = editor

    def reset_all(self):
        for option, default_value in default_settings.items():
            self.current_settings[option].setText("%r" % default_value)


class SaveWindow(QtWidgets.QDialog):
    def __init__(self, parent=None, index=0):
        super(SaveWindow, self).__init__(parent)
        self.setWindowTitle("Save Extracted Spectrum")
        self.parent = parent

        # -- Create Filename Selector:
        basename = './' + parent.output_fname
        self.fname_editor = QtWidgets.QLineEdit(basename)
        self.fname_editor.setMinimumWidth(200)
        file_selector = QtWidgets.QPushButton("...")
        file_selector.clicked.connect(self.open_filedialog)
        fname_row = QtWidgets.QHBoxLayout()
        fname_row.addWidget(self.fname_editor, 1)
        fname_row.addWidget(file_selector)

        # -- Create File Format Group:
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel("File Format:"))
        top_row.addStretch(2)

        self.format_group = QtWidgets.QButtonGroup(self)
        btn_fits = QtWidgets.QRadioButton("FITS")
        self.format_group.addButton(btn_fits)
        self.format_group.setId(btn_fits, 0)
        btn_fits.toggled.connect(self.set_fits)

        btn_fits_table = QtWidgets.QRadioButton("FITS Table")
        btn_fits_table.setChecked(True)
        self.format_group.addButton(btn_fits_table)
        self.format_group.setId(btn_fits_table, 1)
        btn_fits_table.toggled.connect(self.set_fits)

        btn_ascii = QtWidgets.QRadioButton("ASCII")
        self.format_group.addButton(btn_ascii)
        self.format_group.setId(btn_ascii, 2)
        btn_ascii.toggled.connect(self.set_ascii)

        btn_row = QtWidgets.QHBoxLayout()
        # btn_row.addStretch(1)
        btn_row.addWidget(btn_fits)
        btn_row.addWidget(btn_fits_table)
        btn_row.addWidget(btn_ascii)

        # -- Aperture Inclusion:
        self.aper_btn = QtWidgets.QCheckBox("Include 2D Aperture Model")
        middle_row = QtWidgets.QHBoxLayout()
        # middle_row.addStretch(1)
        middle_row.addWidget(self.aper_btn)

        # -- Save and Cancel buttons:
        btn_save = QtWidgets.QPushButton("Save")
        btn_save.clicked.connect(self.save)

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.close)
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.addStretch(1)
        bottom_row.addWidget(btn_save)
        bottom_row.addWidget(btn_cancel)

        # -- Left List View of Objects:
        self.listview = QtWidgets.QListWidget()
        self.listview.setFixedWidth(200)
        for num in range(parent.list_widget.count()):
            parent_item = parent.list_widget.item(num)
            object_name = parent_item.text()
            item = QtWidgets.QListWidgetItem(object_name)
            self.listview.addItem(item)
        self.listview.setCurrentRow(index)

        # -- Manage Layout:
        main_layout = QtWidgets.QHBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setSpacing(3)
        left_layout = QtWidgets.QVBoxLayout()
        bottom_layout = QtWidgets.QHBoxLayout()
        lower_right_layout = QtWidgets.QVBoxLayout()

        lower_right_layout.addLayout(top_row)
        lower_right_layout.addLayout(btn_row)
        lower_right_layout.addStretch(1)
        lower_right_layout.addLayout(middle_row)
        lower_right_layout.addStretch(2)
        lower_right_layout.addLayout(bottom_row)
        bottom_layout.addStretch(1)
        bottom_layout.addLayout(lower_right_layout)

        left_layout.addWidget(QtWidgets.QLabel("Choose object to save:"))
        left_layout.addWidget(self.listview)

        right_layout.addWidget(QtWidgets.QLabel("Filename:"))
        right_layout.addLayout(fname_row)
        # right_layout.addStretch(1)
        right_layout.addLayout(bottom_layout)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout, 1)
        self.setLayout(main_layout)
        self.show()


    def set_ascii(self, checked):
        if checked:
            self.aper_btn.setDisabled(True)
            current_fname = self.fname_editor.text()
            file_root, ext = os.path.splitext(current_fname)
            self.fname_editor.setText("%s.dat" % file_root)

    def set_fits(self, checked):
        if checked:
            self.aper_btn.setEnabled(True)
            current_fname = self.fname_editor.text()
            file_root, ext = os.path.splitext(current_fname)
            self.fname_editor.setText("%s.fits" % file_root)

    def open_filedialog(self):
        basename = self.fname_editor.text()
        abs_path = os.path.abspath(basename)
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save All Extractions', abs_path)
        if fname:
            new_basename = os.path.relpath(fname)
            self.fname_editor.setText(new_basename)
        self.raise_()
        self.setFocus()

    def save(self):
        index = self.listview.currentRow()
        spectrum = self.parent.data1d[index]
        wl = spectrum.wl
        flux = spectrum.data
        err = spectrum.error
        hdr = spectrum.hdr
        bg = spectrum.background

        fname = self.fname_editor.text()
        file_format = self.format_group.checkedId()
        include_aperture = self.aper_btn.isChecked()
        if include_aperture:
            model2d = self.parent.trace_models[index].model2d
        else:
            model2d = None

        if file_format == 0:
            if fname[-5:] != '.fits':
                fname = fname + '.fits'
            saved, msg = save_fits_spectrum(fname, wl, flux, err, hdr, bg, aper=model2d)
        elif file_format == 1:
            if fname[-5:] != '.fits':
                fname = fname + '.fits'
            saved, msg = save_fitstable_spectrum(fname, wl, flux, err, hdr, bg, aper=model2d)
        elif file_format == 2:
            if fname[-4:] != '.dat':
                fname = fname + '.dat'
            saved, msg = save_ascii_spectrum(fname, wl, flux, err, hdr, bg)
        else:
            saved = False
            msg = "Unknown File type... Something went wrong!"

        if saved is False:
            QtWidgets.QMessageBox.critical(None, "Save Error", msg)
        self.close()


class ModelPropertiesWindow(QtWidgets.QDialog):
    def __init__(self, trace_model, list_index, parent=None):
        super(ModelPropertiesWindow, self).__init__(parent)
        self.setWindowTitle("Edit Model Properties")
        self.parent = parent
        self.trace_model = trace_model
        self.color = self.trace_model.color
        self.list_index = list_index
        self.model_fwhm = 0.

        self.color_button = QtWidgets.QPushButton()
        self.color_button.setFixedWidth(35)
        self.color_button.setFixedHeight(25)
        self.color_button.clicked.connect(self.color_selector)
        self.color_button.setStyleSheet("background-color: %s;" % self.trace_model.color)

        self.update_button = QtWidgets.QPushButton("Update")
        self.update_button.clicked.connect(self.update_values)

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)

        self.model_type_editor = QtWidgets.QComboBox()
        model_types = ['Tophat']
        current_model_type = self.trace_model.model_type.capitalize()
        _original_type = self.trace_model._original_type.capitalize()
        if _original_type in ["Moffat", "Gaussian"]:
            model_types.append(_original_type)
        self.model_type_editor.addItems(model_types)
        self.model_type_editor.setCurrentText(current_model_type)
        self.model_type_editor.currentTextChanged.connect(self.update_fwhm)


        cen = self.trace_model.cen
        lower = np.abs(self.trace_model.lower - cen)
        upper = np.abs(self.trace_model.upper - cen)
        self.fwhm_label = QtWidgets.QLabel("FWHM = ...")
        self.lower_editor = QtWidgets.QLineEdit("%.2f" % lower)
        self.lower_editor.setValidator(QtGui.QDoubleValidator())
        self.upper_editor = QtWidgets.QLineEdit("%.2f" % upper)
        self.upper_editor.setValidator(QtGui.QDoubleValidator())
        self.cen_editor = QtWidgets.QLineEdit("%.2f" % cen)
        self.cen_editor.setValidator(QtGui.QDoubleValidator())


        main_layout = QtWidgets.QGridLayout()
        main_layout.addWidget(QtWidgets.QLabel("Color:"), 0, 0)
        main_layout.addWidget(self.color_button, 0, 1)

        main_layout.addWidget(QtWidgets.QLabel("Model Type:"), 1, 0)
        main_layout.addWidget(self.model_type_editor, 1, 1)

        main_layout.addWidget(self.fwhm_label, 2, 0)

        main_layout.addWidget(QtWidgets.QLabel("Upper limit:"), 3, 0)
        main_layout.addWidget(self.upper_editor, 3, 1)

        main_layout.addWidget(QtWidgets.QLabel("Lower limit:"), 4, 0)
        main_layout.addWidget(self.lower_editor, 4, 1)

        main_layout.addWidget(QtWidgets.QLabel("Aperture Centroid:"), 5, 0)
        main_layout.addWidget(self.cen_editor, 5, 1)

        main_layout.addWidget(self.update_button, 6, 0)
        main_layout.addWidget(self.cancel_button, 6, 1)

        self.update_fwhm()
        self.setLayout(main_layout)
        self.show()

    def color_selector(self):
        color_dlg = QtWidgets.QColorDialog(self)
        color_dlg.setCurrentColor(QtGui.QColor(self.color))
        color = color_dlg.getColor()

        if color.isValid():
            self.color = str(color.name())
            self.color_button.setStyleSheet("background-color: %s;" % self.color)
        self.raise_()
        self.update_button.setFocus()

    def update_values(self):
        self.trace_model.set_color(self.color)
        item = self.parent.list_widget.item(self.list_index)
        item.setForeground(QtGui.QColor(self.color))
        new_model_type = self.model_type_editor.currentText().lower()
        self.trace_model.model_type = new_model_type
        if new_model_type == 'tophat':
            self.trace_model.set_visible(True)

        new_lower = float(self.lower_editor.text())
        new_upper = float(self.upper_editor.text())
        new_centroid = float(self.cen_editor.text())
        old_centroid = self.trace_model.cen
        centroid_shift = new_centroid - old_centroid
        if np.abs(centroid_shift) > 0:
            self.trace_model.points['mu'] += centroid_shift
            self.trace_model.fit['mu'] += centroid_shift
            self.trace_model.set_centroid(new_centroid)
        self.trace_model.set_range(new_centroid-new_lower, new_centroid+new_upper)

        self.parent.create_model_trace()
        self.parent.plot_fitted_points()
        self.parent.plot_trace_2d()
        self.parent.canvas_spsf.draw()
        self.parent.canvas_2d.draw()
        self.parent.canvas_1d.draw()
        self.parent.canvas_points.draw()
        self.close()

    def update_fwhm(self):
        new_model_type = self.model_type_editor.currentText().lower()
        if new_model_type == 'moffat':
            mu = self.trace_model.fit['mu']
            alpha = self.trace_model.fit['alpha']
            beta = self.trace_model.fit['beta']
            profile = NN_moffat(self.trace_model.y, np.median(mu), np.median(alpha), np.median(beta), 0.)
            self.fwhm, _ = get_FWHM(profile)
        elif new_model_type == 'gaussian':
            mu = self.trace_model.fit['mu']
            sigma = self.trace_model.fit['sigma']
            profile = NN_gaussian(self.trace_model.y, np.median(mu), np.median(sigma), 0.)
            self.fwhm, _ = get_FWHM(profile)
        else:
            SPSF = self.parent.axis_spsf.lines[0].get_ydata()
            SPSF -= np.median(SPSF)
            SPSF[SPSF < 0] = 0.
            self.fwhm, _ = get_FWHM(SPSF)
        self.fwhm_label.setText("FWHM = %.1f" % self.fwhm)


class HeaderViewer(QtWidgets.QDialog):
    def __init__(self, header, parent=None):
        super(HeaderViewer, self).__init__(parent)
        self.setWindowTitle("View Header")
        self.pattern = ""
        self.header_text = QtGui.QTextDocument(header.__repr__())
        header_font = QtGui.QFont("Courier")
        self.header_text.setDefaultFont(header_font)
        self.header_text.setTextWidth(80)
        self.header_text.adjustSize()

        self.search_bar = QtWidgets.QLineEdit()
        self.search_bar.textChanged.connect(self.search)
        self.search_bar.returnPressed.connect(self.next_match)
        self.search_button = QtWidgets.QPushButton("Find")
        self.search_button.clicked.connect(self.next_match)
        self.search_button.setDefault(False)
        self.search_button.setAutoDefault(False)
        self.search_result = QtWidgets.QLabel("")

        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setDocument(self.header_text)
        self.text_edit.setMinimumWidth(600)
        self.text_edit.setMinimumHeight(400)
        self.cursor = QtGui.QTextCursor(self.header_text)
        self.cursor.setPosition(0)

        main_layout = QtWidgets.QVBoxLayout()
        search_row = QtWidgets.QHBoxLayout()
        results_row = QtWidgets.QHBoxLayout()

        search_row.addWidget(QtWidgets.QLabel("Search:"))
        search_row.addWidget(self.search_bar)
        search_row.addWidget(self.search_button)
        results_row.addStretch(1)
        results_row.addWidget(self.search_result)

        main_layout.addLayout(search_row)
        main_layout.addLayout(results_row)
        main_layout.addWidget(self.text_edit)

        self.setLayout(main_layout)
        self.show()

    def search(self, pattern):
        self.pattern = pattern
        if pattern == '':
            self.search_result.setText("")
            self.cursor.setPosition(0)
        else:
            _matches = list(re.finditer(pattern, self.header_text.toPlainText()))
            N_matches = len(_matches)
            if N_matches == 1:
                self.search_result.setText("1 match found")
            elif N_matches > 1:
                self.search_result.setText("%i matches found" % N_matches)
            else:
                self.search_result.setText("No matches found")

    def next_match(self):
        self.cursor = self.header_text.find(self.pattern, self.cursor)
        self.text_edit.setTextCursor(self.cursor)


class WarningDialog(QtWidgets.QDialog):
    def __init__(self, parent, text, info=""):
        super(WarningDialog, self).__init__(parent)
        title = QtWidgets.QLabel(text)
        bold_font = QtGui.QFont()
        bold_font.setBold(True)
        title.setFont(bold_font)
        info = QtWidgets.QLabel(info)
        btn = QtWidgets.QPushButton("OK")
        btn.clicked.connect(self.close)

        # Handle Layout:
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(btn)
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(title)
        vbox.addWidget(info)
        vbox.addLayout(btn_row)
        self.setLayout(vbox)
        self.show()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Spectral Extraction')
    parser.add_argument("filename", type=str, nargs='?', default='',
                        help="Input 2D spectrum")
    parser.add_argument("--axis", type=int, default=1,
                        help="Dispersion axis 1: horizontal, 2: vertical  [default=1]")
    parser.add_argument("--locked", "-l", action='store_true',
                        help="Lock interface")
    args = parser.parse_args()

    if args.filename == 'test':
        # Load Test Data:
        fname = '/Users/krogager/coding/PyNOT/test/SCI_2D_crr_ALAb170110.fits'
        dispaxis = 2
        # fname = '/Users/krogager/Projects/Johan/close_spectra/obj.fits'
        # dispaxis = 1

    else:
        fname = args.filename
        dispaxis = args.axis
        args.axis


    # Launch App:
    qapp = QtWidgets.QApplication(sys.argv)
    app = ExtractGUI(fname, dispaxis=dispaxis, locked=args.locked, ymin=10, ymax=390, output_fname='tmp1d.fits')
    app.show()
    qapp.exec_()
