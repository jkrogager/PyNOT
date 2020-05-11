
__author__ = "Jens-Kristian Krogager"
__email__ = "krogager.jk@gmail.com"
__credits__ = ["Jens-Kristian Krogager", "Johan Fynbo"]
__version__ = '1.0'

import os
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

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Returns:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def fix_nans(y):
    """Fix NaN values in arrays by interpolating over them.

    Example:
        >>> y = np.array([1, 2, 3, Nan, Nan, 6])
        >>> y_fix = fix_nans(y)
        y_fix: array([ 1.,  2.,  3.,  4.,  5.,  6.])
    """
    nans, x = nan_helper(y)
    y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y


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

def mad(img):
    """Calculate Median Absolute Deviation from the median
    This is a robust variance estimator.
    For a Gaussian distribution:
        sigma ≈ 1.4826 * MAD
    """
    return np.median(np.abs(img - np.median(img)))


def median_filter_data(x, kappa=5., window=21, parname=None):
    med_x = median_filter(x, window)
    MAD = np.median(np.abs(x - med_x))*1.48
    if MAD == 0.:
        MAD = np.std(x - med_x)
    mask = np.abs(x - med_x) < kappa*MAD
    return (med_x, mask)


def gui_label(text, color='black'):
    label_string = "<font color=%s>%s</font>" % (color, text)
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
        wavelength = np.arange(hdr['NAXIS%i' % dispaxis])

    return wavelength


def tophat(x, low, high):
    """Tophat profile: 1 within [low: high], 0 outside"""
    mask = (x >= low) & (x <= high)
    return mask * 1.


def NN_moffat(x, mu, alpha, beta, logamp):
    """One-dimensional non-negative Moffat profile."""
    amp = 10**logamp
    return amp*(1. + ((x-mu)**2/alpha**2))**(-beta)


def NN_gaussian(x, mu, sigma, logamp):
    """ One-dimensional modified non-negative Gaussian profile."""
    amp = 10**logamp
    return amp * np.exp(-0.5*(x-mu)**2/sigma**2)


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


def auto_localize(img2D):
    img2D = img2D.astype(np.float64)

    spsf = np.median(img2D, axis=1)
    spsf = spsf - np.median(spsf)
    spsf[spsf < 0] = 0.

    # Detect peaks:
    kappa = 10.
    noise = mad(img2D) * 1.48 / np.sqrt(img2D.shape[0])
    peaks, properties = find_peaks(spsf, prominence=kappa*noise)
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


class TraceModel(object):
    def __init__(self, cen, amp, axis, shape, fwhm=5, model_type='Moffat', xmin=None, xmax=None, color='RoyalBlue'):
        self.model_type = model_type
        self.xmin = xmin
        self.xmax = xmax
        self.color = color
        self.cmap = make_linear_colormap(color)
        self.cen = cen
        self.lower = cen - fwhm
        self.upper = cen + fwhm
        self.amp = amp
        self.model2d = np.zeros(shape)
        self.x = np.arange(shape[1], dtype=np.float64)
        self.y = np.arange(shape[0], dtype=np.float64)
        self.axis = axis

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
        v_lower.set_visible(False)
        v_upper.set_visible(False)
        self.vlines = [v_lower, v_cen, v_upper]

        # Collector for points:
        self.point_lines = {'mu': [], 'sigma': [],
                            'alpha': [], 'beta': []}
        self.fit_lines = {'mu': None, 'sigma': None,
                          'alpha': None, 'beta': None}
        self.model_image = None
        self.plot_1d = None


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
        unicode_names = {'mu': 'µ', 'sigma': 'σ',
                         'alpha': 'α', 'beta': 'β'}
        return unicode_names[parname]

    def get_data(self):
        if self.model_type.lower() == 'moffat':
            return (self.x_binned, self.points['mu'], self.points['alpha'], self.points['beta'])
        elif self.model_type.lower() == 'gaussian':
            return (self.x_binned, self.points['mu'], self.points['sigma'])
        elif self.model_type.lower() == 'tophat':
            return (self.x_binned, self.points['mu'], self.points['alpha'], self.points['beta'])

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

    def deactivate(self):
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

    def activate(self):
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

    def set_visible(self, vis=True):
        self.vlines[0].set_visible(vis)
        self.vlines[2].set_visible(vis)


class ImageData(object):
    def __init__(self, fname, dispaxis=1):
        self.filename = fname
        data_temp = fits.getdata(fname)
        self.data = data_temp.astype(np.float64)
        self.shape = self.data.shape
        try:
            self.error = fits.getdata(fname, 1)
        except:
            noise = mad(self.data) * 1.48
            self.error = np.ones_like(self.data) * noise

        with fits.open(fname) as hdu:
            self.header = hdu[0].header
            if len(hdu) > 1:
                imghdr = hdu[1].header
                self.header.update(imghdr)

        if dispaxis == 2:
            self.wl = get_wavelength_from_header(self.header, dispaxis)
            self.data = self.data.T
            self.error = self.error.T
            self.shape = self.data.shape
        else:
            self.wl = get_wavelength_from_header(self.header)
        self.x = np.arange(self.data.shape[1], dtype=np.float64)
        self.y = np.arange(self.data.shape[0], dtype=np.float64)


class Spectrum(object):
    def __init__(self, wl=None, data=None, error=None, mask=None, hdr=None):
        self.wl = wl
        self.data = data
        self.error = error
        self.mask = mask
        self.hdr = hdr

        self.plot_line = None


class ExtractGUI(QtWidgets.QMainWindow):
    def __init__(self, fname='', dispaxis=1, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setWindowTitle('PyNOT: Extract')
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        # Set attributes:
        self.model_type = 'Moffat'
        self.data1d = list()
        self.background = None
        self.trace_models = list()
        self.delete_picked_object = False
        self.picked_object = None

        # SPSF controls:
        self.add_btn = QtWidgets.QPushButton("Add Object")
        self.add_btn.clicked.connect(self.add_new_object)
        self.add_bg_btn = QtWidgets.QPushButton("Mark Background")
        self.add_bg_btn.clicked.connect(self.add_bg_range)

        self.remove_btn = QtWidgets.QPushButton("Delete")
        self.remove_btn.clicked.connect(self.delete_object)


        # Limits for profile averaging and fitting:
        self.xmin_edit = QtWidgets.QLineEdit("0")
        self.xmax_edit = QtWidgets.QLineEdit("")
        self.xmin_edit.setValidator(QtGui.QIntValidator(0, 1e6))
        self.xmax_edit.setValidator(QtGui.QIntValidator(0, 1e6))
        self.xmin_edit.returnPressed.connect(self.limits_updated)
        self.xmax_edit.returnPressed.connect(self.limits_updated)

        self.ymin_edit = QtWidgets.QLineEdit("0")
        self.ymax_edit = QtWidgets.QLineEdit("")
        self.ymin_edit.setValidator(QtGui.QIntValidator(0, 1e6))
        self.ymax_edit.setValidator(QtGui.QIntValidator(0, 1e6))
        self.ymin_edit.returnPressed.connect(self.limits_updated)
        self.ymax_edit.returnPressed.connect(self.limits_updated)

        # Fitting Parameters:
        self.model_chooser = QtWidgets.QComboBox()
        self.model_chooser.addItems(["Moffat", "Gaussian", "Tophat"])
        self.model_chooser.currentTextChanged.connect(self.model_change)

        self.bins_edit = QtWidgets.QLineEdit("50")
        self.bins_edit.setValidator(QtGui.QIntValidator(0, 1e6))
        self.bins_edit.returnPressed.connect(self.fit_trace)

        self.med_kappa_edit = QtWidgets.QLineEdit("5")
        self.med_kappa_edit.setValidator(QtGui.QDoubleValidator())
        self.med_kappa_edit.returnPressed.connect(self.median_filter_points)

        self.median_btn = QtWidgets.QPushButton("Median Filter Points")
        self.median_btn.clicked.connect(self.median_filter_points)

        self.c_order_edit = QtWidgets.QLineEdit("3")
        self.c_order_edit.setValidator(QtGui.QIntValidator(0, 100))
        self.c_order_edit.returnPressed.connect(self.fit_trace)
        self.w_order_edit = QtWidgets.QLineEdit("1")
        self.w_order_edit.setValidator(QtGui.QIntValidator(0, 100))
        self.w_order_edit.returnPressed.connect(self.fit_trace)

        self.extract_btn = QtWidgets.QPushButton("Extract")
        self.extract_btn.clicked.connect(self.extract)

        self.fit_btn = QtWidgets.QPushButton("Fit Spectral Trace")
        self.fit_btn.clicked.connect(self.fit_trace)

        # SPSF Viewer:
        self.fig_spsf = Figure(figsize=(4, 3))
        self.canvas_spsf = FigureCanvas(self.fig_spsf)
        self.canvas_spsf.mpl_connect('key_press_event', self.on_key_press)
        self.canvas_spsf.mpl_connect('pick_event', self.on_pick)
        self.canvas_spsf.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas_spsf.mpl_connect('button_release_event', self.on_release)
        self.canvas_spsf.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.spsf_mpl_toolbar = NavigationToolbar(self.canvas_spsf, self)
        self.spsf_mpl_toolbar.setFixedHeight(20)
        self.spsf_mpl_toolbar.setFixedWidth(400)
        self.axis_spsf = self.fig_spsf.add_subplot(111)

        # List of Trace Models:
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.itemChanged.connect(self.toggle_trace_models)


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
        self.axis_2d_bg = self.figure_2d.add_subplot(212)
        self.axis_2d_bg.set_ylabel("Background")
        self.figure_2d.tight_layout()

        self.vmin_edit = QtWidgets.QLineEdit("")
        self.vmax_edit = QtWidgets.QLineEdit("")
        self.vmin_edit.returnPressed.connect(self.update_2d)
        self.vmax_edit.returnPressed.connect(self.update_2d)
        self.vminmax_btn = QtWidgets.QPushButton("Update")
        self.vminmax_btn.clicked.connect(self.update_2d)
        self.vmin_edit.setValidator(QtGui.QDoubleValidator())
        self.vmax_edit.setValidator(QtGui.QDoubleValidator())
        self.bg_fit_btn = QtWidgets.QPushButton("Fit Background")
        self.bg_fit_btn.clicked.connect(self.fit_background)
        row_imvals = QtWidgets.QHBoxLayout()
        row_imvals.addWidget(gui_label("v<sub>min</sub>:", color='#111111'))
        row_imvals.addWidget(self.vmin_edit)
        row_imvals.addWidget(gui_label("v<sub>max</sub>:", color='#111111'))
        row_imvals.addWidget(self.vmax_edit)
        row_imvals.addWidget(self.vminmax_btn)
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
        self.canvas_points = FigureCanvas(self.figure_points)
        self.canvas_points.mpl_connect('pick_event', self.pick_points)
        self.figp_mpl_toolbar = NavigationToolbar(self.canvas_points, self)
        self.figp_mpl_toolbar.setFixedHeight(20)

        row_median = QtWidgets.QHBoxLayout()
        row_median.addWidget(self.median_btn)
        row_median.addStretch(1)
        row_median.addWidget(QtWidgets.QLabel("Kappa: "))
        row_median.addWidget(self.med_kappa_edit)
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
        self.axis_1d.set_xlabel("Dispersion axis")
        self.axis_1d.set_ylabel("Extracted flux")
        self.figure_1d.tight_layout()
        # connect scroll event to smoothing!
        self.canvas_1d = FigureCanvas(self.figure_1d)
        self.fig1d_mpl_toolbar = NavigationToolbar(self.canvas_1d, self)
        self.fig1d_mpl_toolbar.setFixedHeight(20)
        layout_tab3 = QtWidgets.QVBoxLayout()
        layout_tab3.addWidget(self.canvas_1d)
        layout_tab3.addWidget(self.fig1d_mpl_toolbar)
        self.tab3.setLayout(layout_tab3)


        # == Layout ===========================================================
        main_layout = QtWidgets.QHBoxLayout(self._main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # TabWidget Layout:
        main_layout.addWidget(self.tab_widget, 1)

        # Right Panel Layout:
        right_panel = QtWidgets.QVBoxLayout()

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.add_btn)
        button_row.addWidget(self.add_bg_btn)
        button_row.addStretch(1)
        button_row.addWidget(self.remove_btn)
        right_panel.addLayout(button_row)

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
        row_fit.addWidget(self.fit_btn)
        row_fit.addWidget(self.extract_btn)
        right_panel.addLayout(row_fit)

        separatorLine = QtWidgets.QFrame()
        separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        separatorLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        separatorLine.setMinimumSize(3, 20)
        right_panel.addWidget(separatorLine)

        right_panel.addWidget(gui_label("List of Trace Models", color='dimgray'))
        right_panel.addWidget(self.list_widget)

        main_layout.addLayout(right_panel)

        self.canvas_2d.setFocus()


        # -- Set Data:
        if fname:
            self.load_spectrum(fname, dispaxis)

        else:
            self.image2d = None
            self.background = None
            self.last_fit = tuple()


    def load_spectrum(self, fname, dispaxis=1):
        self.image2d = ImageData(fname, dispaxis)
        self.last_fit = tuple()
        self.background = BackgroundModel(self.axis_spsf, self.image2d.data.shape)
        self.background.model2d += np.median(self.image2d.data)
        self.xmax_edit.setText("%i" % self.image2d.data.shape[1])
        self.ymax_edit.setText("%i" % self.image2d.data.shape[0])
        self.update_2d()
        self.update_spsf()
        self.localize_trace()

    def add_new_object(self):
        if self.image2d is None:
            msg = "Load data before defining an object trace"
            QtWidgets.QMessageBox.critical(None, 'No data loaded', msg)
            return
        self.axis_spsf.set_title("Mark Trace Centroid", fontsize=11)
        self.canvas_spsf.draw()
        points = self.fig_spsf.ginput(1, 0)
        if len(points) == 0:
            self.axis_spsf.set_title("")
            self.canvas_spsf.draw()
            return
        center, _ = points[0]
        self.axis_spsf.lines[0]
        x_data, SPSF = self.axis_spsf.lines[0].get_data()
        imin = np.argmin(np.abs(x_data - center))
        height = SPSF[imin]
        trace_model = TraceModel(center, height, self.axis_spsf, shape=self.image2d.data.shape,
                                 color=next(color_cycle))
        self.add_trace(trace_model)
        self.axis_spsf.set_title("")
        self.canvas_spsf.draw()

    def add_bg_range(self):
        if self.image2d is None:
            msg = "Load data before defining background ranges"
            QtWidgets.QMessageBox.critical(None, 'No data loaded', msg)
            return
        self.axis_spsf.set_title("Mark background range limits", fontsize=11)
        self.canvas_spsf.draw()
        points = self.fig_spsf.ginput(2, 0)
        if len(points) == 0:
            self.axis_spsf.set_title("")
            self.canvas_spsf.draw()
            return
        x1, x2 = np.sort([xy[0] for xy in points])
        self.background.add_range(x1, x2)
        self.axis_spsf.set_title("")
        self.canvas_spsf.draw()

    def fit_background(self):
        if self.background is None:
            return
        elif len(self.background.ranges) == 0:
            self.background.model2d *= 0.
            self.background.model2d += np.median(self.image2d.data)
        else:
            bg_order = 3
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
                this_mask = mask * (np.abs(column - med_column) < 10*noise)
                bg_model = Chebyshev.fit(y[this_mask], column[this_mask], bg_order, domain=(y.min(), y.max()))
                self.background.model2d[:, i] = bg_model(y)
                self.progress.setValue(i+1)

            if np.all(self.image2d.error == self.image2d.error[0]):
                # Update the noise model for self-generated noise image:
                noise = mad(self.image2d.data - self.background.model2d) * 1.48
                self.image2d.error = np.ones_like(self.image2d.data) * noise
        self.update_2d()


    def on_pick(self, event):
        artist = event.artist
        if isinstance(artist, matplotlib.patches.Polygon):
            # -- Delete Background Patch
            if self.delete_picked_object:
                index = self.background.patches.index(artist)
                self.background.remove_range(index)

        else:
            for num, model in enumerate(self.trace_models):
                if artist in model.vlines:
                    if self.delete_picked_object:
                        print("Delete this trace!")
                        self.remove_trace(num)
                        self.delete_picked_object = False
                    else:
                        self.picked_object = (num, artist, model)


    def on_motion(self, event):
        if self.picked_object is None:
            return
        elif not event.inaxes:
            return

        num, artist, trace_model = self.picked_object
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
        num, artist, trace_model = self.picked_object
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
        self.picked_object = None


    def on_key_press(self, event):
        if event.key == 'b':
            self.add_bg_range()

        elif event.key == 'a':
            self.add_new_object()

        elif event.key == 'd':
            self.delete_object()


    def delete_object(self):
        # Detect whether the selected object is a background region
        # or whether it's a target region.
        self.axis_spsf.set_title("Select background range to delete", fontsize=11)
        self.canvas_spsf.draw()
        self.delete_picked_object = True
        self.fig_spsf.waitforbuttonpress(5)
        self.delete_picked_object = False
        self.axis_spsf.set_title("")
        self.canvas_spsf.draw()


    def remove_trace(self, index):
        trace_model = self.trace_models[index]
        for vline in trace_model.vlines:
            vline.remove()
        self.canvas_spsf.draw()
        self.list_widget.takeItem(index)
        self.trace_models.pop(index)


    def add_trace(self, model):
        self.trace_models.append(model)
        N = self.list_widget.count() + 1
        object_name = self.image2d.header['OBJECT'] + '_%i' % N
        item = QtWidgets.QListWidgetItem(object_name)
        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        item.setCheckState(QtCore.Qt.Checked)
        item.setForeground(QtGui.QColor(model.color))
        self.list_widget.addItem(item)

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
        if len(self.axis_spsf.lines) == 0:
            self.axis_spsf.plot(self.image2d.y[ymin:ymax], SPSF, color='k', lw=0.7)
        else:
            self.axis_spsf.lines[0].set_data(self.image2d.y[ymin:ymax], SPSF)
        ymin = np.median(SPSF) - 10.*mad(SPSF)
        ymax = np.max(SPSF) + 10.*mad(SPSF)
        self.axis_spsf.set_ylim(ymin, ymax)
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
        self.update_value_range(vmin, vmax)
        self.canvas_2d.draw()

    def plot_trace_2d(self):
        if len(self.last_fit) == 0:
            return

        for num, model in enumerate(self.trace_models):
            trace_model_2d = model.model2d
            if np.max(trace_model_2d) != 0.:
                trace_model_2d /= np.max(trace_model_2d)
            alpha_array = 2 * trace_model_2d.copy()
            alpha_array[alpha_array > 0.1] += 0.2
            alpha_array[alpha_array > 0.3] = 0.5
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

    def localize_trace(self):
        if self.image2d is not None:
            peaks, prominences = auto_localize(self.image2d.data)
            if len(peaks) == 0:
                msg = "Automatic trace detection failed!"
                QtWidgets.QMessageBox.critical(None, 'No trace detected', msg)

            else:
                for center, height in zip(peaks, prominences):
                    trace_model = TraceModel(center, height, self.axis_spsf, shape=self.image2d.data.shape,
                                             color=next(color_cycle))
                    self.add_trace(trace_model)

    def model_change(self, text):
        if text.lower() == 'tophat':
            for num, model in enumerate(self.trace_models):
                listItem = self.list_widget.item(num)
                if listItem.checkState() == 2:
                    model.set_visible()
                else:
                    model.set_visible(False)
            self.canvas_spsf.draw()

        elif text.lower() == 'gaussian' or text.lower() == 'moffat':
            for model in self.trace_models:
                model.set_visible(False)
            self.canvas_spsf.draw()

    def fit_trace(self):
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
                trace_model.xmin = xmin
                trace_model.xmax = xmax
                trace_model.model_type = original_model
                mu = np.array([par['mu_%i' % num] for par in trace_parameters])
                if model_type == 'moffat':
                    alpha = np.array([par['a_%i' % num] for par in trace_parameters])
                    beta = np.array([par['b_%i' % num] for par in trace_parameters])
                    trace_model.set_data(x_binned, mu, alpha=alpha, beta=beta)

                elif model_type == 'gaussian':
                    sig = np.array([par['sig_%i' % num] for par in trace_parameters])
                    trace_model.set_data(x_binned, mu, sigma=sig)
            self.last_fit = this_fit
            self.create_model_trace()
            self.plot_fitted_points()
        else:
            self.create_model_trace()
            self.plot_fitted_points(update_only=True)
        self.plot_trace_2d()
        self.activate_all_listItems()
        self.tab_widget.setCurrentIndex(1)

    def plot_fitted_points(self, update_only=False):
        parameters = self.trace_models[0].get_parnames()
        if update_only:
            for model in self.trace_models:
                parameters = model.get_parnames()
                for ax, parname in zip(self.axes_points, parameters):
                    mask = model.get_mask(parname)
                    for line in model.point_lines[parname]:
                        line.remove()
                    l1, = ax.plot(model.x_binned[mask], model.points[parname][mask],
                                  color=model.color, marker='o', ls='', mec=color_shade(model.color), picker=6.)
                    l2, = ax.plot(model.x_binned[~mask], model.points[parname][~mask],
                                  color=model.color, marker='o', ls='', alpha=0.3, picker=6.)
                    l3, = ax.plot(model.x_binned[~mask], model.points[parname][~mask],
                                  color='k', marker='x', ls='')
                    model.point_lines[parname] = [l1, l2, l3]
                    if parname == 'mu':
                        l4, = self.axis_2d.plot(model.x_binned[mask], model.points[parname][mask],
                                                color=model.color, marker='o', ls='', alpha=0.3, picker=6.)
                        model.point_lines[parname].append(l4)
                    # -- Plot fit to points:
                    if len(model.fit['mu']) > 0:
                        lf = model.fit_lines[parname]
                        lf.set_data(model.x, model.fit[parname])
        else:
            self.figure_points.clear()
            N_rows = len(parameters)
            if N_rows == 1:
                self.axes_points = [self.figure_points.subplots(N_rows, 1)]
            else:
                self.axes_points = self.figure_points.subplots(N_rows, 1)

            for ax, parname in zip(self.axes_points, parameters):
                for model in self.trace_models:
                    mask = model.get_mask(parname)
                    # -- Plot traced points:
                    l1, = ax.plot(model.x_binned[mask], model.points[parname][mask],
                                  color=model.color, marker='o', ls='', mec=color_shade(model.color), picker=6.)
                    l2, = ax.plot(model.x_binned[~mask], model.points[parname][~mask],
                                  color=model.color, marker='o', ls='', alpha=0.3, picker=6.)
                    l3, = ax.plot(model.x_binned[~mask], model.points[parname][~mask],
                                  color='k', marker='x', ls='')
                    model.point_lines[parname] = [l1, l2, l3]
                    if parname == 'mu':
                        l4, = self.axis_2d.plot(model.x_binned[mask], model.points[parname][mask],
                                                color=model.color, marker='o', ls='', alpha=0.3, picker=6.)
                        model.point_lines[parname].append(l4)
                    # -- Plot fit to points:
                    if len(model.fit['mu']) > 0:
                        lf, = ax.plot(model.x, model.fit[parname],
                                      color=model.color, ls='--', lw=1.0)
                        model.fit_lines[parname] = lf
                ax.set_ylabel(r"$\%s$" % parname)
        self.canvas_points.draw()
        self.canvas_2d.draw()

    def create_model_trace(self):
        center_order, width_order = self.get_trace_orders()
        for model in self.trace_models:
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
                delta_lower = lower - cen
                delta_upper = upper - cen
                mu = model.fit['mu']
                lower_array = np.round(mu + delta_lower, 0)
                upper_array = np.round(mu + delta_upper, 0)
                pars_table = np.column_stack([lower_array, upper_array])

            for num, pars in enumerate(pars_table):
                P_i = model_function(model.y, *pars)
                P_i = P_i/np.sum(P_i)
                model.model2d[:, num] = P_i

    def get_trace_orders(self):
        c_order = int(self.c_order_edit.text())
        w_order = int(self.w_order_edit.text())
        return (c_order, w_order)

    def pick_points(self, event):
        artist = event.artist
        if isinstance(artist, matplotlib.lines.Line2D):
            for model in self.trace_models:
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
        window = 5
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
        if len(self.trace_models) == 0:
            return

        data1d_list = []
        for num, model in enumerate(self.trace_models):
            P = model.model2d
            img2d = self.image2d.data - self.background.model2d
            V = self.image2d.error**2
            M = np.ones_like(img2d, dtype=bool)

            data1d = np.sum(M*P*img2d, axis=0)/np.sum(M*P**2, axis=0)
            if model.model_type == 'tophat':
                err1d = np.sqrt(np.sum(V*P, axis=0))
                err1d = fix_nans(err1d)
            else:
                err1d = np.sqrt(np.sum(M*P, axis=0)/np.sum(M*P**2/V, axis=0))
                err1d = fix_nans(err1d)

            wl = self.image2d.wl
            spec1d = Spectrum(wl, data1d, err1d)
            data1d_list.append(spec1d)
        self.data1d = data1d_list
        self.plot_data1d()

    def plot_data1d(self):
        if len(self.data1d) == 0:
            return
        for num, model in enumerate(self.trace_models):
            spec1d = self.data1d[num]
            if model.plot_1d is not None:
                for child in model.plot_1d.get_children():
                    child.remove()
                    # -- find a way to update the data instead...
            model.plot_1d = self.axis_1d.errorbar(spec1d.wl, spec1d.data, spec1d.error,
                                                  color=model.color, lw=1., elinewidth=0.5)
            listItem = self.list_widget.item(num)
            if listItem.checkState() == 2:
                for child in model.plot_1d.get_children():
                    child.set_visible(True)
            else:
                for child in model.plot_1d.get_children():
                    child.set_visible(False)
        self.canvas_1d.draw()
        self.tab_widget.setCurrentIndex(2)


    def toggle_trace_models(self, listItem):
        index = self.list_widget.row(listItem)
        model_type = self.model_chooser.currentText()
        if listItem.checkState() == 2:
            # Active:
            self.trace_models[index].activate()
            if model_type.lower() == 'tophat':
                self.trace_models[index].set_visible(True)
        else:
            # Inactive:
            self.trace_models[index].deactivate()
        self.plot_trace_2d()
        self.canvas_spsf.draw()
        self.canvas_points.draw()

    def activate_all_listItems(self):
        N_rows = self.list_widget.count()
        for num in range(N_rows):
            item = self.list_widget.item(num)
            item.setCheckState(2)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Spectral Extraction')
    parser.add_argument("filename", type=str, nargs='?', default='',
                        help="Input 2D spectrum")
    parser.add_argument("--axis", type=int, default=1,
                        help="Dispersion axis 1: horizontal, 2: vertical  [default=1]")
    args = parser.parse_args()

    if args.filename == 'test':
        # # Load Test Data:
        # fname = '/Users/krogager/coding/PyNOT/test/SCI_2D_crr_ALAb170110.fits'
        # dispaxis = 2
        fname = '/Users/krogager/Projects/Johan/close_spectra/obj.fits'
        dispaxis = 1

    else:
        fname = args.filename
        dispaxis = args.axis


    # Launch App:
    qapp = QtWidgets.QApplication(sys.argv)
    app = ExtractGUI(fname, dispaxis=dispaxis)
    app.show()
    qapp.exec_()
