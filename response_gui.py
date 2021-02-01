#!/usr/bin/env python3
"""
    PyNOT -- Extract GUI

Graphical interface to extract 1D spectra

"""

__author__ = "Jens-Kristian Krogager"
__email__ = "krogager.jk@gmail.com"
__credits__ = ["Jens-Kristian Krogager"]
__version__ = '0.13'

import copy
import os
import re
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.ndimage import median_filter
from numpy.polynomial import Chebyshev
from astropy.io import fits
from PyQt5 import QtCore, QtGui, QtWidgets
import warnings

import alfosc

def run_gui(input_fname, app=None, order=8):
    # global app
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    gui = ResponseGUI(input_fname, locked=True, order=order)
    gui.show()
    app.exec_()
    # Get Response Array:
    response = gui.response
    return response


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
        unicode_names = {'mu': 'µ', 'sigma': 'σ',
                         'alpha': 'α', 'beta': 'β'}
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




class Spectrum(object):
    def __init__(self, wl=None, data=None, error=None, header={}, wl_unit='', flux_unit=''):
        self.wl = wl
        self.data = data
        self.error = error
        self.header = header
        self.wl_unit = wl_unit
        self.flux_unit = flux_unit

        self.plot_line = None



class ResponseGUI(QtWidgets.QMainWindow):
    def __init__(self, fname=None, output_fname='', star_name='', order=8, parent=None, locked=False, **kwargs):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setWindowTitle('PyNOT: Response')
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        # Set attributes:
        self.spectrum = None
        self.response = None
        self.filename = fname
        # Extinction table attributes:
        self.ext_fname = alfosc.path + '/calib/lapalma.ext'
        try:
            ext_wl, ext = np.loadtxt(alfosc.path + '/calib/lapalma.ext', unpack=True)
        except:
            ext_wl, ext = None, None
        self.ext_wl = ext_wl
        self.ext = ext
        # Reference table attributes:
        self.ref_tab = None
        self.flux_bins = None
        self.wl_bins = None
        self.mag_bins = None
        self.resp_bins = None
        self.mask = None


        # Fitting Parameters:
        self.star_chooser = QtWidgets.QComboBox()
        all_names = [alfosc.standard_star_names.values()] + ['']
        self.star_chooser.addItems(all_names)
        self.star_chooser.setCurrentText(star_name)
        self.star_chooser.currentTextChanged.connect(self.set_star)

        self.exptime_edit = QtWidgets.QLineEdit("")
        self.exptime_edit.setValidator(QtGui.QDoubleValidator())
        self.airmass_edit = QtWidgets.QLineEdit("")
        self.airmass_edit.setValidator(QtGui.QDoubleValidator())

        self.order_edit = QtWidgets.QLineEdit("%i" % order)
        self.order_edit.setValidator(QtGui.QIntValidator(0, 100))
        self.order_edit.returnPressed.connect(self.fit_response)

        self.fit_btn = QtWidgets.QPushButton("Fit Response")
        self.fit_btn.setShortcut("ctrl+F")
        self.fit_btn.clicked.connect(self.fit_response)

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
        self.spsf_mpl_toolbar.setFixedWidth(400)
        self.axis_spsf = self.fig_spsf.add_subplot(111)

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

    def save_all_extractions(self, fname=''):
        if len(self.data1d) == 0:
            msg = "No 1D spectra have been extracted. Nothing to save..."
            QtWidgets.QMessageBox.critical(None, "Save Error", msg)
            return False

        if not fname:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            basename = current_dir + '/' + self.image2d.header['OBJECT'] + '_ext.fits'
            filters = "FITS Files (*.fits *.fit)"
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save All Extractions', basename, filters)

        if fname:
            hdu = fits.HDUList()
            prim_hdr = fits.Header()
            prim_hdr['AUTHOR'] = 'PyNOT'
            prim_hdr['OBJECT'] = self.image2d.header['OBJECT']
            prim_hdr['DATE-OBS'] = self.image2d.header['DATE-OBS']
            prim_hdr['EXPTIME'] = self.image2d.header['EXPTIME']
            prim_hdr['AIRMASS'] = self.image2d.header['AIRMASS']
            prim_hdr['ALGRNM'] = self.image2d.header['ALGRNM']
            prim_hdr['ALAPRTNM'] = self.image2d.header['ALAPRTNM']
            prim_hdr['RA'] = self.image2d.header['RA']
            prim_hdr['DEC'] = self.image2d.header['DEC']
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

    def clear_all(self):
        for ax in self.axes:
            ax.clear()
        self.exptime_edit.setText("")
        self.airmass_edit.setText("")
        self.flux_bins = None
        self.wl_bins = None
        self.mag_bins = None
        self.resp_bins = None
        self.response = None
        self.mask = None
        self.filename = ""


    def load_spectrum(self, fname=''):
        if fname is False:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            filters = "FITS files (*.fits | *.fit)"
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open 1D Spectrum', current_dir, filters)
            fname = str(fname)

        if not os.path.exists(fname):
            return

        # Clear all models:
        self.clear_all()

        hdr = fits.getheader(fname)
        table = fits.getdata(fname)
        self.spectrum = Spectrum(wl=table['WAVE'], data=table['FLUX'], header=hdr,
                                 wl_unit=table.columns['WAVE'].unit,
                                 flux_unit=table.columns['FLUX'].unit)
        if 'EXPTIME' in hdr:
            self.exptime_edit.setText("%.1f" % hdr['EXPTIME'])
        if 'AIRMASS' in hdr:
            self.airmass_edit.setText("%.1f" % hdr['AIRMASS'])

        if self.ref_tab is not None:
            self.calculate_flux_in_bins()
            self.calculate_response_bins()
        self.update_plot()


    def set_star(self, text):
        star_name = str(text).lower()
        self.ref_tab = np.loadtxt(alfosc.path+'/calib/std/%s.dat' % star_name)
        self.calculate_flux_in_bins()
        self.calculate_response_bins()
        self.update_plot()


    def calculate_flux_in_bins(self):
        if self.spectrum is None:
            msg = "No spectrum loaded!"
            info = "No spectral data has been loaded."
            WarningDialog(self, msg, info)
            return
        wl = self.spectrum.wl
        flux = self.data
        flux_bins = list()
        for wl_ref, mag_ref, bandwidth in self.ref_tab:
            l1 = wl_ref - bandwidth/2
            l2 = wl_ref + bandwidth/2
            band = (wl >= l1) * (wl <= l2)
            f0 = np.nansum(flux[band])
            flux_bins.append(f0 / bandwidth)
        self.wl_bins = self.ref_tab[:, 0]
        self.flux_bins = np.array(flux_bins)
        self.mag_bins = self.ref_tab[:, 1]

    def calculate_response_bins(self):
        ref_flux = 10**(-(self.mag_bins + 2.406)/2.5) / (self.wl_bins)**2
        exp_str = self.exptime_edit.text()
        airm_str = self.airmass_edit.text()
        if exp_str == '' or airm_str == '':
            WarningDialog(self, "No exposure time or airmass!", "Please set both exposure time and airmass.")
            return
        exptime = float(exp_str)
        airmass = float(airm_str)
        # Calculate Sensitivity:
        self.resp_bins = 2.5*np.log10(self.flux_bins / (exptime * ref_flux)) + airmass*self.ext

    def plot_flux_bins(self):
        pass

    def plot_response_bins(self):
        pass

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
                        fwhm = get_FWHM(profile)
                        lower = trace_model.cen - 2*fwhm
                        upper = trace_model.cen + 2*fwhm
                        trace_model.set_centroid(np.median(mu))
                        trace_model.set_range(lower, upper)

                    elif model_type == 'gaussian':
                        sig = np.array([par['sig_%i' % num] for par in trace_parameters])
                        trace_model.set_data(x_binned, mu, sigma=sig)
                        # Set extraction limits to ±2xFWHM:
                        profile = NN_gaussian(trace_model.y, np.median(mu), np.median(sig), 0.)
                        fwhm = get_FWHM(profile)
                        lower = trace_model.cen - 2*fwhm
                        upper = trace_model.cen + 2*fwhm
                        trace_model.set_centroid(np.median(mu))
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
                                  color=model.color, marker='o', ls='', mec=color_shade(model.color), picker=True, pickradius=10)
                    l2, = ax.plot(model.x_binned[~mask], model.points[parname][~mask],
                                  color=model.color, marker='o', ls='', alpha=0.3, picker=True, pickradius=10)
                    l3, = ax.plot(model.x_binned[~mask], model.points[parname][~mask],
                                  color='k', marker='x', ls='')
                    model.point_lines[parname] = [l1, l2, l3]
                    if parname == 'mu':
                        l4, = self.axis_2d.plot(model.x_binned[mask], model.points[parname][mask],
                                                color=model.color, marker='o', ls='', alpha=0.5, picker=True, pickradius=10)
                        model.point_lines[parname].append(l4)
                    # -- Plot fit to points:
                    if len(model.fit['mu']) > 0:
                        lf = model.fit_lines[parname]
                        lf.set_data(model.x, model.fit[parname])
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
                                  color=model.color, marker='o', ls='', mec=color_shade(model.color), picker=True, pickradius=10)
                    l2, = ax.plot(model.x_binned[~mask], model.points[parname][~mask],
                                  color=model.color, marker='o', ls='', alpha=0.3, picker=True, pickradius=10)
                    l3, = ax.plot(model.x_binned[~mask], model.points[parname][~mask],
                                  color='k', marker='x', ls='')
                    model.point_lines[parname] = [l1, l2, l3]
                    if parname == 'mu':
                        l4, = self.axis_2d.plot(model.x_binned[mask], model.points[parname][mask],
                                                color=model.color, marker='o', ls='', alpha=0.3, picker=True, pickradius=10)
                        model.point_lines[parname].append(l4)
                    # -- Plot fit to points:
                    if len(model.fit['mu']) > 0:
                        lf, = ax.plot(model.x, model.fit[parname],
                                      color=model.color, ls='--', lw=1.0)
                        model.fit_lines[parname] = lf
                    if not model.fixed:
                        ax.set_ylabel(r"$\%s$" % parname)
                    if not ax.is_last_row():
                        ax.set_xticklabels("")
        self.canvas_points.figure.tight_layout()
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
                    model.model2d[:, num] = P_i
                model.model2d /= np.sum(model.model2d, axis=0)

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

                lower, upper = model.get_range()
                dlow = np.abs(model.cen - lower)
                dhigh = np.abs(model.cen - upper)
                for num, pars in enumerate(pars_table):
                    P_i = model_function(model.y, *pars)
                    if model.model_type != 'tophat':
                        il = int(pars[0] - dlow)
                        ih = int(pars[0] + dhigh)
                        P_i[:il] = 0.
                        P_i[ih:] = 0.
                    model.model2d[:, num] = P_i
                model.model2d /= np.sum(model.model2d, axis=0)

        if plot is True:
            self.plot_trace_2d()

    def get_trace_orders(self):
        c_order = int(self.c_order_edit.text())
        w_order = int(self.w_order_edit.text())
        return (c_order, w_order)

    def pick_points(self, event):
        artist = event.artist
        if isinstance(artist, matplotlib.lines.Line2D):
            # -- Can maybe simplify using:
            # line = event.artist
            # xdata, ydata = line.get_data()
            # ind = event.ind
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

        data1d_list = []
        for num, model in enumerate(self.trace_models):
            P = model.model2d
            bg2d = self.background.model2d
            img2d = self.image2d.data - bg2d
            V = self.image2d.error**2
            M = np.ones_like(img2d)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
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
        # model_type = self.model_chooser.currentText()
        if listItem.checkState() == 2:
            # Active:
            self.trace_models[index].activate()
            self.trace_models[index].set_visible(True)
            # if model_type.lower() == 'tophat':
            #     self.trace_models[index].set_visible(True)
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
        basename = './' + parent.image2d.header['OBJECT'] + '_1d.fits'
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
        btn_fits.setChecked(True)
        self.format_group.addButton(btn_fits)
        self.format_group.setId(btn_fits, 0)
        btn_fits.toggled.connect(self.set_fits)

        btn_fits_table = QtWidgets.QRadioButton("FITS Table")
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
            saved, msg = save_fits_spectrum(fname, wl, flux, err, hdr, bg, aper=model2d)
        elif file_format == 1:
            saved, msg = save_fitstable_spectrum(fname, wl, flux, err, hdr, bg, aper=model2d)
        elif file_format == 2:
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
        self.lower_editor = QtWidgets.QLineEdit("%i" % lower)
        self.lower_editor.setValidator(QtGui.QIntValidator(0, 1000))
        self.upper_editor = QtWidgets.QLineEdit("%i" % upper)
        self.upper_editor.setValidator(QtGui.QIntValidator(0, 1000))
        self.cen_editor = QtWidgets.QLineEdit("%i" % cen)
        self.cen_editor.setValidator(QtGui.QIntValidator(0, 10000))


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

        new_lower = int(self.lower_editor.text())
        new_upper = int(self.upper_editor.text())
        new_centroid = int(self.cen_editor.text())
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
            self.fwhm = get_FWHM(profile)
        elif new_model_type == 'gaussian':
            mu = self.trace_model.fit['mu']
            sigma = self.trace_model.fit['sigma']
            profile = NN_gaussian(self.trace_model.y, np.median(mu), np.median(sigma), 0.)
            self.fwhm = get_FWHM(profile)
        else:
            SPSF = self.parent.axis_spsf.lines[0].get_ydata()
            SPSF -= np.median(SPSF)
            SPSF[SPSF < 0] = 0.
            self.fwhm = get_FWHM(SPSF)
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
