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
        self.all_names = ["%s : %s" % item for item in alfosc.standard_star_names.items()] + ['']
        self.star_chooser.addItems(self.all_names)
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


        # -- Plotting
        self.figure_points = Figure(figsize=(8, 6))
        self.axes = self.figure_points.subplots(2, 1)
        self.canvas_points = FigureCanvas(self.figure_points)
        self.canvas_points.mpl_connect('pick_event', self.pick_points)
        self.figp_mpl_toolbar = NavigationToolbar(self.canvas_points, self)
        self.figp_mpl_toolbar.setFixedHeight(20)
        self.data_line = None
        self.fit_line = None
        self.data_points = None
        self.response_points = None

        # == TOP MENU BAR:
        self.save_btn = QtWidgets.QPushButton("Save")
        self.save_btn.clicked.connect(self.save_response)
        self.load_btn = QtWidgets.QPushButton("Load")
        self.load_btn.clicked.connect(self.load_spectrum)
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
        main_layout = QtWidgets.QVBoxLayout(self._main)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        top_menubar = QtWidgets.QHBoxLayout()
        top_menubar.addWidget(self.close_btn)
        top_menubar.addWidget(self.save_btn)
        top_menubar.addWidget(self.load_btn)
        top_menubar.addStretch(1)

        central_layout = QtWidgets.QHBoxLayout()

        main_layout.addLayout(top_menubar)
        main_layout.addLayout(central_layout)

        # TabWidget Layout:
        fig_layout = QtWidgets.QVBoxLayout()
        fig_layout.addWidget(self.canvas_points, 1)
        fig_layout.addWidget(self.figp_mpl_toolbar)
        central_layout.addLayout(fig_layout)

        # Right Panel Layout:
        right_panel = QtWidgets.QVBoxLayout()
        right_panel.setContentsMargins(50, 0, 50, 10)

        separatorLine = QtWidgets.QFrame()
        separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        separatorLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        separatorLine.setMinimumSize(3, 20)
        right_panel.addWidget(separatorLine)

        row_model = QtWidgets.QHBoxLayout()
        row_model.addWidget(QtWidgets.QLabel("Star Name: "))
        row_model.addWidget(self.star_chooser)
        right_panel.addLayout(row_model)

        separatorLine = QtWidgets.QFrame()
        separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        separatorLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        separatorLine.setMinimumSize(3, 20)
        right_panel.addWidget(separatorLine)

        row_exptime = QtWidgets.QHBoxLayout()
        row_exptime.addWidget(QtWidgets.QLabel("Exposure Time: "))
        row_exptime.addWidget(self.exptime_edit)
        right_panel.addLayout(row_exptime)

        row_airmass = QtWidgets.QHBoxLayout()
        row_airmass.addWidget(QtWidgets.QLabel("Airmass: "))
        row_airmass.addWidget(self.airmass_edit)
        right_panel.addLayout(row_airmass)

        separatorLine = QtWidgets.QFrame()
        separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        separatorLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        separatorLine.setMinimumSize(3, 20)
        right_panel.addWidget(separatorLine)

        row_orders = QtWidgets.QHBoxLayout()
        row_orders.addWidget(QtWidgets.QLabel("Polynomial Order:"))
        row_orders.addWidget(self.order_edit)
        row_orders.addStretch(1)
        right_panel.addLayout(row_orders)

        separatorLine = QtWidgets.QFrame()
        separatorLine.setFrameShape(QtWidgets.QFrame.HLine)
        separatorLine.setFrameShadow(QtWidgets.QFrame.Sunken)
        separatorLine.setMinimumSize(3, 20)
        right_panel.addWidget(separatorLine)

        row_fit = QtWidgets.QHBoxLayout()
        row_fit.addStretch(1)
        row_fit.addWidget(self.fit_btn)
        row_fit.addStretch(1)
        right_panel.addLayout(row_fit)

        right_panel.addStretch(1)
        central_layout.addLayout(right_panel)

        self.canvas_points.setFocus()

        # self.create_menu()

        # -- Set Data:
        if fname:
            self.load_spectrum(fname)


    def done(self):
        success = True
        if success:
            self.close()

    def save_response(self, fname=''):
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

        self.filename = fname
        hdr = fits.getheader(fname, 1)
        table = fits.getdata(fname, 1)
        self.spectrum = Spectrum(wl=table['WAVE'], data=table['FLUX'], header=hdr,
                                 wl_unit=table.columns['WAVE'].unit,
                                 flux_unit=table.columns['FLUX'].unit)
        if 'EXPTIME' in hdr:
            self.exptime_edit.setText("%.1f" % hdr['EXPTIME'])
            self.exptime_edit.setEnabled(False)
        if 'AIRMASS' in hdr:
            self.airmass_edit.setText("%.1f" % hdr['AIRMASS'])
            self.airmass_edit.setEnabled(False)
        if 'OBJECT' in hdr:
            for entry_name in self.all_names:
                if hdr['OBJECT'] in entry_name:
                    self.star_chooser.setCurrentText(entry_name)
                    break

        if self.ref_tab is not None:
            self.calculate_flux_in_bins()
            self.calculate_response_bins()
        self.update_plot()


    def set_star(self, text):
        entry_name = str(text).lower()
        star_name = entry_name.split(':')[1].strip()
        self.ref_tab = np.loadtxt(alfosc.path+'/calib/std/%s.dat' % star_name)
        self.calculate_flux_in_bins()
        self.calculate_response_bins()
        self.update_plot()

    def calculate_flux_in_bins(self):
        if self.spectrum is None:
            WarningDialog(self, "No spectrum loaded!", "No spectral data has been loaded.")
            return
        wl = self.spectrum.wl
        flux = self.spectrum.data
        flux_bins = list()
        for wl_ref, mag_ref, bandwidth in self.ref_tab:
            l1 = wl_ref - bandwidth/2
            l2 = wl_ref + bandwidth/2
            band = (wl >= l1) * (wl <= l2)
            if np.sum(band) > 0:
                f0 = np.nanmean(flux[band])
            else:
                f0 = np.nan
            # flux_bins.append(f0 / bandwidth)
            flux_bins.append(f0)

        self.flux_bins = np.array(flux_bins)
        mask = ~np.isnan(self.flux_bins)
        self.wl_bins = self.ref_tab[:, 0][mask]
        self.mag_bins = self.ref_tab[:, 1][mask]
        self.dw = self.ref_tab[:, 2][mask]
        self.flux_bins = self.flux_bins[mask]
        self.mask = np.ones_like(self.flux_bins, dtype=bool)

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
        extinction = np.interp(self.wl_bins, self.ext_wl, self.ext)
        cdelt = np.diff(self.spectrum.wl)[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.resp_bins = 2.5*np.log10(self.flux_bins / (exptime * cdelt * ref_flux)) + airmass*extinction

    def update_plot(self):
        if self.spectrum is None:
            WarningDialog(self, "No spectrum loaded!", "No spectral data has been loaded.")
            return
        if 'OBJECT' in self.spectrum.header:
            object_name = self.spectrum.header['OBJECT']
        else:
            object_name = ''

        if self.data_line is None:
            self.data_line, = self.axes[0].plot(self.spectrum.wl, self.spectrum.data,
                                                color='k', alpha=0.9, lw=0.8, label=object_name)
        else:
            self.data_line.set_data(self.spectrum.wl, self.spectrum.data)
            self.data_line.set_label(object_name)
        xunit = self.spectrum.wl_unit
        yunit = self.spectrum.flux_unit
        self.axes[1].set_xlabel("Wavelength  [%s]" % xunit, fontsize=11)
        self.axes[0].set_ylabel("Flux  [%s]" % yunit, fontsize=11)
        self.axes[1].set_ylabel("Response", fontsize=11)
        self.axes[0].legend()
        self.figure_points.tight_layout()
        self.update_points()

    def update_points(self):
        if self.resp_bins is not None:
            mask = self.mask
            if self.response_points is None:
                self.response_points, = self.axes[1].plot(self.wl_bins[mask], self.resp_bins[mask], 'bo', picker=True, pickradius=5)
                self.masked_response, = self.axes[1].plot(self.wl_bins[~mask], self.resp_bins[~mask], 'rx', picker=True, pickradius=5)
            else:
                self.response_points.set_data(self.wl_bins[mask], self.resp_bins[mask])
                self.masked_response.set_data(self.wl_bins[~mask], self.resp_bins[~mask])

            if self.data_points is None:
                self.data_points, = self.axes[0].plot(self.wl_bins[mask], self.flux_bins[mask], 'bo', picker=True, pickradius=5)
                self.masked_data, = self.axes[0].plot(self.wl_bins[~mask], self.flux_bins[~mask], 'rx', picker=True, pickradius=5)
            else:
                self.data_points.set_data(self.wl_bins[mask], self.flux_bins[mask])
                self.masked_data.set_data(self.wl_bins[~mask], self.flux_bins[~mask])

        if self.response is not None:
            if self.fit_line is None:
                self.fit_line, = self.axes[1].plot(self.spectrum.wl, self.response,
                                                   color='Crimson', lw=1.5, alpha=0.8)
            else:
                self.fit_line.set_data(self.spectrum.wl, self.response)
        self.canvas_points.draw()

    def fit_response(self):
        if self.resp_bins is None:
            WarningDialog(self, "No response data!", "No response data to fit.\nMake sure to load a spectrum and reference star data.")
            return
        wl = self.spectrum.wl
        order = int(self.order_edit.text())
        mask = self.mask
        # Find better interpolation than Chebyshev. Maybe just a spline?
        resp_fit = Chebyshev.fit(self.wl_bins[mask], self.resp_bins[mask], order, domain=[wl.min(), wl.max()])
        self.response = resp_fit(wl)
        self.update_points()


    def pick_points(self, event):
        x0 = event.mouseevent.xdata
        y0 = event.mouseevent.ydata
        is_left_press = event.mouseevent.button == 1
        is_right_press = event.mouseevent.button == 3
        is_on = (event.artist is self.data_points) or (event.artist is self.response_points)
        is_off = (event.artist is self.masked_data) or (event.artist is self.masked_response)
        is_data = (event.artist is self.masked_data) or (event.artist is self.data_points)
        if is_data:
            xrange = self.wl_bins.max() - self.wl_bins.min()
            yrange = self.flux_bins.max() - self.flux_bins.min()
            dist = (self.wl_bins - x0)**2 / xrange**2 + (self.flux_bins - y0)**2 / yrange**2
        else:
            xrange = self.wl_bins.max() - self.wl_bins.min()
            yrange = self.resp_bins.max() - self.resp_bins.min()
            dist = (self.wl_bins - x0)**2 / xrange**2 + (self.resp_bins - y0)**2 / yrange**2
        index = np.argmin(dist)
        if is_on and is_left_press:
            self.mask[index] = ~self.mask[index]
        elif is_off and is_right_press:
            self.mask[index] = ~self.mask[index]
        else:
            return
        self.update_points()


    def create_menu(self):
        load_file_action = QtWidgets.QAction("Load Spectrum", self)
        load_file_action.setShortcut("ctrl+O")
        load_file_action.triggered.connect(self.load_spectrum)

        save_1d_action = QtWidgets.QAction("Save 1D Spectrum", self)
        save_1d_action.setShortcut("ctrl+S")
        save_1d_action.triggered.connect(self.save_response)

        view_hdr_action = QtWidgets.QAction("Display Header", self)
        view_hdr_action.setShortcut("ctrl+shift+H")
        view_hdr_action.triggered.connect(self.display_header)

        main_menu = self.menuBar()
        file_menu = main_menu.addMenu("File")
        file_menu.addAction(load_file_action)
        file_menu.addAction(save_1d_action)

        view_menu = main_menu.addMenu("View")
        view_menu.addAction(view_hdr_action)


    def display_header(self):
        if self.spectrum is not None:
            HeaderViewer(self.spectrum.header, parent=self)
        else:
            msg = "No Data Loaded"
            info = "Load a spectrum first"
            WarningDialog(self, msg, info)



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
    parser.add_argument("--locked", "-l", action='store_true',
                        help="Lock interface")
    args = parser.parse_args()

    # Launch App:
    app = QtWidgets.QApplication(sys.argv)
    gui = ResponseGUI(args.filename, locked=args.locked)
    gui.show()
    app.exec_()
