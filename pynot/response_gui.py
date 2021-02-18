#!/usr/bin/env python3
"""
    PyNOT -- Response GUI

Graphical interface to determine instrumental response

"""

__author__ = "Jens-Kristian Krogager"
__email__ = "krogager.jk@gmail.com"
__credits__ = ["Jens-Kristian Krogager"]


import os
import re
import sys
import numpy as np
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.interpolate import UnivariateSpline
from numpy.polynomial import Chebyshev
from astropy.io import fits
from PyQt5 import QtCore, QtGui, QtWidgets
import warnings

from pynot import alfosc
from pynot.functions import get_version_number


__version__ = get_version_number()


def run_gui(input_fname, output_fname, app=None, order=3, smoothing=0.02):
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    gui = ResponseGUI(input_fname, output_fname=output_fname, locked=True, order=order, smoothing=smoothing)
    gui.show()
    app.exit(app.exec_())
    # Get Response Array:
    response = gui.response
    del gui
    return response



class Spectrum(object):
    def __init__(self, wl=None, data=None, error=None, header={}, wl_unit='', flux_unit=''):
        self.wl = wl
        self.data = data
        self.error = error
        self.header = header
        self.wl_unit = wl_unit
        self.flux_unit = flux_unit



class ResponseGUI(QtWidgets.QMainWindow):
    def __init__(self, fname=None, output_fname='', star_name='', order=3, smoothing=0.02, parent=None, locked=False, **kwargs):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setWindowTitle('PyNOT: Response')
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        # Set attributes:
        self.spectrum = None
        self.response = None
        self.filename = fname
        self.output_fname = output_fname
        self.first_time_open = True
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
        self.all_names = sorted([name.upper() for name in alfosc.standard_stars] + [''])
        self.star_chooser.addItems(self.all_names)
        self.star_chooser.setCurrentText(star_name)
        self.star_chooser.currentTextChanged.connect(self.set_star)

        self.exptime_edit = QtWidgets.QLineEdit("")
        self.exptime_edit.setValidator(QtGui.QDoubleValidator())
        self.airmass_edit = QtWidgets.QLineEdit("")
        self.airmass_edit.setValidator(QtGui.QDoubleValidator())

        self.order_edit = QtWidgets.QLineEdit("%i" % order)
        self.order_edit.setValidator(QtGui.QIntValidator(1, 5))
        self.order_edit.returnPressed.connect(self.fit_response)

        self.smooth_edit = QtWidgets.QLineEdit("%.2f" % smoothing)
        self.smooth_edit.setValidator(QtGui.QDoubleValidator())
        self.smooth_edit.returnPressed.connect(self.fit_response)

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
        row_orders.addWidget(QtWidgets.QLabel("Spline Degree:"))
        row_orders.addWidget(self.order_edit)
        row_orders.addStretch(1)
        right_panel.addLayout(row_orders)

        row_smooth = QtWidgets.QHBoxLayout()
        row_smooth.addWidget(QtWidgets.QLabel("Smoothing factor:"))
        row_smooth.addWidget(self.smooth_edit)
        row_smooth.addStretch(1)
        right_panel.addLayout(row_smooth)

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

        self.create_menu()

        # -- Set Data:
        if fname:
            self.load_spectrum(fname)


    def done(self):
        success = self.save_response(self.output_fname)
        if success:
            self.close()

    def save_response(self, fname=''):
        if self.response is None:
            msg = "No response function has been fitted. Nothing to save..."
            QtWidgets.QMessageBox.critical(None, "Save Error", msg)
            return False

        if not fname:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            basename = os.path.join(current_dir, "response_%s.fits" % (self.spectrum.header['OBJECT']))
            filters = "FITS Files (*.fits *.fit)"
            fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Response Function', basename, filters)

        if fname:
            hdu = fits.HDUList()
            prim_hdr = fits.Header()
            prim_hdr['AUTHOR'] = 'PyNOT version %s' % __version__
            prim_hdr['OBJECT'] = self.spectrum.header['OBJECT']
            prim_hdr['DATE-OBS'] = self.spectrum.header['DATE-OBS']
            prim_hdr['EXPTIME'] = self.spectrum.header['EXPTIME']
            prim_hdr['AIRMASS'] = self.spectrum.header['AIRMASS']
            prim_hdr['ALGRNM'] = self.spectrum.header['ALGRNM']
            prim_hdr['ALAPRTNM'] = self.spectrum.header['ALAPRTNM']
            prim_hdr['RA'] = self.spectrum.header['RA']
            prim_hdr['DEC'] = self.spectrum.header['DEC']
            prim_hdr['COMMENT'] = 'PyNOT response function'
            prim = fits.PrimaryHDU(header=prim_hdr)
            hdu.append(prim)
            col_wl = fits.Column(name='WAVE', array=self.spectrum.wl, format='D', unit=self.spectrum.wl_unit)
            col_resp = fits.Column(name='RESPONSE', array=self.response, format='D', unit='-2.5*log(erg/s/cm2/A)')
            tab = fits.BinTableHDU.from_columns([col_wl, col_resp])
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
        self.canvas_points.draw()


    def load_spectrum(self, fname=''):
        if fname is False:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            filters = "FITS files (*.fits | *.fit)"
            fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open 1D Spectrum', current_dir, filters)
            fname = str(fname)
            if self.first_time_open:
                print(" [INFO] - Don't worry about the warning above. It's an OS warning that can not be suppressed.")
                print("          Everything works as it should")
                self.first_time_open = False

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
        else:
            self.exptime_edit.setEnabled(True)
        if 'AIRMASS' in hdr:
            self.airmass_edit.setText("%.1f" % hdr['AIRMASS'])
            self.airmass_edit.setEnabled(False)
        else:
            self.airmass_edit.setEnabled(True)

        if 'TCSTGT' in hdr:
            TCSname = hdr['TCSTGT']
            TCSname = alfosc.lookup_std_star(TCSname)
            if TCSname:
                star_name = alfosc.standard_star_names[TCSname]
                self.star_chooser.setCurrentText(star_name.upper())
        elif 'OBJECT' in hdr:
            object_name = hdr['OBJECT']
            if object_name.upper() in self.all_names:
                self.star_chooser.setCurrentText(object_name.upper())

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
            WarningDialog(self, "No spectrum loaded!", "No spectral data has been loaded.")
            return
        wl = self.spectrum.wl
        flux = self.spectrum.data
        flux_bins = list()
        for wl_ref, mag_ref, bandwidth in self.ref_tab:
            l1 = wl_ref - bandwidth/2
            l2 = wl_ref + bandwidth/2
            band = (wl >= l1) * (wl <= l2)
            if np.sum(band) > 3:
                f0 = np.nanmean(flux[band])
                if f0 < 0:
                    f0 = np.nan
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
        smoothing = float(self.smooth_edit.text())
        mask = self.mask
        # resp_fit = Chebyshev.fit(self.wl_bins[mask], self.resp_bins[mask], order, domain=[wl.min(), wl.max()])
        resp_fit = UnivariateSpline(self.wl_bins[mask], self.resp_bins[mask], k=order, s=smoothing)
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

        save_1d_action = QtWidgets.QAction("Save", self)
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
    parser.add_argument("filename", type=str,
                        help="Input 1D spectrum of Spectroscopic Standard Star")
    parser.add_argument("--output", type=str, default='',
                        help="Output filename of response function [FITS Table]")
    parser.add_argument("--locked", "-l", action='store_true',
                        help="Lock interface [for pipeline workflow]")
    args = parser.parse_args()

    # Launch App:
    app = QtWidgets.QApplication(sys.argv)
    gui = ResponseGUI(args.filename, output_fname=args.output, locked=args.locked)
    gui.show()
    app.exit(app.exec_())
