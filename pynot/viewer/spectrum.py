from itertools import cycle
from astropy.table import QTable
from dataclasses import dataclass
import astropy.units as u
import numpy as np
from scipy.interpolate import UnivariateSpline as spline
import spectres
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
import pyqtgraph as pg
import os
import logging
from typing import Any

import warnings
warnings.simplefilter('error', RuntimeWarning)

from pynot.txtio import load_ascii_spectrum
from pynot.fitsio import load_fits_spectrum, detect_4most_MEC
from pynot.viewer.dust import SMCDustModel


def gauss(x, mu, sigma):
    return np.exp(-0.5*(x - mu)**2 / sigma**2)


functors = [np.abs, np.sin, np.cos, np.tan, np.cosh, np.sinh, np.tanh,
            np.arccos, np.arcsin, np.arctan, np.arctan2, np.log, np.log10,
            np.sqrt, np.min, np.max, np.ceil, np.floor, np.power, np.exp, np.poly1d,
            gauss]
numpy_functions = {f.__name__: f for f in functors}

function_name_cycle = cycle(['f(x)', 'g(x)', 'h(x)', 'p(x)', 'q(x)'])


@dataclass
class Spectrum:
    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray = None
    noss: np.ndarray = None
    R: float = None
    name: str = ''
    filename: str = ''
    meta: dict = None
    plot_line: pg.PlotItem = None
    error_line: pg.PlotItem = None

    def __post_init__(self):
        if not hasattr(self.wavelength, 'unit') or self.wavelength.unit is None:
            self.wavelength *= u.Angstrom
            logging.warning("No wavelength units given. Assuming Angstrom")
        if str(self.wavelength.unit).lower() == 'angstroms':
            self.wavelength = self.wavelength.value * u.Angstrom
        self.wavelength = self.wavelength.to('Angstrom')

        if not hasattr(self.flux, 'unit'):
            self.flux *= u.Unit("")
            logging.warning("No flux units given. Assuming unitless")
        if 'counts' in str(self.flux.unit).lower():
            flux_unit = str(self.flux.unit).lower().replace('counts', 'count')
            self.flux = self.flux.value * u.Unit(flux_unit)

        if self.meta is None:
            self.meta = {}

        self.parent = None

    def set_parent(self, parent):
        """`parent` must be of type `Target`"""
        self.parent = parent

    def plot(self, plot_graph, color='black', ls=Qt.PenStyle.SolidLine):
        if self.parent is None:
            logging.error("Attempted to plot without a parent `Target`.")
            return

        ydata, yerr = self.apply_smoothing()
        color = QtGui.QColor(color)
        pen = pg.mkPen(color=color, style=ls)
        line = plot_graph.plot(self.wavelength, ydata, pen=pen,
                               name=f"{self.parent.name} {self.name}")
        plot_graph.setLabel("bottom", f"Spectral Axis  [{self.wavelength.unit}]")
        self.plot_line = line
        if yerr is not None:
            err_color = QtGui.QColor('red')
            h, s, l, a = err_color.getHsl()
            err_color.setHsl(h, int(s*0.8), l, 200)
            err_pen = pg.mkPen(color=err_color, style=ls)
            error_line = plot_graph.plot(self.wavelength, yerr, pen=err_pen,
                                         name=f"{self.parent.name} {self.name} 1σ")
            self.error_line = error_line

    def apply_smoothing(self):
        smooth_factor = self.parent.smooth_factor
        if smooth_factor > 1:
            kernel = np.ones(smooth_factor) / smooth_factor
            smooth_flux = np.convolve(self.flux, kernel, mode='same')
            if self.error is not None:
                smooth_error = np.convolve(self.error, kernel, mode='same') / np.sqrt(smooth_factor)
            else:
                smooth_error = None
        else:
            smooth_flux = self.flux
            smooth_error = self.error
        return smooth_flux, smooth_error

    def update_plot_data(self):
        if self.plot_line is None:
            return

        ydata, yerr = self.apply_smoothing()
        self.plot_line.setData(self.wavelength, ydata)
        if self.error_line is not None:
            self.error_line.setData(self.wavelength, yerr)

    @staticmethod
    def read(filename):
        is_fits_file = filename.endswith('.fits') | filename.endswith('.fit') | filename.endswith('.fits.gz')
        if is_fits_file:
            try:
                x, y, err, mask, hdr, output_msg = load_fits_spectrum(filename)
                if output_msg:
                    logging.info(output_msg)

                return Spectrum(x, y, err, filename=filename, meta=hdr)

            except Exception as e:
                logging.exception(e)
                logging.error(f"Failed to load spectrum: {filename}")
                if detect_4most_MEC(filename):
                    msg = "The file seems to be a 4MOST container. "
                    msg += "Try loading with --container or -c"
                    logging.info(msg)
                    print(f"\n\n{msg}\n\n")

        else:
            try:
                table_items = load_ascii_spectrum(filename)
                x, y, err, mask, sky, output_msg = table_items
                if output_msg:
                    logging.info(output_msg)
                return Spectrum(x, y, err, filename=filename)
            except Exception as e:
                logging.exception(e)
                logging.error(f"Failed to load spectrum: {filename}")


class Template:
    def __init__(self, wavelength, flux, name='', filename='', parent=None):
        """ The `parent` is of type `Target` """
        super().__init__()
        self.wavelength = wavelength
        self.flux = flux
        self.interp = 'cubic'
        self.error = None
        self.noss = None
        self.R = None
        self.name = name
        self.filename = filename
        self.meta = {}
        self.plot_line = None
        self.error_line = None
        self.parent = parent

        if np.any(np.diff(self.wavelength) < 0):
            isort = np.argsort(self.wavelength)
            self.wavelength = self.wavelength[isort]
            self.flux = self.flux[isort]
            logging.info("Template x-points must be increasing. I sorted the array for you!")

        if not hasattr(wavelength, 'unit'):
            self.wavelength *= u.Angstrom
            logging.warning("No wavelength units given. Assuming Angstrom")

        # Initiate template model parameters:
        self.z = 0.
        self.C1 = 0. * self.flux
        self.C2 = 1. * self.flux
        self.scaled_flux = self.C1 + self.C2*self.flux
        self.Av = 0.
        self.dust_model = SMCDustModel()

    def set_parent(self, parent):
        self.parent = parent

    # def __call__(self, new_x):
    #     if self.interp == 'linear':
    #         return np.interp(new_x, self.wavelength*(self.z+1), self.scaled_flux)
    #     elif self.interp == 'cubic':
    #         return spline(self.wavelength*(self.z+1), self.scaled_flux, s=0, k=3)(new_x)
    #     else:
    #         return spline(self.wavelength*(self.z+1), self.scaled_flux, s=0, k=2)(new_x)

    def plot(self, plot_graph, color='blue', ls=Qt.PenStyle.DashLine):
        if self.parent is None:
            logging.error("Attempted to plot without a parent `Target`.")
            return
        ydata = self.apply_smoothing()
        pen = pg.mkPen(color=color, style=ls, width=2)
        line = plot_graph.plot(self.wavelength, ydata, pen=pen,
                               name=f"{self.parent.name} {self.name}")
        self.plot_line = line

    def apply_smoothing(self):
        smooth_factor = self.parent.smooth_factor
        if smooth_factor > 1:
            kernel = np.ones(smooth_factor) / smooth_factor
            smooth_flux = np.convolve(self.scaled_flux, kernel, mode='same')
        else:
            smooth_flux = self.scaled_flux
        return smooth_flux

    def update_plot_data(self):
        if self.plot_line is None:
            return

        ydata = self.apply_smoothing() * self.dust_model(self.wavelength.value, self.Av)
        self.plot_line.setData(self.wavelength*(self.z+1), ydata)

    def scale_flux(self, c1=0, c2=1):
        if isinstance(c1, (float, int)):
            self.C1 = c1

        if isinstance(c2, (float, int)):
            self.C2 = c2

        self.scaled_flux = self.C1 + self.C2*self.flux
        self.update_plot_data()

    def set_redshift(self, z):
        if not isinstance(z, (float, u.Quantity)):
            return
        self.z = z
        self.update_plot_data()
        

    @staticmethod
    def read(filename: str):
        tname = os.path.basename(filename)
        is_fits_file = filename.endswith('.fits') | filename.endswith('.fit') | filename.endswith('.fits.gz')
        if is_fits_file:
            x, y, err, mask, hdr = load_fits_spectrum(filename)
            return Template(x, y, name=tname)

        try:
            temp = np.genfromtxt(filename)
            x = temp[:, 0]
            y = temp[:, 1]
        except Exception:
            table_items = load_ascii_spectrum(filename)
            x, y, err, mask, sky, output_msg = table_items

        return Template(x, y, filename=filename)


@dataclass
class ModelSpectrum:
    expression: str
    plot_line: pg.PlotItem = None
    error_line: pg.PlotItem = None
    name: str = None
    xmin: float = 3700
    xmax: float = 9500
    dx: float = 1
    log: bool = False
    parent: Any = None
    spectrum: Spectrum = None
    wavelength: float = 1. * u.Angstrom

    def __post_init__(self):
        if self.name is None:
            self.name = next(function_name_cycle)

    def set_parent(self, parent):
        self.parent = parent

    def set_spectrum(self, spectrum):
        self.spectrum = spectrum

    @property
    def x(self):
        if self.spectrum is not None:
            return self.spectrum.wavelength.value

        if self.xmin and self.xmax and self.dx:
            pass
        else:
            logging.error("Must specify x-axis: xmin, xmax, dx and log")
            return np.zeros(1)

        if self.log:
            x = np.arange(np.log10(self.xmin),
                          np.log10(self.xmax),
                          self.dx)
        else:
            x = np.arange(self.xmin, self.xmax, self.dx)
        return x

    def plot(self, plot_graph, color='red', ls=Qt.PenStyle.SolidLine):
        if self.parent is None:
            logging.error("Attempted to plot without a parent `Target`.")
            return
        ydata = self()
        pen = pg.mkPen(color=color, style=ls, width=2)
        line = plot_graph.plot(self.x, ydata, pen=pen,
                               name=f"{self.parent.name} {self.name}")
        self.plot_line = line

    def update_plot_data(self):
        if self.plot_line is None:
            return

        ydata = self()
        self.plot_line.setData(self.x, ydata)

    def __call__(self, x=None):
        if x is None:
            x = self.x

        variables = {'x': x}
        variables.update(numpy_functions)
        logging.info(f"Evaluating model expression: {self.expression}")
        try:
            ydata = eval(self.expression, variables)
        except (SyntaxError, NameError):
            ydata = np.zeros_like(x)
        return ydata


def join_spectra(spec_group, scale=None):
    """
    Join together the three arms of LRS spectra with automatic scaling by the overlap
    of the blue/green and green/red arms. By default, the green arm is used as reference
    and the blue and red arms are scaled to match the flux level in the green arm.


    Parameters
    ----------
    spec_group : Dict[str, Spectrum]
        Dictionary with keys each of which is an instance of :class:`Spectrum`.

    Returns
    -------
    :class:`Spectrum`
        The joined spectrum.

    """
    all_wave = [spec.wavelength for spec in spec_group.values()]
    wave_units = [spec.wavelength.unit for spec in spec_group.values()]
    if len(set(wave_units)) != 1:
        try:
            for spec in spec_group.values():
                spec.wavelength.to(wave_units[0])
        except Exception:
            raise ValueError("Spectra do not have the same wavelength units!")

    min_wl = np.nanmin(all_wave)
    max_wl = np.nanmax(all_wave)
    dl = np.nanmedian([np.diff(wl) for wl in all_wave])
    wavelength = np.arange(min_wl, max_wl, dl) * u.AA
    flux = []
    weights = []
    flux_units = [spec.flux.unit for spec in spec_group.values()]
    if len(set(flux_units)) != 1:
        raise ValueError("Spectra do not have same flux units!")

    for spec in spec_group.values():
        flux, error = spectres.spectres(wavelength.value,
                                        spec.wavelength.value, spec.flux.value, spec.error.value,
                                        fill=0)
        flux.append(flux)
        weights.append(1/error**2)

    joined_flux = np.nansum(flux * weights, axis=0) / np.nansum(weights, axis=0)
    joined_error = 1 / np.sqrt(np.nansum(weights, axis=0))

    # So we add the units back here:
    flux_unit = flux_units[0]
    joined_flux *= flux_unit
    joined_error *= flux_unit

    joined_meta = {}
    for arm, spec in spec_group.items():
        joined_meta.update(spec.meta)
    return Spectrum(wavelength, joined_flux, joined_error, meta=joined_meta)
