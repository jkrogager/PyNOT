from itertools import cycle
from astropy.table import Table
import numpy as np
from scipy.interpolate import UnivariateSpline as spline
from dataclasses import dataclass
import pyqtgraph as pg
import os
import logging

from pynot.txtio import load_ascii_spectrum
from pynot.fitsio import load_fits_spectrum


@dataclass
class Template:
    x: np.ndarray
    y: np.ndarray
    interp: str = 'cubic'
    plot_line: pg.PlotItem = None
    name: str = None

    def __post_init__(self):
        if np.any(np.diff(self.x) < 0):
            isort = np.argsort(self.x)
            self.x = self.x[isort]
            self.y = self.y[isort]
            logging.info("Template x-points must be increasing. I sorted the array for you!")

    def __call__(self, new_x):
        if self.interp == 'linear':
            return np.interp(new_x, self.x, self.y)
        elif self.interp == 'cubic':
            return spline(self.x, self.y, s=0, k=3)(new_x)
        else:
            return spline(self.x, self.y, s=0, k=2)(new_x)

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

        return Template(x, y, name=tname)
            
            
functors = [np.abs, np.sin, np.cos, np.tan, np.cosh, np.sinh, np.tanh,
            np.arccos, np.arcsin, np.arctan, np.arctan2, np.log, np.log10,
            np.sqrt, np.min, np.max, np.ceil, np.floor, np.power, np.exp, np.poly1d]
numpy_functions = {f.__name__: f for f in functors}

function_name_cycle = cycle(['f(x)', 'g(x)', 'h(x)', 'p(x)', 'q(x)'])


@dataclass
class ModelSpectrum:
    expression: str
    plot_line: pg.PlotItem = None
    name: str = None
    xmin: float = None
    xmax: float = None
    dx: float = None
    log: bool = False

    def __post_init__(self):
        if self.name is None:
            self.name = next(function_name_cycle)

    @property
    def x(self):
        if self.xmin and self.xmax and self.dx:
            pass
        else:
            raise AxisDefinitionError("Must specify x-axis: xmin, xmax, dx and log")

        if self.log:
            x = np.arange(np.log10(self.xmin),
                          np.log10(self.xmax),
                          self.dx)
        else:
            x = np.arange(self.xmin, self.xmax, self.dx)
        return x


    def __call__(self, x=None):
        if x is None:
            x = self.x

        variables = {'x': x}
        variables.update(numpy_functions)
        return eval(self.expression, variables)


class AxisDefinitionError(Exception):
    pass
