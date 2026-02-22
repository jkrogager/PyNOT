from astropy.table import QTable
from dataclasses import dataclass
import astropy.units as u
import numpy as np
import spectres
import pyqtgraph as pg

from pynot.viewer.models import ModelSpectrum, Template

import warnings
warnings.simplefilter('error', RuntimeWarning)

@dataclass
class Spectrum:
    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray = None
    noss: np.ndarray = None
    R: float = None
    filename: str = ''
    meta: dict = None
    plot_line: pg.PlotItem = None
    models: list[ModelSpectrum] = None
    templates: list[Template] = None

    def __post_init__(self):
        if self.models is None:
            self.models = []

        if self.templates is None:
            self.templates = []

    @staticmethod
    def read(filename):
        data = QTable.read(filename)
        try:
            error = data['ERR_FLUX'].flatten()
        except KeyError:
            ivar = data['FLUX_IVAR'].flatten()
            invalid = np.isfinite(ivar) & (ivar > 0)
            ivar[invalid] = np.nan
            with np.errstate(divide='ignore', invalid='ignore'):
                error = 1 / np.sqrt(ivar)
        return Spectrum(data['WAVE'].flatten(),
                        data['FLUX'].flatten(),
                        error,
                        filename=filename,
                        meta=data.meta)


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
