import os
import numpy as np
from astropy.io import fits
import astropy.units as u

from pynot.viewer.spectrum import Spectrum
from pynot.viewer.targets import Target

    
class GenericFileContainer:
    def __init__(self, filelist):
        self.filelist = filelist
        self.view = []
        for f in self.filelist:
            try:
                name = fits.getval(f, 'OBJECT')
            except Exception:
                name = f
            self.view.append(os.path.basename(name))

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        spec = Spectrum.read(self.filelist[index])
        return Target(name=self.view[index], spectra=[spec])


class QMEC:
    def __init__(self, filename, view):
        self.filename = filename
        self.view = view

    def __len__(self):
        return len(self.view)

    def __getitem__(self, index):
        return self.get_target(index)

    def get_target(self, index):
        target = Target(name=self.view[index])
        with fits.open(self.filename) as hdu:
            data = hdu['SPECTAB'].data[index]
            hdr = hdu['SPECTAB'].header

            flux = data['FLUX'].flatten()
            try:
                ivar = data['FLUX_IVAR'].flatten()
                with np.errstate(divide='ignore', invalid='ignore'):
                    error = 1 / np.sqrt(ivar)
            except Exception:
                error = data['ERR_FLUX'].flatten()
            npix = len(flux)
            wavelength = np.arange(npix) * hdr['1CDLT1'] + hdr['1CRVL1']
            wavelength *= u.Unit(hdr['1CUNI1'])
            flux_unit = u.Unit(hdr['1CUNI2'])
            spectrum = Spectrum(wavelength=wavelength,
                                flux=flux*flux_unit,
                                error=error*flux_unit,
                                name=f"[{index}]",
                                filename=self.filename+f"[{index}]",
                                meta=dict(hdr))
            target.add_spectrum(spectrum)
            return target


    @staticmethod
    def read(filename):
        with fits.open(filename) as hdu:
            view = hdu['FIBMETATAB'].data['OBJECT']
        return QMEC(filename, view)
