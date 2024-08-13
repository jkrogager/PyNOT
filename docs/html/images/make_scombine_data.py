import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
import spex

efficiency = fits.getdata('/Users/krogager/Projects/4MOST/py4most/etc/data/efficiency_4most.fits', 2)
response = efficiency['LOW']
response = response / np.mean(response)
sky = fits.getdata('/Users/krogager/Projects/4MOST/py4most/etc/data/sky_model_dark.fits', 2)
wl = sky['WAVE']
sky_flux = gaussian_filter1d(sky['LOW'], 4)

Ny = 200
obj_pos = 149
fname = 'obj1.fits'
seeing = 6.0
Npix = len(sky_flux)
sky2d = np.resize(sky_flux, (Ny, Npix))

qso = spex.QuasarTemplate('selsing')
qso.wavelength *= (3.223 + 1)
obj_flux = np.interp(wl, qso.wavelength, qso.flux)
obj_flux = obj_flux / np.mean(obj_flux) * 1.5 * response

y = np.arange(Ny)
SPSF = np.exp(-0.5*(y - obj_pos)**2 / (seeing/2.35)**2)
obj2d = np.outer(SPSF, obj_flux)

flux2d = obj2d + sky2d
flux2d *= 10
err2d = np.sqrt(flux2d + 1)
flux2d_noise = flux2d + err2d*np.random.normal(0, 1, err2d.shape)
flux2d_noise -= sky2d*10

plt.subplot(2,1,1)
plt.imshow(flux2d, aspect='auto')
plt.subplot(2,1,2)
plt.imshow(flux2d_noise, aspect='auto')

hdr = fits.Header()
hdr['EXPTIME'] = 900
hdr['CRVAL1'] = wl[0]
hdr['CRPIX1'] = 1
hdr['CD1_1'] = np.mean(np.diff(wl))
hdr['CD2_2'] = 0.2
hdr['CD1_2'] = 0
hdr['CD2_1'] = 0
hdr['CRVAL2'] = 0.
hdr['CRPIX2'] = 1
hdr['BUNIT'] = 'adu'
hdr['CUNIT1'] = 'Angstrom'
hdr['CUNIT2'] = 'arcsec'
prim = fits.PrimaryHDU(header=hdr)
img = fits.ImageHDU(data=flux2d_noise, header=hdr, name='FLUX')
err = fits.ImageHDU(data=err2d, header=hdr, name='ERR')
hdu_list = fits.HDUList()
hdu_list.append(img)
hdu_list.append(err)
hdu_list.writeto(fname, overwrite=True)

