# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from astropy.io import fits

from astropy.table import Table


raw_fname = '/Users/krogager/Data/NOT/MALS/test_pynot/raw/ALzh010127.fits'
red_fname = '/Users/krogager/Data/NOT/MALS/test_pynot/MALS_1623+1239_zh010127/CRR_BGSUB2D_MALS_1623+1239.fits'
ext_fname = '/Users/krogager/Data/NOT/MALS/test_pynot/MALS_1623+1239_zh010127/FLUX1D_MALS_1623+1239.fits'

raw = fits.getdata(raw_fname)
med = np.nanmedian(raw)
std = np.median(np.fabs(raw - med))
vmin = med - 5*std
vmax = med + 10*std

red = fits.getdata(red_fname)
med_r = np.nanmedian(red)
std_r = np.median(np.fabs(red - med_r))
vmin_r = med_r - 5*std
vmax_r = med_r + 10*std

cmap = plt.cm.afmhot_r

spec1d = fits.getdata(ext_fname)
wl = spec1d['WAVE']
flux = spec1d['FLUX']*1.e16
err = spec1d['ERR']*1.e16


plt.close('all')
fig = plt.figure(figsize=(3.5, 4.8))
ax = fig.add_subplot(111)
ax.imshow(raw, origin='lower', vmin=vmin, vmax=vmax,
          aspect='auto', cmap=cmap, extent=[0, 400, 0, raw.shape[0]])
plt.tight_layout()
#ax.set_yticklabels('')
#ax.set_xticklabels('')


fig2 = plt.figure(figsize=(5.8, 2.9))
ax2 = fig2.add_subplot(111)
ax2.imshow(red, vmin=vmin_r, vmax=vmax_r, extent=[wl.min(), wl.max(), 0, 400*0.19],
        aspect='auto', cmap=cmap)

fig3 = plt.figure(figsize=(5.8, 3.))
ax3 = fig3.add_subplot(111)
flux = gaussian_filter1d(flux, 1.5)
ax3.errorbar(wl, flux, err, ls='', color='k', elinewidth=0.5, alpha=0.2)
ax3.scatter(wl, flux, marker='.', s=4, c=wl, cmap=plt.cm.jet)
ax3.set_ylim(0., 2.0)
ax3.set_xlim(wl.min(), wl.max())

dpi = 200
fig.savefig("raw_image.png", dpi=dpi)
fig2.savefig("red_image.png", dpi=dpi)
fig3.savefig("spectrum1d.png", dpi=dpi)
