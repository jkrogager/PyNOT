# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from astropy.io import fits
from astropy.wcs import WCS

from astropy.table import Table


data_path = '/Users/krogager/Data/NOT/MALS/test_pynot/raw/'
raw_frames = [
        'ALzh010097.fits',
        'ALzh010098.fits',
        'ALzh010099.fits',
        'ALzh010100.fits',
        ]
red_fname = '/Users/krogager/Data/NOT/MALS/test_pynot/imaging/GRB160801A/GRB160801A_r_SDSS.fits'
seg_fname = '/Users/krogager/Data/NOT/MALS/test_pynot/imaging/GRB160801A/GRB160801A_r_SDSS_seg.fits'

red = fits.getdata(red_fname)
hdr = fits.getheader(red_fname)
med_r = np.nanmedian(red)
std_r = np.median(np.fabs(red - med_r))
vmin_r = med_r - 5*std_r
vmax_r = med_r + 10*std_r

wcs = WCS(hdr)

cmap = plt.cm.afmhot_r



plt.close('all')

dpi = 100
for num, fname in enumerate(raw_frames):
    raw = fits.getdata(data_path+fname)
    med = np.nanmedian(raw)
    std = np.median(np.fabs(raw - med))
    vmin = med - 5*std
    vmax = med + 10*std
    
    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.add_subplot(111)
    ax.imshow(raw, origin='lower', vmin=vmin, vmax=vmax,
              aspect='auto', cmap=cmap, extent=[0, raw.shape[1], 0, raw.shape[0]])
    plt.tight_layout()
    #ax.set_yticklabels('')
    #ax.set_xticklabels('')
    fig.savefig("raw_phot%i.png" % (num+1), dpi=dpi)


fig2 = plt.figure(figsize=(4.8, 4.8))
ax2 = fig2.add_subplot(111, projection=wcs)
ax2.grid(color='k', ls='--', which='major', linewidth=0.5, alpha=0.4)
ax2.imshow(red, vmin=vmin_r, vmax=vmax_r,
        aspect='auto', cmap=cmap)
ax2.set_xlabel(" ")
fig2.subplots_adjust(top=0.98, right=0.98, left=0.12, bottom=0.12)

fig2.savefig("red_phot.png", dpi=dpi)
