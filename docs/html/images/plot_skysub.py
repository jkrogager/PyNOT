import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy import signal

from pynot.skysub import detect_objects_in_slit, fit_background_row
from pynot.functions import mad


filename = '/Users/krogager/Projects/Johan/close_spectra/obj.fits'

data2d = fits.getdata(filename)

spsf = np.nansum(data2d, axis=1) / 1e6
x = np.arange(len(spsf))

mask = detect_objects_in_slit(x, spsf, fwhm_scale=3, obj_kappa=20)

noise = 1.4826 * mad(spsf)
peaks, properties = signal.find_peaks(spsf, prominence=20*noise, width=3)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, spsf, color='0.2', lw=0.5)
ax.plot(x, np.ma.masked_where(~mask, spsf), color='0.2', lw=1.5)

colors = ['red', 'blue', 'green']
for num, center in enumerate(peaks):
    width = properties['widths'][num]
    ax.axvline(center, color=colors[num], alpha=0.5, label=f'Object #{num}')
    ax.axvline(center+width, color=colors[num], alpha=0.5, ls=':')
    ax.axvline(center-width, color=colors[num], alpha=0.5, ls=':')

idx = np.nonzero(~mask)[0]
xmin = np.min(idx)
xmax = np.max(idx)
ax.axvspan(xmin, xmax, alpha=0.1, color='gray', label='Excluded pixels')

ax.set_xlim(0, x.max())
ax.set_xlabel("Spatial axis (along slit)  [pixels]")
ax.set_ylabel("Total flux in slit  [10$^6$ counts]")
ax.legend()
plt.tight_layout()

plt.savefig("skysub1.png", dpi=200)


# Fit single row
row = data2d[:, 1860] + data2d[:, 900]
bg, row_mask = fit_background_row(x, row, mask)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot(x, row, '.:', color='0.6', ms=3, lw=0.5)
ax2.plot(x[mask & row_mask], row[mask & row_mask], 'k.', ms=4, label='Good pixels')
ax2.plot(x[~(mask & row_mask)], row[~(mask & row_mask)], 'rx', ms=4, label='Excluded pixels')
ax2.plot(x, bg, color='RoyalBlue', alpha=0.8, lw=2, label='Best-fit model')

ax2.set_xlim(0, x.max())
ax2.set_xlabel("Spatial axis (along slit)  [pixels]")
ax2.set_ylabel("Flux in row  [counts]")
ax2.legend()

plt.tight_layout()
plt.savefig("skysub2.png", dpi=200)
