from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev
from scipy.ndimage import rotate, median_filter
import scipy.optimize as opt

plt.rcParams['font.family'] = 'Arial'

arc = fits.getdata('/Users/krogager/Projects/Johan/rotcurve_gals/raw_data/ALDh130181.fits')
arc = arc[:-50, 50:-50]
arc = arc - np.median(arc)

noise = np.median(np.fabs(arc))

plt.close('all')

#fig = plt.figure(figsize=(8, 8))
#ax = fig.add_subplot(111)
#ax.imshow(arc, origin='lower', vmin=-5*noise, vmax=35*noise)
#ax.set_ylabel("Dispersion Axis", fontsize=16, fontweight='bold')
#ax.set_xlabel("Spatial Axis", fontsize=16, fontweight='bold')
#fig.tight_layout()
#fig.savefig("rectify_arc2d.png")

# Image region:
x1, x2 = 341, 1830
cutout = arc[:, x1:x2]
cutout = rotate(cutout, 90.)

#ax.axvline(x1, color='white', ls='--', lw=1.)
#ax.axvline(x2, color='white', ls='--', lw=1.)
#fig.savefig("rectify_arc2d_limits.png")


#fig = plt.figure(figsize=(8, 8))
#ax = fig.add_subplot(111)
#ax.imshow(cutout, origin='lower', vmin=-5*noise, vmax=35*noise)
## ax.axhline(50, color='blue', ls='-', lw=2., alpha=0.8)
## ax.axhline(745, color='black', ls='-', lw=2., alpha=0.8)
## ax.axhline(1450, color='red', ls='-', lw=2., alpha=0.8)
#ax.set_xlabel("Dispersion Axis", fontsize=16, fontweight='bold')
#ax.set_ylabel("Spatial Axis", fontsize=16, fontweight='bold')
#fig.tight_layout()
#
#row_numbers = [50, 745, 1450]
#row_colors = ['blue', 'black', 'red']
#fig2, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 8))
#for pixval, col, axis in zip(row_numbers, row_colors, axes[::-1]):
#    axis.plot(cutout[pixval], color=col, lw=1., label="Row no. %i" % pixval)
#    axis.set_xlim(150., 895.)
#    axis.set_yticklabels("")
#    axis.set_ylabel("Row no. %i" % pixval)
#    axis.yaxis.set_label_position("right")
#    ax.axhline(pixval, color=col, ls='-', lw=2, alpha=0.9)
#    for linepos in [216., 430., 633., 845.]:
#        axis.axvline(linepos, color='k', ls='--', lw=0.5)
#
#axes[2].set_xlabel("Dispersion Axis  [pixels]")
#axes[0].set_xticklabels("")
#axes[1].set_xticklabels("")
#fig2.tight_layout()
#
#fig.savefig("rectify_arc2d_rotate.png")
#fig2.savefig("rectify_arc1d_rows.png")

def gauss(x, mu, sigma, amp):
    return amp*np.exp(-(x-mu)**2/(2*sigma**2))

def fitgauss(x, data, params=[15., 1.0, 1.0]):
    # errfunc = lambda p: np.ravel(gauss(*p)(x)-data)
    # p, cov = opt.leastsq(errfunc, params)
    popt, pcov = opt.curve_fit(gauss, x, data, p0=params)
    perr = np.sqrt(pcov.diagonal())
    return popt, perr

x0 = 825
x02 = 870
zoom_in = cutout[:, x0:x02]
fig31 = plt.figure(figsize=(4, 8))
ax2d = fig31.add_subplot(111)
ax2d.imshow(zoom_in, origin='lower', vmin=-5*noise, vmax=1005*noise,
           aspect='auto', interpolation='nearest', extent=[x0, x02, 0, zoom_in.shape[0]])
ax2d.set_xlabel("Dispersion Axis", fontsize=16, fontweight='bold')
ax2d.set_ylabel("Spatial Axis", fontsize=16, fontweight='bold')

peaks = list()
x = np.arange(zoom_in.shape[1])
for num, row in enumerate(zoom_in):
    popt, _ = fitgauss(x, row, [25., 5., np.max(row)])
    loc = popt[0]
    if num % 10 == 0:
        ax2d.plot(x0+loc, num, 'k.', alpha=0.7, ms=4)
    peaks.append(loc)

peaks = np.array(peaks)
col = np.arange(zoom_in.shape[0])
deg = 9

def median_filter_data(x, kappa=2., window=51):
    med_x = median_filter(x, window)
    noise = np.nanmedian(np.abs(x - med_x)) * 1.48
    if noise == 0:
        noise = np.nanstd(x - med_x)
    mask = np.abs(x - med_x) < kappa*noise
    return (med_x, mask)

med_col, mask = median_filter_data(peaks, window=35, kappa=2.)
cheb_polyfit = Chebyshev.fit(col[mask], peaks[mask], deg=deg, domain=(col.min(), col.max()))

fig32 = plt.figure()
ax1 = fig32.add_subplot(211)
ax2 = fig32.add_subplot(212)

ax1.plot(col, x0+peaks, 'k.', alpha=0.8, ms=2, label='Measured position per row')
#ax1.plot(col[mask], x0+peaks[mask], 'k.', alpha=0.8, ms=2)
ax1.plot(col, x0+cheb_polyfit(col), 'r', lw=1.5, alpha=0.8,
        label='Polynomial order: 9')
ax2d.plot(x0+cheb_polyfit(col), col, 'r', lw=1.5, alpha=0.8)
ax1.legend()

residual = peaks - cheb_polyfit(col)
ax2.axhline(0., color='r', alpha=0.7, lw=1.)
ax2.plot(col, residual, marker='.', color='0.2', ls='', ms=2)

ax2.set_xlabel("Spatial Axis", fontsize=16, weight='bold')
ax2.set_ylabel("Residuals  [pixels]", fontsize=14, weight='bold')
ax1.set_ylabel("Arc line position  [pixels]", fontsize=14, weight='bold')
ax1.set_xlim(0, len(col))
ax2.set_xlim(0, len(col))

fig31.tight_layout()
fig32.tight_layout()
fig31.savefig("rectify_linefit2d.png")
fig32.savefig("rectify_linefit1d.png")
