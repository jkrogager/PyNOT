import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

img1 = fits.getdata('obj1.fits')
img2 = fits.getdata('obj2.fits')

img_comb = fits.getdata('obj_comb.fits')
img_comb_x = fits.getdata('obj_comb_x.fits')

images = {
    'obj1': img1,
    'obj2': img2,
    'comb': img_comb,
    'comb_x': img_comb_x,
        }

for figname, im in images.items():
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    vmax = im.mean(0).max()*2
    vmin = im.std()
    ax.imshow(im, aspect='auto', vmin=-vmin, vmax=vmax,
              origin='lower')
    ax.tick_params(labelsize=14)
    ax.set_xlabel("Dispersion Axis  [pixels]", fontsize=18)
    ax.set_ylabel("Spatial Axis  [pixels]", fontsize=18)
    plt.tight_layout()
    fig.savefig(f"scombine_{figname}.png")
    plt.close()

