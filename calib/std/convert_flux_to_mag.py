import pyfits as pf
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Example input:
#fname = 'fits/feige34_005.fits'
#output = 'feige34.dat'
#star = 'Feige 34'
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]
    

def fits_to_dat(fname, star, output='', chunk_size=50):
    if len(output) > 0:
        pass
    else:
        output = star.lower() + '.dat'
    dat = pf.getdata(fname)
    
    wl0 = dat['WAVELENGTH']
    flux0 = dat['FLUX']
    
    wl = np.arange(wl0.min(), wl0.max(), 5.)
    flux = np.interp(wl, wl0, flux0)
    pixels = int(chunk_size / 5.) + 1

    # integrate:
    lam = list()
    mag = list()
    l_step = list()
    wl_chunks = list(chunks(wl, pixels))
    f_chunks = list(chunks(flux, pixels))
    for l, f in zip(wl_chunks, f_chunks):
        dl = l.max() - l.min()
        lam.append(np.mean(l))
        m = -2.5*np.log10(f.mean()) - 2.406 - 5*np.log10(np.mean(l))
        mag.append(m)
        l_step.append(dl)
    
    lam = np.array(lam)
    mag = np.array(mag)
    mag = mag[lam >= 3100]
    l_step = np.array(l_step)
    l_step = l_step[lam >= 3100]
    lam = lam[lam >= 3100]
    
    tab = np.column_stack([lam[:-1], mag[:-1], l_step[:-1]])
    out_file = open(output, 'w')
    out_file.write("# %s, AB magnitudes\n" % star)
    np.savetxt(out_file, tab, fmt="%.1f  %.3f  %.1f")
    out_file.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("fits", type=str,
                        help="FITS file name in table format, containing columns: WAVELENGTH and FLUX")
    parser.add_argument("star", type=str,
                        help="Star name")
    parser.add_argument("-o", "--output", type=str, default='',
                        help="Output filename, default: 'star_name.dat'")
    parser.add_argument("-s", "--size", type=float, default=50.,
                        help="Band pass size in Angstrom")
    args = parser.parse_args()

    fits_to_dat(args.fits, args.star, args.output, args.size)
