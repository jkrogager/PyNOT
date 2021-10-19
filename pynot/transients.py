from astropy.wcs import WCS
from astropy.io import fits
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning

import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

from pynot.wcs import get_gaia_catalog
from pynot.functions import decimal_to_string
from pynot import alfosc


def mad(x):
    return np.median(np.abs(x-np.median(x)))


def find_sources_without_gaia(sep_cat, gaia, limit=1.5):
    no_match_list = list()
    refs = np.array([gaia['ra'], gaia['dec']]).T
    for row in sep_cat:
        xy = np.array([row['ra'], row['dec']])
        dist = np.sqrt(np.sum((refs - xy)**2, axis=1))
        if np.min(dist) < limit/3600.:
            pass
        else:
            no_match_list.append(np.array(row))
    no_match_list = np.array(no_match_list)
    return Table(no_match_list)


def find_new_sources(img_fname, sep_fname, loc_bat=(0., 0., 1.), loc_xrt=(0., 0., 1), mag_lim=20.1, zp=None):
    """
    Cross-match image source catalog with Gaia source catalog to identify new sources.

    Parameters
    ----------
    img_fname : string
        Filename of WCS calibrated image (_wcs.fits)

    sep_fname : string
        Filename of the source extraction table (_phot.fits)

    loc_bat : tuple(ra, dec, radius)
        Localisation of gamma ray detection, ra, dec and radius in deg

    loc_xrt : tuple(ra, dec, radius)
        Localisation of X-ray detection, ra, dec and radius in deg

    mag_lim : float
        Magnitude limit in the given filter

    zp : float  [default=None]
        Magnitude zero point if the source catalog has not been flux calibrated

    Returns
    -------
    new_subset : astropy.table.Table
        Subset of the source catalog which are not in the Gaia catalog above the flux limit.
        Contains the following columns: 'ra', 'dec', 'mag_auto', 'a', 'b', 'theta', 'flux_auto', 'flux_err_auto'

    output_msg : string
        Log of messages from the function call.
    """
    msg = list()
    img = fits.getdata(img_fname)
    hdr = fits.getheader(img_fname)

    # Download Gaia positions:
    base, ext = os.path.splitext(os.path.basename(img_fname))
    dirname = os.path.dirname(img_fname)
    image_radius = np.sqrt(hdr['NAXIS1']**2 + hdr['NAXIS2']**2) / 2
    image_scale = np.sqrt(hdr['CD1_1']**2 + hdr['CD1_2']**2)
    deg_to_arcmin = 60.
    radius = image_scale * image_radius * deg_to_arcmin
    gaia_cat_name = 'gaia_source_%.2f%+.2f_%.1f.csv' % (hdr['CRVAL1'], hdr['CRVAL2'], radius)
    gaia_cat_name = os.path.join(dirname, gaia_cat_name)
    gaia_dr = 'edr3'
    if os.path.exists(gaia_cat_name):
        msg.append("          - Loading Gaia source catalog: %s" % gaia_cat_name)
        ref_cat = Table.read(gaia_cat_name)
    else:
        msg.append("          - Downloading Gaia source catalog... (%s)" % gaia_dr.upper())
        ref_cat = get_gaia_catalog(hdr['CRVAL1'], hdr['CRVAL2'], radius=radius,
                                   catalog_fname=gaia_cat_name, database=gaia_dr)


    # Get the source extraction catalog:
    sep_cat = Table.read(sep_fname)
    sep_hdr = fits.getheader(sep_fname)
    if 'MAG_ZP' not in sep_hdr:
        if zp is None:
            msg.append("[WARNING] - The source table has not been flux calibrated yet. Run task:")
            msg.append("                ]%% pynot autozp %s %s" % (img_fname, sep_fname))
            msg.append("            or provide a zeropoint using the option: -z")
            msg.append("          - Terminating script...")
            msg.append("")
            return [], "\n".join(msg)
        else:
            try:
                sep_cat['mag_auto'] += zp
            except ValueError:
                msg.append(" [ERROR]  - Invalid zeropoint: %r" % zp)
                msg.append("")
                return [], "\n".join(msg)

    # Find sources with r < 20, with no match in Gaia:
    filter = alfosc.filter_translate[alfosc.get_filter(hdr)]
    if '_' in filter:
        band = filter.split('_')[0]

    mag_band = sep_cat['mag_auto']
    new_sources = find_sources_without_gaia(sep_cat[mag_band < mag_lim], ref_cat)
    warnings.simplefilter('ignore', category=AstropyWarning)
    wcs = WCS(hdr)
    if len(new_sources) > 0:
        mag = new_sources['mag_auto']
        q_ab = new_sources['b']/new_sources['a']
        new = new_sources[(q_ab > 0.80) & (mag > 12)]

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111, projection=wcs)
        med_val = np.nanmedian(img)
        ax.imshow(img, vmin=med_val-1*mad(img), vmax=med_val+10*mad(img),
                  origin='lower', cmap=plt.cm.gray_r)
        ylims = ax.get_ylim()
        xlims = ax.get_xlim()
        ax.scatter(ref_cat['ra'], ref_cat['dec'], label='Gaia',
                   transform=ax.get_transform('fk5'),
                   edgecolor='b', facecolor='none', s=40, marker='s')
        ax.set_xlabel("Right Ascension")
        ax.set_ylabel("Declination")
        box = ax.get_position()
        ax.set_position([0.15, box.y0, box.width, box.height])

        sattelite_data = (loc_bat, loc_xrt)
        sat_names = ('BAT', 'XRT')
        linestyles = (':', '--')
        colors = ('k', 'red')
        linewidths = (1., 1.0)
        chi2 = np.zeros_like(new['ra'])
        inside_bat = np.zeros(len(new), dtype=bool)
        for sat_name, burst_data, ls, col, lw in zip(sat_names, sattelite_data, linestyles, colors, linewidths):
            alpha, dec, sigma = burst_data
            err_circle = plt.Circle((alpha, dec), radius=sigma,
                                    transform=ax.get_transform('fk5'),
                                    facecolor='none', edgecolor=col, ls=ls, lw=lw,
                                    label=sat_name)
            ax.add_patch(err_circle)
            # convert 90% CI to 1-sigma (Gaussian):
            sig1 = sigma / 1.66
            # Calculate deviation in sigmas:
            delta_r = np.sqrt((new['ra']-alpha)**2 + (new['dec']-dec)**2)
            if sig1 == 0:
                pass
            else:
                chi2 += delta_r**2 / sig1**2
                if sat_name == 'BAT':
                    inside_bat = delta_r / sigma <= 1
        significance = np.sqrt(chi2)

        msg.append("          - New sources with no Gaia match")
        msg.append("          ---------------------------------")
        for source_id, (row, sig, in_bat) in enumerate(zip(new, significance, inside_bat)):
            if (sig < 5) and (sig > 0):
                color = 'Green'
                term_color = '\033[32m'      # green
                mark_good = '    [NEW] -'
            elif in_bat:
                color = 'DarkOrange'
                term_color = '\033[33m'      # yellow
                mark_good = 11*' '
            else:
                color = 'Red'
                term_color = '\033[31m'      # red
                mark_good = 11*' '

            mag = row['mag_auto']
            ax.scatter(row['ra'], row['dec'],
                       label='(%i)  %s = %.1f mag' % (source_id+1, band, mag),
                       transform=ax.get_transform('fk5'), edgecolor=color,
                       facecolor='none', s=60, lw=1.5)
            ax.text(row['ra']-8/3600., row['dec'], "%i" % (source_id+1),
                    transform=ax.get_transform('fk5'))

            reset = '\033[0m'
            ra_str, dec_str = decimal_to_string(row['ra'], row['dec'])
            msg.append(term_color + "%s %s %s  %s=%5.2f" % (mark_good, ra_str, dec_str, band, mag) + reset)
        ax.legend(title=hdr['OBJECT'].upper(), loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylim(*ylims)
        ax.set_xlim(*xlims)
        # Make a classification array for the type of new sources:
        # 0: not consistent with BAT nor XRT
        # 1: consistent with BAT
        # 2: consistent with BAT and XRT
        class_new = 1*inside_bat + 1*(significance < 5)*(significance > 0)
        new_subset = new['ra', 'dec', 'mag_auto', 'a', 'b', 'theta', 'flux_auto', 'flux_err_auto']
        new_subset['class'] = class_new
        new_table_fname = "new_sources_%s.txt" % base
        new_table_fname = os.path.join(dirname, new_table_fname)
        with open(new_table_fname, 'w') as new_table:
            units = ['(deg)', '(deg)', '(AB)', '(pix)', '(pix)', '(rad)', '(count/s)', '(count/s)', '']
            header = "{:^9}  {:^9}  {:^8}  {:^5}  {:^5}  {:^5}  {:^9}  {:^13}  {:^5}\n".format(*new_subset.colnames)
            unit_header = "{:^9}  {:^9}  {:^8}  {:^5}  {:^5}  {:^5}  {:^9}  {:^13}  {:^5}\n".format(*units)
            new_table.write(header)
            new_table.write(unit_header)
            np.savetxt(new_table, new_subset, fmt="%9.5f  %+9.5f  %8.2f  %5.1f  %5.1f  % 5.2f  %9.2e  %13.2e  %5i")
        msg.append(" [OUTPUT] - Writing detection table: %s" % new_table_fname)

        fig_fname = "new_sources_%s.pdf" % base
        fig_fname = os.path.join(dirname, fig_fname)
        fig.savefig(fig_fname)
        msg.append(" [OUTPUT] - Saving figure: %s" % fig_fname)

    else:
        new_subset = []
        msg.append("          - No new sources identified brighter than %s < %.2f" % (band, mag_lim))
    msg.append("")
    output_msg = "\n".join(msg)
    return new_subset, output_msg
