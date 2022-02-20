"""
Write calibration reports for:
    BIAS frames
    FLAT frames
    ARC frames
    RESPONSE functions
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf
from astropy.io import fits
import os

from pynot import instrument

report_dir = 'reports'

plt.rcParams['font.family'] = 'Arial'

def check_bias(bias_images, bias_fnames, mbias_fname, report_fname=''):
    mbias = fits.getdata(mbias_fname)

    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    N_tot = len(bias_images)-1
    for num, (img, fname) in enumerate(zip(bias_images, bias_fnames)):
        x0 = img.shape[1]//2
        y0 = img.shape[0]//2

        label = os.path.splitext(os.path.basename(fname))[0]

        axes[1, 0].plot(img[:, x0], color=plt.cm.turbo(num/N_tot), lw=0.5)
        axes[1, 1].plot(img[y0, :], color=plt.cm.turbo(num/N_tot), lw=0.5)
        axes[0, 1].plot(-1, -1, marker='s', ls='', color=plt.cm.turbo(num/N_tot), label=label)

        axes[1, 0].set_xlabel("Y-axis at col=%i (pixels)" % x0)
        axes[1, 0].set_xlim(0, img.shape[0])
        axes[1, 0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes[1, 1].set_xlabel("X-axis at row=%i (pixels)" % y0)
        axes[1, 1].set_xlim(0, img.shape[1])

    med = np.nanmedian(mbias)
    noise = np.nanmedian(np.abs(mbias - med))
    vmin = med - 3*noise
    vmax = med + 10*noise
    noise_percent = 1.48*noise / med * 100
    title_font = dict(fontfamily='monospace', fontsize=10)
    report_title = "Bias Report:  MBIAS = %s" % mbias_fname
    report_title += "\nMedian = %.2e  Std.Dev = %.2e  M.A.D. = %.2e  (%.1f%%)" % (med, np.std(mbias), 1.48*noise, noise_percent)
    axes[0, 0].imshow(mbias, vmin=vmin, vmax=vmax, origin='lower')
    axes[0, 0].set_title(report_title, pad=10, loc='left', fontdict=title_font)
    axes[1, 0].plot(mbias[:, x0], color='k', lw=1.)
    axes[1, 1].plot(mbias[y0, :], color='k', lw=1.)
    axes[0, 1].plot(-1, -1, marker='s', ls='', color='k', label='MASTER_BIAS')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].get_xaxis().set_visible(False)
    axes[0, 1].get_yaxis().set_visible(False)
    for spine in axes[0, 1].spines.values():
        spine.set_visible(False)

    # Show legend and adujst it to fit:
    ncol = 2
    fontsize = 6
    legend = axes[0, 1].legend(fontsize=fontsize, ncol=ncol, markerscale=0.5)
    legend.set_in_layout(False)
    # plt.tight_layout()
    # plt.pause(0.00001)
    # # Check if legend bounds are ok:
    # Px = axes[0, 0].bbox.p1[0] + 5
    # Py = axes[1, 1].bbox.p1[1] + 5
    # bbox = legend.get_window_extent()
    # legend_x_ok = bbox.p0[0] > Px
    # legend_y_ok = bbox.p0[1] > Py
    # while legend_x_ok & legend_y_ok is False:
    #     if not legend_x_ok:
    #         # too large in x-direction -> smaller font
    #         fontsize -= 1
    #     if not legend_y_ok:
    #         # too large in y-direction -> add column
    #         ncol += 1
    #     legend.remove()
    #     legend = axes[0, 1].legend(fontsize=fontsize, ncol=ncol, markerscale=0.5)
    #     # plt.draw()
    #     plt.pause(0.00001)
    #     bbox = legend.get_window_extent()
    #     legend_x_ok = bbox.p0[0] > Px
    #     legend_y_ok = bbox.p0[1] > Py

    plt.tight_layout()
    # plt.draw()
    if not report_fname:
        base = os.path.splitext(os.path.basename(mbias_fname))[0]
        report_fname = '%s_report.pdf' % base
    plt.savefig(report_fname)
    plt.close()
    return report_fname


def check_flats(flat_images, flat_fnames, mflat_fname, report_fname=''):
    pass

def check_arcs(arc_fnames, report_fname):
    pdf = backend_pdf.PdfPages(report_fname)
    for fname in arc_fnames:
        fig, axes = plt.subplots(3, 1, figsize=(5.6, 8))
        img = fits.getdata(fname)
        hdr = fits.getheader(fname)
        x0 = img.shape[1]//2
        y0 = img.shape[0]//2

        grism = instrument.get_grism(hdr)
        slit = instrument.get_slit(hdr)
        med = np.nanmedian(img)
        noise = np.nanmedian(np.abs(img - med))
        vmin = med - 3*noise
        vmax = med + 20*noise
        axes[0].imshow(img, vmin=vmin, vmax=vmax, origin='lower')
        axes[1].plot(img[:, x0], color='k', lw=0.5, drawstyle='steps-mid')
        axes[2].plot(img[y0, :], color='k', lw=0.5, drawstyle='steps-mid')

        axes[0].set_title('%s  |  %s  |  %s' % (fname, grism, slit), pad=10, fontsize=10)
        axes[1].set_xlabel("Dispersion Axis at col=%i (pixels)" % x0)
        axes[1].set_xlim(0, img.shape[0])
        axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes[2].set_xlabel("Spatial Axis at row=%i (pixels)" % y0)
        axes[2].set_xlim(0, img.shape[1])
        axes[1].set_ylabel("Counts")
        axes[2].set_ylabel("Counts")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close()
    pdf.close()
