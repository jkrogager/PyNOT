"""
Functions for Imaging Pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.modeling import models, fitting
import os

import astroalign as aa
import sep

from pynot.functions import get_version_number, mad
from pynot.data.organizer import get_filter

__version__ = get_version_number()


def source_detection(fname, threshold=1.5, aperture=3.0, exptime=None, gain=None):
    msg = list()
    # get GAIN from header
    data = fits.getdata(fname)
    error_image = fits.getdata(fname, 'ERR')
    hdr = fits.getheader(fname)
    msg.append("          - Loaded input image: %s" % fname)
    if 'GAIN' in hdr:
        gain = hdr['GAIN']
        msg.append("          - Loaded gain from image header: %.3f" % gain)
    elif gain is not None:
        pass
    else:
        gain = 1.0
        msg.append("[WARNING] - No gain found in image header!")

    if 'EXPTIME' in hdr:
        exptime = hdr['EXPTIME']
        msg.append("          - Loaded exposure time from image header: %.1f" % exptime)
    elif exptime is not None:
        pass
    else:
        exptime = 1.
        msg.append("[WARNING] - No exposure time found in image header!")

    data = data*exptime
    error_image = error_image*exptime

    bkg = sep.Background(data, bw=64, bh=64, fw=3, fh=3)
    data_sub = data - bkg
    msg.append("          - Subtracted sky background")
    msg.append("          - Background RMS: %.2e" % bkg.globalrms)
    if data_sub.dtype.byteorder != '<':
        data_sub = data_sub.byteswap().newbyteorder()
    objects, segmap = sep.extract(data_sub, threshold, err=bkg.globalrms,
                                  segmentation_map=True)

    N_obj = len(objects)

    flux, fluxerr, flag = sep.sum_circle(data_sub, objects['x'], objects['y'],
                                         aperture, err=error_image)


def plot_objects(fig_fname, data, objects):
    # plot background-subtracted image
    fig, ax = plt.subplots()
    m, s = np.median(data), 1.5*mad(data)
    ax.imshow(data, interpolation='nearest', cmap='gray_r',
              vmin=m-3*s, vmax=m+3*s, origin='lower')

    # plot an ellipse for each object
    for i in range(len(objects)):
        e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                    width=6*objects['a'][i],
                    height=6*objects['b'][i],
                    angle=objects['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)
    fig.tight_layout()
    fig.savefig(fig_fname)


def load_fits_image(fname):
    with fits.open(fname) as hdu_list:
        image = hdu_list[0].data
        hdr = hdu_list[0].header
        if 'ERR' in hdu_list:
            error = hdu_list['ERR'].data
        else:
            raise TypeError("No error image detected")

        if 'MASK' in hdu_list:
            mask = hdu_list['MASK'].data
        else:
            mask = np.zeros_like(image, dtype=bool)
    return image, error, mask, hdr


def measure_seeing(img, centers, size=50):
    X = np.arange(img.shape[1])
    Y = np.arange(img.shape[0])
    sigmas = list()
    ratios = list()
    good_x = (centers[:, 0] > size) & (centers[:, 0] < X.max()-size)
    good_y = (centers[:, 1] > size) & (centers[:, 1] < Y.max()-size)
    idx = np.random.choice(np.arange(len(centers))[good_x & good_y], 10, replace=False)
    for x_cen, y_cen in centers[idx]:
        x1, x2 = int(x_cen)-size, int(x_cen)+size
        y1, y2 = int(y_cen)-size, int(y_cen)+size
        cutout = img[y1:y2, x1:x2]
        x, y = np.meshgrid(X[x1:x2], Y[y1:y2])
        A = img[int(y_cen), int(x_cen)]
        p_init = models.Gaussian2D(amplitude=A, x_mean=x_cen, y_mean=y_cen, x_stddev=5, y_stddev=5, theta=0)
        try:
            fitter = fitting.LevMarLSQFitter()
        except TypeError:
            continue
        p_opt = fitter(p_init, x, y, cutout-np.median(cutout))
        sigma_x = p_opt.x_stddev
        sigma_y = p_opt.y_stddev
        sig = np.sqrt(sigma_x**2 + sigma_y**2)
        ba = min(sigma_x, sigma_y) / max(sigma_x, sigma_y)
        sigmas.append(sig)
        ratios.append(ba)

    fwhm = np.median(sigmas) * 2.35
    ratio = np.median(ratios)
    return (fwhm, ratio)


def save_file_log(log_name, image_log, target_hdr):
    with open(log_name, 'w') as out:
        out.write("# PyNOT Combination Log of Target: %s\n" % target_hdr['OBJECT'])
        out.write("# Filter: %s\n" % get_filter(target_hdr))
        out.write("# Col 1: Filename\n")
        out.write("# Col 2: FWHM / pixels  (seeing)\n")
        out.write("# Col 3: PSF axis ratio  (minor/major)\n")
        out.write("# Col 4: Exp. Time / seconds\n")
        out.write("# " + 40*"-" + "\n")
        for line in image_log:
            out.write(" %s   %.1f  %5.2f  %6.1f\n" % tuple(line))


def image_combine(corrected_images, output='', log_name='', fringe_image='', method='weighted', max_control_points=50, detection_sigma=5, min_area=9):
    """
    max_control_points: 50     # Maximum number of control point-sources to find the transformation
    detection_sigma:     5     # Factor of background std-dev above which is considered a detection
    min_area:            9     # Minimum number of connected pixels to be considered a source
    """
    msg = list()
    if fringe_image != '':
        norm_sky = fits.getdata(fringe_image)
        msg.append("          - Loaded normalized fringe image: %s" % fringe_image)
    else:
        norm_sky = 1.
    target_fname = corrected_images[0]
    target, target_err, target_mask, target_hdr = load_fits_image(target_fname)
    target = target - norm_sky*np.median(target)
    exptime = target_hdr['EXPTIME']
    target /= exptime
    target_err /= exptime
    target_hdr['BUNIT'] = 'count / s'
    msg.append("          - Aligning all images to reference: %s" % target_fname)

    msg.append("          - Registering input images:")
    shifted_images = [target]
    shifted_vars = [target_err**2]
    target = target.byteswap().newbyteorder()
    if target.dtype.byteorder != '<':
        target = target.byteswap().newbyteorder()
    final_exptime = exptime
    image_log = list()
    for fname in corrected_images[1:]:
        msg.append("          - Input image: %s" % fname)
        source, source_err, source_mask, hdr_i = load_fits_image(fname)
        source = source - norm_sky*np.median(source)
        source /= hdr_i['EXPTIME']
        source_err /= hdr_i['EXPTIME']
        final_exptime += hdr_i['EXPTIME']
        try:
            transf, (coords) = aa.find_transform(source, target,
                                                 max_control_points=max_control_points,
                                                 detection_sigma=detection_sigma,
                                                 min_area=min_area)
        except:
            msg.append(" [ERROR]  - Failed to find image transformation!")
            msg.append("          - Skipping image")
            continue

        source = source.byteswap().newbyteorder()
        source_err = source_err.byteswap().newbyteorder()
        source_mask = source_mask.byteswap().newbyteorder()
        if source.dtype.byteorder != '<':
            source = source.byteswap().newbyteorder()
        if source_err.dtype.byteorder != '<':
            source_err = source_err.byteswap().newbyteorder()
        if source_mask.dtype.byteorder != '<':
            source_mask = source_mask.byteswap().newbyteorder()

        registered_image, _ = aa.apply_transform(transf, source, target, fill_value=0)
        registered_error, _ = aa.apply_transform(transf, source_err, target, fill_value=0)
        registered_mask, _ = aa.apply_transform(transf, source_mask, target, fill_value=0)
        target_mask += 1 * (registered_mask > 0)
        registered_error[registered_error == 0] = np.mean(registered_error)*10
        shifted_images.append(registered_image)
        shifted_vars.append(registered_error**2)
        source_list, target_list = coords
        if len(image_log) == 0:
            fwhm, ratio = measure_seeing(target, target_list)
            image_log.append([os.path.basename(target_fname), fwhm, ratio, exptime])
        fwhm, ratio = measure_seeing(source, source_list)
        image_log.append([os.path.basename(fname), fwhm, ratio, hdr_i['EXPTIME']])

    if log_name == '':
        log_name = 'filelist_%s_%s.txt' % (target_hdr['OBJECT'], get_filter(target_hdr))
    save_file_log(log_name, image_log, target_hdr)
    msg.append("          - Saved file log and image stats: %s" % log_name)

    if method == 'median':
        final_image = np.nanmedian(shifted_images, axis=0)
        final_error = np.sqrt(np.nanmean(shifted_vars, axis=0))
        target_hdr['COMBINE'] = "Median"
    elif method == 'mean':
        final_image = np.nanmean(shifted_images, axis=0)
        final_error = np.sqrt(np.nanmean(shifted_vars, axis=0))
        target_hdr['COMBINE'] = "Mean"
    else:
        w = 1./np.array(shifted_vars)
        shifted_images = np.array(shifted_images)
        final_image = np.nansum(w*shifted_images, axis=0) / np.sum(w, axis=0)
        final_error = np.sqrt(1. / np.nansum(w, axis=0))
        target_hdr['COMBINE'] = "Inverse Variance Weighted"
    final_mask = 1 * (target_mask > 0)

    target_hdr['NCOMBINE'] = len(shifted_images)
    target_hdr['EXPTIME'] = final_exptime
    # Fix NaN values from negative pixel values:
    err_NaN = np.isnan(final_error)
    final_error[err_NaN] = np.nanmean(final_error)*100
    msg.append("          - Correcting NaNs in noise image: %i pixel(s)" % np.sum(err_NaN))
    target_hdr['DATAMIN'] = np.nanmin(final_image)
    target_hdr['DATAMAX'] = np.nanmax(final_image)
    target_hdr['EXTNAME'] = 'DATA'
    target_hdr['AUTHOR'] = 'PyNOT version %s' % __version__

    mask_hdr = fits.Header()
    mask_hdr.add_comment("0 = Good Pixels")
    mask_hdr.add_comment("1 = Cosmic Ray Hits")

    sci_ext = fits.PrimaryHDU(final_image, header=target_hdr)
    err_ext = fits.ImageHDU(final_error, header=target_hdr, name='ERR')
    mask_ext = fits.ImageHDU(final_mask, header=mask_hdr, name='MASK')
    output_HDU = fits.HDUList([sci_ext, err_ext, mask_ext])
    output_HDU.writeto(output, overwrite=True)
    msg.append("          - Successfully combined the images")
    msg.append(" [OUTPUT] - Saving output: %s" % output)
    msg.append("")
    output_msg = "\n".join(msg)
    return output_msg


def plot_image2D(fname, image, vmin=-2, vmax=2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    med = np.median(image)
    s = mad(image)
    im = ax.imshow(image, origin='lower', vmin=med+vmin*s, vmax=med+vmax*s)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(fname)


def create_fringe_image(input_filenames, output='', fig_fname='', threshold=3.0):
    msg = list()
    hdr = fits.getheader(input_filenames[0])
    img_list = [fits.getdata(fname) for fname in input_filenames]
    exptimes = [fits.getheader(fname)['EXPTIME'] for fname in input_filenames]
    msg.append("          - Loaded input images")
    mask = [np.fabs(im-np.median(im)) < threshold*mad(im) for im in img_list]
    msg.append("          - Created image mask using threshold: %.2f" % threshold)

    N = np.sum(mask, 0)
    skysum = np.sum([im*m/t for im, m, t in zip(img_list, mask, exptimes)], axis=0)
    skysum[N == 0] = np.median(skysum)
    N[N == 0] = 1
    sky = skysum / N
    norm_sky = sky / np.median(sky)
    msg.append("          - Created normalized fringe image")

    if fig_fname:
        plot_image2D(fig_fname, norm_sky, vmin=-2, vmax=2)
        msg.append(" [OUTPUT] - Saving figure: %s" % fig_fname)

    if output == '':
        output = "fringe_%s.fits" % hdr['OBJECT']
    hdr['OBJECT'] = 'Fringe Image'
    hdr['EXTNAME'] = 'MODEL'
    hdr.add_comment('Average Fringe image, median normalized')
    fits.writeto(output, norm_sky, header=hdr, overwrite=True)
    msg.append(" [OUTPUT] - Saving output: %s" % output)
    msg.append("")
    output_msg = "\n".join(msg)
    return output_msg