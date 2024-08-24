import numpy as np
from astropy.io import fits
import operator
from scipy.interpolate import RegularGridInterpolator

from pynot.fitsio import load_fits_image


class FitsImage:
    def __init__(self, data, error=None, mask=None, header=None):
        self.data = data
        self.x = np.arange(data.shape[1])
        self.y = np.arange(data.shape[0])
        self.shape = self.data.shape
        if error is None:
            self.error = np.zeros_like(data)
        elif error.shape == self.shape:
            self.error = error
        else:
            raise ValueError("Error must be of same shape as `data`")

        if mask is None:
            self.mask = np.zeros(data.shape, dtype=bool)
        elif mask.shape == self.shape:
            self.mask = mask.astype(bool)
        else:
            raise ValueError("Mask must be of same shape as `data`")

        if header is None:
            self.header = fits.Header()
            self.header['NAXIS1'] = self.data.shape[1]
            self.header['NAXIS2'] = self.data.shape[0]
            axis_hdr = {'CRPIX1': 1,
                        'CRVAL1': 0,
                        'CDELT1': 1,
                        'CRPIX2': 1,
                        'CRVAL2': 0,
                        'CDELT2': 1,
                        }
            self.header.update(axis_hdr)
        else:
            self.header = header
            self.set_image_axes()

    def set_image_axes(self):
        x, y = None, None
        cdelt1_in_header = 'CD1_1' in self.header or 'CDELT1' in self.header
        cdelt2_in_header = 'CD2_2' in self.header or 'CDELT2' in self.header
        if 'CRVAL1' in self.header and 'CRPIX1' in self.header and cdelt1_in_header:
            crpix1 = self.header['CRPIX1'] - 1
            cdelt1 = self.header.get('CD1_1', self.header.get('CDELT1', 1))
            x = (np.arange(self.header['NAXIS1']) - crpix1) * cdelt1 + self.header['CRVAL1']

        if 'CRVAL2' in self.header and 'CRPIX2' in self.header and cdelt2_in_header:
            crpix1 = self.header['CRPIX2'] - 1
            cdelt1 = self.header.get('CD2_2', self.header.get('CDELT2', 1))
            y = (np.arange(self.header['NAXIS2']) - crpix1) * cdelt1 + self.header['CRVAL2']

        if x is not None and y is not None:
            self.x = x
            self.y = y

    @staticmethod
    def read(fname):
        data, error, mask, header = load_fits_image(fname)
        return FitsImage(data, error=error, mask=mask, header=header)

    def write(self, filename):
        prim = fits.PrimaryHDU(data=self.data, header=self.header)
        prim.name = 'DATA'
        hdu_list = fits.HDUList([
            prim,
            fits.ImageHDU(data=self.error, header=self.header, name='ERR'),
            fits.ImageHDU(data=1*(self.mask >= 1), header=self.header, name='MASK'),
        ])
        hdu_list.writeto(filename, overwrite=True, output_verify='silentfix')

    def __array__(self):
        return self.data

    def __getitem__(self, val):
        values = {}
        attributes = ['data', 'error', 'mask']
        for attr in attributes:
            values[attr] = self.__getattribute__(attr)[val]
        if isinstance(val, tuple):
            yslice, xslice = val
        else:
            yslice = val
            xslice = slice(None, None, None)
        new_x = self.x[xslice]
        new_y = self.y[yslice]
        new_hdr = self.header.copy()
        new_hdr['CRVAL1'] = new_x[0]
        new_hdr['CRVAL2'] = new_y[0]
        new_hdr['NAXIS1'] = len(new_x)
        new_hdr['NAXIS2'] = len(new_y)
        values['header'] = new_hdr
        return self.__class__(**values)

    def __add__(self, other):
        if isinstance(other, (int, float, np.number, np.integer)):
            self.data += other
            return self
        elif isinstance(other, FitsImage):
            return image_operation(self, other, operator.add)
        else:
            raise TypeError(f"Invalid operation with type: {type(other)}")

    def __sub__(self, other):
        if isinstance(other, (int, float, np.number, np.integer)):
            self.data -= other
            return self
        elif isinstance(other, FitsImage):
            return image_operation(self, other, operator.sub)
        else:
            raise TypeError(f"Invalid operation with type: {type(other)}")
        
    def __mult__(self, other):
        if isinstance(other, (int, float, np.number, np.integer)):
            self.data *= other
            self.error *= other
            return self
        elif isinstance(other, FitsImage):
            return image_operation(self, other, operator.mult)
        else:
            raise TypeError(f"Invalid operation with type: {type(other)}")

    def __truediv__(self, other):
        if isinstance(other, (int, float, np.number, np.integer)):
            self.data /= other
            self.error /= other
            return self
        elif isinstance(other, FitsImage):
            return image_operation(self, other, operator.truediv)
        else:
            raise TypeError(f"Invalid operation with type: {type(other)}")

    def __pow__(self, other):
        if isinstance(other, (int, float, np.number, np.integer)):
            self.data = self.data**other
            self.error = np.abs(other) * self.error
            return self
        else:
            raise TypeError(f"Invalid operation with type: {type(other)}")

    def __str__(self):
        return f"<FitsImage: {self.shape}>"

    def interpolate(self, new_x, new_y):
        """
        Interpolate the FitsImage onto a new regular grid defined by axes: `new_x` and `new_y`.
        The new grid will be constructed using :func:`np.meshgrid` of the two input arrays.
        All image attributes of the FitsImage will be interpolated onto the new grid.

        Parameters
        ----------
        new_x : np.ndarray
            The points of the new x-axis corresponding to `self.shape[1]`
        new_y : np.ndarray
            The points of the new y-axis corresponding to `self.shape[0]`

        Returns
        -------
        :class:`pynot.images.FitsImage`
            The new 2D image with the same dimensions as the new x and y coordinate arrays.
        """
        new_points = tuple(np.meshgrid(new_y, new_x, indexing='ij'))
        new_hdr = self.header.copy()
        new_hdr['NAXIS1'] = len(new_x)
        new_hdr['CRPIX1'] = 1
        new_hdr['CRVAL1'] = new_x[0]
        new_hdr['CD1_1'] = np.diff(new_x)[0]
        new_hdr['NAXIS2'] = len(new_y)
        new_hdr['CRPIX2'] = 1
        new_hdr['CRVAL2'] = new_y[0]
        new_hdr['CD2_2'] = np.diff(new_y)[0]
        new_hdr['CD1_2'] = 0
        new_hdr['CD2_1'] = 0
        values = {'header': new_hdr}
        attributes = ['data', 'error', 'mask']
        for attr in attributes:
            array2D = self.__getattribute__(attr)
            values[attr] = RegularGridInterpolator((self.y, self.x), array2D, bounds_error=False)(new_points)
        return self.__class__(**values)

    def shift(self, dx=0, dy=0):
        """
        Shift the image by a given number of pixels in x and y directions.
        Positive values of `dx` and `dy` will shift the image to the right, negative values
        will apply a shift to the left. If non-integer values are applied, the image will
        be interpolated onto a sub-pixel grid and then resampled back to its native sampling.
        All image attributes of the FitsImage will be interpolated onto the new grid.

        Parameters
        ----------
        dx : float | int
            The number of pixels to shift the image along the x-axis, applied to `self.shape[1]`
        dy : float | int
            The number of pixels to shift the image along the y-axis, applied to `self.shape[0]`

        Returns
        -------
        :class:`pynot.images.FitsImage`
            The shifted 2D image with the same dimensions as the old image.
        """
        if isinstance(dx, (int, np.integer)) and isinstance(dy, (int, np.integer)):
            use_interpolation = False
        else:
            use_interpolation = True

        if use_interpolation:
            pixsize_x = np.diff(self.x)[0]
            pixsize_y = np.diff(self.y)[0]
            self.header['CRPIX1'] += dx
            self.header['CRPIX2'] += dy
            return self.interpolate(self.x - dx*pixsize_x, self.y - dy*pixsize_y)

        else:
            self.header['CRPIX1'] += dx
            self.header['CRPIX2'] += dy
            values = {'header': self.header}
            attributes = ['data', 'error', 'mask']
            for attr in attributes:
                array2D = self.__getattribute__(attr)
                shifted = np.roll(array2D, shift=(dx, dy), axis=(1, 0))
                if dy == 0:
                    pass
                elif dy < 0:
                    shifted[dy:] = np.nan
                elif dy > 0:
                    shifted[:dy] = np.nan

                if dx == 0:
                    pass
                elif dx < 0:
                    shifted[:, dx:] = np.nan
                elif dx > 0:
                    shifted[:, :dx] = np.nan
                values[attr] = shifted
            return self.__class__(**values)


def image_operation(img1, img2, func):
    same_size = img1.shape == img2.shape
    if not same_size:
        raise ValueError(f"Cannot operate on images of different sizes!")

    new_data = func(img1.data, img2.data)
    new_mask = (img1.mask + img2.mask) >= 1
    if 'sub' in str(func) or 'add' in str(func):
        new_err = np.sqrt(img1.error**2 + img2.error**2)
    elif 'mult' in str(func) or 'div' in str(func):
        new_err = new_data * np.sqrt((img1.error/img1.data)**2 + (img2.error/img2.data)**2)
    else:
        raise ValueError(f"Invalid operator on FitsImage: {func}")

    return FitsImage(new_data, error=new_err, mask=new_mask, header=img1.header)


def imshift(image, dx=0, dy=0):
    return image.shift(dx, dy)


def resample(image: FitsImage, Nx, Ny):
    if Nx < 2 or Ny < 2:
        raise ValueError(f"Invalid number of pixels: {Nx=}, {Ny=}")
    newx = np.linspace(image.x.min(), image.x.max(), Nx)
    newy = np.linspace(image.y.min(), image.y.max(), Ny)
    return image.interpolate(newx, newy)


def image_mean(*images):
    return FitsImage(data=np.mean([img.data for img in images], axis=0),
                     error=np.mean([img.error for img in images], axis=0)/len(images),
                     mask=np.sum([img.error for img in images], axis=0) > 0,
                     header=images[0].header,
                     )

def image_median(*images):
    return FitsImage(data=np.median([img.data for img in images], axis=0),
                     error=np.median([img.error for img in images], axis=0)/len(images),
                     mask=np.sum([img.error for img in images], axis=0) > 0,
                     header=images[0].header,
                     )

def image_log(image):
    return FitsImage(data=np.log10(image.data),
                     error=image.error / (image.data * np.log(10)),
                     mask=image.mask,
                     header=image.header,
                     )
