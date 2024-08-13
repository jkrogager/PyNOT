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
        hdu_list.writeto(filename, overwrite=True)

    def __array__(self):
        return self.data

    # def __array_wrap__(self):

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
        values = {'header': self.header}
        attributes = ['data', 'error', 'mask']
        for attr in attributes:
            array2D = self.__getattribute__(attr)
            values[attr] = RegularGridInterpolator((self.y, self.x), array2D, bounds_error=False)(new_points)
        return self.__class__(**values)


def image_operation(img1, img2, func):
    same_size = img1.shape == img2.shape
    if same_size:
        dx = img1.x - img2.x
        dy = img1.y - img2.y
        pixsize_x = np.mean(np.diff(img1.x)) / 2
        pixsize_y = np.mean(np.diff(img1.y)) / 2
        array_shape_x = np.all(np.abs(dx) < 0.5*pixsize_x)
        array_shape_y = np.all(np.abs(dy) < 0.5*pixsize_y)
        if array_shape_x and array_shape_y:
            interpolate = False
        else:
            interpolate = True
    else:
        interpolate = True

    if interpolate:
        print(f"          - Interpolating images onto the same grid")
        xmin = np.max([np.min(ar) for ar in (img1.x, img2.x)])
        xmax = np.min([np.max(ar) for ar in (img1.x, img2.x)])
        ymin = np.max([np.min(ar) for ar in (img1.y, img2.y)])
        ymax = np.min([np.max(ar) for ar in (img1.y, img2.y)])
        xnew = np.arange(xmin, xmax, np.mean(np.diff(img1.x)))
        ynew = np.arange(ymin, ymax, np.mean(np.diff(img1.y)))

        img1 = img1.interpolate(xnew, ynew)
        img2 = img2.interpolate(xnew, ynew)
    else:
        xnew = img1.x
        ynew = img1.y

    new_data = func(img1.data, img2.data)
    new_mask = (img1.mask + img2.mask) > 1
    if 'sub' in str(func) or 'add' in str(func):
        new_err = np.sqrt(img1.error**2 + img2.error**2)
    elif 'mult' in str(func) or 'div' in str(func):
        new_err = new_data * np.sqrt((img1.error/img1.data)**2 + (img2.error/img2.data)**2)
    else:
        raise ValueError(f"Invalid operator on FitsImage: {func}")
    new_header = img1.header.copy()
    new_header.update(img2.header)
    new_header['NAXIS1'] = len(xnew)
    new_header['CRPIX1'] = 1
    new_header['CRVAL1'] = xnew.min()
    new_header['CD1_1'] = np.diff(xnew)[0]
    new_header['NAXIS2'] = len(ynew)
    new_header['CRPIX2'] = 1
    new_header['CRVAL2'] = ynew.min()
    new_header['CD2_2'] = np.diff(ynew)[0]
    new_header['CD1_2'] = new_header['CD2_1'] = 0

    return FitsImage(new_data, error=new_err, mask=new_mask, header=new_header)
