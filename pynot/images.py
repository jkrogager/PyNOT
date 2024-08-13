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
        xmin = np.max([np.min(ar) for ar in (img1.x, img2.x)])
        xmax = np.min([np.max(ar) for ar in (img1.x, img2.x)])
        ymin = np.max([np.min(ar) for ar in (img1.y, img2.y)])
        ymax = np.min([np.max(ar) for ar in (img1.y, img2.y)])
        xnew = np.arange(xmin, xmax, np.mean(np.diff(img1.x)))
        ynew = np.arange(ymin, ymax, np.mean(np.diff(img1.y)))
        new_points = tuple(np.meshgrid(xnew, ynew, indexing='ij'))
        data1 = RegularGridInterpolator((img1.x, img1.y), img1.data)(new_points)
        err1 = RegularGridInterpolator((img1.x, img1.y), img1.error)(new_points)
        mask1 = RegularGridInterpolator((img1.x, img1.y), img1.mask)(new_points)
        data2 = RegularGridInterpolator((img2.x, img2.y), img2.data)(new_points)
        err2 = RegularGridInterpolator((img2.x, img2.y), img2.error)(new_points)
        mask2 = RegularGridInterpolator((img2.x, img2.y), img2.mask)(new_points)
    else:
        xnew = img1.x
        ynew = img1.y
        data1 = img1.data
        err1 = img1.error
        mask1 = img1.mask
        data2 = img2.data
        err2 = img2.error
        mask2 = img2.mask

    new_data = func(data1, data2)
    new_mask = (mask1 + mask2) > 1
    if 'sub' in str(func) or 'add' in str(func):
        new_err = np.sqrt(err1**2 + err2**2)
    elif 'mult' in str(func) or 'div' in str(func):
        new_err = new_data * np.sqrt((err1/data1)**2 + (err2/data2)**2)
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
