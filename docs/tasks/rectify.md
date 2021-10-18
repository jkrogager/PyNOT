# PyNOT : rectify

This task will take a *raw* science image and apply a transformation to remove image distortions along the slit. This distortions are most visible in the associated arc line image where the emission lines of the calibration lamp show up as curved lines instead of straight lines. The task ``rectify`` performs the following sub-tasks:

- image edge detection
- arc lamp continuum subtraction
- arc line fitting: the 2D pixel table
- interpolation of 2D pixel table
- image transformation and wavelength linearization



## Image edge detection

The first step in the process is to detect the edges of the exposed region of the image. This is done by finding the location along the slit where the arc lines are no longer visible. This is illustrated in Fig. 1, where the significant curvature of the arc lines is visible as well.

![rectify_arc2d_limits](/Users/krogager/Projects/pynot_docs/rectify_arc2d_limits.png)
Fig. 1 – *Bias and overscan subtracted arc lamp image. The white dashed lines mark the edges of the exposed area.*

The arc and science images are then trimmed and rotated to have dispersion along the horizontal axis. This is done to speed up the calculations, as operations on rows in Python are slightly faster than operations by columns. The resulting trimmed arc image is shown in Fig. 2 together with examples of spectra along three rows (one at either end and one in the middle).



![rectify_rotate_spectra](/Users/krogager/Projects/pynot_docs/rectify_rotate_spectra.png)
Fig. 2 – *Trimmed and rotated arc line image (left) with three spectra from different rows (right). The red, black and blue spectra correspond to the three colored slices along rows in the 2D image on the left. A few emission lines have been identified in the central spectrum and their positions along the dispersion axis are projected to the other plots (as dotted lines). This clearly illustrated the curvature leading to offsets in wavelength as function of image columns.*



## Arc lamp continuum subtraction

Now that the arc line image has been trimmed, we can start to fit the positions of the emission lines for each spatial row. But before the emission lines are fitted, the smoothly varying lamp continuum is subtracted in order not to bias the measurements of the line positions.

The lamp continuum is estimated for each row by masking out the emission lines using a median-filtering procedure. The resulting continuum regions are then interpolated using a Chebyshev polynomial of order given by the parameter ``order_bg``.  Lastly, the continuum estimate is subtracted from the given row. 



## Arc line fitting : the 2D pixel table







## Image transformation and wavelength linearization

The wavelength solution at each spatial row is then obtained as:

​	$\lambda_i (x) = C_{\tt order\_wl}(x\ |\ \lambda_{\mathrm{ref}}, x_{{\rm ref}, i})$

where $C_{\tt order\_wl}$ refers to a Chebyshev polynomium of order determined by the parameter ``order_wl``, given the interpolation reference wavelengths, $\lambda_{\rm ref}$ and the arc line position for the given row, $x_{{\rm ref}, i}$ (see above). 

In order to rectify the science frame, each row along with its associated error and quality mask is now interpolated onto the reference wavelength grid. The central row is by default assumed as the reference position. Hence, all other rows will be aligned to the wavelengths defined by the central row (or whatever row was used to identify the arc lines). Values that fall outside the reference wavelength grid are set to 0.

