# pyNOT
 A Data Processing Pipeline for NOT/ALFOSC

The code is written entirely in Python and depends on the following libraries:
numpy, astropy, matplotlib, scipy, and astroscrappy.

In order to use the shell commands globally the following aliases can be defined (here in bash):
alias alfosc_calibs="python ABS_PATH/PyNOT/calibs.py"
alias alfosc_reduce="python ABS_PATH/PyNOT/PyNOT.py"
alias alfosc_sens="python ABS_PATH/PyNOT/sens.py"


The pipeline performs the following steps of the data processing:

 - Combination of bias frames to create a MASTER_BIAS frame.

 - Combination of spectral flat fields to create a MASTER_FLAT frame.
   The spectral flats are automatically normalized.

 - Background subtraction.

 - Wavelength calibration using a predefined look-up table to fit
   the dispersion solution automatically.

 - Calculation of the sensitivity function.

 - Flux calibration and extraction of 1D spectrum.


There are several steps in the pipeline that the user has to run manually.
A typical workflow will look something like this:

 > python calibs.py --bias bias_files.list  --flat flat_files.list

_bias_files.list_ and _flat_files.list_ are text files containing the filenames of the files to be processed.
This will create the files 'MASTER_BIAS.fits' and 'FLAT_COMBINED_grism_slit.fits'
and 'NORM_FLAT_grism_slit.fits', where *grism* and *slit* refer to the settings of the given flat frames.

 > python sens.py raw_flux_standard_frame.fits  raw_arc_frame.fits  --bias MASTER_BIAS.fits  --flat NORM_FLAT_grism_slit.fits

This will create the sensitivity function by default named as 'sens_grism.fits' where *grism* again refers to the given
setting used for the observations. Ths output name can be changed by using the option '-o output_name.fits'.

 > python PyNOT.py  raw_science_frame.fits  raw_arc_frame.fits  --bias MASTER_BIAS.fits  --flat NORM_FLAT_grism_slit.fits  --sens sens_grism.fits

This will perform cosmis ray rejection and clean the affected pixels using interpolation.
If no cosmic ray rejection should be performed, the user can pass the '--no-crr' option.
Background subtraction, wavelength calibration and flux calibration is performed on the 2D frame,
and lastly, a 1D spectrum is extracted using optimal extraction if there is enough signal to automatically
detect the trace. Otherwise a predefined box extraction will be performed. The parameters for the extraction
box can be controlled using option keywords (see the help description: 'python PyNOT.py --help').
