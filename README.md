# PyNOT-redux
 A Data Processing Pipeline for ALFOSC at the Nordic Optical Telescope


PyNOT handles long-slit spectroscopic data (an extension for imaging data is currently being developed). The pipeline is implemented entirely in Python and can be run directly from the terminal. The main workflow is mostly automated (and can in fact be run fully automated) and includes a graphical user interface for certain tasks (such as line identification for wavelength calibration and spectral 1D extraction).

A special thank you goes out to Prof. Johan Fynbo for helpful discussions and feedback, and for introducing me to the Nordic Optical Telescope in the first place (back in 2012).

```diff
- The pipeline is currently in a testing stage!
  Feel free to test it on your own data and let me know if you find any issues.
  I'll respond as fast as possible.
```

## Installation
The pipeline can be installed using [pip](https://www.pypi.org):

    ]% pip install PyNOT-redux

and requires the following packages : `astroalign`, `astropy`, `astroscrappy`, `lmfit`, `matplotlib`, `numpy`, `PyQt5`, `PyYAML`, `scipy`, `sep`, and `spectres`. I want to give a huge shout out to all the developers of these packages. Thanks for sharing your work!


## Basic Usage
The pipeline is implemented as a series of modules or "tasks" that can either be executed individually or as a fully assembled pipeline. The available tasks can be shown by running:

    ]% pynot -h

and the input parameters for each task can be inspected by running:

    ]% pynot  task-name  -h

Three of the tasks have slightly special behavior:

 - `init` : classifies the data in the given input directory (or directories) and creates a default parameter file in YAML format.

 - `spex` : runs the full spectroscopic pipeline using the parameter file generated by `pynot init spex`. The full pipeline performs bias and flat field correction, wavelength calibration and rectifies the 2D spectrum, subtracts the sky background, corrects cosmic ray hits, flux calibrates the 2D spectrum and performs an automated optimal extraction of all objects identified in the slit (see more details below).

 The extracted 1D spectra are saved as a multi-extension FITS file where each object identified in the slit has its own extension:

    ```
    No.    Name      Ver    Type      Cards   Dimensions   Format
      0  PRIMARY       1 PrimaryHDU       4   ()      
      1  OBJ1          1 BinTableHDU    158   1026R x 3C   [D, D, D]
      2  OBJ2          1 BinTableHDU    158   1026R x 3C   [D, D, D]
      :    :           :     :           :         :           :    
      :    :           :     :           :         :           :    
    ```

 Each spectrum is saved as a Binary Table with three columns 'WAVE', 'FLUX', and 'ERR'. The header of each extension contains the information about the original image such as exposure time and instrument settings.

 - `phot` : runs the full spectroscopic pipeline using the parameter file generated by `pynot init phot`. The photometric pipeline performs bias and flat field correction, correction of cosmis ray hits, fringe correction, image registration and combination, source extraction and WCS calibration (using Gaia as reference). The final combined images are in units of counts per second (see more details below).
 If the observed frames are covered by the SDSS imaging foot print, PyNOT will perform an automatic self-calibration using SDSS photometry of sources in the field.



## Documentation

The full documentation is currently being compiled... stay tuned.



## Example: Spectroscopy
A standard example would be the reduction of the data from one night of observations. All the raw data would be located in a single folder - let's call it `raw_data/`. This folder will contain the necessary raw data: bias frames, flux standard star spectra, arc line frames, spectroscopic flat fields, and the object spectra. Any other data in the folder (imaging files, sky flats, acquisition images, slit images etc.) will be ignored in the pipeline.

A default reduction would require the following steps:


1. Create a parameter file and classify the data:
    `pynot init spex  raw_data  --pars night1.yml`

  This step creates the PyNOT File Classification (dataset.pfc) table which looks something like:

        # PyNOT File Classification Table

        # ARC_HeNe:
        #FILENAME             TYPE      OBJECT     EXPTIME  GRISM     SLIT      FILTER
         raw/ALzh010234.fits  ARC_HeNe  HeNe           3.0  Grism_#4  Slit_1.3  Open
         raw/ALzh010235.fits  ARC_HeNe  HeNe           3.0  Grism_#4  Slit_1.3  Open
         raw/ALzh010247.fits  ARC_HeNe  HeNe           3.0  Grism_#4  Slit_1.0  Open
         raw/ALzh010250.fits  ARC_HeNe  HeNe           3.0  Grism_#4  Slit_1.0  Open

        # BIAS:
        #FILENAME             TYPE  OBJECT     EXPTIME  GRISM        SLIT      FILTER
         raw/ALzh010001.fits  BIAS  bias-full  0.0  Open_(Lyot)  Open      Open
         raw/ALzh010002.fits  BIAS  bias-full  0.0  Open_(Lyot)  Open      Open
         raw/ALzh010003.fits  BIAS  bias-full  0.0  Open_(Lyot)  Open      Open

        ...

  If there are any bad frames (that you know of) you can delete or comment out (using #) the corresponding line to ignore the file in the pipeline.

  This step will also initiate a new parameter file with default values (default filename: 'options_spex.yml'). All available parameters of the steps of the pipeline are laid out in this file. Open the file with your favorite text editor and edit any other values as you see fit. A short description of the parameters is given in the file. For more detail, see the full documentation (coming soon).

  For now we will just focus on the interactive parameters: There are three tasks that can be used in interactive mode, which will start a graphical interface to allow the user more flexibility. These are: line identification (for wavelength calibration), extraction of the 1-dimensional spectra, and calculation of the response function. By default, these are all turned on. Note that the line identification can be defined in two ways:
  (i)  once for all grisms in the given dataset, this line identification information will then automatically be used for all objects observed with the given grism;
  or (ii) for each object in the dataset based on the arc file observed closest in time to the science frame. This provides more accurate rectification of the image, but the difference in low-resolution data is usually negligible.


2. Run the pipeline:
    `pynot spex night1.yml`

  This will start the full pipeline reduction of *all* objects identified in the dataset (with file classification `SPEC_OBJECT`). If you only want to reduce a few targets, you can specify these as: `pynot spex night1.yml --object TARGET1 TARGET2 ...` where the target names must match the value of the `OBJECT` keyword in the FITS headers.

  By default the pipeline creates separate output directories for each target where a detailed log file is saved. This file summarizes the steps of the pipeline and shows any warnings and output generated by the pipeline. By default, the pipeline also generates diagnostic plots of the 2D rectification, response function, sky subtraction and 1D extraction.

  The log is also printed to the terminal as the pipeline progresses. If you want to turn this off, you can run the pipeline with the `-s` (or `--silent`) option.


3. Verify the various steps of the data products and make sure that everything terminated successfully. You should pay special attention to the automated sky subtraction. This can be adjusted during the interactive extraction step, if necessary.


4. Now it's time to do your scientific analysis on your newly calibrated 1D and 2D spectra. Enjoy!



## Example: Imaging

A standard example would be the reduction of the data from one night of observations. All the raw data would be located in a single folder - let's call it `raw_night1/`. This folder will contain the necessary raw data: bias frames, flat field frames in all filters, flux standard star fields (if available), and the raw science images. Any other data in the folder (spectroscopic files, focus images etc.) will be ignored by the pipeline.
A basic automated reduction would require the following steps:

1. Create a parameter file and classify the data:
    `pynot init phot  raw_night1  --pars pars1.yml`

  This step will classify all the data in `raw_night1/` and create the PyNOT classification table (dataset.pfc). This step will also initiate a new parameter file with default values (the filename 'options_phot.yml' is used by default unless the `--pars` option is used). All available parameters of the steps of the pipeline are laid out in this file. Open the file with your favorite text editor and edit any other values as you see fit. A short description for each parameter is given in the file. For more detail, see the full documentation (coming soon).

2. Run the pipeline:
    `pynot phot pars1.yml`

  This will start the full pipeline reduction of *all* objects in *all* filters identified in the dataset (with file classification `IMG_OBJECT`). If you only want to reduce a subset of objects or filters, you can ignore files by editing the 'dataset.pfc' file. Deleting or commenting out (using #) a given line in the .pfc file will tell the pipeline to ignore the file on that line.
  The processed files are structured in sub-directories from the main working directory:

    ```
    working_dir/
         |- imaging/
               |- OBJECT_1/
               |     |- B_band/
               |     |- R_band/
               |     |- combined_B.fits
               |     |- combined_R.fits
               |     |...
               |
               |- OBJECT_2/
                     |- B_band/
                     |- R_band/
                     |- V_band/
                     |- combined_B.fits
                     |- combined_R.fits
                     |- combined_V.fits
                     |...
    ```
  The individual images for each filter of each target are kept in the desginated folders under each object, and are automatically combined. The combined image is in the folder of the given object. The last step of the pipeline as of now is to run a source extraction algorithm (SEP/SExtractor) to provide a final source table with aperture fluxes, a segmentation map as well as a figure showing the identified sources in the field.
  In each designated filter folder, the pipeline also produces a file log showing which files are combined into the final image as well as some basic image statistics: an estimate of the seeing, the PSF ellipticity, and the exposure time. This file can be used as input for further refined image combinations using the task `pynot imcombine  filelist_OBJECT_1.txt  new_combined_R.fits`. Individual frames can be commented out in the file log in order to exclude them in subsequent combinations. The combined images are given in units of counts per second.


3. Verify the various steps of the data products and make sure that everything terminated successfully.

4. Now it's time to do your scientific analysis on your newly calibrated images. Enjoy!
