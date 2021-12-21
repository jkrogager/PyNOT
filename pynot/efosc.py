
"""
Instrument definitions for NTT/EFOSC2
"""

import numpy as np
from os.path import dirname, abspath
from astropy.io import fits
from astropy.table import Table


# Define header keyword to use as object name
# OBJECT is not always reliable, TCS target name is more robust
target_keyword = 'OBJECT'
# or
# target_keyword = 'ESO OBS TARG NAME'

# path = '/Users/krogager/coding/PyNOT'
path = dirname(abspath(__file__))

# path to extinction table:
extinction_fname = path + '/calib/lasilla.ext'

# List grisms, slits and filters, and define filter translations if needed.

grism_translate = {'Gr#3': 'grism3',
                   'Gr#4': 'grism4',
                   'Gr#5': 'grism5',
                   'Gr#6': 'grism6',
                   'Gr#7': 'grism7',
                   'Gr#8': 'grism8',
                   'Gr#10': 'grism10',
                   'Gr#11': 'grism11',
                   'Gr#12': 'grism12',
                   'Gr#14': 'grism14',
                   'Gr#15': 'grism15',
                   'Gr#16': 'grism16',
                   'Gr#17': 'grism17',
                   'Gr#18': 'grism18',
                   'Gr#19': 'grism19',
                   'Gr#20': 'grism20'}

slit_translate = {'slit#1.0': 'slit_1.0',
                  'slit#1.2': 'slit_1.2',
                  'slit#1.5': 'slit_1.5',
                  'slit#2.0': 'slit_2.0',
                  'slit#0.7': 'slit_0.7',
                  'slit#0.5': 'slit_0.5',
                  'slit#5.0': 'slit_5.0',
                  }



def get_header(fname):
    with fits.open(fname) as hdu:
        primhdr = hdu[0].header
        if len(hdu) > 1:
            imghdr = hdu[1].header
            primhdr.update(imghdr)
    if primhdr['INSTRUME'] != 'EFOSC':
        print("[WARNING] - FITS file not originating from NTT/EFOSC!")
    return primhdr


def create_pixel_array(hdr, dispaxis):
    """Load reference array from header using CRVAL, CDELT, CRPIX along dispersion axis"""
    if dispaxis not in [1, 2]:
        raise ValueError("Dispersion Axis must be 1 (X-axis) or 2 (Y-axis)!")
    p = hdr['CRVAL%i' % dispaxis]
    s = hdr['CDELT%i' % dispaxis]
    r = hdr['CRPIX%i' % dispaxis]
    N = hdr['NAXIS%i' % dispaxis]
    # -- If data are from NOT then check for binning and rescale CRPIX:
    pix_array = p + s*(np.arange(N) - (r - 1))
    return pix_array


def get_binning(fname):
    hdr = fits.getheader(fname)
    ccd_setup = get_binning_from_hdr(hdr)
    return ccd_setup


def get_binning_from_hdr(hdr):
    binx = hdr['ESO DET WIN1 BINX']
    biny = hdr['ESO DET WIN1 BINY']
    read = hdr['ESO DET READ SPEED'].strip()
    ccd_setup = "%ix%i_%i" % (binx, biny, read)
    return ccd_setup


def get_filter(hdr):
    filter_name = hdr['ESO INS FILT1 NAME']
    if filter_name == 'Free':
        return "Open"
    return filter_name

def get_grism(hdr):
    grism_name = grism_translate(hdr['ESO INS GRIS1 NAME'])
    return grism_name

def get_slit(hdr):
    # HIERARCH ESO INS SLIT1 NAME    = 'slit#1.2' / Slit common name.
    slit_name = slit_translate(hdr['ESO INS SLIT1 NAME'])
    return slit_name

def get_airmass(hdr):
    """Return the average airmass at mid-exposure"""
    airm_start = hdr['ESO TEL AIRM START']
    airm_end = hdr['ESO TEL AIRM END']
    airmass = 0.5*(airm_start + airm_end)
    return airmass



#Example header:
# SIMPLE  =                    T / Written by IDL:  Sat Mar 17 04:23:05 2012
# BITPIX  =                  -32 / Bits per pixel
# NAXIS   =                    1 / Number of axes
# NAXIS1  =                 1030 /Number of positions along axis 1
# EXTEND  =                    F / File may contain extensions
# IRAF-TLM= '2012-02-02T07:27:58' / Time of last modification
# OBJECT  = 'CQ0831+0930'        /
# DATE    = '2012-02-02T07:27:58'
# IRAF-MAX=           0.000000E0  /  DATA MAX
# IRAF-MIN=           0.000000E0  /  DATA MIN
# ORIGIN  = 'NOAO-IRAF FITS Image Kernel July 2003' / FITS file originator
# DATE    = '2012-02-02T07:23:39' / Date FITS file was generated
# COMMENT NOST 100-2.0: Hanisch,R. et al. 2001, Astron. & Astrophys. 376, 559
#
#
# CRPIX1  =                   1. / Reference pixel
# CRVAL1  =      3804.7008200246 / Coordinate at reference pixel
# CTYPE1  = 'LINEAR  '           / Units of coordinate
# CTYPE2  = 'LINEAR  '           / Units of coordinate
# BUNIT   = 'erg/cm2/s/A'        / Units of data values
#
# MIDASFTP= 'IMAGE   '           / MIDAS File Type
#
# RA      =           127.927598 / MIDAS desc.: O_POS(1)
# DEC     =               9.5059 / MIDAS desc.: O_POS(2)
# EQUINOX =                2000. / MIDAS desc.: O_POS(3)
# DATE-OBS= '2011-11-22'         / MIDAS desc.: O_TIME(1)
# MJD-OBS =         55887.347975 / MIDAS desc.: O_TIME(4)
# TM-START=       30065.04000099 / MIDAS desc.: O_TIME(5)
# EXPTIME =             900.0005 / MIDAS desc.: O_TIME(7)
# TELESCOP= 'ESO-NTT '           / MIDAS desc.: TELESCOP(1)
# INSTRUME= 'EFOSC   '           / MIDAS desc.: INSTRUME(1)
# RADECSYS= 'FK5     '           / MIDAS desc.: RADECSYS(1)
# OBSERVER= 'UNKNOWN '           / MIDAS desc.: OBSERVER(1)
# HISTORY Converted from: crrdatar0019.fits                                      \
# HISTORY          REBIN/ROTATE spec1 rot R1 + NO                                \
# HISTORY
# UTC     =               30057. / 08:20:57.000 UTC at start (sec)
# LST     =              27715.8 / 07:41:55.800 LST at start (sec)
# PI-COI  = 'UNKNOWN '           / PI-COI name.
# CD1_1   =     4.11520195625661 / Translation matrix element.
# CD2_2   =                   1. / Translation matrix element.
# ORIGFILE= 'EFOSC_Spectrum326_0020.fits' / Original File Name
# ARCFILE = 'EFOSC.2011-11-22T08:21:05.040.fits' / Archive File Name
# CHECKSUM= 'fe1Tie1Rfe1Rfe1R'   / HDU checksum updated 2012-03-17T03:23:05
# DATASUM = '1362823054'         / data unit checksum updated 2012-03-17T03:23:05
# O_BZERO =               32768. / Original BZERO Value
# HIERARCH ESO ADA ABSROT END    =  -136.6974 / Abs rot angle at exp end (deg)
# HIERARCH ESO ADA ABSROT START  = -140.79743 / Abs rot angle at exp start (deg)
# HIERARCH ESO ADA GUID STATUS   = 'OFF     ' / Status of autoguider
# HIERARCH ESO ADA POSANG=    -71.698 / Position angle at start
# HIERARCH ESO DET BITS  =         16 / Bits per pixel readout
# HIERARCH ESO DET CHIP1 DATE    = '2000-08-30' / Date of installation [YYYY-MM-DD
# HIERARCH ESO DET CHIP1 ID      = 'ccd40   ' / Detector chip identification
# HIERARCH ESO DET CHIP1 INDEX   =          1 / Chip index
# HIERARCH ESO DET CHIP1 NAME    = 'LORAL   ' / Detector chip name
# HIERARCH ESO DET CHIP1 NX      =       2048 / # of pixels along X
# HIERARCH ESO DET CHIP1 NY      =       2048 / # of pixels along Y
# HIERARCH ESO DET CHIP1 PSZX    =        15. / Size of pixel in X
# HIERARCH ESO DET CHIP1 PSZY    =        15. / Size of pixel in Y
# HIERARCH ESO DET CHIP1 X       =          1 / X location in array
# HIERARCH ESO DET CHIP1 XGAP    =         0. / Gap between chips along x
# HIERARCH ESO DET CHIP1 Y       =          1 / Y location in array
# HIERARCH ESO DET CHIP1 YGAP    =         0. / Gap between chips along y
# HIERARCH ESO DET CHIPS =          1 / # of chips in detector array
# HIERARCH ESO DET DATE  = '2000-08-30' / Installation date
# HIERARCH ESO DET DEC   =         0. / Apparent 00:00:00.0 DEC at start
# HIERARCH ESO DET DID   = 'ESO-VLT-DIC.CCDDCS,ESO-VLT-DIC.FCDDCS' / Diction
# HIERARCH ESO DET EXP NO=       5401 / Unique exposure ID number
# HIERARCH ESO DET EXP RDTTIME   =      9.711 / image readout time
# HIERARCH ESO DET EXP TYPE      = 'Normal  ' / Exposure type
# HIERARCH ESO DET EXP XFERTIM   =     22.315 / image transfer time
# HIERARCH ESO DET FRAM ID       =          1 / Image sequencial number
# HIERARCH ESO DET FRAM TYPE     = 'Normal  ' / Type of frame
# HIERARCH ESO DET ID    = 'CCD FIERA - Rev: 3.96' / Detector system Id
# HIERARCH ESO DET NAME  = 'efosc - ccdefosc' / Name of detector system
# HIERARCH ESO DET OUT1 CHIP     =          1 / Chip to which the output belongs
# HIERARCH ESO DET OUT1 CONAD    =        1.1 / Conversion from ADUs to electrons
# HIERARCH ESO DET OUT1 GAIN     =       0.91 / Conversion from electrons to ADU
# HIERARCH ESO DET OUT1 ID       = 'L       ' / Output ID as from manufacturer
# HIERARCH ESO DET OUT1 INDEX    =          1 / Output index
# HIERARCH ESO DET OUT1 NAME     = 'L       ' / Description of output
# HIERARCH ESO DET OUT1 NX       =       1024 / valid pixels along X
# HIERARCH ESO DET OUT1 NY       =       1024 / valid pixels along Y
# HIERARCH ESO DET OUT1 OVSCX    =          0 / Overscan region in X
# HIERARCH ESO DET OUT1 OVSCY    =          6 / Overscan region in Y
# HIERARCH ESO DET OUT1 PRSCX    =          6 / Prescan region in X
# HIERARCH ESO DET OUT1 PRSCY    =          0 / Prescan region in Y
# HIERARCH ESO DET OUT1 RON      =        7.8 / Readout noise per output (e-)
# HIERARCH ESO DET OUT1 X=       2048 / X location of output
# HIERARCH ESO DET OUT1 Y=          1 / Y location of output
# HIERARCH ESO DET OUTPUTS       =          1 / # of outputs
# HIERARCH ESO DET OUTREF=          0 / reference output
# HIERARCH ESO DET RA    =         0. / Apparent 00:00:00.0 RA at start
# HIERARCH ESO DET READ CLOCK    = 'read L port Fast' / Readout clock pattern used
# HIERARCH ESO DET READ MODE     = 'normal  ' / Readout method
# HIERARCH ESO DET READ NFRAM    =          1 / Number of readouts buffered in sin
# HIERARCH ESO DET READ SPEED    = 'fastL   ' / Readout speed
# HIERARCH ESO DET SHUT ID       = 'ccd shutter' / Shutter unique identifier
# HIERARCH ESO DET SHUT TMCLOS   =      0.065 / Time taken to close shutter
# HIERARCH ESO DET SHUT TMOPEN   =      0.064 / Time taken to open shutter
# HIERARCH ESO DET SHUT TYPE     = 'Slit    ' / type of shutter
# HIERARCH ESO DET SOFW MODE     = 'Normal  ' / CCD sw operational mode
# HIERARCH ESO DET WIN1 BINX     =          2 / Binning factor along X
# HIERARCH ESO DET WIN1 BINY     =          2 / Binning factor along Y
# HIERARCH ESO DET WIN1 DIT1     = 900.000549 / actual subintegration time
# HIERARCH ESO DET WIN1 DKTM     =   900.1025 / Dark current time
# HIERARCH ESO DET WIN1 NDIT     =          1 / # of subintegrations
# HIERARCH ESO DET WIN1 NX       =       1030 / # of pixels along X
# HIERARCH ESO DET WIN1 NY       =       1030 / # of pixels along Y
# HIERARCH ESO DET WIN1 ST       = T / If T, window enabled
# HIERARCH ESO DET WIN1 STRX     =          1 / Lower left pixel in X
# HIERARCH ESO DET WIN1 STRY     =          1 / Lower left pixel in Y
# HIERARCH ESO DET WIN1 UIT1     =       900. / user defined subintegration time
# HIERARCH ESO DET WINDOWS       =          1 / # of windows readout
# HIERARCH ESO DPR CATG  = 'SCIENCE ' / Observation category
# HIERARCH ESO DPR TECH  = 'SPECTRUM' / Observation technique
# HIERARCH ESO DPR TYPE  = 'OBJECT  ' / Observation type
# HIERARCH ESO INS DATE  = '2000-06-16' / Instrument release date (yyyy-mm-d
# HIERARCH ESO INS DID   = 'ESO-VLT-DIC.EFOSC_ICS-1.21' / Data dictionary fo
# HIERARCH ESO INS DPOR POS      =         0. / Position move
# HIERARCH ESO INS DPOR ST       = F / Instrument depolarizer rotating.
# HIERARCH ESO INS FILT1 ID      = 'F1      ' / Filter unique id.
# HIERARCH ESO INS FILT1 NAME    = 'Free    ' / Filter name.
# HIERARCH ESO INS FILT1 NO      =          1 / Filter wheel position index.
# HIERARCH ESO INS GRIS1 ID      = 'F12     ' / OPTIi unique ID.
# HIERARCH ESO INS GRIS1 NAME    = 'Gr#6    ' / OPTIi name.
# HIERARCH ESO INS GRIS1 NO      =         12 / OPTIi slot number.
# HIERARCH ESO INS GRIS1 TYPE    = 'FILTER  ' / OPTIi element.
# HIERARCH ESO INS ID    = 'EFOSC/1.57' / Instrument ID.
# HIERARCH ESO INS MODE  = 'DEFAULT ' / Instrument mode used.
# HIERARCH ESO INS MOS1 LEN      =      1777. / MOSi slit length [arcsec]
# HIERARCH ESO INS MOS1 POSX     =      1002. / MOSi slit position [pix]
# HIERARCH ESO INS MOS1 POSY     =       992. / MOSi slit position [pix]
# HIERARCH ESO INS SLIT1 ENC     =     257870 / Slit absolute position [Enc].
# HIERARCH ESO INS SLIT1 LEN     =         0. / SLIT length [arcsec].
# HIERARCH ESO INS SLIT1 NAME    = 'slit#1.2' / Slit common name.
# HIERARCH ESO INS SLIT1 NO      =          5 / Slide position.
# HIERARCH ESO INS SLIT1 WID     =         0. / SLIT width [arcsec].
# HIERARCH ESO INS SWSIM = 'NORMAL  ' / Software simulation.
# HIERARCH ESO INS WP NAME       = 'HALF    ' / WP name, valid values are HALF/QUA
# HIERARCH ESO INS WP ST = 'OUT     ' / WP position.
# HIERARCH ESO OBS DID   = 'ESO-VLT-DIC.OBS-1.11' / OBS Dictionary
# HIERARCH ESO OBS EXECTIME      =          0 / Expected execution time
# HIERARCH ESO OBS GRP   = '0       ' / linked blocks
# HIERARCH ESO OBS ID    =  100298255 / Observation block ID
# HIERARCH ESO OBS NAME  = 'CQ0831+0930' / OB name
# HIERARCH ESO OBS OBSERVER      = 'UNKNOWN ' / Observer Name
# HIERARCH ESO OBS PI-COI ID     =      75259 / ESO internal PI-COI ID
# HIERARCH ESO OBS PI-COI NAME   = 'UNKNOWN ' / PI-COI name
# HIERARCH ESO OBS PROG ID       = '088.A-0098(A)' / ESO program identification
# HIERARCH ESO OBS START = '2011-11-22T08:16:26' / OB start time
# HIERARCH ESO OBS TARG NAME     = 'CQ0831+0930' / OB target name
# HIERARCH ESO OBS TPLNO =          2 / Template number within OB
# HIERARCH ESO OCS CON WCSFITS   = T / Setup OS to add WCS in fits  .
# HIERARCH ESO OCS DET1 IMGNAME  = 'EFOSC_Spectrum' / Data File Name.
# HIERARCH ESO TEL AIRM END      =      1.297 / Airmass at end
# HIERARCH ESO TEL AIRM START    =      1.316 / Airmass at start
# HIERARCH ESO TEL ALT   =     49.423 / Alt angle at start (deg)
# HIERARCH ESO TEL AMBI FWHM END =       0.57 / Observatory Seeing queried from AS
# HIERARCH ESO TEL AMBI FWHM START       =       0.67 / Observatory Seeing queried
# HIERARCH ESO TEL AMBI PRES END =      768.6 / Observatory ambient air pressure q
# HIERARCH ESO TEL AMBI PRES START       =      768.6 / Observatory ambient air pr
# HIERARCH ESO TEL AMBI RHUM     =        15. / Observatory ambient relative humi
# HIERARCH ESO TEL AMBI TEMP     =       15.6 / Observatory ambient temperature qu
# HIERARCH ESO TEL AMBI WINDDIR  =        35. / Observatory ambient wind directio
# HIERARCH ESO TEL AMBI WINDSP   =        4.8 / Observatory ambient wind speed que
# HIERARCH ESO TEL AZ    =    199.333 / Az angle at start (deg) S=0,W=90
# HIERARCH ESO TEL CHOP ST       = F / True when chopping is active
# HIERARCH ESO TEL DATE  = '2009-06-05T01:00:00' / TCS installation date
# HIERARCH ESO TEL DID   = 'ESO-VLT-DIC.TCS' / Data dictionary for TEL
# HIERARCH ESO TEL DOME STATUS   = 'FULLY-OPEN' / Dome status
# HIERARCH ESO TEL FOCU ID       = 'NB      ' / Telescope focus station ID
# HIERARCH ESO TEL FOCU LEN      =     38.501 / Focal length (m)
# HIERARCH ESO TEL FOCU SCALE    =       5.36 / Focal scale (arcsec/mm)
# HIERARCH ESO TEL FOCU VALUE    =      -3.38 / M2 setting (mm)
# HIERARCH ESO TEL GEOELEV       =      2377. / Elevation above sea level (m)
# HIERARCH ESO TEL GEOLAT=   -29.2584 / Tel geo latitute (+=North) (deg)
# HIERARCH ESO TEL GEOLON=   -70.7345 / Tel geo longitude (+=East) (deg)
# HIERARCH ESO TEL ID    = 'v 8.27+ ' / TCS version number
# HIERARCH ESO TEL MOON DEC      =  -11.00212 / -11:00:07.6 DEC (J2000) (deg)
# HIERARCH ESO TEL MOON RA       = 196.753158 / 13:07:00.7 RA (J2000) (deg)
# HIERARCH ESO TEL OPER  = 'I. CONDOR' / Telescope Operator
# HIERARCH ESO TEL PARANG END    =   -168.034 / Parallactic angle at end (deg)
# HIERARCH ESO TEL PARANG START  =   -162.976 / Parallactic angle at start (deg)
# HIERARCH ESO TEL TARG ALPHA    =   83142.36 / Alpha coordinate for the target
# HIERARCH ESO TEL TARG COORDTYPE= 'M       ' / Coordinate type (M=mean A=apparent
# HIERARCH ESO TEL TARG DELTA    =    93029.8 / Delta coordinate for the target
# HIERARCH ESO TEL TARG EPOCH    =      2000. / Epoch
# HIERARCH ESO TEL TARG EPOCHSYSTEM      = 'J       ' / Epoch system (default J=Ju
# HIERARCH ESO TEL TARG EQUINOX  =      2000. / Equinox
# HIERARCH ESO TEL TARG PARALLAX =         0. / Parallax
# HIERARCH ESO TEL TARG PMA      =         0. / Proper Motion Alpha
# HIERARCH ESO TEL TARG PMD      =         0. / Proper motion Delta
# HIERARCH ESO TEL TARG RADVEL   =         0. / Radial velocity
# HIERARCH ESO TEL TH M1 TEMP    =        -1. / M1 superficial temperature
# HIERARCH ESO TEL TRAK STATUS   = 'NORMAL  ' / Tracking status
# HIERARCH ESO TEL TSS TEMP8     =      14.38 / Temperature Sensing System
# HIERARCH ESO TPL DID   = 'ESO-VLT-DIC.TPL-1.9' / Data dictionary for TPL
# HIERARCH ESO TPL EXPNO =          1 / Exposure number within template
# HIERARCH ESO TPL ID    = 'EFOSC_spec_obs_Spectrum' / Template signature ID
# HIERARCH ESO TPL NAME  = 'Spectroscopy' / Template name
# HIERARCH ESO TPL NEXP  =          1 / Number of exposures within templat
# HIERARCH ESO TPL PRESEQ= 'EFOSC_spec_obs_Spectrum.seq' / Sequencer script
# HIERARCH ESO TPL START = '2011-11-22T08:20:40' / TPL start time
# HIERARCH ESO TPL VERSION       = '1.0     ' / Version of the template
#
# HISTORY  ESO-DESCRIPTORS START   ................
# HISTORY  'LHCUTS','R*4',1,4,'5E14.7'
# HISTORY   0.0000000E+00 0.0000000E+00 1.1754944E-38 7.8100000E+02
# HISTORY
# HISTORY  'ROTANG_FROM_CD-MATRIX','R*8',1,2,'3E23.15'
# HISTORY    1.251366203504492E+00  1.251366203504492E+00
# HISTORY
# HISTORY  ESO-DESCRIPTORS END     ................
# BANDID1 = 'spectrum - background average, weights variance, clean no'
# BANDID2 = 'raw - background average, weights none, clean no'
# BANDID3 = 'background - background average'
# BANDID4 = 'sigma - background average, weights variance, clean no'
# APNUM1  = '1 1 481.35 490.46'
# WCSDIM  =                    3
# CTYPE3  = 'LINEAR  '
# CD3_3   =                   1.
# LTM1_1  =                   1.
# LTM2_2  =                   1.
# LTM3_3  =                   1.
# WAT0_001= 'system=equispec'
# WAT1_001= 'wtype=linear label=Wavelength units=angstroms'
# WAT2_001= 'wtype=linear'
# WAT3_001= 'wtype=linear'
# DC-FLAG =                    0
# DCLOG1  = 'REFSPEC1 = arc_1d.fits'
# AIRMASS =                1.297
# EX-FLAG =                    0
# CA-FLAG =                    0
