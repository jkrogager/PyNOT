from astropy.table import Table
import logging
import numpy as np
from enum import IntFlag
from dataclasses import dataclass
import datetime


class DataFlag(IntFlag):
    """16-bit integer flag"""
    BAD_SPECTRUM      = 2**0    # Bad spectrum or no signal
    BAD_FLUXCALIB     = 2**1    # Problem with flux calibration
    BAD_SKYSUB        = 2**2    # Problem with sky subtraction
    WRONG_Z           = 2**3    # Incorrect redshift
    WRONG_CLASS       = 2**4    # Incorrect classification

    def get_flags(self):
        return [flag.name for flag in DataFlag if flag in self]

    def to_string(self, delimiter='; '):
        return delimiter.join(self.get_flags())


@dataclass
class TargetNote:
    name: str
    note: str
    filenames: list[str]
    date: str | None = None

    def __post_init__(self):
        self.date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:&S")


REDSHIFT_NAMES = ['REDSHIFT', 'Z_SPEC', 'ZBEST', 'Z_PIPE', 'ZSPEC', 'Z', 'ZFIT', 'Z_VI']
ID_NAMES = ['OBJ_NME', 'OBJ_UID', 'SPECUID', 'NAME', 'OBJECT', 'TARGET', 'TARGETID', 'ID', 'UID']
CLASS_NAMES = ['TYPE', 'ZBESTTYPE', 'SPECTYPE', 'CLASS', 'CLASSIFICATION', 'OBJ_CLS']


def load_redshift_table(filename, z_col=None, name_col=None, cls_col=None):
    """
    Load a table of redshifts for a given target, either FITS or ASCII format.
    The table must include a column with a name identifier matching target names
    or an identifier in the spectral meta-data of the targets. The table must
    also include a redshift column.
    If neither of these column names are given explicitly through `z_col` or
    `name_col`, the function will try to guess these based on common names:
    {'redshift', 'z_spec', 'zBest', 'z_pipe', 'z'} for redshifts, and:
    {'obj_uid', 'obj_nme', 'name', 'object', 'target', 'targetid'} for names.

    An optional cls_col can also be given to provide a target classification.

    Returns
    -------
    astropy.table.Table | None
        A redshift catalog with three columns: NAME, REDSHIFT, TYPE.
        If the loading of the table failed, the function returns `None`.
    """
    try:
        tab = Table.read(filename)
    except Exception:
        tab = Table.read(filename, format='ascii')

    # Convert all column names to upper case:
    for colname in tab.colnames:
        tab.rename_column(colname, colname.upper())

    if z_col:
        try:
            redshift = tab[z_col]
        except KeyError:
            logging.error(f"Could not load redshift column: {z_col}")
            return None
    else:
        for colname in REDSHIFT_NAMES:
            if colname in tab.colnames:
                redshift = tab[colname]
                break
        else:
            logging.error(f"Redshift catalog columns: {tab.colnames}")
            logging.error(f"Could not determine a redshift column. Try to give it explicitly with `--zcol=`")
            return None

    # Find the target names
    if name_col:
        try:
            name = tab[name_col]
        except KeyError:
            logging.error(f"Could not load name column: {name_col}")
            return None
    else:
        for colname in ID_NAMES:
            if colname in tab.colnames:
                name = tab[colname]
                name_column = colname
                break
        else:
            logging.error(f"Redshift catalog columns: {tab.colnames}")
            logging.error(f"Could not determine a name column. Try to give it explicitly with `--namecol=`")
            return None

    # Find target classification:
    if cls_col:
        try:
            spectype = tab[cls_col]
        except KeyError:
            logging.error(f"Could not load classification column: {cls_col}")
            spectype = ""
    else:
        for colname in CLASS_NAMES:
            if colname in tab.colnames:
                spectype = tab[colname]
                break
        else:
            logging.warning("No classification information identified in the redshift table")
            spectype = ""

    # Now we have a name and redshift column:
    redshift_table = Table()
    redshift_table['NAME'] = name
    redshift_table['REDSHIFT'] = redshift
    redshift_table['TYPE'] = spectype
    redshift_table.meta['NAME_COLUMN'] = name_column
    redshift_table.add_index('NAME')
    return redshift_table


def redshift_table_lookup(redshift_table, spectrum):
    """
    Find a matching redshift and spectral classification in the redshift table
    for the given spectrum based on the NAME column. If no match is found,
    the function returns (np.nan, "").
    """
    name_column = redshift_table.meta['NAME_COLUMN']
    if not spectrum.meta:
        logging.error("Could not find a redshift. Spectrum has no metadata")
        return np.nan, ""

    try:
        name = spectrum.meta[name_column]
    except KeyError:
        logging.error(f"Could not find a redshift. Spectrum has no {name_column} meta data")
        return np.nan, ""

    try:
        row = redshift_table.loc[name]
        if isinstance(row, Table):
            z = row['REDSHIFT'][0]
            spectype = row['TYPE'][0]
            logging.warning(f"Multiple matches for name: {name}. Redshifts: {row['REDSHIFT']}")
            logging.warning(f"Multiple redshifts: Choosing the first: z = {z}, spectral type: {spectype}")
        else:
            z = row['REDSHIFT']
            spectype = row['TYPE']
        return z, spectype

    except KeyError:
        logging.error(f"No matching redshift for name: {name}")
        return np.nan, ""
