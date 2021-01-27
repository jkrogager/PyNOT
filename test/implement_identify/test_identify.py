import sys

from identify_gui import GraphicInterface

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


def run_identify(fname, grism_name, tab_fname, line_list_fname, dispaxis, calib_dir, locked=True):
    # Launch App:
    qapp = QApplication(sys.argv)
    app = GraphicInterface(fname,
            grism_name=grism_name,
            pixtable=tab_fname,
            linelist_fname=linelist_fname,
            dispaxis=dispaxis,
            locked=locked)
    app.show()
    qapp.exec_()
    poly_order = int(app.poly_order.text())
    pixtab_fname = app.output_fname
    return poly_order, pixtab_fname


# This would be the main loop:
# Load Test Data:
tab_fname = '../../calib/grism4_pixeltable.dat'
grism_name = 'grism4'
linelist_fname = '../../calib/HeNe_linelist.dat'
dispaxis = 2
calib_dir = '/Users/krogager/coding/PyNOT/calib/'

arc_filenames = ['/Users/krogager/Data/NOT/MALS/ALzh010234.fits']
for fname in arc_filenames:
    pixtab_fname = calib_dir + '/%s_pixeltable.dat' % grism_name
    poly_order, pixtab_fname = run_identify(fname, grism_name, tab_fname, linelist_fname, dispaxis, calib_dir, locked=True)
    print(poly_order, pixtab_fname)


