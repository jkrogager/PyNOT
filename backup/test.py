"""
GUI test
"""

import sys
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

import numpy as np


from scipy.optimize import curve_fit
from numpy.polynomial import Chebyshev
from astropy.io import fits


def NN_gaussian(x, bg, mu, sigma, logamp):
    """ One-dimensional modified non-negative Gaussian profile."""
    amp = 10**logamp
    return bg + amp * np.exp(-0.5*(x-mu)**4/sigma**2)

def fit_gaussian_center(x, y):
    bg = np.median(y)
    logamp = np.log10(np.nanmax(y))
    sig = 1.5
    mu = x[len(x)//2]
    p0 = np.array([bg, mu, sig, logamp])
    popt, pcov = curve_fit(NN_gaussian, x, y, p0)
    return popt[1], pcov[1, 1]


def fit_lines(arc1D, pix_table, dx=5, binning=1.):
    x = np.arange(len(arc1D))
    pixels = list()
    wl_vac = list()
    for pix, l_vac in pix_table:
        pix = pix/binning
        xlow = int(pix - dx)
        xhigh = int(pix + dx)
        pix_cen, cov = fit_gaussian_center(x[xlow:xhigh], arc1D[xlow:xhigh])
        if cov is not None:
            pixels.append(pix_cen)
            wl_vac.append(l_vac)

    pixels = np.array(pixels)
    wl_vac = np.array(wl_vac)
    return (pixels, wl_vac)



class ValidatorDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = super(ValidatorDelegate, self).createEditor(parent, option, index)
        if isinstance(editor, QtWidgets.QLineEdit):
            # editor.setValidator(QtGui.QDoubleValidator(editor))
            editor.setValidator(QtGui.QRegExpValidator(QtCore.QRegExp(r"^\s*\d*\s*\.?\d*$")))
        return editor



class GraphicInterface(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setWindowTitle('Test')
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        # Create Toolbar and Menubar
        toolbar = QtWidgets.QToolBar()
        toolbar.setMovable(False)
        toolbar_fontsize = QtGui.QFont()
        toolbar_fontsize.setPointSize(14)

        quit_action = QtWidgets.QAction("Close", self)
        quit_action.setFont(toolbar_fontsize)
        quit_action.setStatusTip("Quit the application")
        quit_action.triggered.connect(self.close)
        quit_action.setShortcut("ctrl+Q")
        toolbar.addAction(quit_action)


        # =============================================================
        # Start Layout:
        layout = QtWidgets.QHBoxLayout(self._main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.line_editor = QtWidgets.QLineEdit("3")
        self.line_editor.setValidator(QtGui.QIntValidator(1, 100))
        self.line_editor.returnPressed.connect(self.func)
        layout.addWidget(self.line_editor)

        # Create Table for Pixel Identifications:
        pixtab_layout = QtWidgets.QVBoxLayout()
        label_pixtab_header = QtWidgets.QLabel("Pixel Table\n",
                                               alignment=QtCore.Qt.AlignCenter)
        self.linetable = QtWidgets.QTableWidget()
        self.linetable.verticalHeader().hide()
        self.linetable.setColumnCount(2)
        self.linetable.setHorizontalHeaderLabels(["Pixel", "Wavelength"])
        self.linetable.setColumnWidth(0, 80)
        self.linetable.setColumnWidth(1, 90)
        self.linetable.setFixedWidth(180)

        delegate = ValidatorDelegate(self.linetable)
        self.linetable.setItemDelegate(delegate)

        pixtab_layout.addWidget(label_pixtab_header)
        pixtab_layout.addWidget(self.linetable)
        layout.addLayout(pixtab_layout)

        # Create Figure Canvas:
        right_layout = QtWidgets.QVBoxLayout()
        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.canvas.setFocus()
        right_layout.addWidget(self.canvas)
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.mpl_toolbar)
        layout.addLayout(right_layout)

        delegate.closeEditor.connect(self.canvas.setFocus)

        # Plot some random data:
        self.ax = self.fig.add_subplot(111)
        self.ax.plot([0, 1, 5, 10, 20], [-1, 2, -4, 5, -2])

    def on_key_press(self, e):
        if e.key == "a":
            self.add_point()

    def add_point(self):
        print("Select point...")
        point = self.fig.ginput(1)
        if not point:
            return
        x0, _ = point[0]

        row = self.linetable.rowCount()
        self.linetable.insertRow(row)

        item = QtWidgets.QTableWidgetItem("%.2f" % x0)
        item.setFlags(QtCore.Qt.ItemIsEnabled)
        item.setBackground(QtGui.QColor('lightgray'))
        self.linetable.setItem(row, 0, item)

        index = self.linetable.model().index(row, 1)
        self.linetable.edit(index)

    def print_val(self):
        rowPosition = self.linetable.rowCount()
        editor = self.linetable.cellWidget(rowPosition-1, 1)
        y_in = float(editor.text())
        print(" Table value: %.2f" % y_in)
        # and set focus back to canvas:
        self.canvas.setFocus()
        # Check which widget has focus:
        focus_widget = self.focusWidget()
        print(focus_widget.__class__)
        print("TableWidget has focus?  %r" % self.linetable.hasFocus())
        print("Cell editor has focus?  %r" % editor.hasFocus())

    def func(self):
        # dome some stuff
        print(self.line_editor.text())
        # and set focus back to canvas:
        self.canvas.setFocus()


# if __name__ == '__main__':
#     # Launch App:
#     qapp = QtWidgets.QApplication(sys.argv)
#     app = GraphicInterface()
#     app.show()
#     qapp.exec_()
