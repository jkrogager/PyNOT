
from dataclasses import dataclass
from itertools import cycle

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt

from argparse import ArgumentParser
from astropy import units as u
from astropy.table import Table, QTable
import numpy as np
import sys
from scipy.interpolate import UnivariateSpline as spline

from pynot.fitsio import load_fits_spectrum, save_fits_spectrum
from pynot.viewer.tablemodels import TableModel, ActiveTableModel, AbstractIndex
from pynot.viewer.spectrum import Spectrum, join_spectra

all_linestyles = cycle([
        QtCore.Qt.PenStyle.SolidLine,
        # QtCore.Qt.PenStyle.DashLine,
        # QtCore.Qt.PenStyle.DotLine,
        ])

color_list = cycle([
    "#3949AB",
    "#D81B60",
    "#009688",
    "#8E24AA",
    "#CCA000",
    "#202020",
])


@dataclass
class Template:
    x: np.ndarray
    template: np.ndarray
    interp: str = 'cubic'
    plot_line: pg.PlotItem = None

    def __call__(self, new_x):
        if self.interp == 'linear':
            return np.interp(new_x, self.x, self.template)
        elif self.interp == 'cubic':
            return spline(self.x, self.template, s=0, k=3)(new_x)
        else:
            return spline(self.x, self.template, s=0, k=2)(new_x)


@dataclass
class ModelSpectrum:
    expression: str
    plot_line: pg.PlotItem = None

    def __call__(self, x):
        return eval(self.expression, {'x': x})



class Target:
    def __init__(self, name=None, spectra=None, models=None, templates=None):
        if name is None:
            self.name = ""
        else:
            self.name = name

        if spectra is None:
            self.spectra = []
        else:
            self.spectra = spectra

        if models is None:
            self.models = []
        else:
            self.models = models

        if templates is None:
            self.templates = []
        else:
            self.templates = templates

    def remove_template(self, num):
        if num < len(self.templates):
            this = self.templates.pop(num)
            return this.plot_line

    def remove_model(self, num):
        if num < len(self.models):
            this = self.models.pop(num)
            return this.plot_line

    def remove_spectrum(self, num):
        if num < len(self.spectra):
            this = self.spectra.pop(num)
            return this.plot_line

    def get_all_plot_lines(self):
        lines = []
        for quantity in [self.spectra, self.models, self.templates]:
            for item in quantity:
                if item.plot_line is None:
                    continue
                lines.append(item.plot_line)
        return lines


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, files=None, assn_table=None, width=None, height=None):
        super().__init__()

        self.setWindowTitle('Pynot Viewer')
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        self.all_targets = TableModel([])
        self.active_targets = ActiveTableModel([])

        self.create_toolbar()

        self.main_menu = self.menuBar()
        self.file_menu = self.main_menu.addMenu("File")
        # Exit QAction
        exit_action = QtWidgets.QAction("Exit", self)
        # exit_action.setShortcut(QtGui.QKeySequence.Quit)
        exit_action.triggered.connect(self.close)

        self.file_menu.addAction(exit_action)

        # Status Bar
        self.status = self.statusBar()
        status_msg = ""
        self.status.showMessage(status_msg)


        # Main Layout
        self.main_layout = QtWidgets.QHBoxLayout(self._main)

        table_layout = QtWidgets.QVBoxLayout()
        self.active_table_label = self.make_table_label("Active targets")
        self.active_table = self.make_active_table()

        self.target_table = self.make_target_table()
        table_layout.addWidget(self.active_table_label)
        table_layout.addWidget(self.active_table, stretch=1)
        table_layout.addWidget(self.target_table, stretch=6)
        self.main_layout.addLayout(table_layout, stretch=1)

        self.plot_graph = pg.PlotWidget()
        self.main_layout.addWidget(self.plot_graph, stretch=5)
        self.plot_graph.setBackground("w")
        self.plot_legend = self.plot_graph.addLegend()
        self.plot_graph.setLabel("left", "Flux")
        self.plot_graph.setLabel("bottom", "Spectral axis")
        self.plot_lines = []
        self.plot_graph.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)

        if files is not None:
            for fname in files:
                self.add_target([fname])
            self.add_active_target(0)

        if assn_table is not None:
            with open(assn_table) as tab:
                lines = tab.readlines()
            assoc = [line.strip().split(',') for line in lines]
            for row in assoc:
                self.add_target([fname.strip() for fname in row])
            self.add_active_target(0)

        if width and height:
            self.resize(int(width), int(height))


    def add_target(self, files):
        target = Target()
        spectra = [Spectrum.read(fname) for fname in files]
        target.spectra = spectra
        target.name = spectra[0].meta.get('OBJECT', 'None')
        self.all_targets._data.append(target)
        self.all_targets.layoutChanged.emit()
        self.target_table.resizeColumnsToContents()


    def add_active_target(self, index):
        if isinstance(index, int):
            row = index
        else:
            row = index.row()

        self.active_targets._data.append(self.all_targets._data[row])
        self.active_targets.layoutChanged.emit()
        self.active_table.resizeColumnsToContents()
        self.plot_spectrum(self.active_targets._data[-1])


    def plot_spectrum(self, target):
        this_style = next(all_linestyles)
        color = next(color_list)
        for spec in target.spectra:
            pen = pg.mkPen(color=color, style=this_style)
            line = self.plot_graph.plot(spec.wavelength, spec.flux, pen=pen,
                                        name=target.name)
            spec.plot_line = line
        self.plot_graph.setLabel("left", f"Flux  [{spec.flux.unit}]")
        self.plot_graph.setLabel("bottom", f"Spectral Axis  [{spec.wavelength.unit}]")


    def make_table_label(self, text):
        label = QtWidgets.QLabel(text)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setFixedHeight(20)
        label.setStyleSheet("""color: #555; font-weight: bold; line-height: 100%;""")
        return label

    def make_target_table(self):
        table = QtWidgets.QTableView()
        table.setModel(self.all_targets)
        table.doubleClicked.connect(self.add_active_target)
        table.resizeColumnsToContents()
        table.setToolTip("Double click on row to load the given spectrum")
        return table

    def make_active_table(self):
        table = QtWidgets.QTableView(
                contextMenuPolicy=Qt.ContextMenuPolicy.CustomContextMenu,
                )
        table.setModel(self.active_targets)
        table.customContextMenuRequested.connect(self.listItemRightClicked)
        table.doubleClicked.connect(self.remove_target)
        table.setToolTip("Double click on row to remove the given spectrum.\n"
                         "Right click for more options...")
        return table


    def create_toolbar(self):
        # Create Toolbar
        toolbar = QtWidgets.QToolBar()
        toolbar.setMovable(False)
        toolbar_fontsize = QtGui.QFont()
        toolbar_fontsize.setPointSize(14)

        self.grid_radio_button = QtWidgets.QCheckBox('Grid Lines')
        self.grid_radio_button.stateChanged.connect(self.toggle_gridlines)
        toolbar.addWidget(self.grid_radio_button)

        join_button = QtWidgets.QPushButton("Merge target")
        join_button.clicked.connect(self.join_arms)
        toolbar.addWidget(join_button)

        self.addToolBar(toolbar)


    def toggle_gridlines(self):
        show_grid = self.grid_radio_button.isChecked()
        self.plot_graph.showGrid(x=show_grid, y=show_grid)

    def get_all_lines(self):
        all_lines = []
        for target in self.all_targets:
            all_lines += target.get_all_plot_lines()
        return all_lines

    def set_visible_obj(self, index, vis):
        if isinstance(index, int):
            row = index
        else:
            row = index.row()
        line_group = self.plot_lines[row]
        if vis is None:
            first_line = list(line_group.values())[0]
            vis = not first_line.isVisible()

        for arm, line in line_group.items():
            line.setVisible(vis)
        self.plot_legend.setVisible(False)
        self.plot_legend.setVisible(True)


    def remove_target(self, index):
        if isinstance(index, int):
            row = index
        else:
            row = index.row()
        target = self.active_targets._data.pop(row)
        self.active_targets.layoutChanged.emit()
        self.active_table.resizeColumnsToContents()
        for line in target.get_all_plot_lines():
            self.plot_graph.removeItem(line)


    def save_spectrum(self, index):
        if isinstance(index, int):
            row = index
        else:
            row = index.row()

        spec_group = self.active_targets._data[row]
        obj_name = [spectrum.meta['OBJ_NME'] for spectrum in spec_group.values()][0]
        if 'joined' in spec_group:
            current_dir = f'./qmost_{obj_name}_LJ1.fits'
        else:
            current_dir = f'./qmost_{obj_name}_L1-RGB.fits'
        filters = "FITS Files (*.fits *.fit)"
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Spectrum',
                                                         current_dir, filters)
        if not fname:
            return

        save_fits_spectrum(spec_group, filename=fname)


    def listItemRightClicked(self, QPos):
        index = self.active_table.currentIndex()
        if not index.isValid():
            return

        row = index.row()
        self.listMenu = QtWidgets.QMenu()
        remove_menu_item = self.listMenu.addAction("Remove")
        remove_menu_item.triggered.connect(lambda x: self.remove_target(index))
        show_menu_item = self.listMenu.addAction("Show all")
        show_menu_item.triggered.connect(lambda x: self.set_visible_obj(row, True))
        hide_menu_item = self.listMenu.addAction("Hide all")
        hide_menu_item.triggered.connect(lambda x: self.set_visible_obj(row, False))
        save_menu_item = self.listMenu.addAction("Save FITS")
        save_menu_item.triggered.connect(lambda x: self.save_spectrum(index))
        parentPosition = self.active_table.mapToGlobal(QtCore.QPoint(0, 0))
        self.listMenu.move(parentPosition + QPos)
        self.listMenu.show()

    def join_arms(self, index=None):
        if len(self.active_targets._data) <= 0:
            QtWidgets.QMessageBox.critical(self, "Cannot join spectra",
                                           "No valid spectra to join")
            return

        all_active_targets, valid_spectra = self.get_joinable_spectra()
        if not index:
            if len(valid_spectra) == 1:
                index = 0
            else:
                index = self.prompt_index_from_user()
                                
                if index is None:
                    return
        spec_group = self.active_targets._data[index]
        joined_spectrum = join_spectra(spec_group, scale=False)
        joined_group = {'joined': joined_spectrum}
        self.active_targets._data.append(joined_group)
        self.active_targets.layoutChanged.emit()
        self.active_table.resizeColumnsToContents()
        self.plot_spectrum(joined_group)


    def get_joinable_spectra(self):
        role = Qt.ItemDataRole.DisplayRole
        all_active_targets = [self.active_targets.data(AbstractIndex(i, 0), role)
                              for i in range(len(self.active_targets._data))]
        valid_spectra = []
        already_joined = []
        for label in all_active_targets:
            if 'JOINED' in label:
                already_joined.append(label)
            else:
                valid_spectra.append(label)

        for label in already_joined:
            root = label.replace('JOINED', 'spec')
            if root in valid_spectra:
                valid_spectra.remove(root)
        return all_active_targets, valid_spectra

    def prompt_index_from_user(self):
        all_active_targets, valid_spectra = self.get_joinable_spectra()
        message = "Please select one of the valid objects:"

        if len(valid_spectra) == 0:
            QtWidgets.QMessageBox.critical(self, "Cannot join spectra",
                                           "No valid spectra to join")
            return

        item, ok = QtWidgets.QInputDialog().getItem(self, "Select Spectra to Join",
                                                    message, valid_spectra,
                                                    0, False)
        if item and ok:
            return all_active_targets.index(item)



def start_gui(args):
    
    app = QtWidgets.QApplication(sys.argv)
    screenSize = app.primaryScreen().size()
    ratio = 0.85
    main = MainWindow(args.files,
                      assn_table=args.table,
                      width=ratio*screenSize.width(),
                      height=ratio*screenSize.height(),
                      )
    main.show()
    app.exec()


def main():
    parser = ArgumentParser(prog='PyNOT View',
                            description='PyNOT Spectral Viewer')
    parser.add_argument("files", type=str, nargs='?',
                        help="Filenames of spectral data to load. Each file is loaded as one target")
    parser.add_argument("-t", "--table", type=str,
                        help="Filename of association table, all files in one row are loaded as a single target")

    args = parser.parse_args()
    start_gui(args)


if __name__ == '__main__':
    main()
