
from dataclasses import dataclass
from itertools import cycle
import json
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt

from argparse import ArgumentParser
from astropy import units as u
from astropy.table import Table, QTable
import numpy as np
import sys
import logging
import os

from pynot.fitsio import load_fits_spectrum, save_fits_spectrum
from pynot.viewer.tablemodels import TableModel, ActiveTableModel, AbstractIndex
from pynot.viewer.spectrum import Spectrum, join_spectra, Template
from pynot.viewer.messages import QtLogHandler, LogViewerDialog
from pynot.viewer.targets import Target, TemplateTarget
from pynot.viewer.containers import QMEC, GenericFileContainer
from pynot.viewer.linelists import LineManagerDialog


color_list = cycle([
    "#3949AB",
    "#D81B60",
    "#009688",
    "#8E24AA",
    "#CCA000",
    "#202020",
])

here = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(here, 'templates')
DEFAULT_LINELIST_JSON = os.path.join(here, 'default_linelists.json')


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, files=None, assn_table=None, container_mode=False, width=None, height=None):
        super().__init__()

        self.setWindowTitle('Pynot Viewer')
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        with open(DEFAULT_LINELIST_JSON, 'r') as f:
            self.linelists = json.load(f)
        superset = []
        for lines in self.linelists.values():
            for line in lines:
                if line not in superset:
                    superset.append(line)
        self.linelists["All"] = superset
        self.line_objects = []
        self.active_lines = []

        self.all_targets = TableModel([])
        self.active_targets = ActiveTableModel([])
        self.container = None

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

        # --- Logging Setup ---
        self.log_dialog = LogViewerDialog(self)
        self.log_handler = QtLogHandler()
        self.log_handler.signaler.signal.connect(self.status.showMessage)
        self.log_handler.signaler.signal.connect(self.log_dialog.append_log)

        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', "%Y-%m-%d %H:%M:%S")
        self.log_handler.setFormatter(formatter)

        # Add the handler to the root logger or a specific logger
        logger = logging.getLogger()
        logger.addHandler(self.log_handler)
        logger.setLevel(logging.INFO)

        self.log_btn = QtWidgets.QToolButton(self)
        self.log_btn.setText("📜 View Log")
        self.log_btn.clicked.connect(self.log_dialog.show)
        self.status.addPermanentWidget(self.log_btn)

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

        # Create the vertical lines and labels
        for name, wl, vis in self.active_lines:
            line = pg.InfiniteLine(pos=wl, angle=90, pen='r', label=name,
                                   labelOpts={'position': 0.9, 'color': (200, 200, 200)})
            self.plot_graph.addItem(line)
            self.line_objects.append(line)

        if files is not None:
            if container_mode:
                self.load_container(files)

            else:
                for fname in files:
                    self.load_target_from_files([fname])

        if assn_table is not None:
            with open(assn_table) as tab:
                lines = tab.readlines()
            assoc = [line.strip().split(',') for line in lines]
            for row in assoc:
                self.load_target_from_files([fname.strip() for fname in row])

        if len(self.all_targets._data) > 0:
            self.add_active_target(0)

        if width and height:
            self.resize(int(width), int(height))

        # -- Define Keyboard Shortcuts
        self.shortcut_plus = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl++"), self)
        self.shortcut_plus.activated.connect(self.increase_smoothing)
        self.shortcut_plus.setContext(Qt.ApplicationShortcut)

        self.shortcut_minus = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+-"), self)
        self.shortcut_minus.activated.connect(self.decrease_smoothing)
        self.shortcut_minus.setContext(Qt.ApplicationShortcut)

        self.shortcut_zero = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+0"), self)
        self.shortcut_zero.activated.connect(self.reset_smoothing)
        self.shortcut_zero.setContext(Qt.ApplicationShortcut)

        self.shortcut_next = QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self)
        self.shortcut_next.activated.connect(self.plot_next_target)
        self.shortcut_next.setContext(Qt.ApplicationShortcut)

        self.shortcut_prev = QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self)
        self.shortcut_prev.activated.connect(self.plot_previous_target)
        self.shortcut_prev.setContext(Qt.ApplicationShortcut)
        self.current_index = None

    def load_linelist(self, name):
        """Clears the current view and loads a new set of lines."""
        # 1. Clear existing plot lines
        for line in self.line_objects:
            self.plot_graph.removeItem(line)
        self.line_objects.clear()
        
        self.active_lines = self.linelists[name]
        self.refresh_plot_lines()

    def load_container(self, filenames):
        if len(filenames) == 1:
            self.container = QMEC.read(filenames[0])
        else:
            self.container = GenericFileContainer(filenames)
        self.all_targets._data = self.container.view
        self.all_targets.layoutChanged.emit()
        self.target_table.resizeColumnsToContents()

    def clear_active_targets(self):
        rows = self.active_targets.rowCount()
        while self.active_targets.rowCount() > 0:
            self.remove_target(-1)

    def change_display_target(self, increment=+1):
        if self.current_index is None:
            self.current_index = self.target_table.currentIndex().row()
            logging.info(f"setting current index: {self.current_index}")

        self.current_index += increment
        # Check that index is in range:
        Nrows = self.all_targets.rowCount()
        if 0 <= self.current_index < Nrows:
            logging.info(f"setting current index: {self.current_index}")
            self.clear_active_targets()
            self.add_active_target(self.current_index)
        elif self.current_index < 0:
            logging.info(f"current index reached the limit: 0")
            self.current_index = 0
        else:
            logging.info(f"current index reached the limit: {Nrows - 1}")
            self.current_index = Nrows - 1

    def plot_next_target(self):
        self.change_display_target(+1)

    def plot_previous_target(self):
        self.change_display_target(-1)

    def reset_smoothing(self):
        index = self.active_table.currentIndex()
        target = self.active_targets._data[index.row()]
        target.smooth_factor = 1
        logging.info(f"Reset smooth level to: {target.smooth_factor} for target {target.name}")
        self.replot_target(target)

    def increase_smoothing(self):
        # Get active target
        index = self.active_table.currentIndex()
        target = self.active_targets._data[index.row()]
        target.smooth_factor += 2
        logging.info(f"Increase smooth level to: {target.smooth_factor} for target {target.name}")
        self.replot_target(target)

    def decrease_smoothing(self):
        # Get active target
        index = self.active_table.currentIndex()
        if index.row() >= self.active_targets.rowCount():
            return
        target = self.active_targets._data[index.row()]
        target.smooth_factor = max(1, target.smooth_factor - 2)
        logging.info(f"Decreased smooth level to: {target.smooth_factor} for target {target.name}")
        self.replot_target(target)

    def replot_target(self, target):
        for spectrum in target.spectra:
            spectrum.update_plot_data()

    def load_spectral_template(self):
        current_dir = TEMPLATE_DIR
        filters = "Spectral files (*.fits | *.fit | *.csv | *.txt | *.dat | *.spec)"
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open 2D Spectrum', current_dir, filters)
        fname = str(fname)
        if not fname:
            return

        template = TemplateTarget(
                        Template.read(fname)
                        )
        self.add_target(template)
        self.add_active_target(None, target_override=template)

    def load_target_from_files(self, files):
        target = Target()
        target.spectra = []
        for fname in files:
            spec = Spectrum.read(fname)
            if spec:
                target.add_spectrum(spec)
        if len(target.spectra) > 0:
            target.name = target.spectra[0].meta.get('OBJECT', 'None')
            self.add_target(target)

    def add_target(self, target):
        if isinstance(target, TemplateTarget):
            pass
        else:
            self.all_targets._data.append(target)
            self.all_targets.layoutChanged.emit()
            self.target_table.resizeColumnsToContents()
        logging.info(f"Loaded target: {target.name}")
        all_wl_units = [spec.wavelength.unit for spec in target.spectra]
        if len(set(all_wl_units)) > 1:
            try:
                for spec in target.spectra:
                    spec.wavelength = spec.wavelength.to('Angstrom')
                logging.info("Converted all wavelength units to Angstrom")
            except u.UnitConversionError:
                logging.warning("Could not convert units to Angstrom")

    def add_active_target(self, index, target_override=None):
        if isinstance(index, int):
            row = index
        elif index is None:
            row = None
        else:
            row = index.row()

        if target_override is None:
            if self.container is None:
                target = self.all_targets._data[row]
            else:
                target = self.container[row]
            self.target_table.selectRow(row)
        else:
            target = target_override

        self.active_targets._data.append(target)
        self.active_targets.layoutChanged.emit()
        self.active_table.resizeColumnsToContents()
        self.plot_spectrum(target)


    def plot_spectrum(self, target):
        color = next(color_list)
        for spec in target.spectra:
            spec.plot(self.plot_graph, color=color)


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
        table.doubleClicked.connect(self.show_target_details)
        table.setToolTip("Double click to show target details.\n"
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

        resample_button = QtWidgets.QPushButton("Resample target spectra", self)
        resample_button.clicked.connect(self.resample_target)
        resample_button.setToolTip("Resample all spectra of the selected target(s) onto a new wavelength grid.")
        resample_button.setWhatsThis("The resampling is done using flux-conservative resampling implemented "
                                     "in the python package `spectres`. The new wavelength grid can be controlled "
                                     "by the user in the pop-up window.")
        toolbar.addWidget(resample_button)

        new_template_btn = QtWidgets.QPushButton("Add Spectral Template", self)
        new_template_btn.clicked.connect(self.load_spectral_template)
        new_template_btn.setToolTip("Load a spectral template from a file")
        toolbar.addWidget(new_template_btn)

        self.addToolBar(toolbar)

        # Make redshift slider and line-list selection toolbar
        lines_toolbar = QtWidgets.QToolBar()
        lines_toolbar.setMovable(True)
        self.linelist_combo = QtWidgets.QComboBox()
        self.linelist_combo.addItems(self.linelists.keys())
        self.linelist_combo.currentTextChanged.connect(self.load_linelist)
        self.linelist_combo.setToolTip("Choose a list of emission or absorption lines to overlay")

        self.linelist_edit_btn = QtWidgets.QPushButton("Edit lines")
        self.linelist_edit_btn.setToolTip("Edit the selected linelist")
        self.linelist_edit_btn.clicked.connect(self.open_line_manager)

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_scale = 100000
        self.slider.setRange(int(-0.1*self.slider_scale), int(10*self.slider_scale))

        self.z_input = QtWidgets.QLineEdit("0.0")
        self.z_input.setFixedWidth(60)

        lines_toolbar.addWidget(self.linelist_combo)
        lines_toolbar.addWidget(self.linelist_edit_btn)
        lines_toolbar.addWidget(QtWidgets.QLabel("Redshift: "))
        lines_toolbar.addWidget(self.slider)
        lines_toolbar.addWidget(self.z_input)
        self.slider.valueChanged.connect(self.update_from_slider)
        self.z_input.editingFinished.connect(self.update_from_text)
        self.addToolBar(lines_toolbar)

    def open_line_manager(self):
        # Create dialog and pass current lines
        self.dialog = LineManagerDialog(self.active_lines, self)
        self.dialog.linesChanged.connect(self.update_line_data)
        self.dialog.show()
        self.dialog.raise_()

    def update_line_data(self, new_line_data):
        self.active_lines = new_line_data
        self.refresh_plot_lines()

    def refresh_plot_lines(self):
        # Clear existing
        for obj in self.line_objects:
            self.plot_graph.removeItem(obj)
        self.line_objects.clear()

        z = self.slider.value() / self.slider_scale
        for label, wl, visible in self.active_lines:
            if visible:
                obs_wl = wl * (1 + z)
                line = pg.InfiniteLine(pos=obs_wl, angle=90, label=label,
                                       labelOpts={
                                            'position': 0.95,
                                            # 'color': (0, 0, 255),
                                            'rotateAxis': (1, 0),
                                            'fill': (255, 255, 255, 200),  # set semi-transparent white background
                                            }
                                       )
                self.plot_graph.addItem(line)
                self.line_objects.append(line)

    def update_from_slider(self):
        z = self.slider.value() / self.slider_scale
        self.z_input.setText(f"{z:.5f}")
        self.refresh_plot_lines()

    def update_from_text(self):
        try:
            z = float(self.z_input.text())
            self.slider.setValue(int(z * self.slider_scale))
            self.refresh_plot_lines()
        except (ValueError, TypeError):
            logging.error("Invalid redshift input: {z}. Must be a numeral")

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
        target = self.all_targets._data[row]
        if vis is None:
            first_line = target.spectra[0].plot_line
            vis = not first_line.isVisible()

        for spec in target.spectra:
            spec.plot_line.setVisible(vis)
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
        logging.info(f"Removed target: {target.name}")


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


    def show_target_details(self, index):
        """Show details for the target at the given `row` from the Active Targets View"""
        if isinstance(index, int):
            row = index
        else:
            row = index.row()
        target = self.active_targets._data[row]
        target.show_details(self)


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
        details_menu_item = self.listMenu.addAction("Show details")
        details_menu_item.triggered.connect(lambda x: self.show_target_details(row))
        parentPosition = self.active_table.mapToGlobal(QtCore.QPoint(0, 0))
        self.listMenu.move(parentPosition + QPos)
        self.listMenu.show()

    def resample_target(self):
        current = self.active_table.selectedIndexes()
        logging.info(f"Currently selected target(s): {current}")

    def show_log_window(self, event):
        self.log_dialog.show()
        self.log_dialog.raise_()



# -- Start main loop

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
