
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
import glob
import logging
import os
import warnings

from pynot.fitsio import load_fits_spectrum, save_fits_spectrum
from pynot.viewer.notes import load_redshift_table, redshift_table_lookup, TargetNote, DataFlag, write_notes_to_file
from pynot.viewer.tablemodels import TableModel, ActiveTableModel, AbstractIndex
from pynot.viewer.spectrum import Spectrum, join_spectra, Template
from pynot.viewer.messages import QtLogHandler, LogViewerDialog
from pynot.viewer.targets import Target
from pynot.viewer.models import TemplateTarget
from pynot.viewer.containers import QMEC, GenericFileContainer
from pynot.viewer.linelists import LineManagerDialog


# if sys.platform == 'win32':
#     # This tells Windows to treat this as a unique application
#     import ctypes
#     myappid = 'pynot.viewer.astro.1.0'
#     ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

# # --- If MacOS:
# # Check if we are on Mac
# elif sys.platform == 'darwin':
#     # This helps macOS associate the icon with the process
#     from Foundation import NSBundle
#     bundle = NSBundle.mainBundle()
#     if bundle:
#         info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
#         if info:
#             info['CFBundleName'] = "PyNOT Viewer"


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
main_icon_path = os.path.join(here, 'icons/main.png')
warnings.simplefilter('ignore', u.UnitsWarning)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, files=None, assn_table=None, container_mode=False, width=None, height=None,
                 redshift_table=None, z_col=None, name_col=None, cls_col=None):
        super().__init__()

        self.setWindowTitle('Pynot Viewer')
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)
        self.setWindowIcon(QtGui.QIcon(main_icon_path))

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
        self.linePen = pg.mkPen(color=(100, 100, 100, 150))

        self.all_targets = TableModel([])
        self.active_targets = ActiveTableModel([])
        self.container = None
        self.redshift_table: Table = None
        self.target_notes: dict[TargetNote] = {}
        self.target_flags: dict[DataFlag] = {}
        self.notes_filename = None

        if os.path.exists("pynot_backup_target_notes.json"):
            N_backup = len(glob.glob('pynot_backup_target_notes*.json'))
            self.notes_backup_filename = f"pynot_backup_target_notes{N_backup:03}.json"
        else:
            self.notes_backup_filename = "pynot_backup_target_notes.json"

        self.create_toolbar()
        self.create_notes_toolbar()
        self.create_flags_toolbar()

        self.create_menubar()

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

        self.proxy = pg.SignalProxy(self.plot_graph.scene().sigMouseMoved, rateLimit=60, slot=self.get_cursor_coordinates)
        self.coordinate_label = QtWidgets.QLabel("(λ, flux)")
        self.status.addPermanentWidget(self.coordinate_label)
        self.status.addPermanentWidget(self.log_btn)

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

        if redshift_table:
            self.redshift_table = load_redshift_table(redshift_table,
                                                      z_col=z_col, name_col=name_col, cls_col=cls_col)
            if self.redshift_table is not None:
                logging.info(f"Loaded redshift catalog: {redshift_table}")
                self.linelist_combo.setCurrentText("Common Lines")

        if len(self.all_targets._data) > 0:
            self.add_active_target(0)

        if width and height:
            self.resize(int(width), int(height))

        # State variables
        self.selection_mode = False
        self.active_region = None
        self.is_dragging = False
        self.start_x = None
        self.data_stats_labels = []
        # Connect to the Scene signals
        self.scene = self.plot_graph.scene()
        self.scene.sigMouseClicked.connect(self.on_mouse_click)
        self.scene.sigMouseMoved.connect(self.on_mouse_move)

        # -- Define Keyboard Shortcuts
        # Smoothing:
        self.shortcut_plus = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl++"), self)
        self.shortcut_plus.activated.connect(self.increase_smoothing)
        self.shortcut_plus.setContext(Qt.ApplicationShortcut)

        self.shortcut_minus = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+-"), self)
        self.shortcut_minus.activated.connect(self.decrease_smoothing)
        self.shortcut_minus.setContext(Qt.ApplicationShortcut)

        self.shortcut_zero = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+0"), self)
        self.shortcut_zero.activated.connect(self.reset_smoothing)
        self.shortcut_zero.setContext(Qt.ApplicationShortcut)

        # Navigation of spectra:
        self.shortcut_next = QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self)
        self.shortcut_next.activated.connect(self.plot_next_target)
        self.shortcut_next.setContext(Qt.ApplicationShortcut)

        self.shortcut_prev = QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self)
        self.shortcut_prev.activated.connect(self.plot_previous_target)
        self.shortcut_prev.setContext(Qt.ApplicationShortcut)
        self.current_index = None

        # Select analysis region:
        self.shortcut_region = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+R"), self)
        self.shortcut_region.activated.connect(self.toggle_selection_mode)


    def toggle_selection_mode(self):
        self.selection_mode = True
        # Disable default panning/zooming
        self.plot_graph.setMouseEnabled(x=False, y=False)
        self.plot_graph.setCursor(Qt.CrossCursor)
        logging.info("Region Selection Activated: Click twice on the plot to select ranges in wavelength...")

    def on_mouse_click(self, event):
        if event.double():
            vb = self.plot_graph.plotItem.vb
            scene_pos = event.scenePos()

            if self.active_region is not None:
                # Check if the double-click happened inside the region bounds
                low, high = self.active_region.getRegion()
                plot_pos = vb.mapSceneToView(scene_pos)
                
                if low <= plot_pos.x() <= high:
                    self.plot_graph.removeItem(self.active_region)
                    self.active_region = None
                    label = self.data_stats_labels.pop()
                    self.plot_graph.removeItem(label)
                    logging.info("Region removed")
                    return

        if not self.selection_mode:
            return

        # Map scene coordinates to plot coordinates
        vb = self.plot_graph.plotItem.vb
        scene_pos = event.scenePos()
        plot_pos = vb.mapSceneToView(scene_pos)

        if not self.is_dragging:
            # FIRST CLICK: Start the region
            self.is_dragging = True
            self.start_x = plot_pos.x()
            
            if self.active_region:
                self.plot_graph.removeItem(self.active_region)
            
            self.active_region = pg.LinearRegionItem([self.start_x, self.start_x],
                                                     brush=(200, 200, 200, 100))
            self.plot_graph.addItem(self.active_region)
        else:
            # SECOND CLICK: Finish the region
            self.is_dragging = False
            self.selection_mode = False
            
            # Finalize stats
            low, high = self.active_region.getRegion()
            self.calculate_stats(low, high)
            
            # Restore defaults
            self.plot_graph.setMouseEnabled(x=True, y=True)
            self.plot_graph.setCursor(Qt.ArrowCursor)

    def on_mouse_move(self, scene_pos):
        if self.selection_mode and self.is_dragging:
            vb = self.plot_graph.plotItem.vb
            plot_pos = vb.mapSceneToView(scene_pos)
            self.active_region.setRegion([self.start_x, plot_pos.x()])

    def calculate_stats(self, x_min, x_max):
        # Filter data within the bounds
        for target in self.active_targets._data:
            if isinstance(target, TemplateTarget):
                continue

            for spec in target.spectra:
                x = spec.wavelength.value
                mask = (x >= x_min) & (x <= x_max)
                selected_y = spec.flux.value[mask]

                if len(selected_y) == 0:
                    continue
                stats = {
                        "Mean   ": np.nanmean(selected_y),
                        "Median ": np.nanmedian(selected_y),
                        "Std Dev": np.nanstd(selected_y),
                        "MAD    ": np.nanmedian(np.abs(selected_y - np.nanmedian(selected_y))),
                        "PTV    ": np.ptp(selected_y),
                    }
                stats_text = f"--- Stats for Range [{x_min:.1f} : {x_max:.1f}] ---\n"
                for k, v in stats.items():
                    stats_text += f"{k}: {v:.3e}\n"
                logging.info(stats_text)
                logging.info(f"Calculated region statistics: [{x_min:.1f} : {x_max:.1f}]")
                color = 'black'
                box_color = QtGui.QColor(250, 250, 250, 156)
                box_font = QtGui.QFont('Courier New')
                text = pg.TextItem(text=stats_text, color=color, fill=box_color, border='gray')
                text.setPos(x_max, np.nanmean(selected_y))
                text.setFont(box_font)
                self.plot_graph.addItem(text)
                self.data_stats_labels.append(text)

    def get_cursor_coordinates(self, event):
        vb = self.plot_graph.getViewBox()
        position = event[0]
        if vb.sceneBoundingRect().contains(position):
            mousePoint = vb.mapSceneToView(position)
            self.coordinate_label.setText(f"{mousePoint.x():.2f}, {mousePoint.y():.2e}")
        else:
            self.coordinate_label.setText("(λ, flux)")

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

        self.current_index += increment
        # Check that index is in range:
        Nrows = self.all_targets.rowCount()
        if 0 <= self.current_index < Nrows:
            self.clear_active_targets()
            logging.info(f"setting current index: {self.current_index}")
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

        # Scale the template to the median of data if a spectrum is already loaded:
        if len(self.active_targets._data) > 0:
            active_targets = self.active_targets._data
            active_specflux = []
            for targ in active_targets:
                active_specflux += [np.array(spec.flux.value) for spec in targ.spectra if isinstance(spec, Spectrum)]
            if len(active_specflux) > 0:
                mean_flux = np.nanmedian([np.nanmedian(flux) for flux in active_specflux])
                f0 = np.nanmedian(template.spectra[0].flux)
                if np.isfinite(mean_flux):
                    template.spectra[0].scale_flux(0, mean_flux / f0)
                    logging.info(f"Rescaled template to median flux of data: {mean_flux:.3e}")
        self.add_target(template)
        self.add_active_target(None, target_override=template)
        dialog = template.show_details(self)
        dialog.show()
        dialog.raise_()

    def load_target_from_files(self, files):
        target = Target()
        target.spectra = []
        for fname in files:
            spec = Spectrum.read(fname)
            if spec:
                target.add_spectrum(spec)
        if len(target.spectra) > 0:
            logging.info(repr(target.spectra[0].meta))
            target_name = target.spectra[0].meta.get('OBJ_NME', 'None')
            if target_name == 'None':
                target.spectra[0].meta.get('OBJ_UID', 'None')
            target.name = target_name
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
        if row is not None:
            self.current_index = row
            index = self.all_targets.index(row, 0)
            self.target_table.setCurrentIndex(index)
        self.active_table.resizeColumnsToContents()
        self.plot_spectrum(target)

        if self.redshift_table and len(target.spectra) > 0:
            z, spectype = redshift_table_lookup(self.redshift_table, target.spectra[0])
            self.update_redshift(z)
        self.sync_target_notes()
        self.sync_target_flags()
        target_flag: DataFlag = self.target_flags.get(target.name, DataFlag(0))
        if DataFlag.Z_VI_CONFIRM in target_flag:
            target_note = self.target_notes.get(target.name, None)
            if target_note is None:
                logging.error(f"No target note for {target.name} but Z_VI_CONFIRM is set?!")
                return
            self.z_input.setText(f"{target_note.redshift}")
            self.update_from_text()

    def plot_spectrum(self, target):
        color = next(color_list)
        for spec in target.spectra:
            spec.plot(self.plot_graph, color=color)
        self.plot_graph.setFocus()

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

    def sync_target_notes(self):
        targets_in_view = self.get_real_active_targets()

        if len(targets_in_view) > 1:
            logging.error("Cannot add notes to more than one target at a time. Remove other targets.")
            note_text = "Cannot add notes when more than one target is loaded."
            self.note_input.setReadOnly(True)
            self.note_input.setText(note_text)
            return
        elif len(targets_in_view) == 0:
            return

        target = targets_in_view[0]
        if target.name in self.target_notes:
            target_note = self.target_notes[target.name]
            note_text = target_note.note
        else:
            note_text = ""
        self.note_input.setReadOnly(False)
        self.note_input.setText(note_text)

    def sync_target_flags(self):
        targets_in_view = self.get_real_active_targets()
        if len(targets_in_view) > 1:
            for editor in self.flag_inputs:
                editor.setCheckable(False)
            return
        elif len(targets_in_view) == 0:
            return

        target = targets_in_view[0]
        for editor in self.flag_inputs:
            editor.setCheckable(True)

        target_flag = self.target_flags.get(target.name, DataFlag(0))

        for checkbox, val in zip(self.flag_inputs, DataFlag):
            if val in target_flag:
                checkbox.blockSignals(True)
                checkbox.setChecked(True)
                checkbox.blockSignals(False)
            else:
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.blockSignals(False)

    def save_notes(self):
        if len(self.active_targets._data) != 1:
            logging.error("Cannot add notes to more than one target at a time. Remove other targets.")
            return

        note_text = self.note_input.text()
        target: Target = self.active_targets._data[0]
        target_note = TargetNote.from_target(target, note=note_text)

        target_flag = self.target_flags.get(target.name, DataFlag(0))
        if DataFlag.Z_VI_CONFIRM in target_flag:
            this_redshift = float(self.z_input.text())
            target_note.redshift = this_redshift

        if DataFlag.CLASS_VI_CONFIRM in target_flag:
            logging.warning("Object classification not yet implemented")
            # target_note.spectype = ""

        self.target_notes[target.name] = target_note
        self.write_notes(self.notes_backup_filename, backup=True)

    def save_flags(self):
        if len(self.active_targets._data) != 1:
            logging.error("Cannot add flags to more than one target at a time. Remove other targets.")
            return
        target_flag = DataFlag(0)
        for checkbox, val in zip(self.flag_inputs, DataFlag):
            if checkbox.isChecked():
                target_flag |= val

        target: Target = self.active_targets._data[0]
        self.target_flags[target.name] = target_flag
        target_note = self.target_notes.get(target.name, None)

        if DataFlag.Z_VI_CONFIRM in target_flag:
            if target.name not in self.target_notes:
                target_note = TargetNote.from_target(target, "Visually updated redshift")
                target_note.redshift = float(self.z_input.text())
                self.target_notes[target.name] = target_note
        elif target_note and target_note.note == "Visually updated redshift":
            target_note.note = ""
            self.note_input.setText("")

    def create_notes_toolbar(self):
        notes_toolbar = QtWidgets.QToolBar()
        self.note_input = QtWidgets.QLineEdit("")
        self.note_input.editingFinished.connect(self.save_notes)
        notes_toolbar.addWidget(QtWidgets.QLabel("Notes: "))
        notes_toolbar.addWidget(self.note_input)
        self.addToolBar(Qt.BottomToolBarArea, notes_toolbar)

    def create_flags_toolbar(self):
        flags_toolbar = QtWidgets.QToolBar()
        flags_toolbar.addWidget(QtWidgets.QLabel("Data Flags: "))
        self.flag_inputs = []
        for val in DataFlag:
            editor = QtWidgets.QCheckBox(val.name)
            editor.clicked.connect(self.save_flags)
            self.flag_inputs.append(editor)
            flags_toolbar.addWidget(editor)
        self.addToolBar(Qt.RightToolBarArea, flags_toolbar)

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
        self.slider_scale = 100_000
        self.slider.setRange(int(-0.1*self.slider_scale), int(7*self.slider_scale))

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

        # Determine data limits on x-axis:
        active_spectra = self.get_active_spectra()
        wl_arrays = [spec.wavelength for spec in active_spectra]
        try:
            data_xmin = np.min(wl_arrays)
            data_xmax = np.max(wl_arrays)
        except Exception:
            data_xmin = 0
            data_xmax = 0

        z = self.slider.value() / self.slider_scale
        for label, wl, visible in self.active_lines:
            if visible:
                obs_wl = wl * (1 + z)
                line = pg.InfiniteLine(pos=obs_wl, angle=90, label=label,
                                       pen=self.linePen,
                                       labelOpts={
                                            'position': 0.9,
                                            'rotateAxis': (1, 0),
                                            'color': (56, 56, 56, 200),
                                            'fill': (255, 255, 255, 100),  # set semi-transparent white background
                                            'movable': True,
                                            }
                                       )
                if (obs_wl < data_xmin) or (obs_wl > data_xmax):
                    line.setVisible(False)
                else:
                    line.setVisible(True)
                self.plot_graph.addItem(line)
                self.line_objects.append(line)
                line.label.setPosition(0.9)

    def update_redshift(self, z):
        if np.isfinite(z):
            self.slider.setValue(int(z * self.slider_scale))
            self.update_from_slider()

    def update_from_slider(self):
        z = self.slider.value() / self.slider_scale
        self.z_input.setText(f"{z:.5f}")
        self.refresh_plot_lines()

    def update_from_text(self):
        try:
            z = float(self.z_input.text())
            self.slider.blockSignals(True)
            self.slider.setValue(int(z * self.slider_scale))
            self.slider.blockSignals(False)
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

    def get_active_spectra(self):
        active_spectra = []
        for target in self.active_targets._data:
            active_spectra += target.spectra
        return active_spectra

    def get_real_active_targets(self):
        targets_in_view = []
        for target in self.active_targets._data:
            if isinstance(target, TemplateTarget):
                continue
            targets_in_view.append(target)
        return targets_in_view

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
        self.sync_target_flags()
        self.sync_target_notes()


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

    def write_notes(self, filename=None, backup=False):
        if len(self.target_notes) == 0:
            if not backup:
                logging.warning("No target notes to save")
            return

        notes_filename = write_notes_to_file(self.target_notes, self.target_flags,
                                             filename=filename)
        if notes_filename and not backup:
            self.notes_filename = str(notes_filename)
            logging.info(f"Saved target notes to file: {notes_filename}")

    def create_menubar(self):
        self.main_menu = self.menuBar()
        self.file_menu = self.main_menu.addMenu("File")
        # load_action = QtWidgets.QAction("Load Spectrum")
        # load_container_action = QtWidgets.QAction("Load Multiple Spectra")

        save_action = QtWidgets.QAction("Save Notes", self)
        save_action.triggered.connect(lambda x: self.write_notes())
        save_action.setShortcut("Ctrl+S")

        # Exit QAction
        exit_action = QtWidgets.QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        
        self.file_menu.addAction(save_action)
        self.file_menu.addAction(exit_action)
