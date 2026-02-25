from PyQt5.QtWidgets import (QDialog, QFileDialog, QHBoxLayout, QVBoxLayout,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QHeaderView)
from PyQt5.QtCore import pyqtSignal, Qt
from astropy.table import Table
import numpy as np
import logging


class LineManagerDialog(QDialog):
    # Signal that sends back a list of [label, wavelength, is_visible]
    linesChanged = pyqtSignal(list)

    def __init__(self, current_lines, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Line List Manager")
        self.setMinimumSize(400, 500)
        self.init_ui(current_lines)

    def init_ui(self, current_lines):
        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Show", "Label", "Rest λ (Å)"])
        self.table.resizeColumnToContents(0)
        
        # Populate with existing data
        for label, wl, visible in current_lines:
            self.add_row(label, wl, visible)

        # Buttons
        button_layout = QHBoxLayout()
        add_btn = QPushButton("Add New Line")
        add_btn.clicked.connect(lambda: self.add_row("New Line", 0.0, True))
        
        del_btn = QPushButton("Remove Selected")
        del_btn.clicked.connect(self.remove_selected)

        # Buttons
        load_btn = QPushButton("Load Line List")
        load_btn.clicked.connect(self.load_linelist)
        load_btn.setToolTip("Load a new linelist from file (FITS or ASCII).\n"
                            "The linelist must contain at least two columns with a label and a wavelength")

        close_btn = QPushButton("Close Window")
        close_btn.clicked.connect(self.close)

        layout.addWidget(self.table)
        button_layout.addWidget(add_btn)
        button_layout.addWidget(del_btn)
        layout.addLayout(button_layout)
        layout.addWidget(load_btn)
        layout.addWidget(close_btn)

        # Trigger update whenever the table is edited
        self.table.itemChanged.connect(self.emit_changes)


    def load_linelist(self):
        current_dir = './'
        filters = "Line lists (*.fits | *.csv | *.txt | *.tsv | *.dat)"
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Line List', current_dir, filters)
        fname = str(fname)
        if not fname:
            return

        if fname.endswith('.fits'):
            linelist = Table.read(fname)
            name, wave = get_label_and_wl_from_table(linelist)

        else:
            try:
                linelist = np.genfromtxt(fname, comments='#', dtype=str)
                name, wave = get_label_and_wl_from_numpy(linelist)
            except Exception:
                linelist = Table.read(fname, format='ascii')
                name, wave = get_label_and_wl_from_table(linelist)

        self.table.blockSignals(True)
        for label, wl in zip(name, wave):
            self.add_row(str(label), float(wl), True)
        self.table.blockSignals(False)
        self.emit_changes()

    def add_row(self, label, wl, visible):
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        check_item = QTableWidgetItem()
        check_item.setCheckState(Qt.Checked if visible else Qt.Unchecked)
        
        self.table.setItem(row, 0, check_item)
        self.table.setItem(row, 1, QTableWidgetItem(label))
        self.table.setItem(row, 2, QTableWidgetItem(str(wl)))

    def remove_selected(self):
        self.table.removeRow(self.table.currentRow())
        self.emit_changes()

    def emit_changes(self):
        data = []
        for r in range(self.table.rowCount()):
            # check that items exist in the row
            check_item = self.table.item(r, 0)
            label_item = self.table.item(r, 1)
            wave_item = self.table.item(r, 2)
            if not all([check_item, label_item, wave_item]):
                continue

            visible = check_item.checkState() == Qt.Checked
            label = label_item.text()
            try:
                wl = float(wave_item.text())
                data.append([label, wl, visible])
            except ValueError:
                logging.error(f"Invalid wavelength in linelist for label: {label}. Must be numeral")
                continue
        self.linesChanged.emit(data)


def get_label_and_wl_from_table(linelist):
    for colname in linelist.colnames:
        linelist.rename_column(colname.upper())

    # Try to find column names for the labels
    label_colnames = ['NAME', 'LINE', 'ID', 'LINE_ID', 'LABEL', 'LABELS']
    for colname in label_colnames:
        if colname in linelist.colnames:
            name = linelist[colname]
            break
    else:
        name = []
    if len(name) == 0:
        logging.error(f"I looked for column names: {label_colnames}")
        logging.error("Could not find any matching labels column.")

    # Try to find column names for the wavelengths
    wavelength_colnames = ['WAVE', 'WAVELENGTH', 'WL', 'REST-FRAME', 'RESTFRAME', 'TRANS', 'TRANSITION', 'L0']
    for colname in wavelength_colnames:
        if colname in linelist.colnames:
            wave = linelist[colname]
            break
    else:
        wave = []
    if len(wave) == 0:
        logging.error(f"I looked for column names: {wavelength_colnames}")
        logging.error("Could not find any matching wavelength column.")

    return name, wave


def get_label_and_wl_from_numpy(linelist):
    try:
        name = linelist[:, 0]
        wave = linelist[:, 1]

    except Exception:
        name = np.array([], dtype=bool)
        wave = np.array([], dtype=bool)

    try:
        wave = wave.astype(float)
    except ValueError:
        logging.error(f"Dumping first row of the linelist: {linelist[0]}")
        logging.error(f"of datatype: {linelist.dtype}")
        logging.error("Could not parse the linelist")
        wave = []
        name = []

    return name, wave
