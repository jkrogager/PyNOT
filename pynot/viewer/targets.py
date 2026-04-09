import logging
from PyQt5 import QtWidgets, QtGui

from pynot.viewer.tablemodels import TargetDetailTableModel, SpectrumNameDelegate, ColorPickerDelegate
from pynot.viewer.spectrum import Spectrum


class Target:
    def __init__(self, name=None, spectra=None):
        self.smooth_factor = 1
        if name is None:
            self.name = ""
        else:
            self.name = name

        if spectra is None:
            self.spectra = []
        else:
            for spec in spectra:
                self.add_spectrum(spec)

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def add_spectrum(self, spectrum):
        if not isinstance(spectrum, Spectrum):
            logging.error(f"Cannot append invalid object of type: {type(spectrum)} as Spectrum")
            return
        spectrum.set_parent(self)
        self.spectra.append(spectrum)

    def remove_spectrum(self, num):
        if num < len(self.spectra):
            this = self.spectra.pop(num)
            return this.plot_line

    def get_all_plot_lines(self):
        lines = []
        for item in self.spectra:
            if item.plot_line is None:
                continue
            lines.append(item.plot_line)
            if item.error_line is not None:
                lines.append(item.error_line)
        return lines

    def show_details(self, parent=None):
        dialog = TargetInspectorDialog(target=self, parent=parent)
        dialog.show()
        dialog.raise_()
        return dialog


# -- Target Inspector

class TargetInspectorDialog(QtWidgets.QDialog):
    def __init__(self, target, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.target = target
        self.setWindowTitle("Target Details")
        self.resize(500, 400)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        target_header = QtWidgets.QLabel("<b>Target Details</b>")
        header_font = QtGui.QFont('Helvetica', 16)
        target_header.setFont(header_font)
        self.layout.addWidget(target_header)

        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Target Name: "))
        self.text_edit = QtWidgets.QLineEdit(target.name)
        self.text_edit.editingFinished.connect(self.update_target_name)
        name_row.addWidget(self.text_edit)

        # Make Spectrum table:
        self.spectral_table = QtWidgets.QTableView()
        self.spectral_model = TargetDetailTableModel(self.target.spectra, parent=self)
        self.name_delegate = SpectrumNameDelegate()
        self.color_delegate = ColorPickerDelegate(self.spectral_table)
        self.spectral_table.setModel(self.spectral_model)
        self.spectral_table.setItemDelegateForColumn(0, self.name_delegate)
        self.spectral_table.setItemDelegateForColumn(1, self.color_delegate)
        self.spectral_table.setEditTriggers(self.spectral_table.AllEditTriggers)
        self.layout.addWidget(self.spectral_table)
        self.spectral_table.resizeColumnsToContents()

        button_row = QtWidgets.QHBoxLayout()
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_row.addSpacing(1)
        button_row.addWidget(close_button)
        self.layout.addLayout(name_row)
        self.layout.addLayout(button_row)

    def update_target_name(self):
        text = self.text_edit.text()
        if ',' in text:
            logging.warning("Invalid character in name: ','. Force conversion to ';'")
            text = text.replace(',', ';')
        self.target.name = text
        for spec in self.target.spectra:
            label = self.parent.plot_legend.getLabel(spec.plot_line)
            label.setText(f"{text} {spec.name}")
            if spec.error_line is not None:
                e_label = self.parent.plot_legend.getLabel(spec.error_line)
                e_label.setText(f"{text} {spec.name} 1σ")
        self.parent.plot_legend.updateSize()
        self.parent.all_targets.layoutChanged.emit()
        self.parent.target_table.resizeColumnsToContents()
        self.parent.active_targets.layoutChanged.emit()
        self.parent.active_table.resizeColumnsToContents()
