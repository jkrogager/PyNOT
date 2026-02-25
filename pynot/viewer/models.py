import logging
from PyQt5 import QtWidgets, QtGui
import inspect
import os

from pynot.viewer.tablemodels import TargetDetailTableModel, SpectrumNameDelegate, ColorPickerDelegate
from pynot.viewer.spectrum import Spectrum, Template
from pynot.viewer.targets import Target
from pynot.viewer import dust

DUST_MODELS = {}
for varname, var in dust.__dict__.items():
    if inspect.isclass(var) and issubclass(var, dust.DustModel) and var is not dust.DustModel:
        DUST_MODELS[var.name] = var


class TemplateTarget(Target):
    def __init__(self, template):
        super().__init__()
        template.set_parent(self)
        self.name = os.path.basename(template.filename)
        self.spectra = [template]

    def show_details(self, parent=None):
        dialog = TemplateConfigurationDialog(target=self, parent=parent)
        dialog.show()
        dialog.raise_()
        return dialog


# -- Target Inspector

class TemplateConfigurationDialog(QtWidgets.QDialog):
    def __init__(self, target, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.target = target
        if len(self.target.spectra) != 1:
            logging.error("Remove the template from the active targets list and try again.")
            logging.error("A spectral template can only hold one spectral model!")
            QtWidgets.QMessageBox.critical(self, "Template error",
                                           "An error occurred in loading the spectral model")
            return
        self.template: Template = target.spectra[0]

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Template Details")
        self.resize(300, 400)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        target_header = QtWidgets.QLabel("<b>Template Details</b>")
        header_font = QtGui.QFont('Helvetica', 16)
        target_header.setFont(header_font)
        self.layout.addWidget(target_header)

        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Template Name: "))
        self.text_edit = QtWidgets.QLineEdit(self.target.name)
        self.text_edit.editingFinished.connect(self.update_target_name)
        name_row.addWidget(self.text_edit)
        self.layout.addLayout(name_row)
        self.layout.addSpacing(1)

        # Make editors for model parameters
        self.z_editor = QtWidgets.QLineEdit(str(self.template.z))
        z_validator = QtGui.QDoubleValidator(-1., 20., 5)
        self.z_editor.setValidator(z_validator)
        self.z_editor.editingFinished.connect(self.update_model)
        
        real_validator = QtGui.QDoubleValidator(-float('inf'), float('inf'), 3)
        self.c1_editor = QtWidgets.QLineEdit(str(self.template.C1))
        self.c1_editor.setValidator(real_validator)
        self.c1_editor.editingFinished.connect(self.update_model)
        self.c2_editor = QtWidgets.QLineEdit(str(self.template.C2))
        self.c2_editor.setValidator(real_validator)
        self.c2_editor.editingFinished.connect(self.update_model)

        self.Av_editor = QtWidgets.QLineEdit(str(self.template.Av))
        self.Av_editor.setValidator(real_validator)
        self.Av_editor.editingFinished.connect(self.update_model)

        self.dust_editor = QtWidgets.QComboBox()
        self.dust_editor.addItems(DUST_MODELS.keys())
        self.dust_editor.currentTextChanged.connect(self.update_model)
        self.dust_editor.setCurrentText(str(self.template.dust_model))
        self.dust_editor.setToolTip("Choose a dust model to apply reddening to the template")

        self.add_model_editor('Redshift: ', self.z_editor)
        self.add_model_editor('Flux Scale: ', self.c2_editor)
        self.add_model_editor('Flux Offset: ', self.c1_editor)
        self.add_model_editor('Dust A(V): ', self.Av_editor)
        self.add_model_editor('Dust Model: ', self.dust_editor)

        button_row = QtWidgets.QHBoxLayout()
        update_button = QtWidgets.QPushButton("Update")
        update_button.clicked.connect(self.update_all)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close_window)
        close_button.clearFocus()
        button_row.addWidget(update_button)
        button_row.addSpacing(1)
        button_row.addWidget(close_button)
        self.layout.addLayout(button_row)
        update_button.setFocus()

    def close_window(self):
        logging.info("Closed template configuration window")
        self.close()

    def add_model_editor(self, label, widget):
        editor_row = QtWidgets.QHBoxLayout()
        editor_row.addWidget(QtWidgets.QLabel(label))
        editor_row.addWidget(widget)
        self.layout.addLayout(editor_row)

    def update_all(self):
        self.update_target_name()
        self.update_model()

    def update_model(self):
        # Update redshift:
        z = float(self.z_editor.text())
        self.template.set_redshift(z)

        c1 = float(self.c1_editor.text())
        c2 = float(self.c2_editor.text())
        self.template.scale_flux(c1, c2)
        self.raise_()

    def update_target_name(self):
        text = self.text_edit.text()
        self.target.name = text
        for spec in self.target.spectra:
            label = self.parent.plot_legend.getLabel(spec.plot_line)
            label.setText(f"{text} {spec.name}")
        self.parent.plot_legend.updateSize()
        self.parent.all_targets.layoutChanged.emit()
        self.parent.target_table.resizeColumnsToContents()
        self.parent.active_targets.layoutChanged.emit()
        self.parent.active_table.resizeColumnsToContents()
        self.raise_()
