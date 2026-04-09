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



class ModelTarget(Target):
    def __init__(self, model):
        super().__init__()
        model.set_parent(self)
        self.name = 'Model Spectrum'
        self.spectra = [model]

    def show_details(self, parent=None):
        dialog = ModelConfigurationDialog(target=self, parent=parent)
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
        Av = float(self.Av_editor.text())
        model_name = self.dust_editor.currentText()
        dust_model = DUST_MODELS[model_name]
        self.template.dust_model = dust_model()
        self.template.Av = Av
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


# -- Model Inspector

class ModelConfigurationDialog(QtWidgets.QDialog):
    def __init__(self, target, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.target = target
        if len(self.target.spectra) != 1:
            logging.error("Remove the template from the active targets list and try again.")
            logging.error("A model template can only hold one spectral model!")
            QtWidgets.QMessageBox.critical(self, "Model error",
                                           "An error occurred in loading the spectral model")
            return
        self.model = target.spectra[0]

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Template Details")
        self.resize(300, 400)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        target_header = QtWidgets.QLabel("<b>Model Details</b>")
        header_font = QtGui.QFont('Helvetica', 16)
        target_header.setFont(header_font)
        self.layout.addWidget(target_header)

        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Model Name: "))
        self.name_edit = QtWidgets.QLineEdit(self.target.name)
        self.name_edit.editingFinished.connect(self.update_target_name)
        name_row.addWidget(self.name_edit)
        self.layout.addLayout(name_row)
        self.layout.addSpacing(1)

        model = self.target.spectra[0]
        self.expr_editor = QtWidgets.QLineEdit(model.expression)
        self.expr_editor.editingFinished.connect(self.update_model)

        self.spec_editor = QtWidgets.QComboBox()
        act_spec = self.parent.get_active_spectra()
        self.active_spectra = {spec.parent.name: spec for spec in act_spec}
        active_names = list(self.active_spectra.keys()) + ['None']
        self.spec_editor.addItems(active_names)
        self.spec_editor.setCurrentText('None')
        self.spec_editor.currentTextChanged.connect(self.update_model)
        self.spec_editor.setToolTip("Choose a spectrum to use as reference for the wavelength (x) axis")

        real_validator = QtGui.QDoubleValidator(-float('inf'), float('inf'), 3)
        self.xmin_editor = QtWidgets.QLineEdit(f"{model.xmin}")
        self.xmin_editor.setValidator(real_validator)
        self.xmin_editor.editingFinished.connect(self.update_model)
        self.xmax_editor = QtWidgets.QLineEdit(f"{model.xmax}")
        self.xmax_editor.setValidator(real_validator)
        self.xmax_editor.editingFinished.connect(self.update_model)
        self.dx_editor = QtWidgets.QLineEdit(f"{model.dx}")
        self.dx_editor.setValidator(real_validator)
        self.dx_editor.editingFinished.connect(self.update_model)

        self.sampling_editor = QtWidgets.QComboBox()
        self.sampling_editor.addItems(['linear', 'log'])
        self.sampling_editor.currentTextChanged.connect(self.update_model)
        self.sampling_editor.setCurrentText('linear')

        self.add_model_editor('y(x) = ', self.expr_editor)
        self.add_model_editor('Spectrum :', self.spec_editor)
        self.add_model_editor('Minimum x :', self.xmin_editor)
        self.add_model_editor('Maximum x :', self.xmax_editor)
        self.add_model_editor('Δx :', self.dx_editor)

        button_row = QtWidgets.QHBoxLayout()
        update_button = QtWidgets.QPushButton("Update")
        update_button.clicked.connect(self.update_all)
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        close_button.clearFocus()
        button_row.addWidget(update_button)
        button_row.addSpacing(1)
        button_row.addWidget(close_button)
        self.layout.addLayout(button_row)
        update_button.setFocus()

    def set_reference_spectrum(self):
        spec_name = self.spec_editor.currentText()
        if spec_name != 'None':
            spec_ref = self.active_spectra[spec_name]
            self.xmin_editor.setEnabled(False)
            self.xmax_editor.setEnabled(False)
            self.dx_editor.setEnabled(False)
            self.sampling_editor.setEnabled(False)
            self.target.spectra[0].set_spectrum(spec_ref)
        else:
            self.xmin_editor.setEnabled(True)
            self.xmax_editor.setEnabled(True)
            self.dx_editor.setEnabled(True)
            self.sampling_editor.setEnabled(True)
            model = self.target.spectra[0]
            model.set_spectrum(None)
            xmin = float(self.xmin_editor.text())
            xmax = float(self.xmax_editor.text())
            dx = float(self.dx_editor.text())
            model.xmin = xmin
            model.xmax = xmax
            model.dx = dx
            model.log = self.sampling_editor.currentText() == 'log'

    def update_all(self):
        self.update_target_name()
        self.update_model()

    def update_model(self):
        model = self.target.spectra[0]
        self.set_reference_spectrum()
        model.expression = self.expr_editor.text()
        model.update_plot_data()
        self.raise_()

    def update_target_name(self):
        text = self.name_edit.text()
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

    def add_model_editor(self, label, widget):
        editor_row = QtWidgets.QHBoxLayout()
        editor_row.addWidget(QtWidgets.QLabel(label))
        editor_row.addWidget(widget)
        self.layout.addLayout(editor_row)
