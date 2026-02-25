
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
import pyqtgraph as pg

from astropy.table import QTable
import numpy as np
import logging

from pynot.viewer.spectrum import Spectrum


class AbstractIndex:
    def __init__(self, row, column, valid=True):
        self._row = row
        self._column = column
        self._valid = valid

    def row(self):
        return self._row

    def column(self):
        return self._column
    
    def isValid(self):
        return self._valid


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if not index.isValid():
            return

        if role == Qt.ItemDataRole.DisplayRole:
            return str(self._data[index.row()])

    def rowCount(self, index=None):
        return len(self._data)

    def columnCount(self, index=None):
        return 1

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return "Target"

            if orientation == Qt.Orientation.Vertical:
                return str(section)


class ActiveTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data):
        super(ActiveTableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if not index.isValid():
            return

        if role == Qt.ItemDataRole.DisplayRole:
            item = self._data[index.row()]
            declension = 'um' if len(item.spectra) == 1 else 'a'
            specID = f"{item.name}: {len(item.spectra)} spectr{declension}"
            return specID

        if role == Qt.ItemDataRole.EditRole:
            return self._data[index.row()]

    def rowCount(self, index=None):
        return len(self._data)

    def columnCount(self, index=None):
        return 1

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return "Active Target"

            if orientation == Qt.Orientation.Vertical:
                return str(section)



class TargetDetailTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data: list[Spectrum], parent=None):
        super().__init__()
        self._data = data
        self.parent = parent
        self.entries = [
            ('Name', lambda x: x.name),
            ('Color', lambda x: x.plot_line.curve.opts['pen'].color().name()),
            ('R (λ/δλ)', lambda x: x.R),
            ('Filename', lambda x: x.filename),
        ]

    def data(self, index, role):
        if not index.isValid():
            return

        if role == Qt.ItemDataRole.DisplayRole:
            spectrum = self._data[index.row()]
            col = index.column()
            return self.entries[col][1](spectrum)

        if role == Qt.ItemDataRole.UserRole:
            return self._data[index.row()]

        if role == Qt.ItemDataRole.FontRole:
            font = QtGui.QFont()
            if index.column() == 3:
                font.setPointSize(10)
            else:
                font.setPointSize(12)
            return font

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole:
            return False

        if not index.isValid():
            return False

        spectrum = self._data[index.row()]
        if index.column() == 0:
            spectrum.name = value
            self.parent.update_target_name()
            logging.info(f"Setting spectrum name: {value}")

        elif index.column() == 1:
            new_color = QtGui.QColor(value)
            if new_color.isValid():
                new_pen = pg.mkPen(color=new_color)
                spectrum.plot_line.setPen(new_pen)
                self.parent.raise_()

        elif index.column() == 2:
            spectrum.R = value
            self.parent.raise_()

        self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole])
        return True

    def rowCount(self, index=None):
        return len(self._data)

    def columnCount(self, index=None):
        return len(self.entries)

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self.entries[section][0]

            if orientation == Qt.Orientation.Vertical:
                return str(section)

    def flags(self, index):
            if not index.isValid():
                return Qt.NoItemFlags

            if index.column() == 3:
                return Qt.NoItemFlags

            # You must include Qt.ItemIsEditable
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable


class SpectrumNameDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def createEditor(self, parent, option, index):
        # This is called when you double-click or press Enter on a cell
        spectrum = index.data(Qt.ItemDataRole.UserRole)
        editor = QtWidgets.QLineEdit(parent)
        editor.setText(spectrum.name)
        return editor

    def setEditorData(self, editor, index):
        spectrum = index.data(Qt.ItemDataRole.UserRole)
        if spectrum.name:
            editor.setText(spectrum.name)

    def setModelData(self, editor, model, index):
        value = editor.text()
        model.setData(index, value, Qt.ItemDataRole.EditRole)


class SpectrumResolutionDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)

    def createEditor(self, parent, option, index):
        # This is called when you double-click or press Enter on a cell
        spectrum = index.data(Qt.ItemDataRole.UserRole)
        editor = QtWidgets.QLineEdit(parent)
        editor.setText(f"{spectrum.R}")
        return editor

    def setEditorData(self, editor, index):
        spectrum = index.data(Qt.ItemDataRole.UserRole)
        if spectrum:
            editor.setText(f"{spectrum.R}")

    def setModelData(self, editor, model, index):
        value = editor.text()
        try:
            value = float(value)
            model.setData(index, value, Qt.ItemDataRole.EditRole)
        except ValueError:
            logging.error(f"Invalid value for spectral resolution: {value}")



class ColorPickerDelegate(QtWidgets.QStyledItemDelegate):
    def paint(self, painter, option, index):
        # ... (same painting logic as before to draw the color box) ...
        color_str = index.data(Qt.DisplayRole)
        color = QtGui.QColor(color_str)
        if color.isValid():
            painter.save()
            rect = option.rect.adjusted(4, 4, -4, -4)
            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawRect(rect)
            painter.restore()
        else:
            super().paint(painter, option, index)

    def createEditor(self, parent, option, index):
        # 1. Get current color to start the dialog at the right spot
        current_color = QtGui.QColor(index.data(Qt.ItemDataRole.DisplayRole))
        
        # 2. Open the standard Color Dialog
        new_color = QtWidgets.QColorDialog.getColor(current_color, parent, "Select Spectrum Color")

        # 3. If the user clicked OK and picked a color
        if new_color.isValid():
            # Convert back to hex string (e.g., "#ff0000")
            color_string = new_color.name()
            
            # 4. Manually tell the model to update
            # We use index.model() because we aren't returning a persistent widget
            index.model().setData(index, color_string, Qt.ItemDataRole.EditRole)
            
            logging.info(f"Color updated to {color_string}")

        # Return None because we don't want an 'in-cell' text editor to appear
        return None
