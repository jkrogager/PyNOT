
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt

from astropy.table import QTable
import numpy as np


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
            target = self._data[index.row()]
            return target.name

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
