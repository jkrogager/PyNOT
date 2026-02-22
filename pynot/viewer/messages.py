import logging
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5 import QtWidgets


class LogSignaler(QObject):
    signal = pyqtSignal(str)


class QtLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.signaler = LogSignaler()

    def emit(self, record):
        msg = self.format(record)
        self.signaler.signal.emit(msg)


# -- Log Viewer

class LogViewerDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Full Log History")
        self.resize(500, 400)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        self.text_edit = QtWidgets.QPlainTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.layout.addWidget(self.text_edit)

    def append_log(self, message):
        self.text_edit.appendPlainText(message)
