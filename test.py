"""
GUI test
"""

import sys
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *



class GraphicInterface(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle('Test')
        self._main = QWidget()
        self.setCentralWidget(self._main)

        # Create Toolbar and Menubar
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar_fontsize = QFont()
        toolbar_fontsize.setPointSize(14)

        quit_action = QAction("Close", self)
        quit_action.setFont(toolbar_fontsize)
        quit_action.setStatusTip("Quit the application")
        quit_action.triggered.connect(self.close)
        quit_action.setShortcut("ctrl+Q")
        toolbar.addAction(quit_action)


        # =============================================================
        # Start Layout:
        layout = QHBoxLayout(self._main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.line_editor = QLineEdit("3")
        self.line_editor.setValidator(QIntValidator(1, 100))
        self.line_editor.returnPressed.connect(self.func)
        layout.addWidget(self.line_editor)

        # Create Table for Pixel Identifications:
        pixtab_layout = QVBoxLayout()
        label_pixtab_header = QLabel("Pixel Table\n")
        label_pixtab_header.setAlignment(Qt.AlignCenter)
        self.linetable = QTableWidget()
        self.linetable.verticalHeader().hide()
        self.linetable.setColumnCount(2)
        self.linetable.setHorizontalHeaderLabels(["Pixel", "Wavelength"])
        self.linetable.setColumnWidth(0, 80)
        self.linetable.setColumnWidth(1, 90)
        self.linetable.setFixedWidth(180)
        pixtab_layout.addWidget(label_pixtab_header)
        pixtab_layout.addWidget(self.linetable)
        layout.addLayout(pixtab_layout)

        # Create Figure Canvas:
        right_layout = QVBoxLayout()
        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()
        right_layout.addWidget(self.canvas)
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.mpl_toolbar)
        layout.addLayout(right_layout)

        # Plot some random data:
        self.ax = self.fig.add_subplot(111)
        self.ax.plot([0, 1, 5, 10, 20], [-1, 2, -4, 5, -2])

    def on_key_press(self, e):
        if e.key == "a":
            self.add_point()

    def add_point(self):
        print("Select point...")
        point = self.fig.ginput(1)
        x0, _ = point[0]
        rowPosition = self.linetable.rowCount()
        self.linetable.insertRow(rowPosition)
        item = QTableWidgetItem("%.2f" % x0)
        item.setFlags(Qt.ItemIsEnabled)
        item.setBackground(QColor('lightgray'))
        self.linetable.setItem(rowPosition, 0, item)

        y_item = QLineEdit("")
        y_item.setValidator(QDoubleValidator())
        y_item.returnPressed.connect(self.print_val)
        self.linetable.setCellWidget(rowPosition, 1, y_item)
        self.linetable.cellWidget(rowPosition, 1).setFocus()

    def print_val(self):
        rowPosition = self.linetable.rowCount()
        editor = self.linetable.cellWidget(rowPosition-1, 1)
        y_in = float(editor.text())
        print(" Table value: %.2f" % y_in)
        # and set focus back to canvas:
        self.canvas.setFocus()
        # Check which widget has focus:
        focus_widget = self.focusWidget()
        print(focus_widget.__class__)
        print("TableWidget has focus?  %r" % self.linetable.hasFocus())
        print("Cell editor has focus?  %r" % editor.hasFocus())

    def func(self):
        # dome some stuff
        print(self.line_editor.text())
        # and set focus back to canvas:
        self.canvas.setFocus()


if __name__ == '__main__':
    # Launch App:
    qapp = QApplication(sys.argv)
    app = GraphicInterface()
    app.show()
    qapp.exec_()
