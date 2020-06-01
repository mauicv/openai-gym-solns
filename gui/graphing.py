from PyQt5 import QtCore, QtGui
import pyqtgraph as pg
from file_system_controller import FileReader
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut


class MainWindow(QtGui.QMainWindow):
    def __init__(
            self,
            data_dirname,
            data_fname,
            loc,
            key='score',
            parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle(key)
        self.setFixedWidth(500)
        self.loc = loc

        self.shortcut = QShortcut(QKeySequence("Ctrl+w"), self)
        self.shortcut.activated.connect(self.close)

        if loc == 'middle':
            self.move(500, 0)
        elif loc == 'left':
            self.move(0, 0)
        elif loc == 'right':
            self.move(1000, 0)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.reader = FileReader(data_dirname, data_fname)
        self.plotter()
        self.key = key

    def plotter(self):
        self.curve = self.graphWidget.getPlotItem().plot()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updater)
        self.timer.start(500)

    def updater(self):
        data = self.reader()
        self.curve.setData(data[self.key])
