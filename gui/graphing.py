from PyQt5 import QtCore, QtGui
import pyqtgraph as pg
from file_system_controller import FileReader


class MainWindow(QtGui.QMainWindow):
    def __init__(self, data_dirname, data_fname, parent=None):
        super(MainWindow, self).__init__(parent)
        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)
        self.reader = FileReader(data_dirname, data_fname)
        self.plotter()

    def plotter(self):
        self.curve = self.graphWidget.getPlotItem().plot()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updater)
        self.timer.start(500)

    def updater(self):
        data = self.reader()
        self.curve.setData(data['score'])
