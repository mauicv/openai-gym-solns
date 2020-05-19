from PyQt5 import QtGui
from gui.graphing import MainWindow


def init_grapher(dirname, fname):
    app = QtGui.QApplication([])
    window = MainWindow(dirname, fname)
    window.show()
    app.exec_()
