from PyQt5 import QtGui
from gui.graphing import MainWindow


def init_grapher(dirname, fname, key='score', loc='middle'):
    app = QtGui.QApplication([])
    window = MainWindow(dirname, fname, loc, key)
    window.show()
    app.exec_()
