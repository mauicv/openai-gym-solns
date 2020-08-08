
try:
    from PyQt5 import QtGui
    from gui.graphing import MainWindow

    def init_grapher(dirname, fname):
        app = QtGui.QApplication([])
        window = MainWindow(dirname, fname)
        window.show()
        app.exec_()
except Exception:
    print('Running in server friendly mode')
    init_grapher = False
