from PyQt5 import QtCore, QtGui
import pyqtgraph as pg


def grapher_and_train(train):
    app = QtGui.QApplication([])
    window = MainWindow(train)
    window.show()
    app.exec_()


class MainWindow(QtGui.QMainWindow):
    def __init__(self, runner, parent=None):
        super(MainWindow, self).__init__(parent)
        self.central_widget = QtGui.QStackedWidget()
        self.setCentralWidget(self.central_widget)
        self.login_widget = LoginWidget(self)
        self.login_widget.button.clicked.connect(self.plotter)
        self.central_widget.addWidget(self.login_widget)
        self.runner = runner

    def plotter(self):
        self.curve = self.login_widget.plot.getPlotItem().plot()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updater)
        self.timer.start(500)
        self.runner.async_start()

    def updater(self):
        self.curve.setData(self.runner.ys)


class LoginWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        super(LoginWidget, self).__init__(parent)
        layout = QtGui.QHBoxLayout()
        self.button = QtGui.QPushButton('Start')
        layout.addWidget(self.button)
        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot)
        self.setLayout(layout)
