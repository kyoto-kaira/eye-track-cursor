# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'first_screen.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_FirstScreen(object):
    def setupUi(self, FirstScreen):
        FirstScreen.setObjectName("FirstScreen")
        FirstScreen.resize(777, 543)
        self.centralwidget = QtWidgets.QWidget(FirstScreen)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(150, 130, 391, 131))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(150, 20, 391, 81))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(90, 310, 601, 81))
        font = QtGui.QFont()
        font.setPointSize(24)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        FirstScreen.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(FirstScreen)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 777, 18))
        self.menubar.setObjectName("menubar")
        FirstScreen.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(FirstScreen)
        self.statusbar.setObjectName("statusbar")
        FirstScreen.setStatusBar(self.statusbar)

        self.retranslateUi(FirstScreen)
        QtCore.QMetaObject.connectSlotsByName(FirstScreen)

    def retranslateUi(self, FirstScreen):
        _translate = QtCore.QCoreApplication.translate
        FirstScreen.setWindowTitle(_translate("FirstScreen", "MainWindow"))
        self.pushButton.setText(_translate("FirstScreen", "スタート"))
        self.label.setText(_translate("FirstScreen", "キャリブレーション実施"))
        self.label_2.setText(_translate("FirstScreen", "左上ー＞右上ー＞右下ー＞右下の順に進める"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    FirstScreen = QtWidgets.QMainWindow()
    ui = Ui_FirstScreen()
    ui.setupUi(FirstScreen)
    FirstScreen.show()
    sys.exit(app.exec_())