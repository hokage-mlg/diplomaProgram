# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainUI.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(804, 714)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_result = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_result.setGeometry(QtCore.QRect(381, 257, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_result.setFont(font)
        self.pushButton_result.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_result.setObjectName("pushButton_result")
        self.pushButton_graphs = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_graphs.setGeometry(QtCore.QRect(580, 120, 141, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_graphs.setFont(font)
        self.pushButton_graphs.setObjectName("pushButton_graphs")
        self.pushButton_show_reliable = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_show_reliable.setGeometry(QtCore.QRect(580, 60, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_show_reliable.setFont(font)
        self.pushButton_show_reliable.setAcceptDrops(False)
        self.pushButton_show_reliable.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_show_reliable.setDefault(False)
        self.pushButton_show_reliable.setFlat(False)
        self.pushButton_show_reliable.setObjectName("pushButton_show_reliable")
        self.textEdit_result = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_result.setGeometry(QtCore.QRect(170, 310, 491, 261))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textEdit_result.sizePolicy().hasHeightForWidth())
        self.textEdit_result.setSizePolicy(sizePolicy)
        self.textEdit_result.setReadOnly(True)
        self.textEdit_result.setObjectName("textEdit_result")
        self.pushButton_save = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_save.setGeometry(QtCore.QRect(371, 606, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_save.setFont(font)
        self.pushButton_save.setObjectName("pushButton_save")
        self.pushButton_clear = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_clear.setGeometry(QtCore.QRect(271, 606, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_clear.setFont(font)
        self.pushButton_clear.setObjectName("pushButton_clear")
        self.pushButton_open = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_open.setGeometry(QtCore.QRect(471, 606, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_open.setFont(font)
        self.pushButton_open.setObjectName("pushButton_open")
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(50, 20, 461, 211))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label_kappa = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_kappa.setFont(font)
        self.label_kappa.setObjectName("label_kappa")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_kappa)
        self.lineEdit_kappa = QtWidgets.QLineEdit(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_kappa.sizePolicy().hasHeightForWidth())
        self.lineEdit_kappa.setSizePolicy(sizePolicy)
        self.lineEdit_kappa.setTabletTracking(False)
        self.lineEdit_kappa.setCursorPosition(0)
        self.lineEdit_kappa.setClearButtonEnabled(False)
        self.lineEdit_kappa.setObjectName("lineEdit_kappa")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_kappa)
        self.label_la = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_la.setFont(font)
        self.label_la.setWordWrap(True)
        self.label_la.setObjectName("label_la")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_la)
        self.lineEdit_la = QtWidgets.QLineEdit(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_la.sizePolicy().hasHeightForWidth())
        self.lineEdit_la.setSizePolicy(sizePolicy)
        self.lineEdit_la.setObjectName("lineEdit_la")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_la)
        self.label_mu = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_mu.setFont(font)
        self.label_mu.setWordWrap(True)
        self.label_mu.setObjectName("label_mu")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_mu)
        self.lineEdit_mu = QtWidgets.QLineEdit(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_mu.sizePolicy().hasHeightForWidth())
        self.lineEdit_mu.setSizePolicy(sizePolicy)
        self.lineEdit_mu.setObjectName("lineEdit_mu")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_mu)
        self.label_alpha = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_alpha.setFont(font)
        self.label_alpha.setObjectName("label_alpha")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_alpha)
        self.lineEdit_alpha = QtWidgets.QLineEdit(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_alpha.sizePolicy().hasHeightForWidth())
        self.lineEdit_alpha.setSizePolicy(sizePolicy)
        self.lineEdit_alpha.setObjectName("lineEdit_alpha")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_alpha)
        self.label_beta = QtWidgets.QLabel(self.formLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_beta.setFont(font)
        self.label_beta.setObjectName("label_beta")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_beta)
        self.lineEdit_beta = QtWidgets.QLineEdit(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_beta.sizePolicy().hasHeightForWidth())
        self.lineEdit_beta.setSizePolicy(sizePolicy)
        self.lineEdit_beta.setObjectName("lineEdit_beta")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lineEdit_beta)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 804, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setObjectName("action")
        self.action_2 = QtWidgets.QAction(MainWindow)
        self.action_2.setObjectName("action_2")
        self.action_3 = QtWidgets.QAction(MainWindow)
        self.action_3.setObjectName("action_3")
        self.action_4 = QtWidgets.QAction(MainWindow)
        self.action_4.setObjectName("action_4")
        self.action_5 = QtWidgets.QAction(MainWindow)
        self.action_5.setObjectName("action_5")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Программа для исследования ненадежной СМО"))
        self.pushButton_result.setText(_translate("MainWindow", "Запуск"))
        self.pushButton_graphs.setText(_translate("MainWindow", "Графики"))
        self.pushButton_show_reliable.setText(_translate("MainWindow", "Характеристики \n"
" надежной СМО"))
        self.pushButton_save.setText(_translate("MainWindow", "Сохранить"))
        self.pushButton_clear.setText(_translate("MainWindow", "Очистить"))
        self.pushButton_open.setText(_translate("MainWindow", "Открыть"))
        self.label_kappa.setText(_translate("MainWindow", "Количество приборов"))
        self.label_la.setText(_translate("MainWindow", "Интенсивность входящего потока требований"))
        self.label_mu.setText(_translate("MainWindow", "Интенсивность обслуживания требования прибором"))
        self.label_alpha.setText(_translate("MainWindow", "Интенсивности наработки на отказ"))
        self.label_beta.setText(_translate("MainWindow", "Интенсивности восстановления"))
        self.action.setText(_translate("MainWindow", "Ненадежная"))
        self.action.setStatusTip(_translate("MainWindow", "Show actions for unreliable system"))
        self.action.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.action_2.setText(_translate("MainWindow", "Надежная"))
        self.action_2.setStatusTip(_translate("MainWindow", "Show actions for reliable system"))
        self.action_2.setShortcut(_translate("MainWindow", "Ctrl+R"))
        self.action_3.setText(_translate("MainWindow", "Ненадежные"))
        self.action_3.setStatusTip(_translate("MainWindow", "Show graph for unreliable system"))
        self.action_3.setShortcut(_translate("MainWindow", "Shift+N"))
        self.action_4.setText(_translate("MainWindow", "Надежные"))
        self.action_4.setStatusTip(_translate("MainWindow", "Show graph for reliable system"))
        self.action_4.setShortcut(_translate("MainWindow", "Shift+R"))
        self.action_5.setText(_translate("MainWindow", "Инфо"))
        self.action_5.setStatusTip(_translate("MainWindow", "Show program info"))
        self.action_5.setShortcut(_translate("MainWindow", "Shift+O"))
