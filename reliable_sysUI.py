# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'reliable_sysUI.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ReliableResults(object):
    def setupUi(self, ReliableResults):
        ReliableResults.setObjectName("ReliableResults")
        ReliableResults.resize(602, 546)
        self.centralwidget = QtWidgets.QWidget(ReliableResults)
        self.centralwidget.setObjectName("centralwidget")
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(50, 40, 501, 380))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldsStayAtSizeHint)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label_use = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_use.setObjectName("label_use")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_use)
        self.lineEdit_use = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_use.setReadOnly(True)
        self.lineEdit_use.setObjectName("lineEdit_use")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_use)
        self.label_busy = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_busy.setObjectName("label_busy")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_busy)
        self.lineEdit_busy = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_busy.setReadOnly(True)
        self.lineEdit_busy.setObjectName("lineEdit_busy")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_busy)
        self.label_free = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_free.setObjectName("label_free")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_free)
        self.lineEdit_free = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_free.setReadOnly(True)
        self.lineEdit_free.setObjectName("lineEdit_free")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_free)
        self.label_queue = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_queue.setObjectName("label_queue")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_queue)
        self.lineEdit_queue = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_queue.setReadOnly(True)
        self.lineEdit_queue.setObjectName("lineEdit_queue")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lineEdit_queue)
        self.label_num_sys = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_num_sys.setObjectName("label_num_sys")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_num_sys)
        self.lineEdit_num_sys = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_num_sys.setReadOnly(True)
        self.lineEdit_num_sys.setObjectName("lineEdit_num_sys")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.lineEdit_num_sys)
        self.label_time_sys = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_time_sys.setObjectName("label_time_sys")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_time_sys)
        self.lineEdit_time_sys = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_time_sys.setReadOnly(True)
        self.lineEdit_time_sys.setObjectName("lineEdit_time_sys")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.lineEdit_time_sys)
        self.label_time_queue = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_time_queue.setObjectName("label_time_queue")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_time_queue)
        self.lineEdit_time_queue = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_time_queue.setReadOnly(True)
        self.lineEdit_time_queue.setObjectName("lineEdit_time_queue")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.lineEdit_time_queue)
        self.label_busy_rate = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_busy_rate.setObjectName("label_busy_rate")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.LabelRole, self.label_busy_rate)
        self.lineEdit_busy_rate = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_busy_rate.setReadOnly(True)
        self.lineEdit_busy_rate.setObjectName("lineEdit_busy_rate")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.FieldRole, self.lineEdit_busy_rate)
        self.label_free_rate = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_free_rate.setObjectName("label_free_rate")
        self.formLayout.setWidget(9, QtWidgets.QFormLayout.LabelRole, self.label_free_rate)
        self.lineEdit_free_rate = QtWidgets.QLineEdit(self.formLayoutWidget)
        self.lineEdit_free_rate.setReadOnly(True)
        self.lineEdit_free_rate.setObjectName("lineEdit_free_rate")
        self.formLayout.setWidget(9, QtWidgets.QFormLayout.FieldRole, self.lineEdit_free_rate)
        self.pushButton_close = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_close.setGeometry(QtCore.QRect(340, 430, 93, 28))
        self.pushButton_close.setObjectName("pushButton_close")
        self.pushButton_show = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_show.setGeometry(QtCore.QRect(140, 430, 93, 28))
        self.pushButton_show.setObjectName("pushButton_show")
        ReliableResults.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(ReliableResults)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 602, 26))
        self.menubar.setObjectName("menubar")
        ReliableResults.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(ReliableResults)
        self.statusbar.setObjectName("statusbar")
        ReliableResults.setStatusBar(self.statusbar)

        self.retranslateUi(ReliableResults)
        QtCore.QMetaObject.connectSlotsByName(ReliableResults)

    def retranslateUi(self, ReliableResults):
        _translate = QtCore.QCoreApplication.translate
        ReliableResults.setWindowTitle(_translate("ReliableResults", "Результаты для надежной системы"))
        self.label_use.setText(_translate("ReliableResults", "Коэффициент использования приборов"))
        self.label_busy.setText(_translate("ReliableResults", "М.о. числа занятых приборов"))
        self.label_free.setText(_translate("ReliableResults", "М.о. числа свободных приборов"))
        self.label_queue.setText(_translate("ReliableResults", "М.о. числа требований, ожидающих в очереди"))
        self.label_num_sys.setText(_translate("ReliableResults", "М.о. числа требований в системе"))
        self.label_time_sys.setText(_translate("ReliableResults", "М.о. длительности пребывания в системе"))
        self.label_time_queue.setText(_translate("ReliableResults", "М.о. длительности пребывания требований в очереди"))
        self.label_busy_rate.setText(_translate("ReliableResults", "Коэффициент загрузки"))
        self.label_free_rate.setText(_translate("ReliableResults", "Коэффициент простоя"))
        self.pushButton_close.setText(_translate("ReliableResults", "Закрыть"))
        self.pushButton_show.setText(_translate("ReliableResults", "Показать"))
