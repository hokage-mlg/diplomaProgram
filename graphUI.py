# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'graphUI.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_GraphWindow(object):
    def setupUi(self, GraphWindow):
        GraphWindow.setObjectName("GraphWindow")
        GraphWindow.resize(973, 570)
        self.centralwidget = QtWidgets.QWidget(GraphWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(90, 170, 221, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(670, 170, 221, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.formLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget.setGeometry(QtCore.QRect(60, 210, 351, 271))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout_2 = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout_2.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.formLayout_2.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.formLayout_2.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_unr_alpha_kg = QtWidgets.QLabel(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_unr_alpha_kg.sizePolicy().hasHeightForWidth())
        self.label_unr_alpha_kg.setSizePolicy(sizePolicy)
        self.label_unr_alpha_kg.setWordWrap(True)
        self.label_unr_alpha_kg.setObjectName("label_unr_alpha_kg")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_unr_alpha_kg)
        self.pushButton_unr_alpha_kg = QtWidgets.QPushButton(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_unr_alpha_kg.sizePolicy().hasHeightForWidth())
        self.pushButton_unr_alpha_kg.setSizePolicy(sizePolicy)
        self.pushButton_unr_alpha_kg.setObjectName("pushButton_unr_alpha_kg")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.pushButton_unr_alpha_kg)
        self.pushButton_unr_la_u = QtWidgets.QPushButton(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_unr_la_u.sizePolicy().hasHeightForWidth())
        self.pushButton_unr_la_u.setSizePolicy(sizePolicy)
        self.pushButton_unr_la_u.setObjectName("pushButton_unr_la_u")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.pushButton_unr_la_u)
        self.label_unr_mu_w = QtWidgets.QLabel(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_unr_mu_w.sizePolicy().hasHeightForWidth())
        self.label_unr_mu_w.setSizePolicy(sizePolicy)
        self.label_unr_mu_w.setMouseTracking(True)
        self.label_unr_mu_w.setWordWrap(True)
        self.label_unr_mu_w.setObjectName("label_unr_mu_w")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_unr_mu_w)
        self.pushButton_unr_mu_w = QtWidgets.QPushButton(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_unr_mu_w.sizePolicy().hasHeightForWidth())
        self.pushButton_unr_mu_w.setSizePolicy(sizePolicy)
        self.pushButton_unr_mu_w.setObjectName("pushButton_unr_mu_w")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.pushButton_unr_mu_w)
        self.label_unr_la_u = QtWidgets.QLabel(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_unr_la_u.sizePolicy().hasHeightForWidth())
        self.label_unr_la_u.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setStyleStrategy(QtGui.QFont.PreferDefault)
        self.label_unr_la_u.setFont(font)
        self.label_unr_la_u.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_unr_la_u.setTextFormat(QtCore.Qt.AutoText)
        self.label_unr_la_u.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_unr_la_u.setWordWrap(True)
        self.label_unr_la_u.setObjectName("label_unr_la_u")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_unr_la_u)
        self.label_unr_beta_b = QtWidgets.QLabel(self.formLayoutWidget)
        self.label_unr_beta_b.setWordWrap(True)
        self.label_unr_beta_b.setObjectName("label_unr_beta_b")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_unr_beta_b)
        self.pushButton_unr_beta_b = QtWidgets.QPushButton(self.formLayoutWidget)
        self.pushButton_unr_beta_b.setObjectName("pushButton_unr_beta_b")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.pushButton_unr_beta_b)
        self.label_unr_beta_kh = QtWidgets.QLabel(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_unr_beta_kh.sizePolicy().hasHeightForWidth())
        self.label_unr_beta_kh.setSizePolicy(sizePolicy)
        self.label_unr_beta_kh.setWordWrap(True)
        self.label_unr_beta_kh.setObjectName("label_unr_beta_kh")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_unr_beta_kh)
        self.pushButton_unr_beta_kh = QtWidgets.QPushButton(self.formLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_unr_beta_kh.sizePolicy().hasHeightForWidth())
        self.pushButton_unr_beta_kh.setSizePolicy(sizePolicy)
        self.pushButton_unr_beta_kh.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.pushButton_unr_beta_kh.setObjectName("pushButton_unr_beta_kh")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.pushButton_unr_beta_kh)
        self.formLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(600, 200, 341, 171))
        self.formLayoutWidget_2.setObjectName("formLayoutWidget_2")
        self.formLayout_3 = QtWidgets.QFormLayout(self.formLayoutWidget_2)
        self.formLayout_3.setContentsMargins(0, 0, 0, 0)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_r_mu_w = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_r_mu_w.setWordWrap(True)
        self.label_r_mu_w.setObjectName("label_r_mu_w")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_r_mu_w)
        self.pushButton_r_la_u = QtWidgets.QPushButton(self.formLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_r_la_u.sizePolicy().hasHeightForWidth())
        self.pushButton_r_la_u.setSizePolicy(sizePolicy)
        self.pushButton_r_la_u.setObjectName("pushButton_r_la_u")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.pushButton_r_la_u)
        self.label_r_la_u = QtWidgets.QLabel(self.formLayoutWidget_2)
        self.label_r_la_u.setWordWrap(True)
        self.label_r_la_u.setObjectName("label_r_la_u")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_r_la_u)
        self.pushButton_r_mu_w = QtWidgets.QPushButton(self.formLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_r_mu_w.sizePolicy().hasHeightForWidth())
        self.pushButton_r_mu_w.setSizePolicy(sizePolicy)
        self.pushButton_r_mu_w.setObjectName("pushButton_r_mu_w")
        self.formLayout_3.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.pushButton_r_mu_w)
        self.formLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.formLayoutWidget_3.setGeometry(QtCore.QRect(360, 40, 311, 81))
        self.formLayoutWidget_3.setObjectName("formLayoutWidget_3")
        self.formLayout_4 = QtWidgets.QFormLayout(self.formLayoutWidget_3)
        self.formLayout_4.setContentsMargins(0, 0, 0, 0)
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_num_steps = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_num_steps.setObjectName("label_num_steps")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_num_steps)
        self.lineEdit_num_steps = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_num_steps.sizePolicy().hasHeightForWidth())
        self.lineEdit_num_steps.setSizePolicy(sizePolicy)
        self.lineEdit_num_steps.setObjectName("lineEdit_num_steps")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_num_steps)
        self.label_step_size = QtWidgets.QLabel(self.formLayoutWidget_3)
        self.label_step_size.setObjectName("label_step_size")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_step_size)
        self.lineEdit_step_size = QtWidgets.QLineEdit(self.formLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_step_size.sizePolicy().hasHeightForWidth())
        self.lineEdit_step_size.setSizePolicy(sizePolicy)
        self.lineEdit_step_size.setObjectName("lineEdit_step_size")
        self.formLayout_4.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_step_size)
        self.pushButton_close = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_close.setGeometry(QtCore.QRect(430, 460, 93, 28))
        self.pushButton_close.setObjectName("pushButton_close")
        GraphWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(GraphWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 973, 26))
        self.menubar.setObjectName("menubar")
        GraphWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(GraphWindow)
        self.statusbar.setObjectName("statusbar")
        GraphWindow.setStatusBar(self.statusbar)

        self.retranslateUi(GraphWindow)
        QtCore.QMetaObject.connectSlotsByName(GraphWindow)

    def retranslateUi(self, GraphWindow):
        _translate = QtCore.QCoreApplication.translate
        GraphWindow.setWindowTitle(_translate("GraphWindow", "Построение графиков"))
        self.label.setText(_translate("GraphWindow", "Для ненадежной системы"))
        self.label_3.setText(_translate("GraphWindow", "Для надежной системы"))
        self.label_unr_alpha_kg.setText(_translate("GraphWindow", "Коэффициент простоя от интенсивности наработки на отказ"))
        self.pushButton_unr_alpha_kg.setText(_translate("GraphWindow", "Показать"))
        self.pushButton_unr_la_u.setText(_translate("GraphWindow", "Показать"))
        self.label_unr_mu_w.setText(_translate("GraphWindow", "М.о. длительности пребывания требований в очереди от интенсивности обслуживания требования одним прибором"))
        self.pushButton_unr_mu_w.setText(_translate("GraphWindow", "Показать"))
        self.label_unr_la_u.setText(_translate("GraphWindow", "М.о. длительности пребывания требований в системе от интенсивности входящего потока требований"))
        self.label_unr_beta_b.setText(_translate("GraphWindow", "М.о. количества требований в очереди от интенсинвости восстановления"))
        self.pushButton_unr_beta_b.setText(_translate("GraphWindow", "Показать"))
        self.label_unr_beta_kh.setText(_translate("GraphWindow", "Коэффициент загрузки от интенсивности восстановления"))
        self.pushButton_unr_beta_kh.setText(_translate("GraphWindow", "Показать"))
        self.label_r_mu_w.setText(_translate("GraphWindow", "М.о. длительности пребывания требований в очереди от интенсивности обслуживания требования одним прибором"))
        self.pushButton_r_la_u.setText(_translate("GraphWindow", "Показать"))
        self.label_r_la_u.setText(_translate("GraphWindow", "М.о. длительности пребывания требований в системе от интенсивности входящего потока требований"))
        self.pushButton_r_mu_w.setText(_translate("GraphWindow", "Показать"))
        self.label_num_steps.setText(_translate("GraphWindow", "Количество шагов"))
        self.label_step_size.setText(_translate("GraphWindow", "Размер шага"))
        self.pushButton_close.setText(_translate("GraphWindow", "Закрыть"))
