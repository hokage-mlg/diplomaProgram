from PyQt5.QtWidgets import QApplication, QDialog, QTabWidget, QWidget, QVBoxLayout, QDialogButtonBox, QLineEdit, QLabel
import sys
from PyQt5 import QtCore, QtWidgets
import numpy as np
import math as m
from itertools import product
from collections import Counter
from matplotlib import pyplot as plt


class TabWidget(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tab Widget Application")
        self.resize(900, 700)
        tabWidget = QTabWidget()
        tabWidget.addTab(FirstTab(), "First Tab")
        tabWidget.addTab(SecondTab(), "Second Tab")
        tabWidget.addTab(ThirdTab(), "Third Tab")
        # buttonbox=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        # buttonbox.accepted.connect(self.accept)
        # buttonbox.rejected.connect(self.reject)
        vbox = QVBoxLayout()
        vbox.addWidget(tabWidget)
        # vbox.addWidget(buttonbox)
        self.setLayout(vbox)


class FirstTab(QWidget):
    def __init__(self):
        super().__init__()

        self.vBox = QVBoxLayout()

        centralWidget = QtWidgets.QWidget(self)
        centralWidget.setEnabled(True)
        centralWidget.setObjectName("centralWidget")
        self.vBox.addWidget(centralWidget)

        self.vBox.formLayoutWidget = QtWidgets.QWidget(centralWidget)
        self.vBox.formLayoutWidget.setGeometry(QtCore.QRect(20, 30, 731, 511))
        self.vBox.formLayoutWidget.setObjectName("formLayoutWidget")
        self.vBox.formLayout = QtWidgets.QFormLayout(self.vBox.formLayoutWidget)
        self.vBox.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldsStayAtSizeHint)
        self.vBox.formLayout.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        self.vBox.formLayout.setLabelAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.vBox.formLayout.setFormAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.vBox.formLayout.setContentsMargins(0, 0, 0, 0)
        self.vBox.formLayout.setHorizontalSpacing(5)
        self.vBox.formLayout.setVerticalSpacing(6)
        self.vBox.formLayout.setObjectName("formLayout")

        self.vBox.label_kappa = QtWidgets.QLabel(self.vBox.formLayoutWidget)
        self.vBox.label_kappa.setObjectName("label_kappa")
        self.vBox.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.vBox.label_kappa)
        self.vBox.lineEdit_kappa = QtWidgets.QLineEdit(self.vBox.formLayoutWidget)
        self.vBox.lineEdit_kappa.setObjectName("lineEdit_kappa")
        self.vBox.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.vBox.lineEdit_kappa)

        self.vBox.label_la = QtWidgets.QLabel(self.vBox.formLayoutWidget)
        self.vBox.label_la.setObjectName("label_la")
        self.vBox.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.vBox.label_la)
        self.vBox.lineEdit_la = QtWidgets.QLineEdit(self.vBox.formLayoutWidget)
        self.vBox.lineEdit_la.setObjectName("lineEdit_la")
        self.vBox.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.vBox.lineEdit_la)

        self.vBox.label_mu = QtWidgets.QLabel(self.vBox.formLayoutWidget)
        self.vBox.label_mu.setObjectName("label_mu")
        self.vBox.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.vBox.label_mu)
        self.vBox.lineEdit_mu = QtWidgets.QLineEdit(self.vBox.formLayoutWidget)
        self.vBox.lineEdit_mu.setObjectName("lineEdit_mu")
        self.vBox.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.vBox.lineEdit_mu)

        self.vBox.label_alpha = QtWidgets.QLabel(self.vBox.formLayoutWidget)
        self.vBox.label_alpha.setObjectName("label_alpha")
        self.vBox.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.vBox.label_alpha)
        self.vBox.lineEdit_alpha = QtWidgets.QLineEdit(self.vBox.formLayoutWidget)
        self.vBox.lineEdit_alpha.setObjectName("lineEdit_alpha")
        self.vBox.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.vBox.lineEdit_alpha)

        # self.vBox.label_beta = QtWidgets.QLabel(self.vBox.formLayoutWidget)
        # self.vBox.label_beta.setObjectName("label_beta")
        # self.vBox.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.vBox.label_beta)
        # self.vBox.lineEdit_beta = QtWidgets.QLineEdit(self.vBox.formLayoutWidget)
        # self.vBox.lineEdit_beta.setObjectName("lineEdit_beta")
        # self.vBox.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.vBox.lineEdit_beta)

        self.radioButton_alpha = QtWidgets.QRadioButton(self.vBox.formLayoutWidget)
        self.radioButton_alpha.setObjectName("radioButton_alpha")
        self.vBox.formLayout.setWidget(9, QtWidgets.QFormLayout.LabelRole, self.radioButton_alpha)

        self.radioButton_beta = QtWidgets.QRadioButton(self.vBox.formLayoutWidget)
        self.radioButton_beta.setObjectName("radioButton_alpha")
        self.vBox.formLayout.setWidget(10, QtWidgets.QFormLayout.LabelRole, self.radioButton_beta)

        self.vBox.pushButton = QtWidgets.QPushButton(self.vBox.formLayoutWidget)
        self.vBox.pushButton.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.vBox.pushButton.setObjectName("pushButton")
        self.vBox.formLayout.setWidget(11, QtWidgets.QFormLayout.FieldRole, self.vBox.pushButton)

        self.vBox.textEdit_result = QtWidgets.QTextEdit(self.vBox.formLayoutWidget)
        self.vBox.textEdit_result.setObjectName("textEdit_result")
        self.vBox.formLayout.setWidget(12, QtWidgets.QFormLayout.SpanningRole, self.vBox.textEdit_result)

        self.vBox.label_kappa.setText("Количество приборов")
        self.vBox.label_la.setText("Интенсивность входящего потока требований")
        self.vBox.label_mu.setText("Интенсивность обслуживания требования прибором")
        self.vBox.label_alpha.setText("Интенсивность: ")
        # self.vBox.label_beta.setText("Интенсивности ")
        self.radioButton_alpha.setText("наработки на отказ")
        self.radioButton_beta.setText("восстановления")
        self.vBox.pushButton.setText("Подсчет")
        self.setLayout(self.vBox)
        self.vBox.pushButton.clicked.connect(
            lambda: self.btn_clicked(self.radioButton_alpha.isChecked()))

    def btn_clicked(self, chk):
        # Очистка текстового поля для вывода результата
        self.vBox.textEdit_result.setText("")
        # Определение входных значений
        k = int(self.vBox.lineEdit_kappa.text())
        la = float(self.vBox.lineEdit_la.text())
        mu = float(self.vBox.lineEdit_mu.text())
        if chk:
            alpha = float(self.vBox.lineEdit_alpha.text())
            beta = 1 - alpha
        else:
            beta = float(self.vBox.lineEdit_alpha.text())
            alpha = 1 - beta
        result = ""
        result += "kappa: " + str(k) + "la: " + str(la) + "mu: " + str(mu) + "alpha: " + str(alpha) + "Beta: " + str(
            beta) + "\n"
        # коэффициент использования
        psi = la / (k * mu)
        # вероятность пребывания в системе 0 требований(все приборы свободны)
        sum1 = 0
        for n in range(0, k):
            sum1 += ((k * psi) ** n) / m.factorial(n)
        p = np.zeros(k + 1)
        p[0] = (((k * psi) ** k) / (m.factorial(k) * (1 - psi)) + sum1) ** -1
        for n in range(1, k + 1):
            p[n] = p[0] * ((k * psi) ** n) / m.factorial(n)
        result += "Вероятности состояний системы: " + str(p) + "\n"
        check1 = 0
        for i in p:
            check1 += i
        result += "Проверка суммы вероятностей: " + str('%.0f' % check1) + "\n"
        combList = [i for i in product(range(2), repeat=k)]
        result += str(combList) + "\n"
        combList = np.array(combList)
        total_1_axis = np.sum(combList, axis=1)
        result += str(total_1_axis) + "\n"
        c = Counter(total_1_axis)
        result += str(c) + "\n"
        p_rob = np.zeros(k + 1)
        for key, value in c.items():
            if key == 0:
                p_rob[0] = (alpha / (alpha + beta)) ** k
            elif key == k:
                p_rob[k] = (beta / (alpha + beta)) ** k
            else:
                p_rob[key] = ((beta / (alpha + beta)) ** key * (alpha / (alpha + beta)) ** (k - key)) * value
        result += str(p_rob) + "\n"
        check2 = 0
        for i in p_rob:
            check2 += i
        result += "Проверка суммы вероятностей: " + str(check2) + "\n"
        # коэффициент использования ненадежных приборов
        sum2 = 0
        for n in range(0, k + 1):
            sum2 += n * p_rob[n]
        psi_n = la / (mu * sum2)
        result += "Коэффициент использования ненадежных приборов: " + str(psi_n) + "\n"
        # м.о. числа занятых и свободных приборов
        h = psi_n * k
        result += "М.о. числа занятых приборов: " + str(h) + "\n"
        g = (1 - psi_n) * k
        result += "М.о. числа свободных приборов: " + str(g) + "\n"
        # м.о. числа требований, ожидающих в очереди
        limit = 50
        b = 0
        for i in range(k + 1, limit + 1):
            for j in range(0, k + 1):
                b += (i - j * p_rob[j]) * ((psi_n ** i * k ** k) / m.factorial(k)) * p[0]
        result += "М.о. числа требований, ожидающих в очереди: " + str(b) + "\n"
        # м.о. числа требований в системе
        q = b + h
        result += "М.о. числа требований в системе: " + str(q) + "\n"
        # м.о. длительности пребывания в системе
        u = q / la
        result += "М.о. длительности пребывания в системе: " + str(u) + "\n"
        # м.о. длительности пребывания требований в очереди
        w = b / la
        result += "М.о. длительности пребывания требований в очереди: " + str(w) + "\n"
        # коэффициент загрузки
        k_h = h / k
        result += "Коэффициент загрузки: " + str(k_h) + "\n"
        # коэффициент простоя
        k_g = g / k
        result += "Коэффициент простоя: " + str(k_g) + "\n"
        # Выводим в правое поле результат
        self.vBox.textEdit_result.setText(result)


class SecondTab(QWidget):
    def __init__(self):
        super().__init__()


class ThirdTab(QWidget):
    def __init__(self):
        super().__init__()
        


app = QApplication(sys.argv)
tabWidget = TabWidget()
tabWidget.show()
app.exec()
