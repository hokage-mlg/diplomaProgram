import sys
from tryna import *
from reliableUI import *
from patternUI import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import numpy as np
import math as m
from itertools import product
from collections import Counter
from matplotlib import pyplot as plt


def is_digit(string):
    if string.isdigit():
        return True
    else:
        try:
            float(string)
            return True
        except ValueError:
            return False


def btn_close_window(window):
    window.hide()


class MyWin(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.windowReliableResults = QtWidgets.QMainWindow()
        self.widget = QtWidgets.QWidget()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_result.clicked.connect(lambda: self.btn_result())
        self.ui.action.triggered.connect(lambda: self.clicked(self.ui.centralwidget.show()))
        self.ui.pushButton_graphs.clicked.connect(lambda: self.btn_show())
        self.ui.pushButton_show_reliable.clicked.connect(lambda: self.btn_reliable_results())
        self.ui.radioButton_alpha.setChecked(True)

    def btn_reliable_results(self):
        window = Ui_ReliableResults()
        window.setupUi(self.windowReliableResults)
        self.windowReliableResults.show()
        window.pushButton_close.clicked.connect(lambda: self.windowReliableResults.hide())

    def btn_show(self):
        form = Ui_Form()
        form.setupUi(self.widget)
        self.widget.show()
        form.pushButton.clicked.connect(lambda: self.widget.hide())

    def clicked(self, form):
        self.ui.centralwidget.setLayout(form)
        self.ui.centralwidget.adjustSize()

    def check_on_format(self, line):
        QtWidgets.QMessageBox.warning(self,
                                      'Ошибка ввода. Некорректный формат.',
                                      'Введите {} в корректном формате'.format(line))

    def check_on_zero(self, line):
        QtWidgets.QMessageBox.warning(self,
                                      'Ошибка ввода. Некорректное значение.', '{} не может принимать нулевое или '
                                                                              'отрицательное значение'.format(line))

    def check_input(self, chk):
        global alpha, la, mu, beta, k
        try:
            if int(self.ui.lineEdit_kappa.text()) > 0:
                k = int(self.ui.lineEdit_kappa.text())
            else:
                self.check_on_zero("Количество приборов")
                self.ui.centralwidget.show()
        except ValueError:
            self.check_on_format("количество приборов")
            self.ui.centralwidget.show()

        try:
            if float(self.ui.lineEdit_mu.text()) > 0:
                mu = float(self.ui.lineEdit_mu.text())
            else:
                self.check_on_zero("Интенсивность обслуживания требования прибором")
                self.ui.centralwidget.show()
        except ValueError:
            self.check_on_format("интенсивность обслуживания требования прибором")
            self.ui.centralwidget.show()

        try:
            if float(self.ui.lineEdit_la.text()) > 0:
                la = float(self.ui.lineEdit_la.text())
            else:
                self.check_on_zero("Интенсивность входящего потока требований")
                self.ui.centralwidget.show()
        except ValueError:
            self.check_on_format("интенсивность входящего потока требований")
            self.ui.centralwidget.show()

        try:
            if chk:
                if float(self.ui.lineEdit_intensity_choice.text()) > 0:
                    alpha = float(self.ui.lineEdit_intensity_choice.text())
                    beta = 1 - alpha
                else:
                    self.check_on_zero("Интенсивность наработки на отказ")
                    self.ui.centralwidget.show()
            else:
                if float(self.ui.lineEdit_intensity_choice.text()) > 0:
                    beta = float(self.ui.lineEdit_intensity_choice.text())
                    alpha = 1 - beta
                else:
                    self.check_on_zero("Интенсивность восстановления")
                    self.ui.centralwidget.show()
        except ValueError:
            self.check_on_format("интенсивность наработки на отказ/восстановления")
            self.ui.centralwidget.show()

    # Функция подсчета СМО с ненадежными приборами
    def btn_result(self):
        # Очистка поля вывода результатов
        self.ui.textEdit_result.setText("")
        self.check_input(self.ui.radioButton_alpha.isChecked())

        result = ""
        result += "kappa: " + str(k) + "la: " + str(la) + "mu: " + str(mu) + "alpha: " + str(
            alpha) + "Beta: " + str(
            beta) + "\n"
        # коэффициент использования
        psi = la / (k * mu)
        # вероятность пребывания в системе 0 требований (все приборы свободны)
        sum1 = 0
        for n in range(0, k):
            sum1 += ((k * psi) ** n) / m.factorial(n)
        p = np.zeros(k + 1)
        p[0] = (((k * psi) ** k) / (m.factorial(k) * (1 - psi)) + sum1) ** -1
        # вероятности пребывания в системе n требований (от 1 до k)
        for n in range(1, k + 1):
            p[n] = p[0] * ((k * psi) ** n) / m.factorial(n)
        result += "Вероятности состояний системы: " + str(p) + "\n"
        # проверка суммы вероятностей на 1
        check1 = 0
        for i in p:
            check1 += i
        result += "Проверка суммы вероятностей: " + str('%.0f' % check1) + "\n"
        # перебор комбинаций работоспособных/неработоспособных приборов
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
        # проверка суммы вероятностей на 1
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
        self.ui.textEdit_result.setText(result)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my_app = MyWin()
    my_app.show()
    sys.exit(app.exec_())
