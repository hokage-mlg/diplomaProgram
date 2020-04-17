import sys
from graphUI import *
from reliableUI import *
from patternUI import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
import numpy as np
import math as m
from itertools import product
from collections import Counter
from collections import defaultdict

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


def btn_result_reliable(window):
    # коэффициент использования
    psi = la / (k * mu)
    window.lineEdit_use.setText(str('%.4f' % psi))
    # вероятность пребывания в системе 0 требований (все приборы свободны)
    sum1 = 0
    for n in range(0, k):
        sum1 += ((k * psi) ** n) / m.factorial(n)
    p = np.zeros(k + 1)
    p[0] = (((k * psi) ** k) / (m.factorial(k) * (1 - psi)) + sum1) ** -1
    # вероятности пребывания в системе n требований (от 1 до k)
    for n in range(1, k + 1):
        p[n] = p[0] * ((k * psi) ** n) / m.factorial(n)
    # м.о. числа занятых и свободных приборов
    h = psi * k
    window.lineEdit_busy.setText(str('%.4f' % h))
    g = (1 - psi) * k
    window.lineEdit_free.setText(str('%.4f' % g))
    limit = 50
    b = 0
    for i in range(k + 1, limit + 1):
        b += (i - k) * ((psi ** i * k ** k) / m.factorial(k)) * p[0]
    window.lineEdit_queue.setText(str('%.4f' % b))
    # м.о. числа требований в системе
    q = b + h
    window.lineEdit_num_sys.setText(str('%.4f' % q))
    # м.о. длительности пребывания в системе
    u = q / la
    window.lineEdit_time_sys.setText(str('%.4f' % u))
    # м.о. длительности пребывания требований в очереди
    w = b / la
    window.lineEdit_time_queue.setText(str('%.4f' % w))
    # коэффициент загрузки
    k_h = h / k
    window.lineEdit_busy_rate.setText(str('%.4f' % k_h))
    # коэффициент простоя
    k_g = g / k
    window.lineEdit_free_rate.setText(str('%.4f' % k_g))


class MyWin(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.windowReliableResults = QtWidgets.QMainWindow()
        self.windowGraphs = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_result.clicked.connect(lambda: self.btn_result_unreliable())
        self.ui.action.triggered.connect(lambda: self.clicked(self.ui.centralwidget.show()))
        self.ui.pushButton_graphs.clicked.connect(lambda: self.btn_show_graphs())
        self.ui.pushButton_show_reliable.clicked.connect(lambda: self.btn_show_reliable_results())

    def clicked(self, form):
        self.ui.centralwidget.setLayout(form)
        self.ui.centralwidget.adjustSize()

    # region Popups with errors
    def popup_error_format(self, line):
        QtWidgets.QMessageBox.warning(self,
                                      'Ошибка ввода. Некорректный формат.',
                                      'Введите {} в корректном формате'.format(line))

    def popup_error_zero_or_negative(self, line):
        QtWidgets.QMessageBox.warning(self,
                                      'Ошибка ввода. Некорректное значение.', '{} не может принимать нулевое или '
                                                                              'отрицательное значение'.format(line))

    def popup_error_quantity(self, line):
        QtWidgets.QMessageBox.warning(self,
                                      'Ошибка ввода. Некорректное количество.',
                                      'Введите {} в соответствии с количеством приборов'.format(line))

    # endregion
    # region Buttons
    def btn_show_reliable_results(self):
        window = Ui_ReliableResults()
        window.setupUi(self.windowReliableResults)
        self.windowReliableResults.show()
        window.pushButton_close.clicked.connect(lambda: self.windowReliableResults.hide())
        window.pushButton_show.clicked.connect(lambda: btn_result_reliable(window))

    def btn_show_graphs(self):
        window = Ui_GraphWindow()
        window.setupUi(self.windowGraphs)
        self.windowGraphs.show()
        window.pushButton_close.clicked.connect(lambda: self.windowGraphs.hide())

    # endregion

    # region Checkers
    def check_input_format(self):
        global alpha, la, mu, beta, k
        try:
            k = int(self.ui.lineEdit_kappa.text())
        except ValueError:
            self.popup_error_format("количество приборов")
            self.ui.centralwidget.show()

        try:
            mu = float(self.ui.lineEdit_mu.text())
        except ValueError:
            self.popup_error_format("интенсивность обслуживания требования прибором")
            self.ui.centralwidget.show()

        try:
            la = float(self.ui.lineEdit_la.text())
        except ValueError:
            self.popup_error_format("интенсивность входящего потока требований")
            self.ui.centralwidget.show()

        try:
            alpha = list(map(float, self.ui.lineEdit_alpha.text().split()))
        except ValueError:
            self.popup_error_format("интенсивность наработки на отказ")
            self.ui.centralwidget.show()

        try:
            beta = list(map(float, self.ui.lineEdit_beta.text().split()))
        except ValueError:
            self.popup_error_format("интенсивность восстановления")
            self.ui.centralwidget.show()

    def check_on_positive(self):
        if k <= 0:
            self.popup_error_zero_or_negative("Количество приборов")
            self.ui.centralwidget.show()
        elif mu <= 0:
            self.popup_error_zero_or_negative("Интенсивность обслуживания требования прибором")
            self.ui.centralwidget.show()
        elif la <= 0:
            self.popup_error_zero_or_negative("Интенсивность входящего потока требований")
            self.ui.centralwidget.show()
        elif len(alpha) != k:
            self.popup_error_quantity("интенсивность наработки на отказ")
            self.ui.centralwidget.show()
        elif len(beta) != k:
            self.popup_error_quantity("интенсивность восстановления")
            self.ui.centralwidget.show()
        elif not all(n > 0 for n in alpha):
            self.popup_error_zero_or_negative("Интенсивность наработки на отказ")
            self.ui.centralwidget.show()
        elif not all(n > 0 for n in beta):
            self.popup_error_zero_or_negative("Интенсивность восстановления")
            self.ui.centralwidget.show()
        else:
            return True

    # endregion

    # Функция подсчета СМО с ненадежными приборами
    def btn_result_unreliable(self):
        # Очистка поля вывода результатов
        self.ui.textEdit_result.setText("")
        self.check_input_format()
        if self.check_on_positive():
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
            total_1_axis = np.sum(combList, axis=1)
            d = defaultdict(list)
            for i, key in enumerate(total_1_axis):
                if combList[i] not in d[key]:
                    d[key].append(combList[i])
            for key, value in d.items():
                result += str(key) + ":" + str(value) + "\n"
            p_rob = np.zeros(k + 1)
            for key, value in d.items():
                pr_ns = 1
                sum_ns = 0
                for i in range(len(d[key])):
                    # print(d[key][i])
                    for j in range(len(d[key][i])):
                        # print(d[key][i][j])
                        if j == 0:
                            pr_ns = pr_ns * alpha[j] / (alpha[j] + beta[j])
                        else:
                            pr_ns = pr_ns * beta[j] / (alpha[j] + beta[j])
                        sum_ns += pr_ns
                    p_rob[key] += sum_ns
            result += "Распределение вероятностей ненадежных приборов: " + str(p_rob) + "\n"
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
            result += "М.о. числа требований, ожидающих в очереди: " + str('%.4f' % b) + "\n"
            # м.о. числа требований в системе
            q = b + h
            result += "М.о. числа требований в системе: " + str('%.4f' % q) + "\n"
            # м.о. длительности пребывания в системе
            u = q / la
            result += "М.о. длительности пребывания в системе: " + str('%.4f' % u) + "\n"
            # м.о. длительности пребывания требований в очереди
            w = b / la
            result += "М.о. длительности пребывания требований в очереди: " + str('%.4f' % w) + "\n"
            # коэффициент загрузки
            k_h = h / k
            result += "Коэффициент загрузки: " + str('%.4f' % k_h) + "\n"
            # коэффициент простоя
            k_g = g / k
            result += "Коэффициент простоя: " + str('%.4f' % k_g) + "\n"
            # Выводим в правое поле результат
            self.ui.textEdit_result.setText(result)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my_app = MyWin()
    my_app.show()
    sys.exit(app.exec_())
