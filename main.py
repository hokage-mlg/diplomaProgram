import sys
import os
from graphUI import *
from reliable_sysUI import *
from mainUI import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QFileDialog
import numpy as np
import math as m
from itertools import product
from collections import defaultdict

from matplotlib import pyplot as plt


# Для .exe приложения (абсолютный путь)
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


Label = resource_path("labelSSU.png")


# region Дополнительные окна

# region Действия на дополнительных окнах
# Функция подсчета характеристик СМО с надежными приборами
def func_for_reliable_sys(window):
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


# График зависимости коэффициента простоя обслуживающих приборов от их интенсивности наработки на отказ
def build_graph_unr_alpha_kg(window):
    if check_input_format(window):
        alpha_graph = np.zeros(num_steps)
        k_g_graph = np.zeros(num_steps)
        alpha_temp = alpha
        for stp in range(0, num_steps):
            alpha_graph[stp] = alpha_temp[0]
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
            # перебор комбинаций работоспособных/неработоспособных приборов
            combList = [i for i in product(range(2), repeat=k)]
            total_1_axis = np.sum(combList, axis=1)
            d = defaultdict(list)
            for i, key in enumerate(total_1_axis):
                if combList[i] not in d[key]:
                    d[key].append(combList[i])
            p_ns = np.zeros(k + 1)
            for key, value in d.items():
                sum_ns = 0
                for i in range(len(d[key])):
                    pr_ns = 1
                    for j in range(len(d[key][i])):
                        if d[key][i][j] == 0:
                            pr_ns *= alpha_temp[j] / (alpha_temp[j] + beta[j])
                        else:
                            pr_ns *= beta[j] / (alpha_temp[j] + beta[j])
                    sum_ns += pr_ns
                p_ns[key] = sum_ns
            # коэффициент использования ненадежных приборов
            sum2 = 0
            for n in range(0, k + 1):
                sum2 += n * p_ns[n]
            psi_ns = la / (mu * sum2)
            # м.о. числа занятых и свободных приборов
            g = (1 - psi_ns) * k
            # коэффициент простоя
            k_g = g / k
            k_g_graph[stp] = k_g
            alpha_temp[0] += step_size
        plt.figure(5)
        plt.gcf().canvas.set_window_title("График зависимости для ненадежной системы")
        plt.ylabel('Коэффициент простоя')
        plt.xlabel('Интенсивность наработки на отказ')
        plt.plot(alpha_graph, k_g_graph)
        plt.show()


# График зависимости м.о. количества требований в системе от интенсивности наработки на отказ
def build_graph_unr_alpha_n(window):
    if check_input_format(window):
        alpha_graph = np.zeros(num_steps)
        n_graph = np.zeros(num_steps)
        alpha_temp = alpha
        for stp in range(0, num_steps):
            alpha_graph[stp] = alpha_temp[0]
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
            # перебор комбинаций работоспособных/неработоспособных приборов
            combList = [i for i in product(range(2), repeat=k)]
            total_1_axis = np.sum(combList, axis=1)
            d = defaultdict(list)
            for i, key in enumerate(total_1_axis):
                if combList[i] not in d[key]:
                    d[key].append(combList[i])
            p_ns = np.zeros(k + 1)
            for key, value in d.items():
                sum_ns = 0
                for i in range(len(d[key])):
                    pr_ns = 1
                    for j in range(len(d[key][i])):
                        if d[key][i][j] == 0:
                            pr_ns = pr_ns * alpha_temp[j] / (alpha_temp[j] + beta[j])
                        else:
                            pr_ns = pr_ns * beta[j] / (alpha_temp[j] + beta[j])
                    sum_ns += pr_ns
                p_ns[key] = sum_ns
            # коэффициент использования ненадежных приборов
            sum2 = 0
            for n in range(0, k + 1):
                sum2 += n * p_ns[n]
            psi_ns = la / (mu * sum2)
            # м.о. числа занятых и свободных приборов
            h = psi_ns * k
            # м.о. числа требований, ожидающих в очереди
            limit = 50
            b = 0
            for i in range(k + 1, limit + 1):
                for j in range(0, k + 1):
                    b += (i - j * p_ns[j]) * ((psi_ns ** i * k ** k) / m.factorial(k)) * p[0]
            # м.о. числа требований в системе
            q = b + h
            for i in range(0, len(alpha_temp)):
                alpha_temp[i] += step_size
            n_graph[stp] = q
        plt.figure(9)
        plt.gcf().canvas.set_window_title("График зависимости для ненадежной системы")
        plt.ylabel('М.о. количества требований в системе')
        plt.xlabel('Интенсивность наработки на отказ')
        plt.plot(alpha_graph, n_graph)
        plt.show()


# График зависимости м.о. длительности пребывания требований в системе от интенсивности наработки на отказ
def build_graph_unr_alpha_u(window):
    if check_input_format(window):
        alpha_graph = np.zeros(num_steps)
        u_graph = np.zeros(num_steps)
        alpha_temp = alpha
        for stp in range(0, num_steps):
            alpha_graph[stp] = alpha_temp[0]
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
            # перебор комбинаций работоспособных/неработоспособных приборов
            combList = [i for i in product(range(2), repeat=k)]
            total_1_axis = np.sum(combList, axis=1)
            d = defaultdict(list)
            for i, key in enumerate(total_1_axis):
                if combList[i] not in d[key]:
                    d[key].append(combList[i])
            p_ns = np.zeros(k + 1)
            for key, value in d.items():
                sum_ns = 0
                for i in range(len(d[key])):
                    pr_ns = 1
                    for j in range(len(d[key][i])):
                        if d[key][i][j] == 0:
                            pr_ns = pr_ns * alpha_temp[j] / (alpha_temp[j] + beta[j])
                        else:
                            pr_ns = pr_ns * beta[j] / (alpha_temp[j] + beta[j])
                    sum_ns += pr_ns
                p_ns[key] = sum_ns
            # коэффициент использования ненадежных приборов
            sum2 = 0
            for n in range(0, k + 1):
                sum2 += n * p_ns[n]
            psi_ns = la / (mu * sum2)
            # м.о. числа занятых и свободных приборов
            h = psi_ns * k
            # м.о. числа требований, ожидающих в очереди
            limit = 50
            b = 0
            for i in range(k + 1, limit + 1):
                for j in range(0, k + 1):
                    b += (i - j * p_ns[j]) * ((psi_ns ** i * k ** k) / m.factorial(k)) * p[0]
            # м.о. числа требований в системе
            q = b + h
            u = q / la
            for i in range(0, len(alpha_temp)):
                alpha_temp[i] += step_size
            u_graph[stp] = u
        plt.figure(10)
        plt.gcf().canvas.set_window_title("График зависимости для ненадежной системы")
        plt.ylabel('М.о. длительности пребывания требований в системе')
        plt.xlabel('Интенсивность наработки на отказ')
        plt.plot(alpha_graph, u_graph)
        plt.show()


# График зависимости коэффициента загрузки обслуживающих приборов от интенсивности их восстановления
def build_graph_unr_beta_kh(window):
    if check_input_format(window):
        beta_graph = np.zeros(num_steps)
        k_h_graph = np.zeros(num_steps)
        beta_temp = beta
        for stp in range(0, num_steps):
            beta_graph[stp] = beta_temp[len(beta_temp) - 1]
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
            # перебор комбинаций работоспособных/неработоспособных приборов
            combList = [i for i in product(range(2), repeat=k)]
            total_1_axis = np.sum(combList, axis=1)
            d = defaultdict(list)
            for i, key in enumerate(total_1_axis):
                if combList[i] not in d[key]:
                    d[key].append(combList[i])
            p_ns = np.zeros(k + 1)
            for key, value in d.items():
                sum_ns = 0
                for i in range(len(d[key])):
                    pr_ns = 1
                    for j in range(len(d[key][i])):
                        if d[key][i][j] == 0:
                            pr_ns *= alpha[j] / (alpha[j] + beta_temp[j])
                        else:
                            pr_ns *= beta_temp[j] / (alpha[j] + beta_temp[j])
                    sum_ns += pr_ns
                p_ns[key] = sum_ns
            # коэффициент использования ненадежных приборов
            sum2 = 0
            for n in range(0, k + 1):
                sum2 += n * p_ns[n]
            psi_ns = la / (mu * sum2)
            # м.о. числа занятых и свободных приборов
            h = psi_ns * k
            # коэффициент загрузки
            k_h = h / k
            k_h_graph[stp] = k_h
            beta_temp[len(beta_temp) - 1] += step_size
        plt.figure(6)
        plt.gcf().canvas.set_window_title("График зависимости для ненадежной системы")
        plt.ylabel('Коэффициент загрузки')
        plt.xlabel('Интенсивность восстановления')
        plt.plot(beta_graph, k_h_graph)
        plt.show()


# График зависимости математического ожидания количества требований в очереди от
# интенсивности восстановления обслуживающих приборов
def build_graph_unr_beta_b(window):
    if check_input_format(window):
        beta_graph = np.zeros(num_steps)
        b_graph = np.zeros(num_steps)
        beta_temp = beta
        for stp in range(0, num_steps):
            beta_graph[stp] = beta_temp[len(beta_temp) - 1]
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
            # перебор комбинаций работоспособных/неработоспособных приборов
            combList = [i for i in product(range(2), repeat=k)]
            total_1_axis = np.sum(combList, axis=1)
            d = defaultdict(list)
            for i, key in enumerate(total_1_axis):
                if combList[i] not in d[key]:
                    d[key].append(combList[i])
            p_ns = np.zeros(k + 1)
            for key, value in d.items():
                sum_ns = 0
                for i in range(len(d[key])):
                    pr_ns = 1
                    for j in range(len(d[key][i])):
                        if d[key][i][j] == 0:
                            pr_ns *= alpha[j] / (alpha[j] + beta_temp[j])
                        else:
                            pr_ns *= beta_temp[j] / (alpha[j] + beta_temp[j])
                    sum_ns += pr_ns
                p_ns[key] = sum_ns
            # коэффициент использования ненадежных приборов
            sum2 = 0
            for n in range(0, k + 1):
                sum2 += n * p_ns[n]
            psi_ns = la / (mu * sum2)
            limit = 50
            b = 0
            for i in range(k + 1, limit + 1):
                for j in range(0, k + 1):
                    b += (i - j * p_ns[j]) * ((psi_ns ** i * k ** k) / m.factorial(k)) * p[0]
            b_graph[stp] = b
            beta_temp[len(beta_temp) - 1] += step_size
        plt.figure(7)
        plt.gcf().canvas.set_window_title("График зависимости для ненадежной системы")
        plt.ylabel('М.о. количества требований в очереди')
        plt.xlabel('Интенсивность восстановления')
        plt.plot(beta_graph, b_graph)
        plt.show()


# График зависимости математического ожидания длительности пребывания требований
# в системе от интенсивности входящего потока требований
def build_graph_unr_la_u(window):
    if check_input_format(window):
        la_graph = np.zeros(num_steps)
        u_graph = np.zeros(num_steps)
        la_temp = la
        for stp in range(0, num_steps):
            la_graph[stp] = la_temp
            # коэффициент использования
            psi = la_temp / (k * mu)
            # вероятность пребывания в системе 0 требований (все приборы свободны)
            sum1 = 0
            for n in range(0, k):
                sum1 += ((k * psi) ** n) / m.factorial(n)
            p = np.zeros(k + 1)
            p[0] = (((k * psi) ** k) / (m.factorial(k) * (1 - psi)) + sum1) ** -1
            # вероятности пребывания в системе n требований (от 1 до k)
            for n in range(1, k + 1):
                p[n] = p[0] * ((k * psi) ** n) / m.factorial(n)
            # перебор комбинаций работоспособных/неработоспособных приборов
            combList = [i for i in product(range(2), repeat=k)]
            total_1_axis = np.sum(combList, axis=1)
            d = defaultdict(list)
            for i, key in enumerate(total_1_axis):
                if combList[i] not in d[key]:
                    d[key].append(combList[i])
            p_ns = np.zeros(k + 1)
            for key, value in d.items():
                sum_ns = 0
                for i in range(len(d[key])):
                    pr_ns = 1
                    for j in range(len(d[key][i])):
                        if d[key][i][j] == 0:
                            pr_ns *= alpha[j] / (alpha[j] + beta[j])
                        else:
                            pr_ns *= pr_ns * beta[j] / (alpha[j] + beta[j])
                    sum_ns += pr_ns
                p_ns[key] = sum_ns
            # коэффициент использования ненадежных приборов
            sum2 = 0
            for n in range(0, k + 1):
                sum2 += n * p_ns[n]
            psi_ns = la_temp / (mu * sum2)
            # м.о. числа занятых и свободных приборов
            h = psi_ns * k
            # м.о. числа требований, ожидающих в очереди
            limit = 50
            b = 0
            for i in range(k + 1, limit + 1):
                for j in range(0, k + 1):
                    b += (i - j * p_ns[j]) * ((psi_ns ** i * k ** k) / m.factorial(k)) * p[0]
            # м.о. числа требований в системе
            q = b + h
            # м.о. длительности пребывания в системе
            u = q / la_temp
            u_graph[stp] = u
            la_temp += step_size
        plt.figure(3)
        plt.gcf().canvas.set_window_title("График зависимости для ненадежной системы")
        plt.ylabel('М.о. длительности пребывания требований в системе')
        plt.xlabel('Интенсивность входящего потока требований')
        plt.plot(la_graph, u_graph)
        plt.show()


# График зависимости математического ожидания длительности пребывания требований
# в очереди от интенсивности обслуживания требования прибором
def build_graph_unr_mu_w(window):
    if check_input_format(window):
        mu_graph = np.zeros(num_steps)
        w_graph = np.zeros(num_steps)
        mu_temp = mu
        for stp in range(0, num_steps):
            mu_graph[stp] = mu_temp
            # коэффициент использования
            psi = la / (k * mu_temp)
            # вероятность пребывания в системе 0 требований (все приборы свободны)
            sum1 = 0
            for n in range(0, k):
                sum1 += ((k * psi) ** n) / m.factorial(n)
            p = np.zeros(k + 1)
            p[0] = (((k * psi) ** k) / (m.factorial(k) * (1 - psi)) + sum1) ** -1
            # вероятности пребывания в системе n требований (от 1 до k)
            for n in range(1, k + 1):
                p[n] = p[0] * ((k * psi) ** n) / m.factorial(n)
                # перебор комбинаций работоспособных/неработоспособных приборов
            combList = [i for i in product(range(2), repeat=k)]
            total_1_axis = np.sum(combList, axis=1)
            d = defaultdict(list)
            for i, key in enumerate(total_1_axis):
                if combList[i] not in d[key]:
                    d[key].append(combList[i])
            p_ns = np.zeros(k + 1)
            for key, value in d.items():
                sum_ns = 0
                for i in range(len(d[key])):
                    pr_ns = 1
                    for j in range(len(d[key][i])):
                        if d[key][i][j] == 0:
                            pr_ns *= alpha[j] / (alpha[j] + beta[j])
                        else:
                            pr_ns *= beta[j] / (alpha[j] + beta[j])
                    sum_ns += pr_ns
                p_ns[key] = sum_ns
            # коэффициент использования ненадежных приборов
            sum2 = 0
            for n in range(0, k + 1):
                sum2 += n * p_ns[n]
            psi_ns = la / (mu_temp * sum2)
            # м.о. числа требований, ожидающих в очереди
            limit = 50
            b = 0
            for i in range(k + 1, limit + 1):
                for j in range(0, k + 1):
                    b += (i - j * p_ns[j]) * ((psi_ns ** i * k ** k) / m.factorial(k)) * p[0]
            # м.о. длительности пребывания требований в очереди
            w = b / la
            w_graph[stp] = w
            mu_temp += step_size
        plt.figure(4)
        plt.gcf().canvas.set_window_title("График зависимости для ненадежной системы")
        plt.ylabel('М.о. длительности пребывания требований в очереди')
        plt.xlabel('Интенсивность обслуживания требования одним прибором')
        plt.plot(mu_graph, w_graph)
        plt.show()


# График зависимости математического ожидания длительности
# пребывания требований в системе от интенсивности входящего потока требований для системы с надежными приборами
def build_graph_r_la_u(window):
    if check_input_format(window):
        check_input_format(window)
        la_graph = np.zeros(num_steps)
        u_graph = np.zeros(num_steps)
        la_temp = la
        for stp in range(0, num_steps):
            la_graph[stp] = la_temp
            # коэффициент использования
            psi = la_temp / (k * mu)
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
            limit = 50
            b = 0
            for i in range(k + 1, limit + 1):
                b += (i - k) * ((psi ** i * k ** k) / m.factorial(k)) * p[0]
            # м.о. числа требований в системе
            q = b + h
            # м.о. длительности пребывания в системе
            u = q / la_temp
            u_graph[stp] = u
            la_temp += step_size
        plt.figure(1)
        plt.gcf().canvas.set_window_title("График зависимости для надежной системы")
        plt.ylabel('М.о. длительности пребывания требований в системе')
        plt.xlabel('Интенсивность входящего потока требований')
        plt.plot(la_graph, u_graph)
        plt.show()


# График зависимости математического ожидания длительности пребывания
# требований в очереди от интенсивности обслуживания требования прибором для системы с надежными приборами
def build_graph_r_mu_w(window):
    if check_input_format(window):
        mu_graph = np.zeros(num_steps)
        w_graph = np.zeros(num_steps)
        mu_temp = mu
        for stp in range(0, num_steps):
            mu_graph[stp] = mu_temp
            # коэффициент использования
            psi = la / (k * mu_temp)
            # вероятность пребывания в системе 0 требований (все приборы свободны)
            sum1 = 0
            for n in range(0, k):
                sum1 += ((k * psi) ** n) / m.factorial(n)
            p = np.zeros(k + 1)
            p[0] = (((k * psi) ** k) / (m.factorial(k) * (1 - psi)) + sum1) ** -1
            # вероятности пребывания в системе n требований (от 1 до k)
            for n in range(1, k + 1):
                p[n] = p[0] * ((k * psi) ** n) / m.factorial(n)
            limit = 50
            b = 0
            for i in range(k + 1, limit + 1):
                b += (i - k) * ((psi ** i * k ** k) / m.factorial(k)) * p[0]
            # м.о. длительности пребывания требований в очереди
            w = b / la
            w_graph[stp] = w
            mu_temp += step_size
        plt.figure(2)
        plt.gcf().canvas.set_window_title("График зависимости для надежной системы")
        plt.ylabel('М.о. длительности пребывания требований в очереди')
        plt.xlabel('Интенсивность обслуживания требования одним прибором')
        plt.plot(mu_graph, w_graph)
        plt.show()


# endregion
# region Checkers
def check_input_format(window):
    global step_size, num_steps
    try:
        num_steps = int(window.lineEdit_num_steps.text())
    except ValueError:
        msg = QMessageBox()
        msg.setWindowIcon(QtGui.QIcon(Label))
        msg.setWindowTitle("Ошибка ввода. Некорректный формат.")
        msg.setText("Введите количество шагов в корректном формате.")
        msg.exec_()
    try:
        step_size = float(window.lineEdit_step_size.text())
    except ValueError:
        msg = QMessageBox()
        msg.setWindowIcon(QtGui.QIcon(Label))
        msg.setWindowTitle("Ошибка ввода. Некорректный формат.")
        msg.setText("Введите размер шага в корректном формате.")
        msg.exec_()

    if num_steps <= 0:
        msg = QMessageBox()
        msg.setWindowIcon(QtGui.QIcon(Label))
        msg.setWindowTitle("Ошибка ввода. Некорректное значение.")
        msg.setText("Количество шагов не может принимать нулевое или отрицательное значение.")
        msg.exec_()

    elif step_size <= 0:
        msg = QMessageBox()
        msg.setWindowIcon(QtGui.QIcon(Label))
        msg.setWindowTitle("Ошибка ввода. Некорректное значение.")
        msg.setText("Размер шага не может принимать нулевое или отрицательное значение.")
        msg.exec_()

    else:
        return True


# endregion
# endregion

# region Главное окно
class MyWin(QtWidgets.QMainWindow):
    # region Инициализация приложения
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.windowReliableResults = QtWidgets.QMainWindow()
        self.windowGraphs = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QtGui.QIcon(Label))
        self.resize(555, 315)
        self.ui.pushButton_result.setGeometry(QtCore.QRect(240, 240, 93, 28))
        self.ui.textEdit_result.hide()
        self.ui.pushButton_show_reliable.hide()
        self.ui.pushButton_graphs.hide()
        self.ui.pushButton_clear.hide()
        self.ui.pushButton_save.hide()
        self.ui.pushButton_open.hide()
        self.ui.pushButton_result.clicked.connect(lambda: self.func_for_unreliable_sys())
        self.ui.action.triggered.connect(lambda: self.clicked(self.ui.centralwidget.show()))
        self.ui.pushButton_graphs.clicked.connect(lambda: self.window_graphs())
        self.ui.pushButton_show_reliable.clicked.connect(lambda: self.window_reliable_sys())
        self.ui.pushButton_clear.clicked.connect(lambda: self.clear_result())
        self.ui.pushButton_save.clicked.connect(lambda: self.save_result())
        self.ui.pushButton_open.clicked.connect(lambda: self.open_result())

    def clicked(self, form):
        self.ui.centralwidget.setLayout(form)
        self.ui.centralwidget.adjustSize()

    # endregion
    # region Всплывающие окна с предупреждениями
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

    # region Установка дополнительных окон
    def window_reliable_sys(self):
        window = Ui_ReliableResults()
        window.setupUi(self.windowReliableResults)
        self.windowReliableResults.show()
        self.windowReliableResults.setWindowIcon(QtGui.QIcon(Label))
        window.pushButton_close.clicked.connect(lambda: self.windowReliableResults.hide())
        window.pushButton_show.clicked.connect(lambda: func_for_reliable_sys(window))

    def window_graphs(self):
        window = Ui_GraphWindow()
        window.setupUi(self.windowGraphs)
        self.windowGraphs.show()
        self.windowGraphs.setWindowIcon(QtGui.QIcon(Label))
        window.pushButton_close.clicked.connect(lambda: self.windowGraphs.hide())
        window.pushButton_r_la_u.clicked.connect(lambda: build_graph_r_la_u(window))
        window.pushButton_r_mu_w.clicked.connect(lambda: build_graph_r_mu_w(window))
        window.pushButton_unr_la_u.clicked.connect(lambda: build_graph_unr_la_u(window))
        window.pushButton_unr_mu_w.clicked.connect(lambda: build_graph_unr_mu_w(window))
        window.pushButton_unr_alpha_kg.clicked.connect(lambda: build_graph_unr_alpha_kg(window))
        window.pushButton_unr_beta_kh.clicked.connect(lambda: build_graph_unr_beta_kh(window))
        window.pushButton_unr_beta_b.clicked.connect(lambda: build_graph_unr_beta_b(window))
        window.pushButton_unr_alpha_n.clicked.connect(lambda: build_graph_unr_alpha_n(window))
        window.pushButton_unr_alpha_u.clicked.connect(lambda: build_graph_unr_alpha_u(window))

    # endregion

    # region Проверка ввода
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
            self.popup_error_format("интенсивности наработки на отказ")
            self.ui.centralwidget.show()

        try:
            beta = list(map(float, self.ui.lineEdit_beta.text().split()))
        except ValueError:
            self.popup_error_format("интенсивности восстановления")
            self.ui.centralwidget.show()
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
            self.popup_error_quantity("интенсивности наработки на отказ")
            self.ui.centralwidget.show()
        elif len(beta) != k:
            self.popup_error_quantity("интенсивности восстановления")
            self.ui.centralwidget.show()
        elif not all(n > 0 for n in alpha):
            self.popup_error_zero_or_negative("Интенсивности наработки на отказ")
            self.ui.centralwidget.show()
        elif not all(n > 0 for n in beta):
            self.popup_error_zero_or_negative("Интенсивности восстановления")
            self.ui.centralwidget.show()
        else:
            return True

    # endregion

    # region Функции главного окна
    def save_result(self):
        file_name = QFileDialog.getSaveFileName(self, "Save File", os.getenv("HOME"))
        with open(file_name[0] + '.txt', "w") as f:
            res = self.ui.textEdit_result.toPlainText()
            f.write(res)

    def clear_result(self):
        self.ui.textEdit_result.clear()

    def open_result(self):
        file_name = QFileDialog.getOpenFileName(self, "Open File", os.getenv("HOME"))
        with open(file_name[0], "r") as f:
            res = f.read()
            self.ui.textEdit_result.setText(res)

    # Функция подсчета характеристик СМО с ненадежными приборами
    def func_for_unreliable_sys(self):
        # Очистка поля вывода результатов
        self.ui.textEdit_result.setText("")
        if self.check_input_format():
            result = ""
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
            # перебор комбинаций работоспособных / неработоспособных приборов
            combList = [i for i in product(range(2), repeat=k)]
            total_1_axis = np.sum(combList, axis=1)
            d = defaultdict(list)
            for i, key in enumerate(total_1_axis):
                if combList[i] not in d[key]:
                    d[key].append(combList[i])
            p_ns = np.zeros(k + 1)
            for key, value in d.items():
                sum_ns = 0
                for i in range(len(d[key])):
                    pr_ns = 1
                    for j in range(len(d[key][i])):
                        if d[key][i][j] == 0:
                            pr_ns = pr_ns * alpha[j] / (alpha[j] + beta[j])
                        else:
                            pr_ns = pr_ns * beta[j] / (alpha[j] + beta[j])
                    sum_ns += pr_ns
                p_ns[key] = sum_ns
            result += "Распределение вероятностей работоспособности ненадежных приборов: " + str(p_ns) + "\n"
            # коэффициент использования ненадежных приборов
            sum2 = 0
            for n in range(0, k + 1):
                sum2 += n * p_ns[n]
            psi_ns = la / (mu * sum2)
            result += "Коэффициент использования ненадежных приборов: " + str(psi_ns) + "\n"
            # м.о. числа занятых и свободных приборов
            h = psi_ns * k
            result += "М.о. числа занятых приборов: " + str(h) + "\n"
            g = (1 - psi_ns) * k
            result += "М.о. числа свободных приборов: " + str(g) + "\n"
            # м.о. числа требований, ожидающих в очереди
            limit = 50
            b = 0
            for i in range(k + 1, limit + 1):
                for j in range(0, k + 1):
                    b += (i - j * p_ns[j]) * ((psi_ns ** i * k ** k) / m.factorial(k)) * p[0]
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
            # выводим в нижнее поле результат
            self.ui.textEdit_result.setText(result)
            self.ui.textEdit_result.show()
            # перестроение окна
            self.ui.pushButton_clear.show()
            self.ui.pushButton_save.show()
            self.ui.pushButton_open.show()
            self.ui.pushButton_show_reliable.show()
            self.ui.pushButton_graphs.show()
            self.ui.pushButton_result.setGeometry(QtCore.QRect(381, 257, 93, 28))
            self.resize(804, 714)


# endregion


# endregion

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my_app = MyWin()
    my_app.show()
    sys.exit(app.exec_())
