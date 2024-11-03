import numpy as np
import matplotlib.pyplot as plt

# Визначення функцій
def f(x, y):
    return y + x**2

def fe(x):
    return -x**2 - 2*x + 2*np.exp(x) - 2

# Метод Рунге-Кутта 4-го порядку з фіксованим кроком
def runge_kutta(a, b, h):
    x_values = []
    y_values = []
    exact_values = []
    local_errors = []
    runge_errors = []

    x = a
    y = 0
    while x <= b:
        exact_y = fe(x)
        x_values.append(x)
        y_values.append(y)
        exact_values.append(exact_y)

        local_error = abs(exact_y - y)
        local_errors.append(local_error)
        
        # Рунге-Кутта 4-го порядку
        k1 = f(x, y)
        k2 = f(x + h/2, y + h * k1 / 2)
        k3 = f(x + h/2, y + h * k2 / 2)
        k4 = f(x + h, y + h * k3)
        y_new = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        # Подвійний крок для оцінки похибки
        h_half = h / 2
        k1 = f(x, y)
        k2 = f(x + h_half / 2, y + h_half * k1 / 2)
        k3 = f(x + h_half / 2, y + h_half * k2 / 2)
        k4 = f(x + h_half, y + h_half * k3)
        y_half_step = y + h_half * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        k1 = f(x + h_half, y_half_step)
        k2 = f(x + 3 * h_half / 2, y_half_step + h_half * k1 / 2)
        k3 = f(x + 3 * h_half / 2, y_half_step + h_half * k2 / 2)
        k4 = f(x + h, y_half_step + h_half * k3)
        y_double_step = y_half_step + h_half * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        runge_error = 16.0/15.0 * abs(y_new - y_double_step)
        runge_errors.append(runge_error)

        # Перехід на наступний крок
        y = y_new
        x += h

    # Побудова графіків
    plt.figure(figsize=(12, 6))
    
    # Графік локальної похибки
    plt.subplot(1, 3, 1)
    plt.plot(x_values, local_errors, label="Локальна похибка")
    plt.xlabel("x")
    plt.ylabel("Локальна похибка")
    plt.legend()
    
    # Графік похибки за методом Рунге
    plt.subplot(1, 3, 2)
    plt.plot(x_values, runge_errors, label="Похибка Рунге", color="orange")
    plt.xlabel("x")
    plt.ylabel("Похибка Рунге")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Автоматичний вибір кроку для досягнення заданої точності
def automatic_step(a, b, eps):
    x_values = []
    h_values = []
    y = 0
    h = 0.01
    x = a
    
    while x < b:
        exact_y = fe(x)
        k1 = f(x, y)
        k2 = f(x + h/2, y + h * k1 / 2)
        k3 = f(x + h/2, y + h * k2 / 2)
        k4 = f(x + h, y + h * k3)
        y_new = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        h_half = h / 2
        y_half_step = y + h_half * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        runge_error = 16.0/15.0 * abs(y_new - y_half_step)
        
        if runge_error > eps:
            h /= 2
        elif runge_error < eps / 32:
            h *= 2
        
        x_values.append(x)
        h_values.append(h)
        
        y = y_new
        x += h
    
    plt.figure(figsize=(6, 6))
    plt.plot(x_values, h_values, label="Крок h(x)", color="green")
    plt.xlabel("x")
    plt.ylabel("Крок h")
    plt.legend()
    plt.show()

# Виконання функцій
a, b = 0, 5
h = 1e-2
eps = 1e-12

runge_kutta(a, b, h)
automatic_step(a, b, eps)
