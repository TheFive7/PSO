import numpy as np


# x est un individu sous forme de liste de valeurs

def sphere(x):
    som = 0
    for i in range (len(x)):
        som += x[i] * x[i]
    return som


def rosenbrock(x):
    cost = 0.0
    for i in range(len(x) - 1):
        cost += 100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2

    return cost


def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + np.sum([x_i**2 - A * np.cos(2 * np.pi * x_i) for x_i in x])


def weierstrass(x):
    a = 0.5
    b = 3
    k_max = 20
    result = 0
    for k in range(k_max+1):
        result += a**k * np.cos(2 * np.pi * b**k * x)
    return result

def michalewicz(x):
    m = 10
    sum = 0
    for i in range(len(x)):
        sum += np.sin(x[i]) * (np.sin((i + 1) * x[i] ** 2 / np.pi)) ** (2 * m)
    return -sum

def griewank(x):
    sum = 0
    product = 1
    for i in range(len(x)):
        sum += x[i] ** 2
        product *= np.cos(x[i] / np.sqrt(i + 1))
    return 1 + sum / 4000 - product


def ackley(x):
    d = len(x)
    sum_sq = 0
    sum_cos = 0
    for i in range(d):
        sum_sq += x[i] ** 2
        sum_cos += np.cos(2 * np.pi * x[i])
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + 20 + np.e

def schwefel(x):
    d = len(x)
    sum_sq = 0
    for i in range(d):
        sum_sq += x[i] * np.sin(np.sqrt(np.abs(x[i])))
    return 418.9829 * d - sum_sq

