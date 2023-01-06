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
    cost = 0.0
    for i in range(len(x) - 1):
        cost += (x[i]**2 - 10 * np.cos(2 * np.pi * x[i])) + (x[i + 1]**2 - 10 * np.cos(2 * np.pi * x[i + 1])) + 20
    return cost


def weierstrass(x):
    cost = 0.0
    a = 3
    b = 0.5
    for n in range(len(x) - 1):
        cost += (a ** n) * np.cos((b ** n) * np.pi * x[n])
    return cost

def michalewicz(x):
    som = 0.0
    m = 10
    for i in range(len(x)):
        som += np.sin(x[i]) * pow(np.sin((i * pow(x[i], 2)) / np.pi), (2 * m))
    return -som

def griewank(x):
    sum = 0
    product = 1
    for i in range(len(x)):
        sum += x[i] ** 2
        product *= np.cos(x[i] / np.sqrt(i + 1))
    return 1 + sum / 4000 - product


def ackley(x):
    d = len(x)
    sum = 0
    for i in range(len(x) - 1):
        sum += -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[i] ** 2 + x[i + 1] ** 2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[i]) + np.cos(2 * np.pi * x[i + 1]))) + 20 + np.e
    return sum

def schwefel(x):
    d = len(x)
    sum = 0
    for i in range(d - 1):
        sum += x[i] * np.sin(np.sqrt(np.abs(x[i])))
    return 420.9687 * (d - 1) - sum

