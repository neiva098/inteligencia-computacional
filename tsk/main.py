import matplotlib.pyplot as plt
import numpy as np

def createFunction():
    x = np.linspace(-10, 10, 1000)

    return (x**2, x)

def putFunction(x, y, label):
    _, ax = plt.subplots()

    ax.plot(x, y, label=label)
    ax.legend(loc='upper right')

def show():
    plt.show()

y,x = createFunction()

putFunction(x, y, 'f(x) = x²')

show()