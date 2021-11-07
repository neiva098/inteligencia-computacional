import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from numpy.core.fromnumeric import mean

# Tarefa
# Utilizando o Método do gradiente mostrados nos slides e vídeos do tópico 3,
# aproximar a saída para f(x) = x^2

LIMIT = 10


def generate_points():
    return np.linspace(-LIMIT, LIMIT, 1000)


def create_base_function(x):
    return x**2


def plot_function(x, y, color, label):
    plt.plot(x, y, label=label, color=color)


def plot_base_function(x):
    y = create_base_function(x)

    plot_function(x, y, 'r', 'f(x) = x²')

    return (y, x)


def plot_approximate_function(x, base):
    y = tsk(x)

    err = mean(((y - base) ** 2) / 2)

    plt.text(-10.7, 0.1, 'Erro ' + str(err))

    plot_function(x, y, 'b', 'Curva estimada ajustada')

    return y


def configure_plot():
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='upper right')
    plt.title('Curva da função f(x)=x² e Curva estimada ajustada')


def show_plot():
    plt.show()


def show():
    configure_plot()
    show_plot()


def tsk(x):
    alpha = 0.01

    def avaliatePertinence(x1, x2, sd1, sd2,  xr):
        w1 = fuzz.gaussmf(xr, x1, sd1)
        w2 = fuzz.gaussmf(xr, x2, sd2)

        return (w1, w2)

    def avaliateFuzzyReturn(p1, p2, q1, q2, w1, w2, xr):
        y1 = p1 * xr + q1
        y2 = p2 * xr + q2
        ye = (np.multiply(w1, y1) + np.multiply(w2, y2)) / (w1 + w2)

        return (y1, y2, ye)

    def avaliatePartialDerivates(ye, yd, w1, w2, y1, y2, x1, x2, sd1, sd2, xr):
        dedy = (ye - yd)
        dydy1 = w1 / (w1 + w2)
        dydy2 = w2 / (w1 + w2)
        dy1dp1 = xr
        dy2dp2 = xr
        dydw1 = w1 * w2 * (y1 - y2) / ((w1 + w2) ** 2)
        dydw2 = w2 * w1 * (y2 - y1) / ((w1 + w2) ** 2)
        dw1dx1b = ((xr - x1) / sd1 ** 2)
        dw2dx2b = ((xr - x2) / sd2 ** 2)
        dw1dsd1 = ((xr - x1) / sd1 ** 3)
        dw2dsd2 = ((xr - x2) / sd2 ** 3)

        return (dedy, dydy1, dydy2, dy1dp1, dy2dp2, dydw1, dydw2, dw1dx1b, dw2dx2b, dw1dsd1, dw2dsd2)

    def avaliateDerivatedErrors(dedy, dydy1, dydy2, dy1dp1, dy2dp2, dydw1, dydw2, dw1dx1b, dw2dx2b, dw1dsd1, dw2dsd2):
        dedp1 = dedy * dydy1 * dy1dp1
        dedp2 = dedy * dydy2 * dy2dp2
        dedq1 = dedy * dydy1
        dedq2 = dedy * dydy2
        dedx1b = dedy * dydw1 * dw1dx1b
        dedx2b = dedy * dydw2 * dw2dx2b
        dedsid1 = dedy * dydw1 * dw1dsd1
        dedsid2 = dedy * dydw2 * dw2dsd2

        return (dedp1, dedp2, dedq1,  dedq2, dedx1b, dedx2b, dedsid1, dedsid2)

    def updateParameters(p1, alpha, dedp1, dedp2, dedq1,  dedq2, dedx1b, dedx2b, dedsid1, dedsid2, p2, q1, q2, x1, x2, sd1, sd2):
        p1 = p1 - alpha * dedp1
        p2 = p2 - alpha * dedp2
        q1 = q1 - alpha * dedq1
        q2 = q2 - alpha * dedq2
        x1 = x1 - alpha * dedx1b
        x2 = x2 - alpha * dedx2b
        sd1 = sd1 - alpha * dedsid1
        sd2 = sd2 - alpha * dedsid2

        return (p1, p2, q1, q2, x1, x2, sd1, sd2)

    def avaliateGaussian(p1, p2, q1, q2, x1, x2, sd1, sd2, x):
        y1 = p1 * x + q1
        y2 = p2 * x + q2
        w1 = fuzz.gaussmf(x,  x1, sd1)
        w2 = fuzz.gaussmf(x,  x2, sd2)

        return (y1, y2, w1, w2)

    sd1 = 1.7
    x1 = 2
    sd2 = 1.7
    x2 = - 2

    p1 = - 2
    q1 = 0
    p2 = 2
    q2 = 0

    for _ in range(1000):
        for _ in range(100):
            alpha = 0.01
            xr = np.random.uniform(-LIMIT, LIMIT)
            yd = xr ** 2

            w1, w2 = avaliatePertinence(x1, x2, sd1, sd2,  xr)

            y1, y2, ye = avaliateFuzzyReturn(p1, p2, q1, q2, w1, w2, xr)

            dedy, dydy1, dydy2, dy1dp1, dy2dp2, dydw1, dydw2, dw1dx1b, dw2dx2b, dw1dsd1, dw2dsd2 = avaliatePartialDerivates(
                ye, yd, w1, w2, y1, y2, x1, x2, sd1, sd2, xr)

            dedp1, dedp2, dedq1,  dedq2, dedx1b, dedx2b, dedsid1, dedsid2 = avaliateDerivatedErrors(
                dedy, dydy1, dydy2, dy1dp1, dy2dp2, dydw1, dydw2, dw1dx1b, dw2dx2b, dw1dsd1, dw2dsd2)

            p1, p2, q1, q2, x1, x2, sd1, sd2 = updateParameters(
                p1, alpha, dedp1, dedp2, dedq1,  dedq2, dedx1b, dedx2b, dedsid1, dedsid2, p2, q1, q2, x1, x2, sd1, sd2)

    y1, y2, w1, w2 = avaliateGaussian(p1, p2, q1, q2, x1, x2, sd1, sd2, x)

    return (np.multiply(w1, y1) + np.multiply(w2, y2)) / (w1 + w2)


def main():
    x = generate_points()
    plot_base_function(x)
    plot_approximate_function(x, x**2)
    show()


if __name__ == '__main__':
    main()
