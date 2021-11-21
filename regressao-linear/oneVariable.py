from numpy import loadtxt, zeros, ones
from numpy.core.defchararray import array
from pylab import scatter, show, title, xlabel, ylabel, plot

ITERATIONS = 2000
ALPHA = 0.01


def readFile():
    data = loadtxt(
        'data1.txt', delimiter=',')

    x = data[:, 0]
    y = data[:, 1]

    return (x, y)


def initializeTheta():
    return zeros(shape=(2, 1))


def getParameters(x, y):
    features = ones(shape=(y.size, 2))
    features[:, 1] = x
    targets = y

    return (features, targets)


def computeCost(x, y, theta):
    predictions = x.dot(theta).flatten()
    sqErrors = (predictions - y) ** 2
    cost = (1.0 / (2 * y.size)) * sqErrors.sum()
    return cost


def gradientDescent(x, y, theta, alpha, iterations):
    costHistory = zeros(shape=(iterations, 1))

    for i in range(iterations):
        predictions = x.dot(theta).flatten()
        x1Errors = (predictions - y) * x[:, 0]
        x2Errors = (predictions - y) * x[:, 1]
        theta[0][0] = theta[0][0] - alpha * (1.0 / y.size) * x1Errors.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / y.size) * x2Errors.sum()
        costHistory[i, 0] = computeCost(x, y, theta)

    return theta, costHistory


def main():
    (x, y) = readFile()

    scatter(x, y, marker='X', c='r')
    title('1.2 - Distribuição de lucros')
    xlabel('População em 10.000')
    ylabel('Lucro em R$10,000')
    show()

    (features, targets) = getParameters(x, y)

    theta, costHistory = gradientDescent(
        features, targets, initializeTheta(), ALPHA, ITERATIONS)

    title('1.3 - Custo / iteração')
    xlabel('Iteração')
    ylabel('Custo')
    plot(costHistory)
    show()

    results = features.dot(theta).flatten()

    scatter(x, y, marker='x', c='r')
    title('1.3 - Gradiente descendente')
    xlabel('População em 10.000')
    ylabel('Lucro em R$10,000')
    plot(x, results)
    show()


if __name__ == '__main__':
    main()
