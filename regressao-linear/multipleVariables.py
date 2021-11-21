from numpy import loadtxt, zeros, ones, arange, matmul
from numpy.linalg import inv
from numpy.core.fromnumeric import transpose
from numpy.lib.function_base import average
from pylab import scatter, show, title, xlabel, ylabel, plot
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from statistics import stdev

ITERATIONS = 1000
ALPHA = 0.001


def readFile():
    data = loadtxt(
        'data2.txt', delimiter=',')
    size = data[:, 0]
    rooms = data[:, 1]
    price = data[:, 2]

    return (size, rooms, price)


def plot_graph(x, y, label_title, x_label, y_label):
    scatter(x, y, marker='X', c='r')
    title(label_title)
    xlabel(x_label)
    ylabel(y_label)
    show()


def feature_normalization(feature):
    normalized_features = feature

    avg = average(feature)
    std_dev = stdev(feature)
    for i, feat in enumerate(feature):
        normalized_features[i] = (feat - avg)/std_dev

    return normalized_features


def normalize_features(size, rooms, price):
    normalized_size = feature_normalization(size)
    normalized_rooms = feature_normalization(rooms)
    normalized_price = feature_normalization(price)

    return (normalized_size, normalized_rooms, normalized_price)


def getParameters(x_1, x_2, y):
    features = ones(shape=(y.size, 3))
    features[:, 1] = x_1
    features[:, 2] = x_2
    targets = y

    return (features, targets)


def initializeTheta():
    return zeros(shape=(3, 1))


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
        x3Errors = (predictions - y) * x[:, 2]
        theta[0][0] = theta[0][0] - alpha * (1.0 / y.size) * x1Errors.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / y.size) * x2Errors.sum()
        theta[2][0] = theta[1][0] - alpha * (1.0 / y.size) * x3Errors.sum()
        costHistory[i, 0] = computeCost(x, y, theta)

    return theta, costHistory


def normal_equation(features, targets):
    transposed_features = transpose(features)
    aux = inv(matmul(transposed_features, features))
    theta = matmul(matmul(aux, transposed_features), targets)

    return theta


def main():
    (size, rooms, price) = readFile()

    (normalized_size, normalized_rooms,
     normalized_price) = normalize_features(size, rooms, price)

    plot_graph(normalized_size, normalized_price, '2.1 - Distribuição de preços por tamanho normalizado',
               'Tamanho', 'Preço')
    plot_graph(normalized_rooms, normalized_price, '2.1 - Distribuição de preços por numero de quartos normalizado',
               'Quartos', 'Preço')
    plot_graph(normalized_size, normalized_rooms, '2.1 - Distribuição de tamanhos por numero de quartos normalizado',
               'Tamanho', 'Quartos')

    (normalized_features, normalized_targets) = getParameters(
        normalized_size, normalized_rooms, normalized_price)
    gradient_theta, cost_history = gradientDescent(
        normalized_features, normalized_targets, initializeTheta(), ALPHA, ITERATIONS)

    title('2.2 - Custo / iteração')
    xlabel('Iteração')
    ylabel('Custo')
    plot(arange(ITERATIONS), cost_history)
    show()

    (features, targets) = getParameters(size, rooms, price)
    normal_theta = normal_equation(features, targets)

    normal_results = features.dot(normal_theta).flatten()
    gradient_results = features.dot(gradient_theta).flatten()

    rms_normal = sqrt(mean_squared_error(targets, normal_results))
    print('Normal Error: ', rms_normal)
    rms_gradient = sqrt(mean_squared_error(targets, gradient_results))
    print('Gradient Error: ', rms_gradient)

    plt.ylim(0, 200)
    scatter(targets, calculate_error(normal_results, targets),
             marker='x', c='r')
    scatter(targets,calculate_error(gradient_results, targets),
             marker='o', c='b')
    title('3 - Erros x Preço')
    xlabel('Preço')
    ylabel('Erro (%)')
    show()


def calculate_error(pred, target):
    return (abs(pred - target)/abs(target)) * 100


if __name__ == '__main__':
    main()
