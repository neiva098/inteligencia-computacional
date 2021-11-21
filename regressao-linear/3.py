import pandas as pd

from numpy import loadtxt, zeros, ones, transpose, dot, linalg
from numpy.core.defchararray import array
from numpy.lib.function_base import average
from pylab import scatter, show, title, xlabel, ylabel, plot
from statistics import stdev

ITERATIONS = 2000
ALPHA = 0.001


def readFile():
    data = loadtxt(
        'data2.txt', delimiter=',')
    size = data[:, 0]
    rooms = data[:, 1]
    price = data[:, 2]

    return (size, rooms, price)


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


def calcEqNormalTheta():
    data2_en = pd.read_csv("data2.txt", names=["t", "x", "y"])

    data2_en["x0"] = [1 for i in range(0, len(data2_en["t"]))]

    x = data2_en[["x0", "t", "x"]].to_numpy()
    y = data2_en[["y"]].to_numpy()
    xt = transpose(x)

    return dot(dot(linalg.inv(dot(xt, x)), xt), y)


def main():
    thetaNormal = calcEqNormalTheta()


main()
