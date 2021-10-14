import random
from typing import List
import time
import copy
import math


ABS_MAX_DIFF = 0.03


class TrainPattern:
    def __init__(self, xs: List, y):
        self.xs_train = xs
        self.y = y


def duplicate_set(pattern: TrainPattern, num_of_samples):
    patterns = []
    for i in range(num_of_samples):
        xs_new = [random.uniform(pattern.xs_train[i] - ABS_MAX_DIFF, pattern.xs_train[i] + ABS_MAX_DIFF) for i in range(len(pattern.xs_train))]
        patterns.append(TrainPattern(xs_new, pattern.y))
    return patterns


def prepare_training_sets(num_of_samples):
    base_patterns = [TrainPattern([0, 0], 0), TrainPattern([0, 1], 0), TrainPattern([1, 0], 0)]
    patterns = []
    for pattern in base_patterns:
        patterns.extend(duplicate_set(pattern, num_of_samples))

    class_1_training_set = TrainPattern([1, 1], 1)
    patterns.extend(duplicate_set(class_1_training_set, 3*num_of_samples))
    return patterns


def unipolar_fun(arg, theta):
    return 1 if arg > theta else 0


def bipolar_fun(arg, theta):
    return 1 if arg > theta else -1


def train_perceptron(xs_train, y_train, weights, theta, alpha, act_fun):
    assert len(xs_train) == len(weights), f"xs and weights should have the same length! xs: {len(xs_train)}, weights: {len(weights)}"

    if act_fun == bipolar_fun:
        xs_train = [1 if abs(x-1) < ABS_MAX_DIFF else -1 for x in xs_train]
        y_train = -1 if y_train == 0 else y_train

    z = sum((x * w for x, w in zip(xs_train, weights)))
    y_obtained = act_fun(z, theta)
    error = y_train - y_obtained
    if error == 0:
        return error
    for i in range(len(weights)):
        weights[i] += alpha * error * xs_train[i]
    return error


def speed_vs_theta(weights_base, thetas, train_patterns):
    alpha = 0.0001
    for theta in thetas:
        for i in range(10):
            weights = weights_base.copy()
            epoch = 0
            no_errors = False
            while not no_errors:
                epoch += 1
                no_errors = True
                for train_pattern in train_patterns:
                    if train_perceptron(train_pattern.xs_train, train_pattern.y, weights, theta, alpha, act_fun=unipolar_fun) != 0:
                        no_errors = False
            print(f"theta: {theta} próba nr {i} epoki: {epoch}")


def speed_vs_interval(weights_intervals, train_patterns):
    alpha = 0.0001
    theta = 0
    for interval in weights_intervals:
        weights_base = [random.uniform(interval[0], interval[1]) for i in range(3)]
        for i in range(10):
            weights = weights_base.copy()
            epoch = 0
            no_errors = False
            while not no_errors:
                epoch += 1
                no_errors = True
                for train_pattern in train_patterns:
                    xs = [1]
                    xs.extend(train_pattern.xs_train)
                    if train_perceptron(xs, train_pattern.y, weights, theta, alpha, act_fun=unipolar_fun) != 0:
                        no_errors = False
            print(f"interwał: <{interval[0]}:{interval[1]}> próba nr {i} epoki: {epoch}")


def speed_vs_alpha(weights_base, alphas, train_patterns):
    theta = 10
    for alpha in alphas:
        for i in range(10):
            weights = copy.copy(weights_base)
            epoch = 0
            no_errors = False
            while not no_errors:
                epoch += 1
                no_errors = True
                for train_pattern in train_patterns:
                    if train_perceptron(train_pattern.xs_train, train_pattern.y, weights, theta, alpha, act_fun=unipolar_fun) != 0:
                        no_errors = False
            print(f"alpha: {alpha} próba nr {i} epoki: {epoch}")


def speed_vs_function(weights_base, train_patterns):
        theta = 10
        alpha = 0.0001
        for i in range(10):
            weights = copy.copy(weights_base)
            epoch = 0
            no_errors = False
            while not no_errors:
                epoch += 1
                no_errors = True
                for train_pattern in train_patterns:
                    if train_perceptron(train_pattern.xs_train, train_pattern.y, weights, theta, alpha, act_fun=unipolar_fun) != 0:
                        no_errors = False
            print(f"unipolar próba nr {i} epoki: {epoch}")
        for i in range(10):
            weights = copy.copy(weights_base)
            epoch = 0
            no_errors = False
            while not no_errors:
                epoch += 1
                no_errors = True
                for train_pattern in train_patterns:
                    if train_perceptron(train_pattern.xs_train, train_pattern.y, weights, theta, alpha, act_fun=bipolar_fun) != 0:
                        no_errors = False
            print(f"bipolar próba nr {i} epoki: {epoch}")


def train_perceptron_alc(xs_train, y_train, weights, alpha, act_fun):
    assert len(xs_train) == len(
        weights), f"xs and weights should have the same length! xs: {len(xs_train)}, weights: {len(weights)}"

    if act_fun == bipolar_fun:
        xs_train = [1 if abs(x-1) < ABS_MAX_DIFF else -1 for x in xs_train]
        y_train = -1 if y_train == 0 else y_train

    z = sum((x * w for x, w in zip(xs_train, weights)))
    error = (y_train - z)
    if error == 0:
        return error
    for i in range(len(weights)):
        weights[i] += 2 * alpha * error * xs_train[i]
    return error


def speed_vs_interval_alc(weights_intervals, train_patterns):
    alpha = 0.0001
    for interval in weights_intervals:
        weights = [random.uniform(interval[0], interval[1]) for i in range(3)]
        for i in range(10):
            _, epochs = train_weights_alc(train_patterns, weights.copy(), alpha)
            print(f"interwał <{interval[0]}:{interval[1]}> próba nr {i} epoki: {epochs}")


def speed_vs_alpha_alc(alphas, train_patterns):
    weights = [random.uniform(-1, 1) for i in range(3)]
    for alpha in alphas:
        for i in range(10):
            _, epochs = train_weights_alc(train_patterns, weights.copy(), alpha)
            print(f"alfa: {alpha} próba nr {i} epoki: {epochs}")


def speed_vs_min_error_alc(min_errors, train_patterns):
    weights = [random.uniform(-1, 1) for i in range(3)]
    alpha = 0.0001
    for min_error in min_errors:
        for i in range(10):
            _, epochs = train_weights_alc(train_patterns, weights.copy(), alpha, min_error)
            print(f"min_error: {min_error} próba nr {i} epoki: {epochs}")


def train_weights(train_patterns):
    weights = [random.uniform(-1, 1) for i in range(3)]
    no_errors = False
    alpha = 0.00001
    epochs = 0
    while not no_errors:
        epochs += 1
        no_errors = True
        for train_pattern in train_patterns:
            xs = [1]
            xs.extend(train_pattern.xs_train)
            if train_perceptron(xs, train_pattern.y, weights, 0, alpha, bipolar_fun) != 0:
                no_errors = False
    print(f"num of epochs: {epochs}")
    return weights


def train_weights_alc(train_patterns, weights, alpha, mean_square_error_limit=0.1):
    epochs = 0
    mean_square_error = 1
    while abs(mean_square_error) >= mean_square_error_limit:
        error_sum = 0
        epochs += 1
        for train_pattern in train_patterns:
            xs = [1]
            xs.extend(train_pattern.xs_train)
            error = train_perceptron_alc(xs, train_pattern.y, weights, alpha, bipolar_fun)
            error_sum += error
        mean_square_error = error_sum / len(train_patterns)
    return weights, epochs


def try_perceptron(train_pattern: TrainPattern, weights):
    xs_train = [1]
    xs_train.extend([1 if abs(x-1) < ABS_MAX_DIFF else -1 for x in train_pattern.xs_train])

    y_train = -1 if train_pattern.y == 0 else train_pattern.y
    z = sum((x * w for x, w in zip(xs_train, weights)))
    y_obtained = bipolar_fun(z, 0)
    print(f"xs: {xs_train} and weights: {weights} z: {z} should produce: {y_train}. Model produced: {y_obtained}")


def main():
    thetas = [0.1, 0.2, 0.5, 0.7, 1, 1.5, 2, 5, 10, 30, 100]
    alphas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]
    num_of_samples = 200
    train_patterns = prepare_training_sets(num_of_samples)
    intervals = [(-1, 1), (-0.8, 0.8), (-0.5, 0.5), (-0.2, 0.2), (-0.1, 0.1), (-0.01, 0.01), (-0.005, 0.005)]
    weights_base = [random.uniform(-0.05, 0.05) for i in range(2)]

    # speed_vs_theta(weights_base, thetas, train_patterns)
    # print("\n\n\n\n")
    # speed_vs_interval([(-1, 1), (-0.8, 0.8), (-0.5, 0.5), (-0.2, 0.2), (-0.1, 0.1), (-0.01, 0.01), (-0.005, 0.005)], train_patterns)
    # print("\n\n\n\n")
    # speed_vs_alpha(weights_base, alphas, train_patterns)
    # print("\n\n\n\n")
    # speed_vs_function(weights_base, train_patterns)


    print("\n\n\n\n")
    speed_vs_interval_alc(intervals, train_patterns)
    print("\n\n\n\n")
    speed_vs_alpha_alc(alphas, train_patterns)
    print("\n\n\n\n")
    speed_vs_min_error_alc([1, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.005], train_patterns)

    # weights = train_weights_alc(train_patterns)
    # try_perceptron(TrainPattern([1, 1], 1), weights)
    # try_perceptron(TrainPattern([1, 0.98], 1), weights)
    # try_perceptron(TrainPattern([0.98, 0.98], 1), weights)
    # try_perceptron(TrainPattern([1, 0.89], 0), weights)
    # try_perceptron(TrainPattern([0, 0.98], 0), weights)
    # try_perceptron(TrainPattern([0, 0.01], 0), weights)


if __name__ == "__main__":
    main()
