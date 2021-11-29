import gzip
from pathlib import Path
import numpy as np
import requests
import shutil
import zipfile
import cv2
from typing import List, cast
import time
from enum import Enum
import matplotlib.pyplot as plt
import sys
import copy
import random


URL_TO_MNIST_ARCHIVE = "https://data.deepai.org/mnist.zip"

PATH_TO_IMGS = Path.cwd() / "imgs"
PATH_TO_ARCHIVES = PATH_TO_IMGS / "archives"


def download_mnist():
    print("Downloading mnist")
    PATH_TO_ARCHIVES.mkdir(parents=True, exist_ok=True)
    with requests.get(URL_TO_MNIST_ARCHIVE, stream=True) as archive:
        with open(PATH_TO_ARCHIVES / "mnist.zip", "wb") as file:
            shutil.copyfileobj(archive.raw, file)
            with zipfile.ZipFile(PATH_TO_ARCHIVES / "mnist.zip", "r") as arch:
                arch.extractall(PATH_TO_ARCHIVES)
    print("Mnist downloaded")


class Images:
    class ImgsSet:
        def __init__(self, imgs_tuple):
            self.imgs = imgs_tuple[0]
            self.labels = imgs_tuple[1]

    def __init__(self, train_set, test_set):
        self.train = self.ImgsSet(train_set)
        self.test = self.ImgsSet(test_set)


def _load_imgs(path: Path, imgs_group: str):
    images_path = path / f"{imgs_group}-images-idx3-ubyte.gz"
    labels_path = path / f"{imgs_group}-labels-idx1-ubyte.gz"

    with gzip.open(labels_path, 'rb') as labels_zip:
        labels = np.frombuffer(labels_zip.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgs_zip:
        images = np.frombuffer(imgs_zip.read(), dtype=np.uint8, offset=16)\
            .reshape((len(labels), 784))

    return images, labels


def load_mnist(redownload=False) -> Images:
    PATH_TO_IMGS.mkdir(parents=True, exist_ok=True)
    if redownload:
        download_mnist()

    train_set = _load_imgs(PATH_TO_ARCHIVES, "train")
    test_set = _load_imgs(PATH_TO_ARCHIVES, "t10k")

    return Images(train_set, test_set)


# ================================================================================================ #
# ================================================================================================ #
# MODEL #
# ================================================================================================ #
# ================================================================================================ #

class ActFun:
    def __init__(self, fun, derivative):
        self.fun = fun
        self.derivative = derivative


def softmax(xs):
    """
    :param xs: wektor wartosci 10 x N
    :return: wektor prawdopodobieństw 10 x N
    """
    val = np.exp(xs - xs.max())
    soft = val / np.sum(val, axis=0)
    soft[np.isnan(soft)] = 0.001
    return soft


def softmax_derivative(x):
    soft = softmax(x)
    return soft * (1 - soft)


def relu(x):
    #return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    return np.maximum(0, x)


def relu_deriv(x):
    #return 1/(1 + np.exp(-x))
    return np.sign(x)
    #return 0*(x < 0) + 1 * (x > 0) + 0.5*(x == 0)


def identity_function(x):
    return x


def identity_function_deriv(x):
    return 1 * (x == x)


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1-sig)


def tanh(x):
    exp_x = np.exp(x)
    exp_min_x = np.exp(-x)
    return (exp_x - exp_min_x) / (exp_x + exp_min_x)


def tanh_derivative(x):
    return 1 - tanh(x) ** 2


def calc_z(xs, weights):
    assert xs.shape[0] == weights.shape[1], f"Wrong shape of xs or weights weights: {weights.shape}, xs: {xs.shape}, {xs.shape[0]} != {weights.shape[1]}"
    return (weights @ xs)


def hot_encode(ys, num_classes):
    if ys.shape == ():
        ys_encoded = np.zeros(shape=(num_classes, 1))
        ys_encoded[ys, 0] = 1
    else:
        ys_encoded = np.zeros(shape=(num_classes, ys.shape[0]))
        for i in range(ys_encoded.shape[1]):
            ys_encoded[ys[i], i] = 1
    return ys_encoded


def weights_init_he(size):
    # size[0] = num of weights = n_in
    # size[1] = num of neurons = n_out
    st_dev = np.sqrt(2/size[0])
    return np.random.normal(0, st_dev, size[0] * size[1])


def weights_init_xavier(size):
    st_dev = np.sqrt(2 / (size[0] + size[1]))
    return np.random.normal(0, st_dev, size[0] * size[1])


class Layer:
    def __init__(self, num_of_neurons, weights_initializer=None):
        self.num_of_neurons = num_of_neurons
        self.outputs = None
        self.weights = None
        if weights_initializer:
            self.weights_initializer = weights_initializer
        else:
            self.weights_initializer = lambda size: np.random.normal(0, 1, size[0] * size[1])
        self.bias = None
        self.z = 0

    def apply(self, xs):
        pass

    def init_weights(self, num_of_weights, batch_size):
        if self.weights is None:
            self.weights = self.weights_initializer((num_of_weights, self.num_of_neurons)) \
                .reshape((self.num_of_neurons, num_of_weights))
        if self.outputs is None:
            self.outputs = np.zeros(shape=(self.num_of_neurons, batch_size))
        if self.bias is None:
            self.bias = np.random.normal(0, 0.1, self.num_of_neurons)\
                .reshape((self.num_of_neurons, 1))


class InputLayer(Layer):
    def __init__(self, num_of_neurons):
        super().__init__(num_of_neurons)

    def apply(self, xs):
        self.outputs = np.zeros(shape=(self.num_of_neurons, xs.shape[1]))
        self.outputs[:, :] = xs
        return self.outputs


class ActFunction(Enum):
    Relu = 1
    Identity = 2
    Sigmoid = 3
    Softmax = 4
    Tanh = 5


class HiddenLayer(Layer):
    ACT_FUNCTIONS = {
        ActFunction.Relu: ActFun(relu, relu_deriv),
        ActFunction.Identity: ActFun(identity_function, identity_function_deriv),
        ActFunction.Sigmoid: ActFun(sigmoid, sigmoid_derivative),
        ActFunction.Softmax: ActFun(softmax, softmax_derivative),
        ActFunction.Tanh: ActFun(tanh, tanh_derivative)
    }

    def __init__(self, num_of_neurons, act_fun: ActFunction, weights_initializer=None):
        super().__init__(num_of_neurons, weights_initializer=weights_initializer)
        self.act_fun = self.ACT_FUNCTIONS[act_fun]

    def apply(self, xs):
        self.outputs = np.zeros(shape=(self.num_of_neurons, xs.shape[1]))
        self.z = calc_z(xs, self.weights) + self.bias
        self.outputs[:, :] = self.act_fun.fun(self.z)
        return self.outputs

    def calc_error(self, weights_lay_higher, error_lay_higher, optimizer):
        weights = optimizer.get_weights_to_calc_gradient(weights_lay_higher)
        return (weights.T @ error_lay_higher) * self.act_fun.derivative(self.z)


class OutputLayer(HiddenLayer):
    def __init__(self, num_of_neurons, end_fun: ActFunction, weights_initializer=None):
        super().__init__(num_of_neurons, ActFunction.Identity, weights_initializer=weights_initializer)
        self.end_function = self.ACT_FUNCTIONS[end_fun]

    def apply(self, xs):
        self.outputs = np.zeros(shape=(self.num_of_neurons, xs.shape[1]))
        self.z = calc_z(xs, self.weights) + self.bias
        self.outputs[:, :] = self.end_function.fun(self.z)
        return self.outputs

    def calc_error(self, y_expected):
        cost_fun_derivative = (self.outputs - y_expected)
        return cost_fun_derivative * self.end_function.derivative(self.z)


class Optimizers(Enum):
    NoneOptimizer = 0
    Momentum = 1
    NesterovMomentum = 2
    Adagrad = 3
    Adadelta = 4
    Adam = 5


class NoneOptimizer:
    def apply(self, alpha, mini_batch_size, errors, prev_layer_outputs, bias_shape, t):
        weights_delta = alpha / mini_batch_size * (errors @ prev_layer_outputs)
        bias_delta = alpha / mini_batch_size * np.sum(errors, axis=1).reshape(bias_shape)
        return weights_delta, bias_delta

    def get_weights_to_calc_gradient(self, input_weights):
        return input_weights


class Momentum:
    def __init__(self):
        self.mom_param = 0.8
        self.previous_weights_delta = 0
        self.previous_bias_delta = 0

    def apply(self, alpha, mini_batch_size, errors, prev_layer_outputs, bias_shape, t):
        weights_delta, bias_delta = NoneOptimizer().apply(alpha, mini_batch_size, errors, prev_layer_outputs, bias_shape, t)
        self.previous_weights_delta = weights_delta + self.mom_param * self.previous_weights_delta
        self.previous_bias_delta = bias_delta + self.mom_param * self.previous_bias_delta
        return self.previous_weights_delta, self.previous_bias_delta

    def get_weights_to_calc_gradient(self, input_weights):
        return input_weights


class NesterovMomentum:
    def __init__(self):
        self.mom_param = 0.8
        self.previous_weights_delta = 0
        self.previous_bias_delta = 0

    def apply(self, alpha, mini_batch_size, errors, prev_layer_outputs, bias_shape, t):
        weights_delta, bias_delta = NoneOptimizer().apply(alpha, mini_batch_size, errors, prev_layer_outputs,
                                                          bias_shape, t)
        self.previous_weights_delta = weights_delta + self.mom_param * self.previous_weights_delta
        self.previous_bias_delta = bias_delta + self.mom_param * self.previous_bias_delta
        return self.previous_weights_delta, self.previous_bias_delta

    def get_weights_to_calc_gradient(self, input_weights):
        return input_weights - self.mom_param * self.previous_weights_delta


class Adagrad:
    def __init__(self):
        self.weights_delta_sum = None
        self.bias_delta_sum = None

    def apply(self, alpha, mini_batch_size, errors, prev_layer_outputs, bias_shape, t):
        weights_delta = (errors @ prev_layer_outputs)
        bias_delta = np.sum(errors, axis=1).reshape(bias_shape)
        if self.weights_delta_sum is None:
            self.weights_delta_sum = np.zeros(shape=weights_delta.shape)
            self.bias_delta_sum = np.zeros(shape=bias_delta.shape)

        self.weights_delta_sum += (weights_delta/mini_batch_size) ** 2
        self.bias_delta_sum += (bias_delta/mini_batch_size) ** 2

        previous_weights_delta = alpha / mini_batch_size * weights_delta / np.sqrt(self.weights_delta_sum + 1e-8)
        previous_bias_delta = alpha / mini_batch_size * bias_delta / np.sqrt(self.bias_delta_sum + 1e-8)
        #print(f"t: {t}, {self.weights_delta_sum[0]}")
        return previous_weights_delta, previous_bias_delta

    def get_weights_to_calc_gradient(self, input_weights):
        return input_weights


class Adadelta:
    def __init__(self):
        self.gamma = 0.9
        self.weights_grad_avg = None
        self.bias_grad_avg = None
        self.weights_delta_avg = None
        self.bias_delta_avg = None

    def apply(self, alpha, mini_batch_size, errors, prev_layer_outputs, bias_shape, t):
        weights_grad = (errors @ prev_layer_outputs)
        bias_grad = np.sum(errors, axis=1).reshape(bias_shape)
        if self.weights_grad_avg is None:
            self.weights_grad_avg = np.zeros(shape=weights_grad.shape)
            self.bias_grad_avg = np.zeros(shape=bias_grad.shape)
            self.weights_delta_avg = np.zeros(shape=weights_grad.shape)
            self.bias_delta_avg = np.zeros(shape=bias_grad.shape)

        self.weights_grad_avg = (self.gamma * self.weights_grad_avg) + (1 - self.gamma) * ((weights_grad/mini_batch_size) ** 2)
        self.bias_grad_avg = (self.gamma * self.bias_grad_avg) + (1 - self.gamma) * ((bias_grad/mini_batch_size) ** 2)

        weights_delta = (np.sqrt(self.weights_delta_avg + 1e-8)) / (np.sqrt(self.weights_grad_avg + 1e-8)) * weights_grad
        bias_delta = (np.sqrt(self.bias_delta_avg + 1e-8)) / (np.sqrt(self.bias_grad_avg + 1e-8)) * bias_grad

        self.weights_delta_avg = (self.gamma * self.weights_delta_avg) + (1 - self.gamma) * ((weights_delta/mini_batch_size) ** 2)
        self.bias_delta_avg = (self.gamma * self.bias_delta_avg) + (1 - self.gamma) * ((bias_delta/mini_batch_size) ** 2)

        return weights_delta, bias_delta

    def get_weights_to_calc_gradient(self, input_weights):
        return input_weights


class Adam:
    def __init__(self):
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.previous_m_weights = 0
        self.previous_m_bias = 0
        self.previous_v_weights = 0
        self.previous_v_bias = 0

    def apply(self, alpha, mini_batch_size, errors, prev_layer_outputs, bias_shape, t):
        weights_delta = (errors @ prev_layer_outputs)
        bias_delta = np.sum(errors, axis=1).reshape(bias_shape)

        self.previous_m_weights = self.beta_1 * self.previous_m_weights + (1 - self.beta_1) * weights_delta
        self.previous_m_bias = self.beta_1 * self.previous_m_bias + (1 - self.beta_1) * bias_delta

        self.previous_v_weights = self.beta_2 * self.previous_v_weights + (1 - self.beta_2) * ((weights_delta) ** 2)
        self.previous_v_bias = self.beta_2 * self.previous_v_bias + (1 - self.beta_2) * ((bias_delta) ** 2)

        m_weights_corrected = self.previous_m_weights / (1 - self.beta_1 ** (max(1, t//mini_batch_size)))
        m_bias_corrected = self.previous_m_bias / (1 - self.beta_1 ** (max(1, t//mini_batch_size)))

        v_weights_corrected = self.previous_v_weights / (1 - self.beta_2 ** (max(1, t//mini_batch_size)))
        v_bias_corrected = self.previous_v_bias / (1 - self.beta_2 ** (max(1, t//mini_batch_size)))

        previous_weights_delta = (alpha / mini_batch_size) / (np.sqrt(v_weights_corrected) + 1e-8) * m_weights_corrected
        previous_bias_delta = (alpha / mini_batch_size) / (np.sqrt(v_bias_corrected) + 1e-8) * m_bias_corrected
        return previous_weights_delta, previous_bias_delta

    def get_weights_to_calc_gradient(self, input_weights):
        return input_weights


class Model:
    OPTIMIZERS = {Optimizers.NoneOptimizer: NoneOptimizer,
                Optimizers.Momentum: Momentum,
                Optimizers.NesterovMomentum: NesterovMomentum,
                Optimizers.Adagrad: Adagrad,
                Optimizers.Adadelta: Adadelta,
                Optimizers.Adam: Adam}

    def __init__(self, images: Images, val_set_part, train_on_part=1.0, alpha=0.00001, mini_batch=30, optimizer=Optimizers.NoneOptimizer):
        self.layers: List[Layer] = []
        num_of_imgs_to_use = int(images.train.imgs.shape[0] * train_on_part)
        num_of_train_imgs = int(num_of_imgs_to_use * (1-val_set_part))
        self.xs = images.train.imgs[0: num_of_train_imgs].reshape(num_of_train_imgs, 784).T / 255
        self.ys = images.train.labels[0: num_of_train_imgs]
        num_of_val_imgs = num_of_imgs_to_use - num_of_train_imgs
        self.xs_val = images.train.imgs[num_of_train_imgs: num_of_imgs_to_use].reshape(num_of_val_imgs, 784).T / 255
        self.ys_val = images.train.labels[num_of_train_imgs: num_of_imgs_to_use]

        self.val_costs = []
        self.training_costs = []
        self.alpha = alpha
        self.mini_batch_size = mini_batch
        self.best_weights = {}
        self.last_cost_when_weights_were_saved = 1e10
        self.cost_was_not_increasing_before = False
        self.optimizerType = optimizer
        self.optimizers = {}

    def add_layer(self, layer):
        self.layers.append(layer)
        return self

    def __init_weights(self, batch_size):
        for i in range(0, len(self.layers)):
            self.layers[i].init_weights(self.layers[i-1].num_of_neurons, batch_size)
            self.optimizers[i] = Model.OPTIMIZERS[self.optimizerType]()

    def shuffle_training_set(self):
        rng_state = np.random.get_state()
        self.xs = self.xs.T
        np.random.shuffle(self.xs)
        self.xs = self.xs.T
        np.random.set_state(rng_state)
        np.random.shuffle(self.ys)

    def __calc_errors(self, ys):
        y_expected = hot_encode(ys, self.layers[-1].num_of_neurons)
        errors = {}
        layer = self.layers[-1]
        prev_error = layer.calc_error(y_expected)
        errors[len(self.layers) - 1] = prev_error

        for i in reversed(range(1, len(self.layers) - 1)):
            layer = self.layers[i]
            weights_lay_higher = self.layers[i + 1].weights
            prev_error = layer.calc_error(weights_lay_higher, errors[i+1], self.optimizers[i+1])
            errors[i] = prev_error
        return errors

    def __update_weights_and_biases(self, ys, alpha, t):
        errors = self.__calc_errors(ys)
        for i in reversed(range(1, len(self.layers))):
            layer = self.layers[i]
            weights_delta, bias_delta = self.optimizers[i].apply(alpha, self.mini_batch_size,
                                                                 errors[i], self.layers[i-1].outputs.T, layer.bias.shape, t)
            layer.weights -= weights_delta
            layer.bias -= bias_delta

    def train_using_minibatch(self):
        self.shuffle_training_set()
        for i in range(0, self.xs.shape[1], self.mini_batch_size):
            prev_output = self.layers[0].apply(self.xs[:, i: i + self.mini_batch_size])
            for j in range(1, len(self.layers)):
                prev_output = self.layers[j].apply(prev_output)
            self.__update_weights_and_biases(self.ys[i: i + self.mini_batch_size], self.alpha, self.t)
            self.t += 1

    def check_early_stopping(self, cost, epochs):
        if cost < self.last_decreasing_cost:
            self.last_decreasing_cost = cost
            self.cost_was_not_increasing_before = True
        else:
            # Started rising
            if self.cost_was_not_increasing_before and cost < self.last_cost_when_weights_were_saved:
                self.cost_was_not_increasing_before = False
                self.last_cost_when_weights_were_saved = cost
                self.best_weights = {}
                self.epoch_with_best_weights = epochs
                for i in range(1, len(self.layers)):
                    self.best_weights[i] = (copy.deepcopy(self.layers[i].weights), copy.deepcopy(self.layers[i].bias))
        self.count_of_last_epochs += 1
        if cost - self.last_decreasing_cost > self.last_decreasing_cost * self.COST_INCREASE_LIMI_COEFF:
            print(f"\nEarly stopping due to significant cost increase")
            self.early_stopping = True
        elif self.count_of_last_epochs >= self.max_count_of_epochs_to_check:
            if abs(self.first_cost - cost) < self.minimum_required_change * self.first_cost:
                print(f"\nEarly stopping due to no better change in cost for last {self.max_count_of_epochs_to_check} epochs")
                self.early_stopping = True
            else:
                self.count_of_last_epochs = 0
                self.first_cost = cost

    def restore_lase_best_weights(self):
        for i in range(1, len(self.layers)):
            weights, bias = self.best_weights[i]
            self.layers[i].weights = weights
            self.layers[i].bias = bias

    def train(self):
        self.layers.insert(0, InputLayer(self.xs.shape[0]))
        self.__init_weights(self.mini_batch_size)
        epochs = 0
        self.last_cost = 1e10
        self.first_cost = 1e10
        self.last_decreasing_cost = 1e10
        self.COST_INCREASE_LIMI_COEFF = 0.2  # 5% of last cost
        self.minimum_required_change = 0.005  # 0.5%
        self.count_of_last_epochs = 0
        self.max_count_of_epochs_to_check = 20
        self.last_cost_when_weights_were_saved = 1e10
        self.epoch_with_best_weights = 1
        self.early_stopping = False
        MAX_NUM_OF_EPOCHS = 3000
        self.cost_was_not_increasing_before = False
        self.t = 1
        cost = 0
        while not self.early_stopping:
            epochs += 1
            self.train_using_minibatch()
            cost, _ = self.calc_cost()
            print(f"\r\rEpoch nr {epochs} cost {cost}", end="")
            self.check_early_stopping(cost, epochs)
            if epochs >= MAX_NUM_OF_EPOCHS:
                self.early_stopping = True
        if self.last_cost_when_weights_were_saved < cost:
            print(f"restoring weights from epoch {self.epoch_with_best_weights}, cost: {self.last_cost_when_weights_were_saved}")
            self.restore_lase_best_weights()
        print(f"Number of epochs to learn: {epochs}")

    def calc_cost(self):
        def __calc_cost(xs, ys):
            ys_encoded = hot_encode(ys, self.layers[-1].num_of_neurons)
            output = self.layers[0].apply(xs)
            for j in range(1, len(self.layers)):
                output = self.layers[j].apply(output)
            return -np.sum(ys_encoded * np.log(output + 1e-8))

        val_cost = __calc_cost(self.xs_val, self.ys_val)
        train_cost = __calc_cost(self.xs, self.ys)
        self.val_costs.append(val_cost)
        self.training_costs.append(train_cost)
        return val_cost, train_cost

    def test(self, xs, ys):
        self.layers[0] = InputLayer(xs.shape[0])
        self.layers[0].outputs = np.zeros(shape=(self.layers[0].num_of_neurons, xs.shape[1]))
        prev_output = self.layers[0].apply(xs)
        for j in range(1, len(self.layers)):
            self.layers[j].outputs = np.zeros(shape=(self.layers[j].num_of_neurons, xs.shape[1]))
            prev_output = self.layers[j].apply(prev_output)
        ys_pred = np.argmax(prev_output, axis=0)
        print(f"Accuracy: {np.sum(ys == ys_pred)/ys_pred.shape[0]*100} % cost {self.val_costs[-1]}")

    def show_last_training_plot(self, name):
        plt.plot(self.val_costs)
        plt.xlabel("Liczba epok")
        plt.ylabel("Wartosc funkcji kosztu")
        plt.gcf().canvas.manager.set_window_title(name)
        plt.show()


def main():
    images = load_mnist()

    weights_init = weights_init_he
    model = Model(images, val_set_part=0.2, train_on_part=0.01, alpha=0.1, optimizer=Optimizers.Adadelta)
    model.add_layer(HiddenLayer(20, ActFunction.Sigmoid, weights_initializer=weights_init))\
        .add_layer(OutputLayer(10, ActFunction.Softmax, weights_initializer=weights_init))
    start = time.time_ns()
    model.train()
    training_time = (time.time_ns() - start) / 1000000000
    print(f"Training time {int(training_time*100)/100} s")
    model.test(images.test.imgs[0:10000].reshape(10000, 784).T/255, images.test.labels[0:10000])


if __name__ == "__main__":
    main()



