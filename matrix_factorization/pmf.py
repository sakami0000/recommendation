import copy

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
from sklearn.metrics import mean_squared_error
from tqdm import tqdm, trange

plt.style.use('seaborn')


class PMF:
    """Probabilistic Matrix Factorization.

    Arguments
    ---------
        lambda_alpha: Float. Regularizer coefficient of user matrix.
        lambda_beta: Float. Regularizer coefficient of item matrix.
        latent_size: Integer. The dimention of user and item vector.
        optimizer: String. The optimizer used in learning.
            choosed from ('sgd', 'momentum_sgd', 'adagrad', 'adam').
        lr: Float. learning rate.
        iters: Integer. How many times paramters updated in training.
        out: How to output the training logs.
        seed: Integer. random seed.
        kwargs: params of learning optimizer.
    """

    def __init__(self, lambda_alpha=1e-2, lambda_beta=1e-2, latent_size=50,
                 optimizer='momentum_sgd', lr=1e-3, iters=200, out=tqdm.write, seed=None, **kwargs):
        assert optimizer in ('sgd', 'momentum_sgd', 'adagrad', 'adam'), f'Invalid optimizer: {optimizer}'

        self.lambda_alpha = lambda_alpha
        self.lambda_beta = lambda_beta
        self.latent_size = latent_size
        self.optimizer = optimizer
        self.lr = lr
        self.iterations = iters
        self.out = out
        self.random_state = RandomState(seed)
        self.kwargs = kwargs

        self.R = None
        self.I = None
        self.U = None
        self.V = None

        self.train_loss = []

    def fit(self, user_item_matrix, verbose=1):
        self.R = user_item_matrix
        self.R = np.nan_to_num(self.R)
        self.I = copy.deepcopy(user_item_matrix)
        self.I[~(np.isnan(self.I))] = 1
        self.I = np.nan_to_num(self.I)

        self.U = 0.1 * self.random_state.rand(np.size(self.R, 0), self.latent_size)
        self.V = 0.1 * self.random_state.rand(np.size(self.R, 1), self.latent_size)

        if self.optimizer == 'sgd':
            optimize = self._sgd
        elif self.optimizer == 'momentum_sgd':
            optimize = self._momentum_sgd
            self.momentum_rate = self.kwargs.get('momentum_rate', 0.8)
            self.momentum_u = np.zeros(self.U.shape)
            self.momentum_v = np.zeros(self.V.shape)
        elif self.optimizer == 'adagrad':
            optimize = self._adagrad
            self.D_u = np.zeros(self.U.shape)
            self.D_v = np.zeros(self.V.shape)
        elif self.optimizer == 'adam':
            optimize = self._adam
            self.beta1 = self.kwargs.get('beta1', 0.9)
            self.beta2 = self.kwargs.get('beta2', 0.999)
            self.mean_u = np.zeros(self.U.shape)
            self.mean_v = np.zeros(self.V.shape)
            self.variance_u = np.zeros(self.U.shape)
            self.variance_v = np.zeros(self.V.shape)

        if verbose == 0:
            freq = self.iterations
        elif verbose == 1:
            freq = self.iterations // 5
        else:
            freq = self.iterations // 10

        for it in trange(self.iterations):

            update_u, update_v = optimize()

            self.U = self.U - update_u
            self.V = self.V - update_v

            loss = self._loss()
            self.train_loss.append(loss)

            if it % freq == 0 or it == self.iterations - 1:
                self.out(f'training iteration: {it}, loss: {loss:.2f}')

    def _loss(self):
        loss = np.sum(self.I * (self.R - np.dot(self.U, self.V.T))**2) + \
            self.lambda_alpha * np.sum(np.square(self.U)) + \
            self.lambda_beta * np.sum(np.square(self.V))
        return loss

    def _grads(self):
        grads_u = np.dot(self.I * (self.R - np.dot(self.U, self.V.T)), -self.V) + \
            self.lambda_alpha * self.U
        grads_v = np.dot((self.I * (self.R - np.dot(self.U, self.V.T))).T, -self.U) + \
            self.lambda_beta * self.V
        return grads_u, grads_v

    def _sgd(self):
        grads_u, grads_v = self._grads()
        grads_u = self.lr * grads_u
        grads_v = self.lr * grads_v
        return grads_u, grads_v

    def _momentum_sgd(self):
        grads_u, grads_v = self._grads()

        self.momentum_u = (self.momentum_rate * self.momentum_u) + self.lr * grads_u
        self.momentum_v = (self.momentum_rate * self.momentum_v) + self.lr * grads_v

        return self.momentum_u, self.momentum_v

    def _adagrad(self):
        grads_u, grads_v = self._grads()

        self.D_u += np.square(grads_u)
        self.D_v += np.square(grads_v)

        update_u = self.lr * grads_u / (np.sqrt(self.D_u) + 1e-8)
        update_v = self.lr * grads_v / (np.sqrt(self.D_v) + 1e-8)

        return update_u, update_v

    def _adam(self):
        grads_u, grads_v = self._grads()

        self.mean_u = self.beta1 * self.mean_u + (1 - self.beta1) * grads_u
        self.mean_v = self.beta1 * self.mean_v + (1 - self.beta1) * grads_v

        self.variance_u = self.beta2 * self.variance_u + (1 - self.beta2) * grads_u ** 2
        self.variance_v = self.beta2 * self.variance_v + (1 - self.beta2) * grads_v ** 2

        mean_u = self.mean_u / (1 - self.beta1)
        mean_v = self.mean_v / (1 - self.beta1)

        variance_u = self.variance_u / (1 - self.beta2)
        variance_v = self.variance_v / (1 - self.beta2)

        update_u = self.lr * mean_u / (np.sqrt(variance_u) + 1e-8)
        update_v = self.lr * mean_v / (np.sqrt(variance_v) + 1e-8)

        return update_u, update_v

    def predict(self):
        assert self.R is not None, 'Not trained yet.'
        return np.dot(self.U, self.V.T)

    def rmse(self, test_data):
        """calcurate rooted mean squared error of test data.

        Arguments
        ---------
            test_data: np.ndarray(float)[:, :]. test data.
        """
        test_data_values = copy.deepcopy(test_data)
        test_data_values[self.I.astype(bool)] = np.nan
        test_vals = test_data_values[~(np.isnan(test_data_values))].flatten()

        pred_vals = self.predict()
        pred_vals = pred_vals[~(np.isnan(test_data_values))].flatten()

        return np.sqrt(mean_squared_error(test_vals, pred_vals))

    def plot(self, figsize=(12, 6)):
        """plot history of train data losses.
        """
        assert len(self.train_loss) != 0, 'Not trained yet.'

        plt.figure(figsize=figsize)

        plt.plot(range(len(self.train_loss)), self.train_loss)

        plt.xlabel('Train Iteration')
        plt.ylabel('Train Loss')
        plt.show()
