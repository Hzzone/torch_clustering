# -*- coding: UTF-8 -*-
'''
@Project : torch_clustering 
@File    : beta_mixture.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 12:21 PM 
'''

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class BetaMixture1D(object):
    def __init__(self,
                 max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.eps_nan = 1e-12

    def fit_beta_weighted(self, x, w):
        def weighted_mean(x, w):
            return np.sum(w * x) / np.sum(w)

        x_bar = weighted_mean(x, w)
        s2 = weighted_mean((x - x_bar) ** 2, w)
        alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
        beta = alpha * (1 - x_bar) / x_bar
        return alpha, beta

    def likelihood(self, x, y):
        import scipy.stats as stats
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def responsibilities(self, x):
        r = np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r.T

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x).T

            # M-step
            self.alphas[0], self.betas[0] = self.fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = self.fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        # plt.plot(x, self.probability(x), lw=2, label='mixture')
        plt.legend()

    def __repr__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)
