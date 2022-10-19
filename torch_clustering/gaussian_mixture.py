# -*- coding: UTF-8 -*-
'''
@Project : torch_clustering 
@File    : gaussian_mixture.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 12:22 PM 
'''

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D
import torch.distributed as dist
from .__base__ import BasicClustering
from .kmeans.kmeans import PyTorchKMeans


class PyTorchGaussianMixture(BasicClustering):
    def __init__(self,
                 covariance_type='diag',
                 metric='euclidean',
                 reg_covar=1e-6,
                 init='k-means++',
                 random_state=0,
                 n_clusters=8,
                 n_init=10,
                 max_iter=300,
                 tol=1e-4,
                 distributed=False,
                 verbose=True):
        '''
        pytorch_gaussian_mixture = PyTorchGaussianMixture(metric='cosine',
                                                  covariance_type='diag',
                                                  reg_covar=1e-6,
                                                  init='k-means++',
                                                  random_state=0,
                                                  n_clusters=10,
                                                  n_init=10,
                                                  max_iter=300,
                                                  tol=1e-5,
                                                  verbose=True)
        pseudo_labels = pytorch_gaussian_mixture.fit_predict(torch.from_numpy(features).cuda())
        :param metric:
        :param reg_covar:
        :param init:
        :param random_state:
        :param n_clusters:
        :param n_init:
        :param max_iter:
        :param tol:
        :param verbose:
        '''
        super().__init__(n_clusters=n_clusters,
                         init=init,
                         distributed=distributed,
                         random_state=random_state,
                         n_init=n_init,
                         max_iter=max_iter,
                         tol=tol,
                         verbose=verbose)
        self.reg_covar = reg_covar
        self.metric = metric
        self._estimate_gaussian_covariances = {'diag': self._estimate_gaussian_covariances_diag,
                                               'spherical': self._estimate_gaussian_covariances_spherical}[
            covariance_type]
        self.covariances, self.weights, self.lower_bound_ = None, None, None

    def _estimate_gaussian_covariances_diag(self, resp: torch.Tensor, X: torch.Tensor, nk: torch.Tensor,
                                            means: torch.Tensor):
        avg_X2 = torch.matmul(resp.T, X * X) / nk[:, None]
        avg_means2 = means ** 2
        avg_X_means = means * torch.matmul(resp.T, X) / nk[:, None]
        return avg_X2 - 2 * avg_X_means + avg_means2 + self.reg_covar
        # N * K * L
        # return (((X.unsqueeze(1) - means.unsqueeze(0)) ** 2) * resp.unsqueeze(-1)).sum(0) / nk[:, None] + self.reg_covar

    def _estimate_gaussian_covariances_spherical(self, resp: torch.Tensor, X: torch.Tensor, nk: torch.Tensor,
                                                 means: torch.Tensor):
        return self._estimate_gaussian_covariances_diag(resp, X, nk, means).mean(1, keepdim=True)

    def initialize(self, X: torch.Tensor, resp: torch.Tensor):
        """Initialization of the Gaussian mixture parameters.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = X.shape
        weights, means, covariances = self._estimate_gaussian_parameters(X, resp)
        return means, covariances, weights

    def _estimate_gaussian_parameters(self, X: torch.Tensor, resp: torch.Tensor):
        # N * K * L
        nk = resp.sum(dim=0) + 10 * torch.finfo(resp.dtype).eps
        # means = torch.sum(X[:, None, :] * resp[:, :, None], dim=0) / nk[:, None]
        means = resp.T.mm(X) / nk[:, None]
        if self.metric == 'cosine':
            means = F.normalize(means, dim=-1)
        covariances = self._estimate_gaussian_covariances(resp, X, nk, means)
        weights = nk / X.size(0)
        return weights, means, covariances

    def fit_predict(self, X: torch.Tensor):
        if self.metric == 'cosine':
            X = F.normalize(X, dim=1)
        best_means, best_covariances, best_weights, best_resp = None, None, None, None
        max_lower_bound = - float("Inf")

        g = torch.Generator()
        g.manual_seed(self.random_state)
        random_states = torch.randperm(10000, generator=g)[:self.n_init * self.world_size]
        random_states = random_states[self.rank:self.n_init * self.world_size:self.world_size]

        for n_init in range(self.n_init):

            random_state = int(random_states[n_init])
            # KMeans init
            pseudo_labels = PyTorchKMeans(metric=self.metric,
                                          init=self.init,
                                          n_clusters=self.n_clusters,
                                          random_state=random_state,
                                          n_init=self.n_init,
                                          max_iter=self.max_iter,
                                          tol=self.tol,
                                          distributed=self.distributed,
                                          verbose=self.verbose).fit_predict(X)
            resp = F.one_hot(pseudo_labels, self.n_clusters).to(X)
            means, covariances, weights = self.initialize(X, resp)
            previous_lower_bound_ = self.log_likehood(resp.log())

            for n_iter in range(self.max_iter):
                # E step
                log_resp = self._e_step(X, means, covariances, weights)

                resp = F.softmax(log_resp, dim=1)

                lower_bound_ = self.log_likehood(log_resp)

                shift = torch.abs(previous_lower_bound_ - lower_bound_)

                if shift < self.tol:
                    if self.verbose:
                        print('converge at Iteration {} with shift: {}'.format(n_iter, shift))
                    break

                if self.verbose:
                    print(f'Iteration {n_iter}, loglikehood: {lower_bound_.item()}, shift: {shift.item()}')
                previous_lower_bound_ = lower_bound_

                if lower_bound_ > max_lower_bound:
                    max_lower_bound = lower_bound_
                    best_means, best_covariances, best_weights, best_resp = \
                        means, covariances, weights, resp

                # M step
                means, covariances, weights = self._m_step(X, resp)

        if self.distributed:
            max_lower_bound = self.distributed_sync(max_lower_bound)
            best_idx = torch.argmax(max_lower_bound).item()
            max_lower_bound = max_lower_bound[best_idx]
            dist.broadcast(best_means, src=best_idx)
            dist.broadcast(best_covariances, src=best_idx)
            dist.broadcast(best_weights, src=best_idx)
            dist.broadcast(best_resp, src=best_idx)
        if self.verbose:
            print(f"Final loglikehood {max_lower_bound.item()}.")

        if self.verbose:
            print(f'Converged with loglikehood {max_lower_bound.item()}')
        self.cluster_centers_, self.covariances, self.weights, self.lower_bound_ = \
            best_means, best_covariances, best_weights, max_lower_bound
        return self.predict_score(X)

    def _e_step(self, X: torch.Tensor, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor):
        estimate_precision_error_message = (
            "Fitting the mixture model failed because some components have "
            "ill-defined empirical covariance (for instance caused by singleton "
            "or collapsed samples). Try to decrease the number of components, "
            "or increase reg_covar.")
        if torch.any(torch.le(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        log_resp = self.log_prob(X, means, covariances, weights)
        return log_resp

    def log_prob(self, X: torch.Tensor, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor):
        log_resp = D.Normal(loc=means.unsqueeze(0),
                            scale=covariances.unsqueeze(0).sqrt()).log_prob(X.unsqueeze(1)).sum(dim=-1)
        log_resp = log_resp + weights.unsqueeze(0).log()
        return log_resp

    def log_prob_sklearn(self, X: torch.Tensor, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor):
        n_samples, n_features = X.size()
        n_components, _ = means.size()

        precisions_chol = 1. / torch.sqrt(covariances)

        log_det = torch.sum(precisions_chol.log(), dim=1)
        precisions = precisions_chol ** 2
        log_prob = (torch.sum((means ** 2 * precisions), dim=1) -
                    2. * torch.matmul(X, (means * precisions).T) +
                    torch.matmul(X ** 2, precisions.T))
        log_p = -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det
        weighted_log_p = log_p + weights.unsqueeze(0).log()

        # seems not work
        # weighted_log_p = weighted_log_p - weighted_log_p.logsumexp(dim=1, keepdim=True)
        return weighted_log_p

    def _m_step(self, X: torch.Tensor, resp: torch.Tensor):
        n_samples, _ = X.shape
        weights, means, covariances = self._estimate_gaussian_parameters(X, resp)
        return means, covariances, weights

    def log_likehood(self, log_resp: torch.Tensor):
        # N * K
        return log_resp.logsumexp(dim=1).mean()

    def predict_score(self, X: torch.Tensor):
        return F.softmax(self._e_step(X, self.cluster_centers_, self.covariances, self.weights), dim=1)
