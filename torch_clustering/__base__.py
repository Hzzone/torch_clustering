# -*- coding: UTF-8 -*-
'''
@Project : torch_clustering 
@File    : __base__.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 12:20 PM 
'''

import torch
import torch.nn.functional as F
import torch.distributed as dist


class BasicClustering:
    def __init__(self,
                 n_clusters,
                 init='k-means++',
                 n_init=10,
                 random_state=0,
                 max_iter=300,
                 tol=1e-4,
                 distributed=False,
                 verbose=True):
        '''
        :param n_clusters:
        :param init: {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.
        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.
        :param n_init: int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
        :param random_state: int, RandomState instance or None, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
        :param max_iter:
        :param tol:
        :param verbose: int, default=0 Verbosity mode.
        '''
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_ = None
        self.init = init
        self.random_state = random_state
        self.is_root_worker = True if not dist.is_initialized() else (dist.get_rank() == 0)
        self.verbose = verbose and self.is_root_worker
        self.distributed = distributed and dist.is_initialized()
        if verbose and self.distributed and self.is_root_worker:
            print('Perform K-means in distributed mode.')
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.rank = dist.get_rank() if self.distributed else 0

    def fit_predict(self, X):
        pass

    def distributed_sync(self, tensor):
        tensors_gather = [torch.ones_like(tensor)
                          for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.stack(tensors_gather)
        return output


def pairwise_cosine(x1: torch.Tensor, x2: torch.Tensor, pairwise=True):
    x1 = F.normalize(x1)
    x2 = F.normalize(x2)
    if not pairwise:
        return (1 - (x1 * x2).sum(dim=1))
    return 1 - x1.mm(x2.T)


def pairwise_euclidean(x1: torch.Tensor, x2: torch.Tensor, pairwise=True):
    if not pairwise:
        return ((x1 - x2) ** 2).sum(dim=1).sqrt()
    return torch.cdist(x1, x2, p=2.)
