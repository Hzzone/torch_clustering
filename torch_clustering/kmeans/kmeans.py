# -*- coding: UTF-8 -*-
'''
@Project : torch_clustering 
@File    : kmeans.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 12:23 PM 
'''

import numpy as np
import torch
import tqdm
import torch.distributed as dist
from ..__base__ import BasicClustering, pairwise_euclidean, pairwise_cosine
from .kmeans_plus_plus import _kmeans_plusplus


class PyTorchKMeans(BasicClustering):
    def __init__(self,
                 metric='euclidean',
                 init='k-means++',
                 random_state=0,
                 n_clusters=8,
                 n_init=10,
                 max_iter=300,
                 tol=1e-4,
                 distributed=False,
                 verbose=True):
        super().__init__(n_clusters=n_clusters,
                         init=init,
                         random_state=random_state,
                         n_init=n_init,
                         max_iter=max_iter,
                         tol=tol,
                         verbose=verbose,
                         distributed=distributed)
        self.distance_metric = {'euclidean': pairwise_euclidean, 'cosine': pairwise_cosine}[metric]
        # self.distance_metric = lambda a, b: torch.cdist(a, b, p=2.)
        if isinstance(self.init, (np.ndarray, torch.Tensor)): self.n_init = 1

    def initialize(self, X: torch.Tensor, random_state: int):
        num_samples = len(X)
        if isinstance(self.init, str):
            g = torch.Generator()
            g.manual_seed(random_state)
            if self.init == 'random':
                indices = torch.randperm(num_samples, generator=g)[:self.n_clusters]
                init_state = X[indices]
            elif self.init == 'k-means++':
                init_state, _ = _kmeans_plusplus(X,
                                                 random_state=random_state,
                                                 n_clusters=self.n_clusters,
                                                 pairwise_distance=self.distance_metric)
                # init_state = X[torch.randperm(num_samples, generator=g)[0]].unsqueeze(0)
                # for k in range(1, self.n_clusters):
                #     d = torch.min(self.distance_metric(X, init_state), dim=1)[0]
                #     init_state = torch.cat([init_state, X[torch.argmax(d)].unsqueeze(0)], dim=0)
            else:
                raise NotImplementedError
        elif isinstance(self.init, (np.ndarray, torch.Tensor)):
            init_state = self.init.to(X)
        else:
            raise NotImplementedError

        return init_state

    def fit_predict(self, X: torch.Tensor):

        tol = torch.mean(torch.var(X, dim=0)) * self.tol

        min_inertia, best_states, best_labels = float('Inf'), None, None

        random_states = torch.arange(self.n_init * self.world_size) + self.random_state
        random_states = random_states[self.rank:len(random_states):self.world_size]
        # g = torch.Generator()
        # g.manual_seed(self.random_state)
        # random_states = torch.randperm(10000, generator=g)[:self.n_init * self.world_size]
        # random_states = random_states[self.rank:self.n_init * self.world_size:self.world_size]

        self.stats = {'state': [], 'inertia': [], 'label': []}
        for n_init in range(self.n_init):
            random_state = int(random_states[n_init])
            old_state = self.initialize(X, random_state=random_state)
            old_labels, inertia = self.predict(X, old_state)

            labels = old_labels

            progress_bar = tqdm.tqdm(total=self.max_iter, disable=not self.verbose)

            for n_iter in range(self.max_iter):

                # https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335/7
                # n_samples = X.size(0)
                # weight = torch.zeros(self.n_clusters, n_samples, dtype=X.dtype, device=X.device)  # L, N
                # weight[labels, torch.arange(n_samples)] = 1
                # weight = F.normalize(weight, p=1, dim=1)  # l1 normalization
                # state = torch.mm(weight, X)  # L, F
                state = torch.zeros(self.n_clusters, X.size(1), dtype=X.dtype, device=X.device)
                counts = torch.zeros(self.n_clusters, dtype=X.dtype, device=X.device) + 1e-6
                classes, classes_counts = torch.unique(labels, return_counts=True)
                counts[classes] = classes_counts.to(X)
                state.index_add_(0, labels, X)
                state = state / counts.view(-1, 1)

                # d = self.distance_metric(X, state)
                # inertia, labels = d.min(dim=1)
                # inertia = inertia.sum()
                labels, inertia = self.predict(X, state)

                if inertia < min_inertia:
                    min_inertia = inertia
                    best_states, best_labels = state, labels

                if self.verbose:
                    progress_bar.set_description(
                        f'nredo {n_init + 1}/{self.n_init:02d}, iteration {n_iter:03d} with inertia {inertia:.2f}')
                    progress_bar.update(n=1)

                center_shift = self.distance_metric(old_state, state, pairwise=False)

                if torch.equal(labels, old_labels):
                    # First check the labels for strict convergence.
                    if self.verbose:
                        print(f"Converged at iteration {n_iter}: strict convergence.")
                    break
                else:
                    # center_shift = self.distance_metric(old_state, state).diag().sum()
                    # No strict convergence, check for tol based convergence.
                    # center_shift_tot = (center_shift ** 2).sum()
                    center_shift_tot = center_shift.sum()
                    if center_shift_tot <= tol:
                        if self.verbose:
                            print(
                                f"Converged at iteration {n_iter}: center shift "
                                f"{center_shift_tot} within tolerance {tol} "
                                f"and min inertia {min_inertia.item()}."
                            )
                        break

                old_labels[:] = labels
                old_state = state
            progress_bar.close()
            self.stats['state'].append(old_state)
            self.stats['inertia'].append(inertia)
            self.stats['label'].append(old_labels)

        self.stats['state'] = torch.stack(self.stats['state'])
        self.stats['inertia'] = torch.stack(self.stats['inertia'])
        self.stats['label'] = torch.stack(self.stats['label'])
        if self.distributed:
            min_inertia = self.distributed_sync(min_inertia)
            best_idx = torch.argmin(min_inertia).item()
            min_inertia = min_inertia[best_idx]
            dist.broadcast(best_labels, src=best_idx)
            dist.broadcast(best_states, src=best_idx)
            self.stats['state'] = self.distributed_sync(self.stats['state'])
            self.stats['inertia'] = self.distributed_sync(self.stats['inertia'])
            self.stats['label'] = self.distributed_sync(self.stats['label'])

        if self.verbose:
            print(f"Final min inertia {min_inertia.item()}.")

        self.cluster_centers_ = best_states
        return best_labels

    def predict(self, X: torch.Tensor, cluster_centers_=None):
        if cluster_centers_ is None:
            cluster_centers_ = self.cluster_centers_
        split_size = min(4096, X.size(0))
        inertia, pred_labels = 0., []
        for f in X.split(split_size, dim=0):
            d = self.distance_metric(f, cluster_centers_)
            inertia_, labels_ = d.min(dim=1)
            inertia += inertia_.sum()
            pred_labels.append(labels_)
        return torch.cat(pred_labels, dim=0), inertia


if __name__ == '__main__':
    torch.cuda.set_device(1)
    clustering_model = PyTorchKMeans(metric='cosine',
                                     init='k-means++',
                                     random_state=0,
                                     n_clusters=1000,
                                     n_init=10,
                                     max_iter=300,
                                     tol=1e-4,
                                     distributed=False,
                                     verbose=True)
    X = torch.randn(1280000, 256).cuda()
    clustering_model.fit_predict(X)
