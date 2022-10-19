# -*- coding: UTF-8 -*-
'''
@Project : torch_clustering 
@File    : faiss_kmeans.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 12:22 PM 
'''

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

try:
    import faiss
except:
    print('faiss not installed')
from .__base__ import BasicClustering


class FaissKMeans(BasicClustering):
    def __init__(self,
                 metric='euclidean',
                 n_clusters=8,
                 n_init=10,
                 max_iter=300,
                 random_state=1234,
                 distributed=False,
                 verbose=True):
        super().__init__(n_clusters=n_clusters,
                         n_init=n_init,
                         max_iter=max_iter,
                         distributed=distributed,
                         verbose=verbose)

        if metric == 'euclidean':
            self.spherical = False
        elif metric == 'cosine':
            self.spherical = True
        else:
            raise NotImplementedError
        self.random_state = random_state

    def apply_pca(self, X, dim):
        n, d = X.shape
        if self.spherical:
            X = F.normalize(X, dim=1)
        mat = faiss.PCAMatrix(d, dim)
        mat.train(n, X)
        X = mat.apply_py(X)

    def fit_predict(self, input: torch.Tensor, device=-1):
        n, d = input.shape

        assert isinstance(input, (torch.Tensor, np.ndarray))
        is_torch_tensor = isinstance(input, torch.Tensor)
        if is_torch_tensor:
            if self.spherical:
                input = F.normalize(input, dim=1)

            if input.is_cuda:
                device = input.device.index
            X = input.cpu().numpy().astype(np.float32)
        else:
            if self.spherical:
                X = input / np.linalg.norm(input, 2, axis=1)[:, np.newaxis]
            else:
                X = input
            X = X.astype(np.float32)

        random_states = torch.arange(self.world_size) + self.random_state
        random_state = random_states[self.rank]
        if device >= 0:
            # faiss implementation of k-means
            clus = faiss.Clustering(int(d), int(self.n_clusters))

            # Change faiss seed at each k-means so that the randomly picked
            # initialization centroids do not correspond to the same feature ids
            # from an epoch to another.
            #                 clus.seed = np.random.randint(1234)
            clus.seed = int(random_state)

            clus.niter = self.max_iter
            clus.max_points_per_centroid = 10000000
            clus.min_points_per_centroid = 10
            clus.spherical = self.spherical
            clus.nredo = self.n_init
            clus.verbose = self.verbose
            res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.useFloat16 = False
            flat_config.device = device
            flat_config.verbose = self.verbose
            flat_config.spherical = self.spherical
            flat_config.nredo = self.n_init
            index = faiss.GpuIndexFlatL2(res, d, flat_config)

            # perform the training
            clus.train(X, index)
            D, I = index.search(X, 1)
        else:
            clus = faiss.Kmeans(d=d,
                                k=self.n_clusters,
                                niter=self.max_iter,
                                nredo=self.n_init,
                                verbose=self.verbose,
                                spherical=self.spherical)
            X = X.astype(np.float32)
            clus.train(X)
            # self.cluster_centers_ = self.kmeans.centroids
            D, I = clus.index.search.search(X, 1)  # for each sample, find cluster distance and assignments

        tensor_device = 'cpu' if device < 0 else f'cuda:{device}'

        best_labels = torch.from_numpy(I.flatten()).to(tensor_device)
        min_inertia = torch.from_numpy(D.flatten()).to(tensor_device).sum()
        best_states = faiss.vector_to_array(clus.centroids).reshape(self.n_clusters, d)
        best_states = torch.from_numpy(best_states).to(tensor_device)

        if self.distributed:
            min_inertia = self.distributed_sync(min_inertia)
            best_idx = torch.argmin(min_inertia).item()
            min_inertia = min_inertia[best_idx]
            dist.broadcast(best_labels, src=best_idx)
            dist.broadcast(best_states, src=best_idx)

        if self.verbose:
            print(f"Final min inertia {min_inertia.item()}.")

        self.cluster_centers_ = best_states
        return best_labels


if __name__ == '__main__':
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(dist.get_rank())
    X = torch.randn(1280, 256).cuda()
    clustering_model = FaissKMeans(metric='euclidean',
                                   n_clusters=10,
                                   n_init=2,
                                   max_iter=1,
                                   random_state=1234,
                                   distributed=True,
                                   verbose=True)
    clustering_model.fit_predict(X)
