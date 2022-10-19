# -*- coding: UTF-8 -*-
'''
@Project : torch_clustering 
@File    : kmeans_plus_plus.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 12:23 PM 
'''

import torch
import numpy as np
import warnings


def stable_cumsum(arr, dim=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat.
    axis : int, default=None
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float, default=1e-05
        Relative tolerance, see ``np.allclose``.
    atol : float, default=1e-08
        Absolute tolerance, see ``np.allclose``.
    """
    if dim is None:
        arr = arr.flatten()
        dim = 0
    out = torch.cumsum(arr, dim=dim, dtype=torch.float64)
    expected = torch.sum(arr, dim=dim, dtype=torch.float64)
    if not torch.all(torch.isclose(out.take(torch.Tensor([-1]).long().to(arr.device)),
                                   expected, rtol=rtol,
                                   atol=atol, equal_nan=True)):
        warnings.warn('cumsum was found to be unstable: '
                      'its last element does not correspond to sum',
                      RuntimeWarning)
    return out


def _kmeans_plusplus(X,
                     n_clusters,
                     random_state,
                     pairwise_distance,
                     n_local_trials=None):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.
    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for.
    n_clusters : int
        The number of seeds to choose.
    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.
    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The inital centers for k-means.
    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    n_samples, n_features = X.shape

    generator = torch.Generator(device=str(X.device))
    generator.manual_seed(random_state)

    centers = torch.empty((n_clusters, n_features), dtype=X.dtype, device=X.device)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    #     center_id = random_state.randint(n_samples)
    center_id = torch.randint(n_samples, (1,), generator=generator, device=X.device)

    indices = torch.full((n_clusters,), -1, dtype=torch.int, device=X.device)
    centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = pairwise_distance(
        centers[0, None], X)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        #         rand_vals = random_state.random_sample(n_local_trials) * current_pot
        rand_vals = torch.rand(n_local_trials, generator=generator, device=X.device) * current_pot

        candidate_ids = torch.searchsorted(stable_cumsum(closest_dist_sq),
                                           rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        torch.clip(candidate_ids, None, closest_dist_sq.numel() - 1,
                   out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = pairwise_distance(
            X[candidate_ids], X)

        # update closest distances squared and potential for each candidate
        torch.minimum(closest_dist_sq, distance_to_candidates,
                      out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(dim=1)

        # Decide which candidate is the best
        best_candidate = torch.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices
