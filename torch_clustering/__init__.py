# -*- coding: UTF-8 -*-
'''
@Project : torch_clustering 
@File    : __init__.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 12:21 PM 
'''

from .kmeans.kmeans import PyTorchKMeans
from .faiss_kmeans import FaissKMeans
from .gaussian_mixture import PyTorchGaussianMixture
from .beta_mixture import BetaMixture1D

import numpy as np
from munkres import Munkres
from sklearn import metrics
import warnings

def evaluate_clustering(label, pred, eval_metric=['nmi', 'acc', 'ari'], phase='train'):
    mask = (label != -1)
    label = label[mask]
    pred = pred[mask]
    results = {}
    if 'nmi' in eval_metric:
        nmi = metrics.normalized_mutual_info_score(label, pred, average_method='arithmetic')
        results[f'{phase}_nmi'] = nmi
    if 'ari' in eval_metric:
        ari = metrics.adjusted_rand_score(label, pred)
        results[f'{phase}_ari'] = ari
    if 'f' in eval_metric:
        f = metrics.fowlkes_mallows_score(label, pred)
        results[f'{phase}_f'] = f
    if 'acc' in eval_metric:
        n_clusters = len(set(label))
        if n_clusters == len(set(pred)):
            pred_adjusted = get_y_preds(label, pred, n_clusters=n_clusters)
            acc = metrics.accuracy_score(pred_adjusted, label)
        else:
            acc = 0.
            warnings.warn('TODO: the number of classes is not equal...')
        results[f'{phase}_acc'] = acc
    return results


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred
