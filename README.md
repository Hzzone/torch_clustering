## torch_clustering

This repo contains a pure PyTorch implementation of the following:
* Kmeans with kmeans++ initialization;
* Gaussian Mixture Model (GMM);
* Support for `euclidean` and `cosine` distance;
* Support for both `cpu` and `gpu` tensors, and distributed clustering!

In addition, we provide a [Faiss](https://github.com/facebookresearch/faiss) wrapper that can be used with my code without any changes!

**If you found this code helps your work, do not hesitate to cite my paper or star this repo!**

### Install

```shell
git clone --depth https://github.com/Hzzone/torch_clustering
cd torch_clustering && pip install -e .
```

### Example

There are two files for examples:
* distribute_kmeans_example.py demonstrates how to use distributed clustering;
* example_and_benchmark.ipynb [![Explore in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hzzone/torch_clustering/blob/master/example_and_benchmark.ipynb)

Snippet:

```python
from torch_clustering import PyTorchKMeans, FaissKMeans, PyTorchGaussianMixture, evaluate_clustering
clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
psedo_labels = clustering_model.fit_predict(features)
```

### Benchmark

Tested on colab

...

### Citation

```
@misc{huang2022learning,
    title={Learning Representation for Clustering via Prototype Scattering and Positive Sampling},
    author={Zhizhong Huang and Jie Chen and Junping Zhang and Hongming Shan},
    year={2021},
    eprint={2111.11821},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```