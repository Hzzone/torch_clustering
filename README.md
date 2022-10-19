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

Tested on colab (Tesla T4)

| CIFAR-10                | NMI    | ACC    | Speed       |
|-------------------------|--------|--------|-------------|
| Faiss kmeans (nredo 5 maxiter 50)            | 0.8551 | 0.9236 | 0.73+-0.09  |
| torch_clustering kmeans (nredo 10 maxiter 300) | 0.8552 | 0.9235 | 4.59+-0.05  |
| torch_clustering GMM (nredo 10x10 maxiter 300)   | 0.8559 | 0.9238 | 11.67+-0.33 |

On ImageNet, the performance of torch_clustering will be much better Faiss.


### Citation

```
@article{huang2022learning,
  title={Learning Representation for Clustering via Prototype Scattering and Positive Sampling},
  author={Zhizhong Huang and Jie Chen and Junping Zhang and Hongming Shan},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
}
```