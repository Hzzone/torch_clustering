# -*- coding: UTF-8 -*-
'''
@Project : torch_clustering 
@File    : distribute_kmeans_example.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 3:57 PM 
'''

if __name__ == '__main__':
    # export CUDA_VISIBLE_DEVICES=3,4
    # torchrun --master_port 17673 --nproc_per_node=2 distribute_kmeans_example.py
    # or python -m torch.distributed.launch --nproc_per_node=2 distribute_kmeans_example.py
    import torch.distributed as dist
    import torch

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(dist.get_rank())
    torch.autograd.set_grad_enabled(False)

    data = torch.load('/home/zzhuang/cifar10_features.pth', map_location='cpu')
    features, labels = data[:, :-1], data[:, -1]
    features, labels = features.cuda().float(), labels.cuda().long()

    kwargs = {
        'metric': 'cosine',  # euclidean if not l2_normalize
        'distributed': True,
        'random_state': 0,
        'n_clusters': int(labels.max() + 1),
        'verbose': False
    }

    from torch_clustering import PyTorchKMeans, evaluate_clustering

    clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
    psedo_labels = clustering_model.fit_predict(features)
    results = evaluate_clustering(labels.cpu().numpy(), psedo_labels.cpu().numpy(), eval_metric=['nmi', 'acc'], phase='train')
    print(results)
