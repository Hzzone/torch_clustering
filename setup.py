# -*- coding: UTF-8 -*-
'''
@Project : torch_clustering 
@File    : setup.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 1:39 PM 
'''

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch_clustering",
    version="0.0.1",
    author="Zhizhong Huang",
    author_email="zzhuang19@fudan.edu.cn",
    description="A pure pytorch implementation of kmeans and GMM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hzzone/torch_clustering",
    packages=['torch_clustering', ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_data={"": ["*"]},  # 数据文件全部打包
)