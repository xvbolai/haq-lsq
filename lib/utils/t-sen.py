import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np


def tsne_wrapper(feat, n_components = 2):
    r"""
    n_components(默认值:2):嵌入空间的维度，需要降到几维写几

    init: 初始化方法，多采用 PCA 初始化

    perplexity(默认值:30): perplexity 与其他流形学习算法中使用的最近邻的数量有关。考虑选择 5 到 50 之间的值。

    n_iter(默认值: 1000): 优化的最大迭代次数。应至少为 250。

    random_state: 随机种子

    还有其他参数可以调整。有关详细信息
    """
    
    # tsne [num, 2]
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=42).fit_transform(feat)

    # tsne 归一化， 这一步可做可不做
    x_min, x_max = tsne.min(0), tsne.max(0)
    tsne_norm = (tsne - x_min) / (x_max - x_min)
    
    return tsne_norm


# 设置散点形状
maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
# 设置散点颜色
colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
          'hotpink']
# 图例名称
Label_Com = ['a', 'b', 'c', 'd']
# 设置字体格式
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 32,
         }


def tsne_plt(feat, lable, size, n_components = 2, figsize = (8, 8), file_name = 'digits_tsne-plot.png'):
    
    r'''
    feat: [nums, feat_size]
    lable: [nums]
    size: 种类数
    '''
    
    # tsne_normal[i, 0]为横坐标，X_norm[i, 1]为纵坐标，1为散点图的面积， color给每个类别设定颜色
    plt.figure(figsize= figsize)
    
    feat = tsne_wrapper(feat, n_components = n_components)
    
    for i in range(size):
        idxs = (lable == i)
        data = feat[idxs]
        # data[:, 0]为横坐标，data[:, 1]为纵坐标，1为散点图的面积， color给每个类别设定颜色
        plt.scatter(data[:, 0], data[:, 1], 10, c = colors[i], label=f'{i + 1}')

    plt.legend(loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')


if __name__ == "__main__":
    feat = torch.rand(128, 1024)  # 128个特征，每个特征的维度为1024
    label_test1 = [0 for index in range(40)]
    label_test2 = [1 for index in range(40)]
    label_test3 = [2 for index in range(48)]

    label_test = np.array(label_test1 + label_test2 + label_test3)


    tsne_plt(feat, label_test, 10)
