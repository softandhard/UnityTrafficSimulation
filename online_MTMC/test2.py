from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成示例数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 初始化谱聚类算法
spectral_clustering = SpectralClustering(n_clusters=4, affinity='nearest_neighbors', n_neighbors=10)

# 对数据进行聚类
clusters = spectral_clustering.fit_predict(X)

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7)
plt.show()

import numpy as np

# 创建一个示例的 ndarray
arr = np.array([[1, 5, 3],
                [4, 2, 7],
                [9, 6, 8]])

# 获取最小值
min_value = np.mean(arr)

print("最小值:", min_value)