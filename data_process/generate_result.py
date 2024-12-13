import matplotlib.pyplot as plt

# 数据
detectors = ['Mask-RCNN', 'YOLOv7-W6', 'YOLOv7-E6', 'YOLOv7-D6', 'YOLOv7-E6E']
feature_extractors = ['ResNet50-IBN-A + GAP', 'ResNet50-IBN-A + GEM',
                     'ResNet101-IBN-A + GAP', 'ResNet101-IBN-A + GEM']
idf1 = [[71.9, 71.0, 71.9, 72.4], [77.6, 78.0, 76.9, 75.9],
        [76.1, 74.6, 78.4, 76.9], [76.2, 71.1, 74.7, 73.2],
        [77.9, 77.1, 77.5, 75.5]]
idp = [[74.6, 73.3, 73.4, 74.4], [79.7, 80.0, 79.4, 77.2],
       [79.5, 78.0, 78.8, 77.9], [78.1, 75.3, 76.9, 75.3],
       [80.6, 80.0, 79.3, 78.2]]

# 创建图形
fig, ax = plt.subplots()

# 设置图形标题
ax.set_title("Ablation Study on Object Detectors, Feature Extractors, and Pooling Methods")

# 设置 x 轴和 y 轴标签
ax.set_xlabel("Feature Extractor")
ax.set_ylabel("Performance (IDF1, IDP)")

# 设置 x 轴刻度
ax.set_xticks(range(len(feature_extractors)))
ax.set_xticklabels(feature_extractors)

# 设置 y 轴刻度
ax.set_yticks(range(60, 90, 5))

# 绘制数据点
for i, detector in enumerate(detectors):
    ax.plot(range(len(feature_extractors)), idf1[i], label=detector + " (IDF1)", marker='o')
    ax.plot(range(len(feature_extractors)), idp[i], label=detector + " (IDP)", marker='x')

# 设置图例
ax.legend(loc='lower right')

# 添加网格线
ax.grid(True)

# 保存图片
plt.savefig('ablation_study.png')

# 显示图片
plt.show()