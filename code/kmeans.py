import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# 读取PCA处理后的数据
data_path = '../data/pca_data.csv'
df = pd.read_csv(data_path)

# 选择PCA特征进行聚类
features = df.loc[:, 'pca_feature1':'pca_feature100']

# 进行KMeans聚类
print("Performing KMeans clustering...")
kmeans = KMeans(n_clusters=9, random_state=0, n_init=10)
# 添加错误处理和替代方案
try:
    df['label'] = kmeans.fit_predict(features)
except AttributeError:
    # 使用替代方法：先fit再predict
    kmeans.fit(features)
    df['label'] = kmeans.predict(features)

# 使用PCA将特征降至2维以便可视化
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

# 绘制聚类结果
plt.figure(figsize=(10, 6))
plt.scatter(features_2d[:, 0], features_2d[:, 1], c=df['label'], cmap='tab10')
plt.title('KMeans Clustering')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.colorbar(label='Cluster Label')
plt.show()

# 保存带标签的完整数据
output_path = '../data/labeled_data.csv'
df.to_csv(output_path, index=False)
print(f"\nLabeled data saved to {output_path}")

# 根据标签分组并连接相同标签的文本
labeled_text_df = df.groupby('label')['text'].apply(lambda x: ','.join(x)).reset_index()

# 保存分组后的文本
text_output_path = '../data/labeled_text.csv'
labeled_text_df.to_csv(text_output_path, index=False)
print(f"Labeled text saved to {text_output_path}")
