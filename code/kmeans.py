import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
from wordcloud import WordCloud, STOPWORDS
import jieba

N_CLUSTER = 10
# 读取PCA处理后的数据
data_path = '../data/pca_data.csv'
df = pd.read_csv(data_path)

# 选择PCA特征进行聚类
features = df.loc[:, 'pca_feature1':'pca_feature100']

# 尝试不同的聚类数量
best_k = 4
best_inertia = np.inf
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=50)
    kmeans.fit(features)
    if kmeans.inertia_ < best_inertia:
        best_k = k
        best_inertia = kmeans.inertia_

print(f"Best number of clusters: {best_k}")

# 使用最佳聚类数量进行KMeans聚类
kmeans = KMeans(n_clusters=N_CLUSTER, random_state=20000, n_init=100)
results = kmeans.fit_predict(features)

# 将results中小于5个元素的聚类标签替换为-1，其余结果改变成从0开始的连续整数
solid_label = 0
for i in range(N_CLUSTER):
    if np.sum(results == i) < 10:
        results[results == i] = -1
    else:
        results[results == i] = solid_label
        solid_label += 1
df['label'] = results

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

# 自定义x轴y轴的范围
plt.xlim(-150, 25)  # 根据需要调整范围
plt.ylim(-150, 30)  # 根据需要调整范围

plt.savefig('../data/kmeans.png')
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
# 创建输出目录
output_dir = '../data/wordclouds'
os.makedirs(output_dir, exist_ok=True)

# 生成词云并保存

# 添加中文字体路径
font_path = '/System/Library/Fonts/STHeiti Light.ttc'

for label in labeled_text_df['label'].unique():
    if label >= 0:
        text = labeled_text_df[labeled_text_df['label'] == label]['text'].values[0]
        # 使用jieba分词
        text = ' '.join(jieba.cut(text))
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white', 
            font_path=font_path, 
            stopwords=STOPWORDS,
            contour_width=3, 
            contour_color='steelblue',
            colormap='viridis'
        ).generate(text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for Cluster {label}', fontsize=20, color='navy')
        wordcloud_path = os.path.join(output_dir, f'wordcloud_cluster_{label}.png')
        plt.savefig(wordcloud_path)
        plt.close()
        print(f"Word cloud for cluster {label} saved to {wordcloud_path}")