import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
csv_path = r'E:\pythonProject\9topics\pca_data.csv'
df = pd.read_csv(csv_path)
features = df.loc[:, 'pca_feature1':'pca_feature500']
# 进行KMeans聚类
kmeans = KMeans(n_clusters=9, random_state=0)
df['label'] = kmeans.fit_predict(features)

# 初始化PCA模型，设置目标维度为2
pca = PCA(n_components=2)
# 对特征进行降维
features = pca.fit_transform(features)
plt.figure(figsize=(10, 6))
plt.scatter(features[:, 0], features[:, 1], c=df['label'], cmap='tab10')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Label')
plt.show()
labeled_data_csv_path = r'E:\pythonProject\9topics\labeled_data.csv'
df.to_csv(labeled_data_csv_path, index=False)
# 根据标签进行分组并连接相同标签的文本
labeled_text_df = df.groupby('label')['text'].apply(lambda x: ','.join(x)).reset_index()
# 保存到CSV文件
labeled_text_csv_path = r'E:\pythonProject\9topics\labeled_text.csv'
labeled_text_df.to_csv(labeled_text_csv_path, index=False)
