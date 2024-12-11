import pandas as pd
import numpy as np
import ast
from sklearn.decomposition import PCA
# 从CSV文件读取数据
csv_path = r'E:\pythonProject\9topics\vec_data.csv'
df = pd.read_csv(csv_path)
# 将字符串表示的列表转为实际的列表
df['vec'] = df['vec'].apply(lambda x: np.array(ast.literal_eval(x)))
# 将每一行展平为一维数组
df['vec'] = df['vec'].apply(lambda x: np.concatenate(x))
# 计算最大长度
max_len = max(df['vec'].apply(len))
print(max_len)
# 补零
df['vec'] = df['vec'].apply(lambda x: np.pad(x, (0, max_len - len(x)), 'constant'))

# # 创建新的DataFrame，每一位作为一列特征
features_df = pd.DataFrame(df['vec'].to_list(), columns=[f'feature{i+1}' for i in range(max_len)])
# # 合并原始DataFrame和新的特征DataFrame
df = pd.concat([df, features_df], axis=1)
# 选择 'feature1' 到 'feature6150' 这些特征进行降维
features_to_pca = df.loc[:, 'feature1':'feature'+str(max_len)]
# 使用PCA进行降维到500维
pca = PCA(n_components=500)
df_pca = pd.DataFrame(pca.fit_transform(features_to_pca), columns=[f'pca_feature{i+1}' for i in range(500)])
df = pd.concat([df, df_pca], axis=1)
# # 保存带有标签的数据
labeled_data_csv_path = r'E:\pythonProject\9topics\pca_data.csv'
df.to_csv(labeled_data_csv_path, index=False)
