import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from word2vec import vectorize

def pad_vectors(vectorized_corpus):
    """将不同长度的词向量补零到相同长度"""
    # 计算最大长度
    max_len = 0
    for doc_vectors in vectorized_corpus:
        total_len = doc_vectors.shape[0] * doc_vectors.shape[1]
        if total_len > max_len:
            max_len = total_len

    # 将每个文档的词向量展平并补零
    padded_vectors = []
    for doc_vectors in vectorized_corpus:
        # 展平词向量
        flattened = doc_vectors.reshape(-1)
        # 补零
        padded = np.pad(flattened, (0, max_len - len(flattened)), 'constant')
        padded_vectors.append(padded)

    return np.array(padded_vectors), max_len

# 主程序
if __name__ == "__main__":
    # 读取数据
    data_path = '../data/processed_data.csv'
    df = pd.read_csv(data_path)

    # 使用word2vec获取词向量
    print("Converting documents to vectors...")
    vectorized_corpus = vectorize(df)

    # 补零处理
    print("Padding vectors to same length...")
    padded_vectors, max_len = pad_vectors(vectorized_corpus)
    print(f"Max vector length: {max_len}")

    # 使用PCA进行降维
    print("Performing PCA...")
    pca = PCA(n_components=100)  # 降至100维
    X_pca = pca.fit_transform(padded_vectors)

    # 计算解释方差比
    explained_variance_ratio = pca.explained_variance_ratio_
    print(f"Total explained variance: {sum(explained_variance_ratio):.4f}")

    # 创建包含PCA结果的DataFrame
    pca_features = pd.DataFrame(
        X_pca,
        columns=[f'pca_feature{i+1}' for i in range(100)]
    )

    # 将PCA结果添加到原始数据中
    result_df = pd.concat([df, pca_features], axis=1)

    # 保存结果
    output_path = '../data/pca_data.csv'
    result_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # 绘制解释方差比累积图
    plt.figure(figsize=(10, 6))
    cumsum_variance_ratio = np.cumsum(explained_variance_ratio)
    plt.plot(range(1, len(explained_variance_ratio) + 1),
            cumsum_variance_ratio, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio vs. Number of Components')
    plt.grid(True)
    plt.show()
