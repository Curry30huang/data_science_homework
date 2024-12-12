import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os

def print_top_words(model, feature_names, n_top_words):
    """打印每个主题的前N个关键词"""
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx}:")
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword

def plot_perplexity(n_topics_range, perplexities):
    """绘制困惑度折线图"""
    plt.figure(figsize=(12, 6))
    plt.plot(list(n_topics_range), perplexities, 'bo-')
    plt.xlabel("number of topics")
    plt.ylabel("perplexity")
    plt.grid(True)
    plt.savefig('../data/perplexity_plot.png')
    plt.show()

# 主程序
if __name__ == "__main__":
    # 创建输出目录
    # os.makedirs('output', exist_ok=True)

    # 读取数据
    data_path = '../data/processed_data.csv'
    data = pd.read_csv(data_path)
    # 剔除小于1000词语的文本
    data = data[data.text.str.split(' ').str.len() > 1000]

    # 特征提取
    n_features = 1000  # 提取1000个特征词语
    tf_vectorizer = CountVectorizer(
        strip_accents='unicode',
        max_features=n_features,
        max_df=0.5, #  词语在文档中出现的频率大于的词语，
        min_df=10 # 词语在文档中出现的频率小于10的词语
    )
    tf = tf_vectorizer.fit_transform(data.text)

    # 计算不同主题数量下的困惑度
    print("Calculating perplexity for different numbers of topics...")
    plexs = []
    n_max_topics = 16

    for i in range(1, n_max_topics):
        print(f"Training LDA with {i} topics...")
        lda = LatentDirichletAllocation(
            n_components=i,
            max_iter=50,
            learning_method='batch',
            learning_offset=50,
            random_state=0
        )
        lda.fit(tf)
        plexs.append(lda.perplexity(tf))

    # 绘制困惑度折线图
    n_t = 15  # 区间最右侧的值
    x = list(range(1, n_t+1))
    plot_perplexity(x, plexs[0:n_t])

    # TODO: 选择最佳主题数量（这里使用7个主题作为示例）
    n_topics = 7
    print(f"\nTraining final LDA model with {n_topics} topics...")

    # 训练最终模型
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=50,
        learning_method='batch',
        learning_offset=50,
        random_state=0
    )
    lda.fit(tf)

    # 打印每个主题的关键词
    print("\nTop words in each topic:")
    n_top_words = 8
    tf_feature_names = tf_vectorizer.get_feature_names_out()
    topic_word = print_top_words(lda, tf_feature_names, n_top_words)

    # 获取文档-主题分布
    topics = lda.transform(tf)

    # 为每个文档分配主题
    topic = []
    for t in topics:
        topic.append("Topic #" + str(list(t).index(np.max(t))))

    # 添加结果到数据框
    data['概率最大的主题序号'] = topic
    data['每个主题对应概率'] = list(topics)

    # 保存结果
    output_path = '../data/data_topic.csv'
    data.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
