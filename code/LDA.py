import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
from collections import defaultdict
import warnings
import matplotlib.font_manager as fm
warnings.filterwarnings('ignore')  # 忽略警告信息

def compute_coherence_score(model, term_frequency, feature_names):
    """计算主题连贯性分数"""
    topic_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-10:-1]  # 获取前10个词的索引
        topic_words.append([feature_names[i] for i in top_words_idx])
    
    # 计算平均PMI分数作为连贯性分数
    coherence = 0
    word_doc_freq = defaultdict(int)
    total_docs = term_frequency.shape[0]
    
    # 计算词频
    for word_idx in range(len(feature_names)):
        word_doc_freq[feature_names[word_idx]] = np.sum(term_frequency[:, word_idx] > 0)
    
    # 计算每个主题的PMI
    for topic in topic_words:
        topic_coherence = 0
        for i in range(len(topic)):
            for j in range(i + 1, len(topic)):
                # 计算共现频率
                w1, w2 = topic[i], topic[j]
                w1_idx = np.where(feature_names == w1)[0][0]
                w2_idx = np.where(feature_names == w2)[0][0]
                co_doc_freq = np.sum((term_frequency[:, w1_idx] > 0) & (term_frequency[:, w2_idx] > 0))
                
                # 计算PMI
                if co_doc_freq > 0:
                    pmi = np.log(co_doc_freq * total_docs / (word_doc_freq[w1] * word_doc_freq[w2]))
                    topic_coherence += pmi
        
        coherence += topic_coherence
    
    return coherence / len(topic_words)

def print_top_words(model, feature_names, n_top_words):
    """打印每个主题的前N个关键词"""
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx}:")
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword

def plot_metrics(n_topics_range, perplexities, coherence_scores):
    """绘制困惑度和连贯性分数的对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制困惑度
    ax1.plot(n_topics_range, perplexities, 'bo-')
    ax1.set_xlabel('主题数量')
    ax1.set_ylabel('困惑度')
    ax1.set_title('困惑度随主题数量的变化')
    ax1.grid(True)
    
    # 绘制连贯性分数
    ax2.plot(n_topics_range, coherence_scores, 'ro-')
    ax2.set_xlabel('主题数量')
    ax2.set_ylabel('连贯性分数')
    ax2.set_title('连贯性分数随主题数量的变化')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('../data/topic_metrics.png')
    plt.show()

# 设置中文字体
def set_chinese_font():
    # 找到系统中可用的中文字体
    font_path = '/System/Library/Fonts/STHeiti Light.ttc'  # macOS系统的中文字体路径
    if not os.path.exists(font_path):
        # 如果路径不存在，可以尝试其他路径或下载字体
        raise FileNotFoundError("找不到中文字体，请检查路径或安装中文字体。")
    
    # 设置字体
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()

# 主程序
if __name__ == "__main__":
    set_chinese_font()  # 设置中文字体

    # 读取数据
    data_path = '../data/processed_data.csv'
    data = pd.read_csv(data_path)
    # 剔除小于1000词语的文本
    data = data[data.text.str.split(' ').str.len() > 1000]

    # 特征提取
    n_features = 1000
    tf_vectorizer = CountVectorizer(
        strip_accents='unicode',
        max_features=n_features,
        max_df=0.5,
        min_df=10
    )
    tf = tf_vectorizer.fit_transform(data.text)
    feature_names = tf_vectorizer.get_feature_names_out()

    # 计算不同主题数量下的困惑度和连贯性分数
    print("计算不同主题数量下的评估指标...")
    perplexities = []
    coherence_scores = []
    n_topics_range = range(2, 21, 2)  # 从2到20，步长为2

    for n_topics in n_topics_range:
        print(f"训练 {n_topics} 个主题的LDA模型...")
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=50,
            learning_method='batch',
            learning_offset=50,
            random_state=0
        )
        lda.fit(tf)
        
        perplexity = lda.perplexity(tf)
        coherence = compute_coherence_score(lda, tf.toarray(), feature_names)
        
        perplexities.append(perplexity)
        coherence_scores.append(coherence)
        
        print(f"主题数量: {n_topics}")
        print(f"困惑度: {perplexity:.2f}")
        print(f"连贯性分数: {coherence:.4f}\n")

    # 绘制评估指标图
    plot_metrics(list(n_topics_range), perplexities, coherence_scores)

    # 根据评估指标选择最佳主题数量
    # 这里使用连贯性分数最高的主题数量
    best_n_topics = n_topics_range[np.argmax(coherence_scores)]
    print(f"\n根据连贯性分数选择的最佳主题数量: {best_n_topics}")

    # 使用最佳主题数量训练最终模型
    print(f"\n使用 {best_n_topics} 个主题训练最终模型...")
    final_lda = LatentDirichletAllocation(
        n_components=best_n_topics,
        max_iter=50,
        learning_method='batch',
        learning_offset=50,
        random_state=0
    )
    final_lda.fit(tf)

    # 打印每个主题的关键词
    print("\n每个主题的关键词:")
    n_top_words = 8
    topic_word = print_top_words(final_lda, feature_names, n_top_words)

    # 获取文档-主题分布
    topics = final_lda.transform(tf)

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
    print(f"\n结果已保存到 {output_path}")
