import pandas as pd
from sklearn.model_selection import train_test_split
import jieba
import os
from gensim.models import Word2Vec

VECTOR_SIZE = 50
WINDOW = 5
MIN_COUNT = 1
WORD2VEC_EPOCHS = 20

file_root = "data_20231118125833.csv"
stop_words_root = "stopwords-master"
stop_words_file_list = ["baidu_stopwords.txt", "cn_stopwords.txt", "hit_stopwords.txt", "scu_stopwords.txt"]

def read_data(root = "./data_20231118125833.csv"):
    data = pd.read_csv(root)
    return data

# 读取停止词列表
def read_stopwords_file(root = "stopwords-master", file_list = []):
    stop_words = set()
    for file_name in file_list:
        path = os.path.join(root, file_name)
        with open(path, 'r', encoding='utf-8') as f:
            stop_words.update([line.rstrip("\n") for line in f.readlines()])
    return list(stop_words)

def preprocess_text(text, stop_words_list):
    # 使用jieba进行分词
    words = jieba.cut(text)
    # 去除停止词
    words = [word.strip() for word in words if (word.strip() not in stop_words_list and word.strip() != '')]

    clean_text = ' '.join(words)
    return clean_text

# 分割训练集和测试集
def split_data(data):
    data['time'] = pd.to_datetime(data['time'])
    data = data.sort_values(by='time')

    # 按时间均匀划分训练集和测试集
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    return train_data, test_data

# word2vec词向量化
def vectorize(data):
    assert "text" in data.columns.to_list(), "Not exist text."
    vec_data = data.copy()
    vec_data["text"] = vec_data["text"].str.split(' ')
    tokenized_corpus = vec_data["text"].tolist()
    model = Word2Vec(sentences=tokenized_corpus, vector_size=VECTOR_SIZE, window=WINDOW, min_count=MIN_COUNT, workers=4, epochs=WORD2VEC_EPOCHS)
    vectorized_corpus = [model.wv[words] for words in tokenized_corpus]
    return vectorized_corpus

# 导出
def export(data, path = "vec_data.csv"):
    data.to_csv(path, index=False)
    print("Success export.")

if __name__ == '__main__':
    data = read_data(file_root)
    stop_words = read_stopwords_file(root=stop_words_root, file_list=stop_words_file_list)
    data["text"] = data["text"].apply(preprocess_text, args=(stop_words,))
    data = data[data["text"] != '']
    data["vec"] = vectorize(data)
    print(data.head())
    print(len(data))
    export(data)