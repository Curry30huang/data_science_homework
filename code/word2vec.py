import pandas as pd
from sklearn.model_selection import train_test_split
import os
from gensim.models import Word2Vec

VECTOR_SIZE = 50
WINDOW = 5
MIN_COUNT = 1
WORD2VEC_EPOCHS = 20



# 分割训练集和测试集
def split_data(data):
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
def export(data, path = "../data/vec_data.csv"):
    data.to_csv(path, index=False)
    print("Success export.")

if __name__ == '__main__':
    # 读取处理好的数据
    file_path = "../data/processed_data.csv"
    data = pd.read_csv(file_path)
    vec = vectorize(data)
    total_length = 0
    for v in vec:
        total_length += len(v)
    print(total_length)
    data["vec"] = vec
    print(len(data))
    export(data)