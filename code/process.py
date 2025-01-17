import pandas as pd
import jieba
import os


# 读取文件
def read_data(file_path:str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    data = pd.read_csv(file_path)
    return data

# 读取停止词列表，传入停用词目录
def read_stopwords(file_dir:str):
    if not os.path.exists(file_dir):
        raise FileNotFoundError(f"File not found: {file_dir}")
    stop_words = set()
    # 读取file_dir 下所有的 停用词 txt 文件
    for file_name in os.listdir(file_dir):
        if file_name.endswith(".txt"):
            with open(os.path.join(file_dir, file_name), 'r', encoding='utf-8') as f:
                stop_words.update([line.rstrip("\n") for line in f.readlines()])
    return list(stop_words)

def preprocess_text(text, stop_words_list):
    # 检查是否为空值或非字符串类型
    if pd.isna(text) or not isinstance(text, str):
        return ''

    # 使用jieba进行分词
    words = jieba.cut(text)
    # 去除停止词
    words = [word.strip() for word in words if (word.strip() not in stop_words_list and word.strip() != '')]

    clean_text = ' '.join(words)
    return clean_text

def save_data(data: pd.DataFrame, output_path: str, columns: list = None):
    """
    保存数据到CSV文件

    Args:
        data: 要保存的DataFrame
        output_path: 输出文件路径
        columns: 要保存的列名列表，如果为None则保存所有列
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 如果指定了列名，则只保存指定列
    if columns:
        data = data[columns]

    # 保存数据
    data.to_csv(output_path, index=False, encoding='utf-8')

if __name__ == '__main__':
    file_path = "../data/merged_data.csv"
    data = read_data(file_path)
    file_dir = "../stop_words"
    stop_words = read_stopwords(file_dir)
    
    # 统计预处理前的总词汇量
    original_word_count = 0
    for text in data["content_text"]:
        if pd.isna(text) or not isinstance(text, str):
            continue
        words = jieba.lcut(text)  # 使用jieba分词
        original_word_count += len(words)
    
    print(f"预处理前总词汇量: {original_word_count} 个词")
    
    # 处理文本
    data["text"] = data["content_text"].apply(preprocess_text, args=(stop_words,))
    data = data[data["text"] != '']
    
    # 统计预处理后的总词汇量
    processed_word_count = 0
    for text in data["text"]:
        words = text.split()
        processed_word_count += len(words)
    
    print(f"预处理后总词汇量: {processed_word_count} 个词")
    print(f"词汇量减少: {original_word_count - processed_word_count} 个词")
    print(f"词汇量减少比例: {((original_word_count - processed_word_count) / original_word_count * 100):.2f}%")

    # 保存处理后的数据
    output_path = "../data/processed_data.csv"
    columns_to_save = ['text','content_id','content_text','created_time']
    save_data(data, output_path, columns_to_save)

    print("\n处理后数据预览:")
    print(data.head())
    print(f"总文档数: {len(data)}")
