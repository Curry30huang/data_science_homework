import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 读取数据
data_path = r'../data/processed_data.csv'
df = pd.read_csv(data_path)

# 合并所有文本
all_text = ' '.join(df['text'].astype(str))

# 创建图形
plt.figure(figsize=(20, 10))

# 生成词云
wordcloud = WordCloud(
    width=1600,
    height=800,
    background_color='white',
    font_path='../data/SimHei.ttf',  # 使用中文字体
    max_words=200,
    max_font_size=150,
    min_font_size=10,
    random_state=42,
).generate(all_text)

# 显示词云
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

# 显示图形
plt.show()

# 保存词云图片
wordcloud.to_file('../data/wordcloud.png')