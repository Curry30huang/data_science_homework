import pandas as pd
import numpy as np
from wordcloud import WordCloud
from wordcloud import STOPWORDS, ImageColorGenerator
from PIL import Image
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
# 生成创意词云

# 创建一个简单的圆形遮罩
def create_circle_mask(diameter):
    mask = np.ones((diameter, diameter), dtype=np.uint8) * 255
    center = diameter // 2
    y, x = np.ogrid[:diameter, :diameter]
    mask_area = (x - center) ** 2 + (y - center) ** 2 <= center ** 2
    mask[mask_area] = 0
    return mask

# 使用圆形遮罩
mask = create_circle_mask(800)

# 将遮罩转换为3通道图像
mask = np.stack((mask,) * 3, axis=-1)

# 生成创意词云
creative_wordcloud = WordCloud(
    width=1600,
    height=800,
    background_color='white',
    font_path='../data/SimHei.ttf',  # 使用中文字体
    max_words=200,
    max_font_size=150,
    min_font_size=10,
    random_state=42,
    mask=mask,
    stopwords=STOPWORDS,
    colormap='rainbow'  # 使用彩虹色彩映射
).generate(all_text)

# 显示创意词云
plt.figure(figsize=(20, 10))
plt.imshow(creative_wordcloud, interpolation='bilinear')
plt.axis('off')

# 显示图形
plt.show()

# 保存创意词云图片
creative_wordcloud.to_file('../data/creative_wordcloud.png')