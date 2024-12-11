import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 读取数据
data_path = r'E:\pythonProject\9topics\labeled_text.csv'
df = pd.read_csv(data_path)

# 生成3x3的九宫格
fig = plt.figure(figsize=(18, 18))
grid = GridSpec(3, 3, wspace=0.05, hspace=0.05)

# 对每一行文本生成词云并绘制到对应的子图中
for i, (index, row) in enumerate(df.iterrows()):
    text = row['text']

    # 生成词云
    wordcloud = WordCloud(width=400, height=400, background_color='white',font_path='C:/Windows/Fonts/simhei.ttf').generate(text)

    # 在九宫格中绘制词云
    ax = fig.add_subplot(grid[i])
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f'Label: {row["label"]}')
    ax.axis('off')

# 显示九宫格
plt.show()
