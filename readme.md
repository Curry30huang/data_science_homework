# 项目结构

- origin_data: 原始数据
- data: 实际使用的数据
- example_data: 示例数据
- process_data: 处理后的数据
- stop_words: 停用词表
- code: 代码
    - merge_data: 多源数据集整合
    - process: 数据处理，将处理后的数据存储到process_data目录下
    - topiccloud: 主题词云
    - word2vec: word2vec词向量化
    - pca: pca降维
    - kmeans: kmeans聚类
    - classify: 热点话题预测分类

# 开源资源下载

## 停用词表

- 中文停用词表下载：https://github.com/goto456/stopwords

## 爬虫

- 使用了[MediaCrawler](https://github.com/NanmiCoder/MediaCrawler)对新闻数据进行爬取

# 实验结果说明

- 首先运行merge_data.py，将三个数据集整合为一个数据集
- 运行process.py 处理数据存储到 data/processed_data.csv
- 运行topiccloud.py 生成主题词云, 保存到 data/wordcloud.png
- 运行word2vec.py 生成词向量，保存到 data/vec_data.csv
- 运行LDA.py 进行LDA主题模型分析,绘制困惑度折线图,选择最佳主题数量为7，如果需要查看图像与打印词需要重新运行一下
- pca.py 进行pca降维,保存到 data/pca_data.csv
- 运行kmeans.py 进行kmeans聚类，保存结果到 data/labeled_data.csv 和 data/labeled_text.csv。如果需要得到PCA降维到2维图像结果需要重新运行一下。
- 运行classify.py 进行热点话题预测分类
