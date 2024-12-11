import vectorize
import pandas as pd
def pure_data():
    # 读取原始数据
    input_file_path = r'E:\pythonProject\9topics\data_20231118125833.csv'
    df = pd.read_csv(input_file_path)
    strs = df['text'].tolist()
    print(strs[:5])
    for i in range(len(strs)):
        start_index = strs[i].find('/') + 1  # 跳过 "@"
        end_index = strs[i].find('：', start_index)  # 寻找 ":" 的位置，从 "@" 之后开始查找
        if start_index > 0 and end_index > start_index:
            extracted_content = strs[i][start_index:end_index]
            strs[i] = strs[i].replace(extracted_content,"")
        strs[i] = strs[i].replace("[图片] 暂不支持查看图片", "")
        strs[i] = strs[i].replace("此内容暂时不可见", "")
    new_df = pd.DataFrame({
        'text': strs,'time':df['time']})
    # 保存为新的 CSV 文件
    output_file_path = r'E:\pythonProject\9topics\pure_data.csv'
    new_df.to_csv(output_file_path, index=False)
pure_data()
file_root = "pure_data.csv"
vectorize.export(vectorize.vectorize_file(file_root),"vec_data.csv")