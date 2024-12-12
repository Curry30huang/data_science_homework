import pandas as pd
import os

def standardize_dataset_type1(df):
    """
    处理第一类数据集 (包含 content_id, content_text, created_time)
    """
    # 选择需要的列并确保列名正确
    return df[['content_id', 'content_text', 'created_time']]

def standardize_dataset_type2(df):
    """
    处理第二类数据集 (包含 comment_id, content, publish_time)
    将comment_id的值赋给content_id，然后删除comment_id
    """
    # 重命名列以匹配标准格式
    df = df.rename(columns={
        'content': 'content_text',
        'publish_time': 'created_time'
    })

    # 如果存在content_id列，使用comment_id的值覆盖它
    if 'content_id' in df.columns:
        df['content_id'] = df['comment_id']
    else:
        # 如果不存在content_id列，将comment_id重命名为content_id
        df = df.rename(columns={'comment_id': 'content_id'})

    # 确保只返回需要的列，且每列只出现一次
    return df[['content_id', 'content_text', 'created_time']]

def merge_datasets():
    """
    合并所有数据集
    """
    try:
        # 读取数据集
        contents1 = pd.read_csv('../data/1_search_contents_2024-12-09 copy 2.csv')
        contents2 = pd.read_csv('../data/1_search_contents_2024-12-09.csv')
        comments = pd.read_csv('../data/1_search_comments_2024-12-09.csv')

        # 打印每个数据集的信息
        print("数据集1列名:", contents1.columns.tolist())
        print("数据集2列名:", contents2.columns.tolist())
        print("数据集3列名:", comments.columns.tolist())

        # 标准化每个数据集
        contents_std1 = standardize_dataset_type1(contents1)
        contents_std2 = standardize_dataset_type1(contents2)
        comments_std = standardize_dataset_type2(comments)

        # 检查标准化后的数据集
        print("\n标准化后的列名:")
        print("数据集1:", contents_std1.columns.tolist())
        print("数据集2:", contents_std2.columns.tolist())
        print("数据集3:", comments_std.columns.tolist())

        # 检查每个数据集是否有重复的列名
        for i, df in enumerate([contents_std1, contents_std2, comments_std], 1):
            if len(df.columns) != len(set(df.columns)):
                print(f"警告：数据集{i}中存在重复的列名")
                print(f"列名：{df.columns.tolist()}")

        # 合并数据集
        merged_df = pd.concat([contents_std1, contents_std2, comments_std], ignore_index=True)

        # 检查合并后的数据
        print("\n合并后的数据信息:")
        print("总行数:", len(merged_df))
        print("列名:", merged_df.columns.tolist())
        print("content_id是否有重复:", merged_df['content_id'].duplicated().sum())

        # 确保时间戳格式一致
        merged_df['created_time'] = pd.to_numeric(merged_df['created_time'], errors='coerce')

        # 删除重复项和空值
        merged_df = merged_df.drop_duplicates(subset=['content_id'], keep='first')
        merged_df = merged_df.dropna(subset=['content_text', 'created_time'])

        # 保存结果
        output_dir = '../data'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, 'merged_data.csv')
        merged_df.to_csv(output_path, index=False, encoding='utf-8')

        print(f"\n数据合并完成。最终行数: {len(merged_df)}")

    except Exception as e:
        print(f"处理数据时出错: {str(e)}")
        # 打印更详细的错误信息
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    merge_datasets()
