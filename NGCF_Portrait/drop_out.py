import pandas as pd

# 读取原始评分数据
ratings = pd.read_csv(
    './Data/ml-1m/ratings.dat',
    sep='::',
    header=None,
    engine='python',
    names=['UserID', 'MovieID', 'Rating', 'Timestamp']
)


# 定义减少函数：每个用户保留 (原始数量 * 0.5或0.2) 条记录
def reduce_interactions(group):
    keep_num = int(len(group) * 0.2)  # 确保每个用户随机减少一半或五分之一交互
    return group[['UserID', 'MovieID', 'Rating', 'Timestamp']].sample(n=keep_num, random_state=42)  # 固定随机种子保证可复现


# 按用户分组并减少交互
reduced_ratings = ratings.groupby('UserID', group_keys=False).apply(reduce_interactions)
# 重置索引为 0,1,2,... 的连续整数
reduced_ratings_reset = reduced_ratings.reset_index(drop=True)

# 验证结果（示例检查）
print("[验证样例] 处理后用户交互数量:")
print(reduced_ratings['UserID'].value_counts().head())

# 手动写入文件，使用 :: 作为分隔符
with open('./Data/ml-1m/reduced_ratings_2.dat', 'w', encoding='utf-8') as f:
    for row in reduced_ratings_reset.itertuples(index=False):
        line = f"{row.UserID}::{row.MovieID}::{row.Rating}::{row.Timestamp}\n"
        f.write(line)
