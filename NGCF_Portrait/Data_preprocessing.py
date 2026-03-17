import pandas as pd
import numpy as np

import os

# 获取当前工作目录
current_directory = os.getcwd()
print("当前工作目录是:", current_directory)

# 用户数据----------------------------------------------------------------------
# 1. 导入数据集
users_path = './Data/ml-1m/users.dat'

user_portrait_name = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']

users_data = pd.read_csv(users_path, sep='::', header=None, names=user_portrait_name, engine='python')

# 将性别字段转换为数值型特征
users_data['Gender'] = users_data['Gender'].map({'F': 0., 'M': 1.})

# 定义年龄组
age_groups = {
    1: 'Teen',
    18: 'Teen',
    25: 'Adult',
    35: 'Middle-aged',
    45: 'Middle-aged',
    50: 'Elderly',
    56: 'Elderly'
}

# 将年龄转换为四大类
users_data['AgeGroup'] = users_data['Age'].map(age_groups)

# 定义职业类别
occupation_mapping = {
    0: 'Other',  # 不明确
    1: 'Professional_Technical',  # 老师
    2: 'Professional_Technical',  # 艺术家
    3: 'Service_Management',  # 行政人员
    4: 'Other',  # 大学生
    5: 'Service_Management',  # 服务人员
    6: 'Professional_Technical',  # 医生
    7: 'Service_Management',  # 管理者
    8: 'Other',  # 农民
    9: 'Other',  # 家庭主妇
    10: 'Other',  # 青少年
    11: 'Service_Management',  # 律师
    12: 'Professional_Technical',  # 程序员
    13: 'Other',  # 退休人员
    14: 'Service_Management',  # 销售
    15: 'Professional_Technical',  # 科学家
    16: 'Service_Management',  # 个体经营
    17: 'Professional_Technical',  # 工程师
    18: 'Professional_Technical',  # 技工
    19: 'Other',  # 失业者
    20: 'Professional_Technical'  # 作家
}

# 将职业转换为三大类
users_data['OccupationCategory'] = users_data['Occupation'].map(occupation_mapping)

from sklearn.preprocessing import OneHotEncoder

# 独热编码职业类别
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(users_data[['AgeGroup', 'OccupationCategory']])

# 获取独热编码后的列名
encoded_columns = encoder.get_feature_names_out(['AgeGroup', 'OccupationCategory'])

# 将独热编码结果转换为DataFrame，并指定索引
encoded_features = pd.DataFrame(encoded_features.toarray(), columns=encoded_columns, index=users_data.index)

# 合并用户数据和独热编码的职业类别
users_data = pd.concat([users_data, encoded_features], axis=1)

# 去掉不含用户信息的用户id和邮编，只保留数值型特征，即用户画像特征向量
user_portrait = users_data.drop(['AgeGroup', 'OccupationCategory', 'Age', 'Occupation', 'UserID', 'Zip-code'], axis=1)

# 物品数据------------------------------------------------------------------------------
# 1. 导入数据集
items_path = './Data/ml-1m/movies.dat'

items_data = pd.read_csv(items_path, sep='::', header=None, engine='python')

print("Original data head:")
print(items_data.head())

# 评分数据------------------------------------------------------------------------------
# 1. 导入数据集
file_path = './Data/ml-1m/reduced_ratings_2.dat'  # 手动更改路径

# 读取评分数据
ratings = pd.read_csv(file_path, sep='::', names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python')

# 2. 按用户分组并按时间戳排序
ratings.sort_values(['UserID', 'Timestamp'], inplace=True)

# 计算每个用户的评分记录数量
user_ratings_count = ratings['UserID'].value_counts()
print(user_ratings_count)
print('每个用户至少有20/10/4次评分')

# 初始化训练集和测试集
train = pd.DataFrame()
test = pd.DataFrame()

# 遍历每个用户，划分80%训练集和20%测试集
for user_id in ratings['UserID'].unique():
    user_ratings = ratings[ratings['UserID'] == user_id]
    # 计算80%的索引位置
    split_index = int(len(user_ratings) * 0.8)
    # 分割训练集和测试集
    train = pd.concat([train, user_ratings[:split_index]])
    test = pd.concat([test, user_ratings[split_index:]])

# 保存数据集
test.to_csv('test_2.csv', index=False)
train.to_csv('train_2.csv', index=False)

# 肘部法则确定簇数----------------------------------------------------------------------
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg

# 接收用户画像数据
data = np.array(user_portrait)
print(user_portrait.shape)

# 尝试不同的 K 值
K_range = range(2, 20)
SSE = []
for K in K_range:
    kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
    SSE.append(kmeans.inertia_)  # inertia_ 是簇内距离平方和

# 绘制肘部图
plt.plot(K_range, SSE, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.show()

plt.savefig('elbow_method.png')  # 保存图像文件
# 聚类效果不佳！！！！

# 层次聚类---------------------------------------------------------------------
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
# 接收用户画像数据
data = np.array(user_portrait)

# 计算汉明距离矩阵
def hamming_distance(x, y):
    return np.sum(x != y)

# 使用 pdist 计算每对样本之间的汉明距离
distance_matrix = pdist(data, metric=hamming_distance)

# 使用 linkage 函数进行层次聚类
Z = linkage(distance_matrix, method='average')  # method='average' 表示使用平均链接法

# 绘制树状图
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 设置后端为Agg

# 绘制树状图
plt.figure(figsize=(15, 10))

# 显示树状图x轴标签的代码（标签过于密集）
# selected_labels = [f"{i+1}" for i in range(len(data))]
# dendrogram(
#     Z,
#     labels = selected_labels,  # 显示标签
#     leaf_rotation=90,  # 旋转标签
#     leaf_font_size=5  # 调整标签字体大小
# )

dendrogram(
    Z,
    no_labels=True,  # 不显示x轴上的标签
)

plt.title("Dendrogram")
plt.xlabel("Users")
plt.ylabel("Hamming Distance")
plt.axhline(y=3, color='r', linestyle='--')  # 添加高度阈值线
plt.show()
plt.savefig("dendrogram.png")  # 保存为文件
plt.close()  # 关闭图形窗口

# 根据树状图选择合适的聚类数量
num_clusters = 3
clusters = fcluster(Z, num_clusters, criterion='maxclust')

# 计算每个聚类的代表特征向量
representative_vectors = []
for i in range(1, num_clusters + 1):
    cluster_data = data[clusters == i].astype(int)  # 将浮点数转换为整数
    if cluster_data.size > 0:  # 检查聚类是否包含数据点
        # 多数投票确定每个位置的值
        representative_vector = np.array([
            np.bincount(cluster_data[:, j], minlength=2).argmax()
            for j in range(data.shape[1])
        ])
        print(f"聚类 {i} 的代表特征向量：{representative_vector}")
        representative_vectors.append(representative_vector)  # 将代表向量添加到列表中
    else:
        print(f"聚类 {i} 中没有数据点")

# 创建一个新的数组来存储替换后的向量
data_replaced = np.array([representative_vectors[cluster - 1] for cluster in clusters]).astype(float)
# 保存
df = pd.DataFrame(data_replaced, columns=[f'Feature_{j+1}' for j in range(data.shape[1])])
df.to_csv('users_portrait.csv', index=False)
