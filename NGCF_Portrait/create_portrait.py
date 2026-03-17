import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
import os

# 获取当前工作目录
current_directory = os.getcwd()
print("当前工作目录是:", current_directory)

# 加载数据
users = pd.read_csv('./Data/ml-1m/users.dat', sep='::', engine='python',
                    names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
ratings = pd.read_csv('./Data/ml-1m/ratings.dat', sep='::', engine='python',
                      names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
movies = pd.read_csv('./Data/ml-1m/movies.dat', sep='::', engine='python', names=['MovieID', 'Title', 'Genres'])

# 映射性别和职业
users['Gender'] = users['Gender'].map({'M': 0, 'F': 1})
users['Occupation'] = users['Occupation'].astype('category').cat.codes

# 归一化年龄和职业
users['Age'] = (users['Age'] - users['Age'].min()) / (users['Age'].max() - users['Age'].min())
users['Occupation'] = (users['Occupation'] - users['Occupation'].min()) / (users['Occupation'].max() - users['Occupation'].min())

# 计算自然画像
user_natural_profile = users.set_index('UserID').drop(columns=['Zip-code'])

# 处理电影类型
genres = set('|'.join(movies['Genres']).split('|'))
for genre in genres:
    movies[genre] = movies['Genres'].apply(lambda x: int(genre in x))

# 计算兴趣画像
user_genre_prefs = ratings.merge(movies, on='MovieID').drop(columns=['Title', 'Genres'])  # 得到用户对每部电影的评分以及电影的类型信息。
user_interest_profile = user_genre_prefs.groupby('UserID')[list(genres)].mean()  # 计算每个用户对每种电影类型的评分均值

# 归一化兴趣画像
user_interest_profile = (user_interest_profile - user_interest_profile.min()) / (
            user_interest_profile.max() - user_interest_profile.min())

# 合并用户画像
user_profiles = user_natural_profile.join(user_interest_profile, how='left').fillna(0)

# 使用肘部法找最佳簇数
sse = []
k_values = range(1, 21)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(user_profiles)
    sse.append(kmeans.inertia_)

# 绘制肘部法曲线
matplotlib.use('Agg')  # 设置后端为Agg

plt.plot(k_values, sse, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method For Optimal k')
plt.show()
plt.savefig("elbow_of_k.png")  # 保存为文件
plt.close()  # 关闭图形窗口

# 选择最佳簇数（假设人为选择肘点处）
best_k = 3  # 可根据肘部法曲线调整

# 进行 KMeans 聚类
kmeans = KMeans(n_clusters=best_k, random_state=42)
user_profiles['Cluster'] = kmeans.fit_predict(user_profiles)

# 用簇中心替换用户画像
cluster_centers = kmeans.cluster_centers_
user_profiles_clustered = np.array([cluster_centers[label] for label in user_profiles['Cluster']])

# 输出结果
print(user_profiles_clustered[:5])

# 导出用户画像聚类结果到CSV
user_profiles_clustered_df = pd.DataFrame(user_profiles_clustered)
user_profiles_clustered_df.to_csv('users_portrait_2.csv', index=False, header=False)
