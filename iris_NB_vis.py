import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 降维到2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 定义分类器
clf = GaussianNB()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, random_state=0)
clf.fit(X_train, y_train)

# 计算分类准确度
accuracy = clf.score(X_test, y_test)

# 根据数据范围创建网格
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 绘制决策边界
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']), edgecolor='k', s=20)
plt.title(f"GaussianNB Classification Boundaries (Accuracy: {accuracy:.2f})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()
