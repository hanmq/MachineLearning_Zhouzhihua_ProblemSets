from sklearn import svm, discriminant_analysis
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris['data']

y = iris['target']

# 取出 0，1类作为训练样本
target = pd.Series(y).isin([0, 1])

# lda
lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=1)

lda.fit(X[:, [0, 2]][target], y[target])

fig, ax = plt.subplots()

ax.scatter(X[y == 0, 0], X[y == 0, 2])
ax.scatter(X[y == 1, 0], X[y == 1, 2])

x_tmp = np.linspace(4, 7.2, 2040)
y_tmp = np.linspace(0, 5.2, 2040)
X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)

Z_lda = lda.predict(np.c_[X_tmp.ravel(), Y_tmp.ravel()]).reshape(X_tmp.shape)

C_lda = ax.contour(X_tmp, Y_tmp, Z_lda, [0.5], colors='orange', linewidths=1)
plt.clabel(C_lda, fmt={C_lda.levels[0]: 'lda decision boundary'}, inline=True, fontsize=8)

y[y == 0] = -1

linear_svm = svm.SVC(kernel='linear', C=10000)
linear_svm.fit(X[:, [0, 2]][target], y[target])

Z_svm = linear_svm.predict(np.c_[X_tmp.ravel(), Y_tmp.ravel()]).reshape(X_tmp.shape)

C_svm = ax.contour(X_tmp, Y_tmp, Z_svm, [0], colors='g', linewidths=1)

ax.scatter(linear_svm.support_vectors_[:, 0], linear_svm.support_vectors_[:, 1], marker='o', c='', edgecolors='g',
           s=150)

plt.clabel(C_svm, fmt={C_svm.levels[0]: 'svm decision boundary'}, inline=True, fontsize=8, manual=True)
plt.show()
