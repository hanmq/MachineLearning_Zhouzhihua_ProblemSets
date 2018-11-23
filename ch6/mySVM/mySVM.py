import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


def set_ax_gray(ax):
    ax.patch.set_facecolor("gray")
    ax.patch.set_alpha(0.1)
    ax.spines['right'].set_color('none')  # 设置隐藏坐标轴
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.grid(axis='y', linestyle='-.')


def plt_support_(clf, X_, y_, kernel, c):
    pos = y_ == 1
    neg = y_ == -1
    ax = plt.subplot()

    x_tmp = np.linspace(0, 1, 600)
    y_tmp = np.linspace(0, 0.8, 600)

    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)

    Z_rbf = clf.predict(np.c_[X_tmp.ravel(), Y_tmp.ravel()]).reshape(X_tmp.shape)

    # ax.contourf(X_, Y_, Z_rbf, alpha=0.75)
    cs = ax.contour(X_tmp, Y_tmp, Z_rbf, [0], colors='orange', linewidths=1)
    ax.clabel(cs, fmt={cs.levels[0]: 'decision boundary'})

    set_ax_gray(ax)

    ax.scatter(X_[pos, 0], X_[pos, 1], label='1', color='c')
    ax.scatter(X_[neg, 0], X_[neg, 1], label='0', color='lightcoral')

    ax.scatter(X_[clf.support_, 0], X_[clf.support_, 1], marker='o', c='', edgecolors='g', s=150,
               label='support_vectors')

    ax.legend()
    ax.set_title('{} kernel, C={}'.format(kernel, c))
    plt.show()


class mySVM(object):

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', tol=1e-3, max_iter=-1, seed=16):
        self.C = C
        assert kernel in ('linear', 'poly', 'rbf')
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.seed = seed
        self._gamma = None

        self.alphas = None
        self._nonzero_alpha = None
        self._e_cache = None
        self._non_bound_alpha = None
        self.b = None

        self.support_ = None
        self.support_vector_ = None
        self.support_y_ = None
        self.support_alpha_ = None

    def predict(self, X):
        if self.support_vector_ is None:
            raise Exception('you have to fit first before predict')

        m = X.shape[0]
        output = np.ones((m,))
        for i in range(m):

            kX_i = self._kX_i(self.support_vector_, X[i])
            gx = np.dot((self.support_alpha_.reshape(-1, 1) * self.support_y_).T, kX_i) + self.b

            if gx < 0:
                output[i] = -1

        return output

    def fit(self, X, y):

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if self.gamma == 'auto':
            self._gamma = 1.0 / X.shape[1]
        else:
            self._gamma = self.gamma

        self.alphas = np.zeros((X.shape[0],))
        self.b = 0
        self._nonzero_alpha = np.zeros((X.shape[0],), dtype=bool)  # 用于记录非零值的alphas的bool数组，
        self._non_bound_alpha = np.zeros((X.shape[0],))  # 用于记录 (0, C)之间的alphas(即间隔边界点), 1表示当前位置的alphas为non_bound
        self._e_cache = np.zeros((X.shape[0],))

        self._smo(X, y)

        self.support_ = self._nonzero_alpha.nonzero()[0]
        self.support_vector_ = X[self.support_]
        self.support_y_ = y[self.support_]
        self.support_alpha_ = self.alphas[self._nonzero_alpha]

        return self

    def _smo(self, X, y):
        '''
        第一层循环选择第一个变量，smo选择第一个变量的逻辑是：
        1：首先遍历所有 0 < ai < C 的样本点，若不满足KKT条件则选择第二变量，进行着两个变量的调优。
        2：直到所有 0 < ai < C 的样本点都满足KKT条件，那么而后遍历所有样本点，对不满足KKT条件的ai进行调优。
        3：若2步骤有对找到可以进行调优的ai，那么回到1。否则返回所有ai。
        '''
        iter = 0  # 记录迭代次数
        entire_flag = True  # 表示是否需要遍历全部样本的flag。因将所有ai初始化为0，没有0 < ai < C的样本，所以第一次必定遍历全部样本。
        m = X.shape[0]
        while (self.max_iter == -1) or (iter < self.max_iter):  # 若参max_iter == -1, 则直到所有样本都符合kkt条件才停止。
            # out loop 寻找第一个alpha
            alpha_pairs_changed = 0

            if entire_flag:
                for i in range(m):
                    alpha_pairs_changed += self._inner_loop(X, y, i)

            else:
                non_bound_index = self._non_bound_alpha.nonzero()[0]
                for i in non_bound_index:
                    alpha_pairs_changed += self._inner_loop(X, y, i)
            iter += 1

            if entire_flag:  # 若这次遍历所有样本，则下一次将遍历 0 < ai < C 的样本点
                if alpha_pairs_changed == 0:  # 若遍历所有样本都没有更新ai(等价于所有样本都满足KKT条件)，则直接停止更新。
                    return self
                entire_flag = False

            elif alpha_pairs_changed == 0:  # 若遍历 0 < ai < C的所有样本，都没有更新ai，则下一次遍历所有样本
                entire_flag = True

        return self

    def _inner_loop(self, X, y, i):
        '''
        inner loop 寻找第二个alpha
        若更新成功返回1，没有更新返回0
        '''
        if self._non_bound_alpha[i]:
            Ei = self._e_cache[i]
        else:
            Ei = self._clac_Ei(X, y, i)
        m = X.shape[0]
        yi = y[i, :]
        ai = self.alphas[i]

        if ((yi * Ei < -self.tol) and (ai < self.C)) or ((yi * Ei > self.tol) and (ai > 0)):  # 该alpha 违背kkt条件
            best_j = None
            if np.sum(self._non_bound_alpha) > 0:  # 存在non_bound的alpha 则从中选择一个是的|Ei - Ej|最大的一个j
                j, Ej = self._select_second_alpha(Ei)
                if self._update_alpha(X, y, i, j, Ei, Ej):  # 若成功更新了alpha 则直接返回1
                    return 1
                best_j = j

            non_bound_index = self._non_bound_alpha.nonzero()[0]  # 在上面方法更新失败时，则遍历所有non_bound样本，

            # 这里原论文中描述：为了避免结果倾向于第一个样本，需要从一个随机位置开始遍历。不过这里使用打乱顺序的方式
            for j in np.random.permutation(non_bound_index):
                if j == best_j:  # 既然上面的方法失败了，就直接跳过这个alpha
                    continue
                Ej = self._e_cache[j]
                if self._update_alpha(X, y, i, j, Ei, Ej):
                    return 1

            # 如果还找不到合适的aj就遍历所有alpha
            for j in np.random.permutation(m):  # 原文同样是使用从一个随机位置开始遍历。这里也同样使用打乱顺序的方式
                if j in non_bound_index:  # 忽略上面已经失败的alpha
                    continue
                Ej = self._clac_Ei(X, y, j)
                if self._update_alpha(X, y, i, j, Ei, Ej):
                    return 1

        return 0

    def _update_alpha(self, X, y, i, j, Ei, Ej=None):
        if i == j:
            return 0
        alpha_i_old = self.alphas[i].copy()
        alpha_j_old = self.alphas[j].copy()

        yi = y[i, 0]
        yj = y[j, 0]

        if yi != yj:
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
        else:
            L = max(0, alpha_j_old + alpha_i_old - self.C)
            H = min(self.C, alpha_j_old + alpha_i_old)
        if L == H:
            return 0
        kii = self._kij(X, i, i)
        kjj = self._kij(X, j, j)
        kij = self._kij(X, i, j)

        eta = kii + kjj - 2 * kij  # 参考《统计学习方法》p127 式7.107

        if eta <= 0:  # eta = ||\phi{xi} - \phi{xj}||^{2}, 有： eta>=0, 这里eta<=0判断条件，实际上等价于eta==0
            # 在eta == 0时, alpha 的更新公式就失效了，其实在原论文中，这种情况是有专门的对应方式的。
            # 不过因为这种情况极少发生，这里参考《机器学习实战》中代码，就直接设置为更新失败，返回0了
            return 0

        alpha_j_new = alpha_j_old + yj * (Ei - Ej) / eta
        alpha_j_new = self._clip_alpha(alpha_j_new, H, L)

        if abs(alpha_j_new - alpha_j_old) < 0.00001:  # 没有足够的更新，就直接更新失败。
            return 0

        alpha_i_new = alpha_i_old + yi * yj * (alpha_j_old - alpha_j_new)

        bj = self.b - Ej - yj * kjj * (alpha_j_new - alpha_j_old) - yi * kij * (alpha_i_new - alpha_i_old)
        bi = self.b - Ei - yi * kij * (alpha_j_new - alpha_j_old) - yi * kii * (alpha_i_new - alpha_i_old)

        # 更新alpha
        if 0 < alpha_j_new < self.C:
            self.b = bj
        elif 0 < alpha_i_new < self.C:
            self.b = bi
        else:
            self.b = (bi + bj) / 2

        self.alphas[i] = alpha_i_new
        self.alphas[j] = alpha_j_new

        self._update_alpha_adjoint_arr(alpha_i_new, i)
        self._update_alpha_adjoint_arr(alpha_j_new, j)

        self._update_e_cache(X, y)
        return 1

    def _update_alpha_adjoint_arr(self, alpha_new, i):
        '''
        更新 self._non_bound_alpha 和 self._nonzero_alpha
        '''
        self._nonzero_alpha[i] = (alpha_new > 0)
        self._non_bound_alpha[i] = int(0 < alpha_new < self.C)

    def _update_e_cache(self, X, y):
        '''更新non_bound对应的误差缓存'''

        for i in self._non_bound_alpha.nonzero()[0]:
            self._e_cache[i] = self._clac_Ei(X, y, i)

    def _select_second_alpha(self, Ei):
        '''
        在从non_bound alpha中选择最佳的alpha时，在原论文和《统计学习方法》中都是这样描述的：
        选择alpha_j 使得|Ei - Ej|最大，由于Ei已经定了，所以若Ei > 0, 则选择最小的E作为Ej; 若Ei < 0, 则选择最大的E 作为Ej。
        个人感觉这里逻辑其实不对, Ei > 0 时选择最小的E作为Ej并不一定使得|Ei - Ej|最大,
        比如Ei = 1, min(E) = -1, max(E) = 10, 这时显然选择max(E)作为Ej更合理. Ei < 0 时同理.

        所以这里是直接计算最大的|Ei - Ej|
        '''
        non_bound_index = self._non_bound_alpha.nonzero()[0]
        delta_E = np.abs(Ei - self._e_cache[non_bound_index])
        j = non_bound_index[np.argmax(delta_E)]
        Ej = self._e_cache[j]

        return j, Ej

    def _clac_Ei(self, X, y, i):
        '''参考《统计学习方法中》p127, 式7.104 - 7.105'''

        xi = X[[i], :]
        yi = y[i, 0]
        if self._nonzero_alpha.any():
            valid_X = X[self._nonzero_alpha]  # 这里仅使用非零值的
            valid_y = y[self._nonzero_alpha]
            valid_alphas = self.alphas[self._nonzero_alpha]
            kX_i = self._kX_i(valid_X, xi)
            gx = np.dot((valid_alphas.reshape(-1, 1) * valid_y).T, kX_i) + self.b

        else:  # 当所有alpha
            gx = self.b

        Ei = gx - yi
        return np.squeeze(Ei)

    def _kX_i(self, X, xi):
        '''

        :param X:  (n_non_bound, m)
        :param xi: (1, m)
        :return:
        '''

        if self.kernel == 'linear':
            kX_i = np.dot(X, xi.T)
        elif self.kernel == 'poly':
            kX_i = np.power(np.dot(X, xi.T), self.degree)
        else:
            kX_i = np.exp(-self._gamma * np.sum(np.power(X - xi, 2), axis=1))

        return kX_i

    def _kij(self, X, i, j):
        '''

        :param X:
        :param i:
        :param j:
        :return:    K(i, j)
        '''
        xi = X[[i], :]
        xj = X[[j], :]

        if self.kernel == 'linear':
            kX_i = np.dot(xj, xi.T)
        elif self.kernel == 'poly':
            kX_i = np.power(np.dot(xj, xi.T), self.degree)
        else:
            kX_i = np.exp(-self._gamma * np.sum(np.power(xj - xi, 2), axis=1))

        return np.squeeze(kX_i)

    def _clip_alpha(self, aj, H, L):
        if aj > H:
            aj = H
        if aj < L:
            aj = L
        return aj


if __name__ == '__main__':
    clf = mySVM(C=10000)

    path = r'C:\Users\hanmi\Documents\xiguabook\watermelon3_0a_Ch.txt'
    data = pd.read_table(path, delimiter=' ', dtype=float)

    X = data.iloc[:, [0, 1]].values
    y = data.iloc[:, 2].values

    y[y == 0] = -1

    start = time.clock()
    clf.fit(X, y)

    plt_support_(clf, X, y, 'rbf', 10000)
