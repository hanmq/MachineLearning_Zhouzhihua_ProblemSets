import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class LDA(object):

    def fit(self, X_, y_, plot_=False):
        pos = y_ == 1
        neg = y_ == 0
        X0 = X_[neg]
        X1 = X_[pos]

        u0 = X0.mean(0, keepdims=True)  # (1, n)
        u1 = X1.mean(0, keepdims=True)

        sw = np.dot((X0 - u0).T, X0 - u0) + np.dot((X1 - u1).T, X1 - u1)
        w = np.dot(np.linalg.inv(sw), (u0 - u1).T).reshape(1, -1)  # (1, n)

        if plot_:
            fig, ax = plt.subplots()
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.spines['left'].set_position(('data', 0))
            ax.spines['bottom'].set_position(('data', 0))

            plt.scatter(X1[:, 0], X1[:, 1], c='k', marker='o', label='good')
            plt.scatter(X0[:, 0], X0[:, 1], c='r', marker='x', label='bad')

            plt.xlabel('密度', labelpad=1)
            plt.ylabel('含糖量')
            plt.legend(loc='upper right')

            x_tmp = np.linspace(-0.05, 0.15)
            y_tmp = x_tmp * w[0, 1] / w[0, 0]
            plt.plot(x_tmp, y_tmp, '#808080', linewidth=1)

            wu = w / np.linalg.norm(w)

            # 正负样板店
            X0_project = np.dot(X0, np.dot(wu.T, wu))
            plt.scatter(X0_project[:, 0], X0_project[:, 1], c='r', s=15)
            for i in range(X0.shape[0]):
                plt.plot([X0[i, 0], X0_project[i, 0]], [X0[i, 1], X0_project[i, 1]], '--r', linewidth=1)

            X1_project = np.dot(X1, np.dot(wu.T, wu))
            plt.scatter(X1_project[:, 0], X1_project[:, 1], c='k', s=15)
            for i in range(X1.shape[0]):
                plt.plot([X1[i, 0], X1_project[i, 0]], [X1[i, 1], X1_project[i, 1]], '--k', linewidth=1)

            # 中心点的投影
            u0_project = np.dot(u0, np.dot(wu.T, wu))
            plt.scatter(u0_project[:, 0], u0_project[:, 1], c='#FF4500', s=60)
            u1_project = np.dot(u1, np.dot(wu.T, wu))
            plt.scatter(u1_project[:, 0], u1_project[:, 1], c='#696969', s=60)

            ax.annotate(r'u0 投影点',
                        xy=(u0_project[:, 0], u0_project[:, 1]),
                        xytext=(u0_project[:, 0] - 0.2, u0_project[:, 1] - 0.1),
                        size=13,
                        va="center", ha="left",
                        arrowprops=dict(arrowstyle="->",
                                        color="k",
                                        )
                        )

            ax.annotate(r'u1 投影点',
                        xy=(u1_project[:, 0], u1_project[:, 1]),
                        xytext=(u1_project[:, 0] - 0.1, u1_project[:, 1] + 0.1),
                        size=13,
                        va="center", ha="left",
                        arrowprops=dict(arrowstyle="->",
                                        color="k",
                                        )
                        )
            plt.axis("equal")  # 两坐标轴的单位刻度长度保存一致
            plt.show()

        self.w = w
        self.u0 = u0
        self.u1 = u1
        return self

    def predict(self, X):
        project = np.dot(X, self.w.T)

        wu0 = np.dot(self.w, self.u0.T)
        wu1 = np.dot(self.w, self.u1.T)

        return (np.abs(project - wu1) < np.abs(project - wu0)).astype(int)


if __name__ == '__main__':
    data_path = r'C:\Users\hanmi\Documents\xiguabook\watermelon3_0_Ch.csv'

    data = pd.read_csv(data_path).values

    X = data[:, 7:9].astype(float)
    y = data[:, 9]

    y[y == '是'] = 1
    y[y == '否'] = 0
    y = y.astype(int)

    lda = LDA()
    lda.fit(X, y, plot_=True)
    print(lda.predict(X))  # 和逻辑回归的结果一致
    print(y)

