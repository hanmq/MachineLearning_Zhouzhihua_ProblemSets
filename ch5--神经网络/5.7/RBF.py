'''
这里使用均方根误差作为损失函数的RBF神经网络。
'''
import numpy as np
import matplotlib.pyplot as plt


def RBF_forward(X_, parameters_):
    m, n = X_.shape
    beta = parameters_['beta']
    W = parameters_['W']
    c = parameters_['c']
    b = parameters_['b']

    t_ = c.shape[0]
    p = np.zeros((m, t_))  # 中间隐藏层的激活值     对应书上5.19式
    x_c = np.zeros((m, t_))  # 5.19式中 x - c_{i}
    for i in range(t_):
        x_c[:, i] = np.linalg.norm(X_ - c[[i],], axis=1) ** 2

        p[:, i] = np.exp(-beta[0, i] * x_c[:, i])

    a = np.dot(p, W.T) + b
    return a, p, x_c


def RBF_backward(a_, y_, x_c, p_, parameters_):
    m, n = a_.shape
    grad = {}
    beta = parameters_['beta']
    W = parameters_['W']

    da = (a_ - y_)      # 损失函数对输出层的偏导 ，这里的a其实对应着  输出层的y_hat

    dw = np.dot(da.T, p_) / m
    db = np.sum(da, axis=0, keepdims=True) / m
    dp = np.dot(da, W)   # dp即损失函数对隐藏层激活值的偏导

    dbeta = np.sum(dp * p_ * (-x_c), axis=0, keepdims=True) / m

    assert dbeta.shape == beta.shape
    assert dw.shape == W.shape
    grad['dw'] = dw
    grad['dbeta'] = dbeta
    grad['db'] = db

    return grad


def compute_cost(y_hat_, y_):
    m = y_.shape[0]
    loss = np.sum((y_hat_ - y) ** 2) / (2 * m)
    return np.squeeze(loss)


def RBF_model(X_, y_, learning_rate, num_epochs, t):
    '''

    :param X_:
    :param y_:
    :param learning_rate:  学习率
    :param num_epochs:     迭代次数
    :param t:   隐藏层节点数量
    :return:
    '''
    parameters = {}
    np.random.seed(16)
    # 定义中心点，本来这里的中心点应该由随机采用或者聚类等非监督学习来获得的，这里为了简单就直接定义好了

    parameters['beta'] = np.random.randn(1, t)  # 初始化径向基的方差
    parameters['W'] = np.zeros((1, t))  # 初始化
    parameters['c'] = np.random.rand(t, 2)
    parameters['b'] = np.zeros([1, 1])
    costs = []

    for i in range(num_epochs):
        a, p, x_c = RBF_forward(X_, parameters)
        cost = compute_cost(a, y_)
        costs.append(cost)
        grad = RBF_backward(a, y_, x_c, p, parameters)

        parameters['beta'] -= learning_rate * grad['dbeta']
        parameters['W'] -= learning_rate * grad['dw']
        parameters['b'] -= learning_rate * grad['db']

    return parameters, costs


def predict(X_, parameters_):
    a, p, x_c = RBF_forward(X_, parameters_)

    return a


X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
y = np.array([[1], [1], [0], [0]])
#

parameters, costs = RBF_model(X, y, 0.003, 10000, 8)

plt.plot(costs)
plt.show()

print(predict(X, parameters))

# 梯度检验
# parameters = {}
# parameters['beta'] = np.random.randn(1, 2)  # 初始化径向基的方差
# parameters['W'] = np.random.randn(1, 2)  # 初始化
# parameters['c'] = np.array([[0.1, 0.1], [0.8, 0.8]])
# parameters['b'] = np.zeros([1, 1])
# a, p, x_c = RBF_forward(X, parameters)
#
# cost = compute_cost(a, y)
# grad = RBF_backward(a, y, x_c, p, parameters)
#
#
# parameters['b'][0, 0] += 1e-6
#
# a1, p1, x_c1 = RBF_forward(X, parameters)
# cost1 = compute_cost(a1, y)
# print(grad['db'])
#
# print((cost1 - cost) / 1e-6)
