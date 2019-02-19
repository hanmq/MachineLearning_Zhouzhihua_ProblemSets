import numpy as np
from matplotlib import pyplot as plt


def xavier_initializer(layer_dims_, seed=16):
    np.random.seed(seed)

    parameters_ = {}
    num_L = len(layer_dims_)
    for l in range(num_L - 1):
        temp_w = np.random.randn(layer_dims_[l + 1], layer_dims_[l]) * np.sqrt(1 / layer_dims_[l])
        temp_b = np.zeros((1, layer_dims_[l + 1]))

        parameters_['W' + str(l + 1)] = temp_w
        parameters_['b' + str(l + 1)] = temp_b

    return parameters_


def he_initializer(layer_dims_, seed=16):
    np.random.seed(seed)

    parameters_ = {}
    num_L = len(layer_dims_)
    for l in range(num_L - 1):
        temp_w = np.random.randn(layer_dims_[l + 1], layer_dims_[l]) * np.sqrt(2 / layer_dims_[l])
        temp_b = np.zeros((1, layer_dims_[l + 1]))

        parameters_['W' + str(l + 1)] = temp_w
        parameters_['b' + str(l + 1)] = temp_b

    return parameters_


def cross_entry_sigmoid(y_hat_, y_):
    '''
    计算在二分类时的交叉熵
    :param y_hat_:  模型输出值
    :param y_:      样本真实标签值
    :return:
    '''

    m = y_.shape[0]
    loss = -(np.dot(y_.T, np.log(y_hat_)) + np.dot(1 - y_.T, np.log(1 - y_hat_))) / m

    return np.squeeze(loss)


def cross_entry_softmax(y_hat_, y_):
    '''
    计算多分类时的交叉熵
    :param y_hat_:
    :param y_:
    :return:
    '''
    m = y_.shape[0]
    loss = -np.sum(y_ * np.log(y_hat_)) / m
    return loss


def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a


def relu(z):
    a = np.maximum(0, z)
    return a


def softmax(z):
    z -= np.max(z)  # 防止过大，超出限制，导致计算结果为 nan
    z_exp = np.exp(z)
    softmax_z = z_exp / np.sum(z_exp, axis=1, keepdims=True)
    return softmax_z


def sigmoid_backward(da_, cache_z):
    a = 1 / (1 + np.exp(-cache_z))
    dz_ = da_ * a * (1 - a)
    assert dz_.shape == cache_z.shape
    return dz_


def softmax_backward(y_, cache_z):
    #
    a = softmax(cache_z)
    dz_ = a - y_
    assert dz_.shape == cache_z.shape
    return dz_


def relu_backward(da_, cache_z):
    dz = np.array(da_, copy=True)
    dz[cache_z <= 0] = 0
    assert (dz.shape == cache_z.shape)

    return dz


def update_parameters_with_gd(parameters_, grads, learning_rate):
    L_ = int(len(parameters_) / 2)

    for l in range(1, L_ + 1):
        parameters_['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters_['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    return parameters_


def update_parameters_with_sgd(parameters_, grads, learning_rate):
    L_ = int(len(parameters_) / 2)

    for l in range(1, L_ + 1):
        parameters_['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters_['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    return parameters_


def initialize_velcoity(paramters):
    v = {}

    L_ = int(len(paramters) / 2)

    for l in range(1, L_ + 1):
        v['dW' + str(l)] = np.zeros(paramters['W' + str(l)].shape)
        v['db' + str(l)] = np.zeros(paramters['b' + str(l)].shape)

    return v


def update_parameters_with_sgd_momentum(parameters, grads, velcoity, beta, learning_rate):
    L_ = int(len(parameters) / 2)

    for l in range(1, L_ + 1):
        velcoity['dW' + str(l)] = beta * velcoity['dW' + str(l)] + (1 - beta) * grads['dW' + str(l)]
        velcoity['db' + str(l)] = beta * velcoity['db' + str(l)] + (1 - beta) * grads['db' + str(l)]

        parameters['W' + str(l)] -= learning_rate * velcoity['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * velcoity['db' + str(l)]

    return parameters, velcoity


def initialize_adam(paramters_):
    l = int(len(paramters_) / 2)
    square_grad = {}
    velcoity = {}
    for i in range(l):

        for i in range(l):
            square_grad['dW' + str(i + 1)] = np.zeros(paramters_['W' + str(i + 1)].shape)
            square_grad['db' + str(i + 1)] = np.zeros(paramters_['b' + str(i + 1)].shape)
            velcoity['dW' + str(i + 1)] = np.zeros(paramters_['W' + str(i + 1)].shape)
            velcoity['db' + str(i + 1)] = np.zeros(paramters_['b' + str(i + 1)].shape)
        return velcoity, square_grad


def update_parameters_with_sgd_adam(parameters_, grads_, velcoity, square_grad, epoch, learning_rate=0.1, beta1=0.9,
                                    beta2=0.999, epsilon=1e-8):
    l = int(len(parameters_) / 2)

    for i in range(l):
        velcoity['dW' + str(i + 1)] = beta1 * velcoity['dW' + str(i + 1)] + (1 - beta1) * grads_['dW' + str(i + 1)]
        velcoity['db' + str(i + 1)] = beta1 * velcoity['db' + str(i + 1)] + (1 - beta1) * grads_['db' + str(i + 1)]

        vw_correct = velcoity['dW' + str(i + 1)] / (1 - np.power(beta1, epoch))         # 这里是对迭代初期的梯度进行修正
        vb_correct = velcoity['db' + str(i + 1)] / (1 - np.power(beta1, epoch))

        square_grad['dW' + str(i + 1)] = beta2 * square_grad['dW' + str(i + 1)] + (1 - beta2) * (
                    grads_['dW' + str(i + 1)] ** 2)
        square_grad['db' + str(i + 1)] = beta2 * square_grad['db' + str(i + 1)] + (1 - beta2) * (
                    grads_['db' + str(i + 1)] ** 2)

        sw_correct = square_grad['dW' + str(i + 1)] / (1 - np.power(beta2, epoch))
        sb_correct = square_grad['db' + str(i + 1)] / (1 - np.power(beta2, epoch))

        parameters_['W' + str(i + 1)] -= learning_rate * vw_correct / np.sqrt(sw_correct + epsilon)
        parameters_['b' + str(i + 1)] -= learning_rate * vb_correct / np.sqrt(sb_correct + epsilon)

    return parameters_, velcoity, square_grad


def set_ax_gray(ax):
    ax.patch.set_facecolor("gray")
    ax.patch.set_alpha(0.1)
    ax.spines['right'].set_color('none')  # 设置隐藏坐标轴
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.grid(axis='y', linestyle='-.')


def plot_costs(costs, labels, colors=None):
    if colors is None:
        colors = ['C', 'lightcoral']

    ax = plt.subplot()
    assert len(costs) == len(labels)
    for i in range(len(costs)):
        ax.plot(costs[i], color=colors[i], label=labels[i])
    set_ax_gray(ax)
    ax.legend(loc='upper right')
    ax.set_xlabel('num epochs')
    ax.set_ylabel('cost')
    plt.show()
