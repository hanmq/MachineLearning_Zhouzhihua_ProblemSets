'''
拟合函数中，X支持pd.DataFrame数据类型；y暂只支持pd.Series类型，其他数据类型未测试，
目前在西瓜数据集上和sklearn中自带的iris数据集上运行正常，以后若发现有其他bug，再修复。
'''

import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
import treePlottter
import pruning


class Node(object):
    def __init__(self):
        self.feature_name = None
        self.feature_index = None
        self.subtree = {}
        self.impurity = None
        self.is_continuous = False
        self.split_value = None
        self.is_leaf = False
        self.leaf_class = None
        self.leaf_num = None
        self.high = -1


class DecisionTree(object):
    '''
    没有针对缺失值的情况作处理。
    '''

    def __init__(self, criterion='gini', pruning=None):
        '''

        :param criterion: 划分方法选择，'gini', 'infogain', 'gainratio', 三种选项。
        :param pruning:   是否剪枝。 'pre_pruning' 'post_pruning'
        '''
        assert criterion in ('gini', 'infogain', 'gainratio')
        assert pruning in (None, 'pre_pruning', 'post_pruning')
        self.criterion = criterion
        self.pruning = pruning

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        '''
        生成决策树
        -------
        :param X:  只支持DataFrame类型数据，因为DataFrame中已有列名，省去一个列名的参数。不支持np.array等其他数据类型
        :param y:
        :return:
        '''

        if self.pruning is not None and (X_val is None or y_val is None):
            raise Exception('you must input X_val and y_val if you are going to pruning')

        X_train.reset_index(inplace=True, drop=True)
        y_train.reset_index(inplace=True, drop=True)

        if X_val is not None:
            X_val.reset_index(inplace=True, drop=True)
            y_val.reset_index(inplace=True, drop=True)

        self.columns = list(X_train.columns)  # 包括原数据的列名
        self.tree_ = self.generate_tree(X_train, y_train)

        if self.pruning == 'pre_pruning':
            pruning.pre_pruning(X_train, y_train, X_val, y_val, self.tree_)
        elif self.pruning == 'post_pruning':
            pruning.post_pruning(X_train, y_train, X_val, y_val, self.tree_)

        return self

    def generate_tree(self, X, y):
        my_tree = Node()
        my_tree.leaf_num = 0
        if y.nunique() == 1:  # 属于同一类别
            my_tree.is_leaf = True
            my_tree.leaf_class = y.values[0]
            my_tree.high = 0
            my_tree.leaf_num += 1
            return my_tree

        if X.empty:  # 特征用完了，数据为空，返回样本数最多的类
            my_tree.is_leaf = True
            my_tree.leaf_class = pd.value_counts(y).index[0]
            my_tree.high = 0
            my_tree.leaf_num += 1
            return my_tree

        best_feature_name, best_impurity = self.choose_best_feature_to_split(X, y)

        my_tree.feature_name = best_feature_name
        my_tree.impurity = best_impurity[0]
        my_tree.feature_index = self.columns.index(best_feature_name)

        feature_values = X.loc[:, best_feature_name]

        if len(best_impurity) == 1:  # 离散值
            my_tree.is_continuous = False

            unique_vals = pd.unique(feature_values)
            sub_X = X.drop(best_feature_name, axis=1)

            max_high = -1
            for value in unique_vals:
                my_tree.subtree[value] = self.generate_tree(sub_X[feature_values == value], y[feature_values == value])
                if my_tree.subtree[value].high > max_high:  # 记录子树下最高的高度
                    max_high = my_tree.subtree[value].high
                my_tree.leaf_num += my_tree.subtree[value].leaf_num

            my_tree.high = max_high + 1

        elif len(best_impurity) == 2:  # 连续值
            my_tree.is_continuous = True
            my_tree.split_value = best_impurity[1]
            up_part = '>= {:.3f}'.format(my_tree.split_value)
            down_part = '< {:.3f}'.format(my_tree.split_value)

            my_tree.subtree[up_part] = self.generate_tree(X[feature_values >= my_tree.split_value],
                                                          y[feature_values >= my_tree.split_value])
            my_tree.subtree[down_part] = self.generate_tree(X[feature_values < my_tree.split_value],
                                                            y[feature_values < my_tree.split_value])

            my_tree.leaf_num += (my_tree.subtree[up_part].leaf_num + my_tree.subtree[down_part].leaf_num)

            my_tree.high = max(my_tree.subtree[up_part].high, my_tree.subtree[down_part].high) + 1

        return my_tree

    def predict(self, X):
        '''
        同样只支持 pd.DataFrame类型数据
        :param X:  pd.DataFrame 类型
        :return:   若
        '''
        if not hasattr(self, "tree_"):
            raise Exception('you have to fit first before predict.')
        if X.ndim == 1:
            return self.predict_single(X)
        else:
            return X.apply(self.predict_single, axis=1)

    def predict_single(self, x, subtree=None):
        '''
        预测单一样本。 实际上这里也可以写成循环，写成递归样本大的时候有栈溢出的风险。
        :param x:
        :param subtree: 根据特征，往下递进的子树。
        :return:
        '''
        if subtree is None:
            subtree = self.tree_

        if subtree.is_leaf:
            return subtree.leaf_class

        if subtree.is_continuous:  # 若是连续值，需要判断是
            if x[subtree.feature_index] >= subtree.split_value:
                return self.predict_single(x, subtree.subtree['>= {:.3f}'.format(subtree.split_value)])
            else:
                return self.predict_single(x, subtree.subtree['< {:.3f}'.format(subtree.split_value)])
        else:
            return self.predict_single(x, subtree.subtree[x[subtree.feature_index]])

    def choose_best_feature_to_split(self, X, y):
        assert self.criterion in ('gini', 'infogain', 'gainratio')

        if self.criterion == 'gini':
            return self.choose_best_feature_gini(X, y)
        elif self.criterion == 'infogain':
            return self.choose_best_feature_infogain(X, y)
        elif self.criterion == 'gainratio':
            return self.choose_best_feature_gainratio(X, y)

    def choose_best_feature_gini(self, X, y):
        features = X.columns
        best_feature_name = None
        best_gini = [float('inf')]
        for feature_name in features:
            is_continuous = type_of_target(X[feature_name]) == 'continuous'
            gini_idex = self.gini_index(X[feature_name], y, is_continuous)
            if gini_idex[0] < best_gini[0]:
                best_feature_name = feature_name
                best_gini = gini_idex

        return best_feature_name, best_gini

    def choose_best_feature_infogain(self, X, y):
        '''
        以返回值中best_info_gain 的长度来判断当前特征是否为连续值，若长度为 1 则为离散值，若长度为 2 ， 则为连续值
        :param X: 当前所有特征的数据 pd.DaraFrame格式
        :param y: 标签值
        :return:  以信息增益来选择的最佳划分属性，第一个返回值为属性名称，

        '''
        features = X.columns
        best_feature_name = None
        best_info_gain = [float('-inf')]
        entD = self.entroy(y)
        for feature_name in features:
            is_continuous = type_of_target(X[feature_name]) == 'continuous'
            info_gain = self.info_gain(X[feature_name], y, entD, is_continuous)
            if info_gain[0] > best_info_gain[0]:
                best_feature_name = feature_name
                best_info_gain = info_gain

        return best_feature_name, best_info_gain

    def choose_best_feature_gainratio(self, X, y):
        '''
        以返回值中best_gain_ratio 的长度来判断当前特征是否为连续值，若长度为 1 则为离散值，若长度为 2 ， 则为连续值
        :param X: 当前所有特征的数据 pd.DaraFrame格式
        :param y: 标签值
        :return:  以信息增益率来选择的最佳划分属性，第一个返回值为属性名称，第二个为最佳划分属性对应的信息增益率
        '''
        features = X.columns
        best_feature_name = None
        best_gain_ratio = [float('-inf')]
        entD = self.entroy(y)

        for feature_name in features:
            is_continuous = type_of_target(X[feature_name]) == 'continuous'
            info_gain_ratio = self.info_gainRatio(X[feature_name], y, entD, is_continuous)
            if info_gain_ratio[0] > best_gain_ratio[0]:
                best_feature_name = feature_name
                best_gain_ratio = info_gain_ratio

        return best_feature_name, best_gain_ratio

    def gini_index(self, feature, y, is_continuous=False):
        '''
        计算基尼指数， 对于连续值，选择基尼系统最小的点，作为分割点
        -------
        :param feature:
        :param y:
        :return:
        '''
        m = y.shape[0]
        unique_value = pd.unique(feature)
        if is_continuous:
            unique_value.sort()  # 排序, 用于建立分割点
            # 这里其实也可以直接用feature值作为分割点，但这样会出现空集， 所以还是按照书中4.7式建立分割点。好处是不会出现空集
            split_point_set = [(unique_value[i] + unique_value[i + 1]) / 2 for i in range(len(unique_value) - 1)]

            min_gini = float('inf')
            min_gini_point = None
            for split_point_ in split_point_set:  # 遍历所有的分割点，寻找基尼指数最小的分割点
                Dv1 = y[feature <= split_point_]
                Dv2 = y[feature > split_point_]
                gini_index = Dv1.shape[0] / m * self.gini(Dv1) + Dv2.shape[0] / m * self.gini(Dv2)

                if gini_index < min_gini:
                    min_gini = gini_index
                    min_gini_point = split_point_
            return [min_gini, min_gini_point]
        else:
            gini_index = 0
            for value in unique_value:
                Dv = y[feature == value]
                m_dv = Dv.shape[0]
                gini = self.gini(Dv)  # 原书4.5式
                gini_index += m_dv / m * gini  # 4.6式

            return [gini_index]

    def gini(self, y):
        p = pd.value_counts(y) / y.shape[0]
        gini = 1 - np.sum(p ** 2)
        return gini

    def info_gain(self, feature, y, entD, is_continuous=False):
        '''
        计算信息增益
        ------
        :param feature: 当前特征下所有样本值
        :param y:       对应标签值
        :return:        当前特征的信息增益, list类型，若当前特征为离散值则只有一个元素为信息增益，若为连续值，则第一个元素为信息增益，第二个元素为切分点
        '''
        m = y.shape[0]
        unique_value = pd.unique(feature)
        if is_continuous:
            unique_value.sort()  # 排序, 用于建立分割点
            split_point_set = [(unique_value[i] + unique_value[i + 1]) / 2 for i in range(len(unique_value) - 1)]
            min_ent = float('inf')  # 挑选信息熵最小的分割点
            min_ent_point = None
            for split_point_ in split_point_set:

                Dv1 = y[feature <= split_point_]
                Dv2 = y[feature > split_point_]
                feature_ent_ = Dv1.shape[0] / m * self.entroy(Dv1) + Dv2.shape[0] / m * self.entroy(Dv2)

                if feature_ent_ < min_ent:
                    min_ent = feature_ent_
                    min_ent_point = split_point_
            gain = entD - min_ent

            return [gain, min_ent_point]

        else:
            feature_ent = 0
            for value in unique_value:
                Dv = y[feature == value]  # 当前特征中取值为 value 的样本，即书中的 D^{v}
                feature_ent += Dv.shape[0] / m * self.entroy(Dv)

            gain = entD - feature_ent  # 原书中4.2式
            return [gain]

    def info_gainRatio(self, feature, y, entD, is_continuous=False):
        '''
        计算信息增益率 参数和info_gain方法中参数一致
        ------
        :param feature:
        :param y:
        :param entD:
        :return:
        '''

        if is_continuous:
            # 对于连续值，以最大化信息增益选择划分点之后，计算信息增益率，注意，在选择划分点之后，需要对信息增益进行修正，要减去log_2(N-1)/|D|，N是当前特征的取值个数，D是总数据量。
            # 修正原因是因为：当离散属性和连续属性并存时，C4.5算法倾向于选择连续特征做最佳树分裂点
            # 信息增益修正中，N的值，网上有些资料认为是“可能分裂点的个数”，也有的是“当前特征的取值个数”，这里采用“当前特征的取值个数”。
            # 这样 (N-1)的值，就是去重后的“分裂点的个数” , 即在info_gain函数中，split_point_set的长度，个人感觉这样更加合理。有时间再看看原论文吧。

            gain, split_point = self.info_gain(feature, y, entD, is_continuous)
            p1 = np.sum(feature <= split_point) / feature.shape[0]  # 小于或划分点的样本占比
            p2 = 1 - p1  # 大于划分点样本占比
            IV = -(p1 * np.log2(p1) + p2 * np.log2(p2))

            grain_ratio = (gain - np.log2(feature.nunique()) / len(y)) / IV  # 对信息增益修正
            return [grain_ratio, split_point]
        else:
            p = pd.value_counts(feature) / feature.shape[0]  # 当前特征下 各取值样本所占比率
            IV = np.sum(-p * np.log2(p))  # 原书4.4式
            grain_ratio = self.info_gain(feature, y, entD, is_continuous)[0] / IV
            return [grain_ratio]

    def entroy(self, y):
        p = pd.value_counts(y) / y.shape[0]  # 计算各类样本所占比率
        ent = np.sum(-p * np.log2(p))
        return ent


if __name__ == '__main__':

    # 4.3
    # data_path = r'C:\Users\hanmi\Documents\xiguabook\watermelon3_0_Ch.csv'
    # data3 = pd.read_csv(data_path, index_col=0)
    #
    # tree = DecisionTree(criterion='infogain')
    # tree.fit(data3.iloc[:, :8], data3.iloc[:, 8])
    # treePlottter.create_plot(tree.tree_)

    # 4.4
    data_path2 = r'C:\Users\hanmi\Documents\xiguabook\watermelon2_0_Ch.txt'
    data = pd.read_table(data_path2, encoding='utf8', delimiter=',', index_col=0)

    train = [1, 2, 3, 6, 7, 10, 14, 15, 16, 17]
    train = [i - 1 for i in train]
    X = data.iloc[train, :6]
    y = data.iloc[train, 6]

    test = [4, 5, 8, 9, 11, 12, 13]
    test = [i - 1 for i in test]

    X_val = data.iloc[test, :6]
    y_val = data.iloc[test, 6]

    tree = DecisionTree('gini', 'pre_pruning')
    tree.fit(X, y, X_val, y_val)

    print(np.mean(tree.predict(X_val) == y_val))
    treePlottter.create_plot(tree.tree_)
