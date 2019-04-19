#### 5.1 试述将线性函数 $ f(x) = w^{T}x $ 用作神经元激活函数的缺陷。
**答：**   

使用线性函数作为激活函数时，无论是在隐藏层还是在输出层（无论传递几层），其单元值（在使用激活函数之前）都还是输入 x 的线性组合，
这个时候的神经网络其实等价于逻辑回归（即原书中的对率回归，输出层仍然使用Sigmoid函数）的，若输出层也使用线性函数作为激活函数，那么就等价于线性回归 。

#### 5.2 试述使用图 5.2(b) 激活函数的神经元与对率回归的联系。
**答：**   
使用 $ Sigmoid $ 激活函数，每个神经元几乎和对率回归相同，只不过对率回归在 $ sigmoid(x)>0.5 $ 时输出为1，而神经元直接输出 $ sigmoid(x) $ 。

#### 5.3 对于图 5.7 中的 $ v_{ih} $ ，试推导出 BP 算法中的更新公式 (5.13)。
**答：**   
$ \triangle{v_{ih}} = -\eta\frac{\partial{E_{k}}}{\partial{v_{ih}}} $，因$ v_{ih} $ 只在计算 $ b_{h} $ 时用上，
所以 $ \frac{\partial{E_{k}}}{\partial{v_{ih}}}=\frac{\partial{E_{k}}}{\partial{b_{h}}} \frac{\partial{b_{h}}}{\partial{v_{ih}}} $ ，
其中 $ \frac{\partial{b_{h}}}{\partial{v_{ih}}}=\frac{\partial{b_{h}}}{\partial{a_{h}}} \frac{\partial{a_{h}}}{\partial{v_{ih}}}=\frac{\partial{b_{h}}}{\partial{a_{h}}} x_{i} $，
所以 $ \frac{\partial{E_{k}}}{\partial{v_{ih}}}=\frac{\partial{E_{k}}}{\partial{b_{h}}} \frac{\partial{b_{h}}}{\partial{a_{h}}} x_{i} =-e_{h}x_{i} $，即得原书中5.13式。

#### 5.4 试述式 (5.6) 中学习率的取值对神经网络训练的影响。
**答：**   

用一张网上找到的图来说明吧。   
![1](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/blob/master/ch5--%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/image/1.jpg)   
简单说就是学习率太高会导致误差函数来回震荡，无法收敛；而学习率太低则会收敛太慢，影响训练效率。

在原书p104也提到过。

#### 5.5 试编程实现标准 BP 算法和累积 BP 算法，在西瓜数据集 3.0 上分别用这两个算法训练一个单隐层网络，并进行比较。
**答：**   

标准 BP 算法和累积 BP 算法在原书（P105）中也提到过，就是对应标准梯度下降和随机梯度下降，差别就是后者每次迭代用全部数据计算梯度，前者用一个数据计算梯度。

代码在：[5.5-5.6](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/tree/master/ch5--%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/5.5-5.6)

具体两种情况的结果如下图：可以看出来gd的成本函数收敛过程更加稳定，而sgd每次迭代并不一定向最优方向前进，但总体方向是收敛的，且同样是迭代200次，最后结果相差不大，但由于sgd每次迭代只使用一个样本，计算量大幅度下降，显然sgd的速度会更快。

ps.关于随机梯度下降的实现，好像有两种方式，一种是每次将样本打乱，然后遍历所有样本，而后再次打乱、遍历；另一种是每次迭代随机抽取样本。这里采取的是后一种方式，貌似两种方式都可以。

此外，BP神经网络代码在以前学吴恩达老师深度学习课程的时候就写过，这次整理了一下正好放上来，所以很多代码和课程代码类似，添加了应用多分类的情况的代码。下面的5.6题也一并在这里实现。

![2](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/blob/master/ch5--%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/image/2.jpg)   


#### 5.6 试设计一个 BP 改进算法，能通过动态调整学习率显著提升收敛速度。编程实现该算法，并选择两个 UCI 数据集与标准 BP 算法进行实验比较。
**答：**   

动态调整学习率有很多现成的算法，RMSProp、Adam、NAdam等等。也可以手动实现一个简单指数式衰减 $ \eta(t)=\eta_{0}10^{-\frac{t}{r}} $ ，$ r $ 是一个超参。这里代码实现了Adam，下面

代码和5.5一同实现，同样在：[5.5-5.6](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/tree/master/ch5--%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/5.5-5.6)

这里只尝试了sklearn 中自带的iris数据集试了一下。同样学习率下，两者训练时损失函数如下：   
![3](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/blob/master/ch5--%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/image/3.jpg)   

可以明显看出adam的速度更快的。

#### 5.7 根据式 (5.18)和 (5.19) ，试构造一个能解决异或问题的单层 RBF 神经网络。
**答：**   

这里可以使用X = array([[1, 0], [0, 1], [0, 0], [1, 1]])，y = array([[1], [1], [0], [0]])作为数据，训练一个RBF神经网络。

这里使用均方根误差作为损失函数；输出层和书上一致，为隐藏层的线性组合，且另外加上了一个偏置项（这是书上没有）。

代码在：[5.7](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/tree/master/ch5--%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/5.7)

最后输出是：

>[[ 9.99944968e-01]   
[ 9.99881045e-01]   
[ 8.72381056e-05]   
[ 1.26478454e-04]]

感觉，分类的时候在输出层使用sigmoid作为激活函数也可以。

#### 5.8 从网上下载或自己编程实现 SOM 网络，并观察其在西瓜数据集 3.0α上产生的结果。
**答：**   

花了挺长时间看，写完代码的发现结果和预期有点不太符合，先暂时放一下吧还是...代码不完整就不放了。

这里提一个迷惑了我很久的一点，有些博客说SOM神经网络的聚类类别不需要自己定义，其实是不对的，
SOM神经网络输出聚类类别是需要自己定义，每个输出节点对应着一个类别，通过计算样本和输出节点的权重向量的相似度来确定样本属于哪个类别（节点）；
输入节点的数量和样本的维度一样（和BP网络相同）；输出的节点常常是以二维矩阵（这里可以是正方形也可以多边形等）或者一维直线的形式，每一个输出节点对应着一个权重向量和输入节点实现全连接。

想了解SOM建议参考下面几个链接：

https://link.zhihu.com/?target=https%3A//www.jianshu.com/p/41fc86728928   
https://link.zhihu.com/?target=https%3A//github.com/KeKe-Li/tutorial/blob/master/assets/src/SOM/SOM.md
https://link.zhihu.com/?target=http%3A//www.cs.bham.ac.uk/~jxb/NN/l16.pdf

#### 5.9* 试推导用于 Elman 网络的 BP 算法.
**答：**   

Elman 网络在西瓜书原书上说的是“递归神经网络”，但是在网上找资料说的
>“递归神经网络”是空间维度的展开，是一个树结构。   
“循环神经网络”是时间维度的展开，代表信息在时间维度从前往后的的传递和积累。

从书上p111描述来看感觉更像“循环神经网络”。最近时间不多（lan..），就不去啃原论文了。
关于“循环神经网络”或者递归神经网络的BP可以参考下面链接。

[零基础入门深度学习(5) - 循环神经网络](https://link.zhihu.com/?target=https%3A//zybuluo.com/hanbingtao/note/541458)

另外关于循环神经网络也可以看看吴恩达老师的深度学习课程“序列模型”那部分。

#### 5.10 从网上下载或自己编程实现一个卷积神经网络并在手写字符识别数据 MNIST 上进行实验测试。
**答：**  

正好前段时间做过Kaggle上手写数字识别的题目。这里正好放上来，CNN是用Tensorflow实现的，
之前看吴恩达老师深度学习课程的时候也拿numpy实现过（课程作业），等以后有时间再整理放上来吧。

[手写字识别 Kaggle](https://link.zhihu.com/?target=https%3A//github.com/han1057578619/kaggle_competition/tree/master/Digit_Recogniz)

