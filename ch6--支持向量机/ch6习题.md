#### 6.11 自己编程实现SVM，并在西瓜数据集 3.0αα 测试。
**答：**   

**自己加的题目。虽然书上没要求，但还是自己写了一遍。**   
代码在：[mySVM](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/tree/master/ch6--%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/mySVM)

其实主要就是把SMO实现了一遍。

参考：
- 《统计学习方法》
- 《机器学习实战》
- 《Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines》
- 《THE IMPLEMENTATION OF SUPPORT VECTOR MACHINES USING THE SEQUENTIAL MINIMAL OPTIMIZATION ALGORITHM》

写代码的时候参考以上资料。代码主要根据原论文中提供的伪代码和《机器学习实战》的源码写的，《机器学习实战》中给出的代码和原论文有几个地方有差异：

1. 在选择第二个 $ \alpha $ 变量时，原论文给出的方法是
1、首先从间隔边界上的支持向量 $ ( 0<\alpha<C ) $ 中，找到使得 $ \lvert{E_{i}-E_{j}}\rvert $ 最大的 $ \alpha $ 。
2、若上面选的 $ \alpha $ 不行，则遍历所有的支持向量。
3、还是不行则遍历所有的样本点。
4、若所有样本点都不行，则放弃一个变量。关于认定 $ \alpha $ 不行的原则，原论文描述的是：不能使得目标函数有足够下降。
实际上在伪代码中是使得 $ \alpha $ 本身有足够下降就认为不行。而《机器学习实战》中代码从更新过的 $ \alpha $ 中选择使得 $ \lvert{E_{i}-E_{j}}\rvert $ 最大的 $ \alpha $作为第二个变量。
若不行则直接放弃第一个变量。不知道这一点是改进还是简化。代码中是按照论文的方式实现的。
2. 为了选择第二个变量时方便，SMO会将所有支持向量的误差 $ E $ 建立缓存，且在每次更新完 $ \alpha $ 之后，都同时更新误差缓存。
《机器学习实战》源码中，在给支持向量建立误差缓存时，虽然有更新 $ E_{k} $ 的步骤，但只更新了每次更新的两个变量 $ \alpha_{i} $ , $ \alpha_{j} $ 对应的误差，
并且更新之后也没有使用，在选择第二个变量计算 $ \lvert{E_{i}-E_{j}}\rvert $ 时，都重新计算了 $ E $ 。在自己实现的时候，这一点也是按照论文来的。
3. 最后一点，在更新 $ a $ 时，需要计算 $ \eta=K_{11}+K_{22}-2K_{12} $ （参考统计学习方法 p127 式7.107），
在极少数情况下 $ \eta $ 会等于零，在原论文中给出了针对这种情况下 $ a $ 的更新方式，在《机器学习实战》中，
这种情况会直接跳过更新。这里代码和《机器学习实战》一致，直接跳过了，这一点其实影响不大。

另外实际上更新误差缓存 E 有更高效的方法，是在第四个参考文献里面发现的。不过在代码里面没有实现。因为感觉有点复杂。。。有兴趣的可以看看那篇论文，在3.4小节解释了更新误差缓存的方式。

最后自己写的代码在西瓜数据集3.0α上测试了一下，训练后决策边界如下：   
![1](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/blob/master/ch6--%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/image/1.jpg)

训练结果和使用sklearn中结果（习题6.2）一致，支持向量也是相同的，决策边界差不多相同，其他数据未测试。不过速度上，sklearn会快很多，测试了一下训练西瓜数据集，自己写的代码需要5e-4秒，而sklearn只需要1e-8。还是有很大差距的。

代码有点乱，这里只为深刻理解一下SMO，也不做工程使用，暂时就不优化了。。以后闲下来再看看。

以上。

---

#### 6.1 试证明样本空间中任意点 $ x $ 到超平面 $ (w,b) $ 的的距离为式 (6.2)。
**答：**   

图中，令A点到超平面（点B）的距离为 $ \gamma $ ，于是 $ \bar{BA}=\gamma*\frac{w}{\left| w \right|} $ ( $ \frac{w}{\left| w \right|} $ 是 $ w $ 同向的单位向量， 
对于超平面 $ (w,b) $ 其垂直方向即 $ w $ )，对于B点有： $ w^{T}\bar{B} +b=0 $ ，而 $ \bar{B}= \bar{A}-\bar{BA} $ ，
于是 $ w^{T}(\bar{A}-\gamma*\frac{w}{\left| w \right|}) + b = 0 $ ，
可得 $ w^{T}\ast\bar{A}-\gamma\ast\left| w \right|+b=0\Rightarrow\gamma=\frac{w^{T}\bar{A}+b}{\left| w \right|} $ ，
这里的 $ \bar{A} $ 即书中 $ x $，即可的式6.2。

![2](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/blob/master/ch6--%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/image/2.jpg)

这个问题在吴恩达老师的《机器学习》课程（斯坦福版本CS 229）里面讲解过，有兴趣的可以自己去网易公开课看看，图片截图自该课程课件。

#### 6.2 试使用 LIBSVM，在西瓜数据集 3.0α 上分别用线性核和高斯核训练一个 SVM，并比较其支持向量的差别。
**答：**   

这里没用LIBSVM，用的sklearn中的sklearn.svm.svc，它的实现也是基于libsvm的。

**使用不同参数的时候，支持向量是不同的（没有对高斯核中的gamma调参）。**
由于西瓜数据集3.0a线性不可分，所以使用线性核时，无论惩罚系数多高 ，还是会出现误分类的情况；
而使用高斯核时在惩罚系数设置较大时，是可以完全拟合训练数据。所以在惩罚系数设置较小时，
两者支持向量都类似，而在惩罚系数较大（支持向量机中，惩罚系数越大，正则化程度越低）时，
高斯核的支持向量数目会较少，而线性核的会几乎没有变化。

代码在：[6.2](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/tree/master/ch6--%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/6.2)

C = 100时训练情况如下：   
高斯核支持向量： [ 8 9 11 12 13 14 16 2 3 4 5 6 7]   
![3](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/blob/master/ch6--%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/image/3.jpg)   
线性核支持向量： [ 8 9 11 12 13 14 16 2 3 4 5 6 7]   
![4](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/blob/master/ch6--%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/image/4.jpg)   

C = 10000时训练情况如下：   
高斯核支持向量： [11 13 14 1 5 6]
![5](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/blob/master/ch6--%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/image/5.jpg)   

线性核支持向量： [ 9 11 12 13 14 16 2 3 4 5 6 7]
![6](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/blob/master/ch6--%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/image/6.jpg)   

#### 6.3 选择两个 UCI 数据集，分别用线性核和高斯核训练一个 SVM，并与BP 神经网络和 C4.5 决策树进行实验比较。
**答：**   

代码在：[6.3](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/tree/master/ch6--%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/6.3)

#### 6.4 试讨论线性判别分析与线性核支持向量机在何种条件下等价。
**答：**   

**这道题想不出很明确的答案，这仅讨论一下。**
有一点很明确的是：在数据线性可分时并不会导致线性判别分析与线性核支持向量机等价。   
![7](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/blob/master/ch6--%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/image/7.jpg)   

上图是以 iris 数据中第 0 列和第 2 列数据作为训练集。分别 LDA 和线性核 SVM 训练，得到图中两者的决策边界，
可以看出在数据线性可分的情况下，两者决策边界还是有很大差别的。如果这里等价理解为两者决策边界相同，
即两个模型决策函数是相同的，那么两者决策边界重合时，两者等价的。那么什么时候两者会重叠？

事实上，可以想象到，LDA 的决策边界的斜率可以由投影面 $ w $ 得到，其斜率是垂直于 $ w $ 的，
而截距则可由两类样本的中心点 在 $ w $ 投影 $ w^{T}u_{0},w^{T}u_{1} $ 得到，即LDA决策边界通过 $ w^{T}u_{0},w^{T}u_{1} $ 的中点（公式参考原书p60）。

而线性核 SVM 的决策边界则由模型参数 $ w_{svm},b $ 得到（对应原书式6.12），所以当 SVM 中的参数 $ w_{svm} $ 和 LDA 中投影面 $ w_{lda} $ 垂直，
且 SVM 的 $ w_{svm},b $ 通过两类样本中心在 $ w_{lda} $ 的投影的中点时，两者等价。只讨论到这里了。

查了很多资料没找到相关信息。感觉LDA和SVM其实没有多大相似度。   
ps.这里解答其实就按照结果倒推了一下。貌似都是些废话。   

[画图代码](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/tree/master/ch6--%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/6.4)

#### 6.5 试述高斯核 SVM 与 RBF 神经网络之间的联系。
**答：**   

其实这个题目在 p145 的《休息一会儿》的注释里面已经给出答案了。

RBF神经网络中，将隐藏层神经元个数设置为训练样本数，每个样本设置为一个神经元中心，此时RBF的预测函数和SVM激活函数相同。

个人理解，两个模型还是有挺大差别的。

- RBF中径向基激活函数中控制方差的参数 $ \beta $ 是由模型自动习得，而在 SVM 中是一个可调节的超参。
- 目标函数、优化方式也不同。

但是如果将RBF中 $ \beta $ 固定为和SVM一致，最后训练结果应该会比较相似。

以上是个人理解。就不写代码验证了。。。

#### 6.6 试析 SVM 对噪声敏感的原因。
**答：**   

SVM的决策只基于少量的支持向量，若噪音样本出现在支持向量中，容易对决策造成影响，所以SVM对噪音敏感。

#### 6.7 试给出式 (6.52) 的完整 KKT 条件。
**答：**   

6.52 式是经过将完整的KKT条件

$ \nabla_{w}L =0,\nabla_{b}L =0,\nabla_{\xi_{i}}L =0,\nabla_{\hat{\xi_{i}}}L = 0 $ ，这里对应着原书式6.47-6.50合并之后的。完整的如下：

$$ \alpha_{i}(f(x_{i})-y_{i}-\epsilon-\xi_{i})=0,a_{i}\geq0,f(x_{i})-y_{i}-\epsilon-\xi_{i}\leq0 $$
$$ \hat\alpha_{i}(y_{i}-f(x_{i})-\epsilon-\hat{\xi_{i}})=0,\hat{a_{i}}\geq0,y_{i}-f(x_{i})-\epsilon-\hat{\xi_{i}}\leq0 $$
$$ u_{i}\xi_{i}=0, u_{i}\geq0, -\xi_{i}\leq0 $$
$$ \hat{u_{i}}\hat{\xi_{i}}=0, \hat{u_{i}}\geq0, -\hat{\xi_{i}}\leq0 $$

6.52中其他公式的推导出来的。

#### 6.8 以西瓜数据集 3.0α 的"密度"为输入"含糖率"为输出，试使用LIBSVM 训练一个 SVR。
**答：** 

关于SVR没有理解很深，简单了解了一下。这道题就简单看一下不同参数，训练结果的变换吧。   
![8](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/blob/master/ch6--%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/image/8.jpg)   
![9](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/blob/master/ch6--%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/image/9.jpg)   

直观上看，含糖率和密度无明显关系。所以无论模型参数怎么调，看上去对数据的拟合都不是很好，预测值和真实值还是有较大差异。
不过还是可以看出来随着gamma或者C的增大，模型都会趋于更加复杂。
   
这里代码很简单，还是放上来。
[6.8](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/tree/master/ch6--%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/6.8)

#### 6.9 试使用核技巧推广对率回归，产生"核对率回归"。
**答：** 

对于 $ w^{T}x+b $ 形式的模型，即线性模型，使用核技巧的关键点在于最优的 $ w^{\ast} $ 可以由训练集的线性组合表示，
即 $ w^{\ast}=\sum_{i}\beta_{i}x_{i} $ ，使得模型可表示为 $ \sum_{i}\beta_{i}\langle x_{i},x\rangle+b $ ，
进而使用核函数直接计算数据点在高维空间内积，而不显式的计算数据点从低维到高维的映射。

原命题：事实上对于任何 L2 正则化的线性模型： $ \min_{w} \frac{\lambda}{N}w^{T}w+\frac{1}{N}\sum_{n=1}^{N}err(y_{n},w^{T}x_{n}) $ ，
这里，其最优值都可以表示为 $ w^{\ast}=\sum_{i}\beta_{i}x_{i} $ 。其证明参考下图：（截图自林轩田讲授的《机器学习技法》课程第五章课件）   
![10](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/blob/master/ch6--%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/image/10.jpg)   
上图中 $ z_{n} $ 可以理解为数据点 $ x_{n} $ 或者 $ x_{n} $ 在高维空间的映射 $ \phi(x_{n}) $   

上图通过反证法来证明：

将 $ w_{\ast} $ 分解为与 $ z_{n} $ 空间平行的 $ w_{\lVert} $ 和垂直的 $ w_{\bot} $ ，若 $ w_{\bot}=0 $ 则表示 $ w_{\ast} $ 可以表示为 $ z_{n} $ 的线性组合。

假设 $ w_{\ast} $ 为最优解且 $ w_{\bot}\ne0 $ 。由于 $ w_{\bot} $ 与 $ z_{n} $ 空间垂直，于是 $ w_{\bot}^{T}z_{n}=0 $ ， 
因此 $ w_{\bot} $ 不会对目标函数中 $ err $ 项的大小产生影响，而对于 $ w_{\ast}^{T}w_{\ast} $ ，
在 $ w_{\bot}\ne0 $ 的情况下必定有： $ w_{\ast}^{T}w_{\ast}>w_{\lVert}^{T}w_{\lVert} $ ，
显然 $ w_{\lVert} $ 比 $ w_{\ast} $ “更优”，即 $ w_{\ast} $ 不是最优解。于是原命题得证。

那么对于L2正则化的逻辑回归，其核形式即如下图：

![11](https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/blob/master/ch6--%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA/image/11.jpg)   

可直接使用梯度下降等优化算法求解上图的目标函数即

题目没要求。就偷个懒不写代码了。

说个题外的。SVM 中只有少数支持向量对应的 $ a_{i} $ 非零，所以对于SVM来说，
训练完成后只需要储存个别非零的 $ a_{i} $ 和对应数据点即可；而不同于SVM， 
核逻辑回归并没有这一性质，需要储存所有训练数据。就这一点来说核逻辑回归并不高效。

#### 6.10* 试设计一个能显著减少 SVM 中支持向量的数目而不显著降低泛化性能的方法。（未完成）
**答：** 

这个应该也是某个论文。最近时间不多，暂时不深究了就。。

