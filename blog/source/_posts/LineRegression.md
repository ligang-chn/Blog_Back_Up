---
title:  机器学习-线性回归
related_posts: true
date: 2019-09-30 22:16:23
copyright: true
categories: 机器学习
mathjax: true
tags: 
    - 机器学习
    - 梯度下降法
    - Python
---

​		线性回归——一种有监督的学习算法，即在建模过程中必须同时具备自变量x和因变量y。更为概括地说，线性模型就是对输入特征加权求和，再加上一个我们称为偏置项的常数。

<!-- more -->

​		**两种不同的训练模型的方法**：

- 通过“闭式”方程——直接计算出最适合训练集的模型参数（也就是使训练集上的成本函数最小化的模型参数）。
- 使用迭代优化的方法，即梯度下降法（GD）。逐渐调整模型参数直至训练集上的成本函数调至最低。包括梯度下降的几种变体：批量梯度下降、小批量梯度下降、随机梯度下降。



​		

####  一元线性回归

#####  定义

​		又称为简单线性回归模型，是指模型中只有一个自变量和一个因变量，给模型的数学表达式可以表示成：
$$
y=ax+b+\xi
$$
​		类似于一次函数，其中$\xi$为模型的误差，$a$和$b$统称为回归系数。**误差项$\xi$的存在主要是为了平衡等号两边的值**，通常被称为模型无法解释的部分。如下图：

![](https://img-blog.csdnimg.cn/20190913141802837.png?80)



#####  确定目标函数

​		我们希望预测值和实际值的差距尽量小，那么如何表示该差距呢？
<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913141833519.png">
</div>

​		由于误差项$\xi$是y与ax+b的差，结果可能是正值或负值，因此误差项$\xi$达到最小的问题需转换为误差平方和最小的问题（**最小二乘法的思路**）。
$$
J(a,b)=\sum^{n} _{i=1} {\xi^2}=\sum^{n}_{i=1}{(y_i-ax_i-b)^2}
$$
​		上面的函数可以称为`损失函数（loss function)`或`效用函数（utility function）`。通过分析问题，确定问题的损失函数或者效用函数；通过最优化损失函数或者效用函数，获得机器学习的模型。

​		求解误差项最小就是求解$J(a,b)$的最小值。该目标函数其实就是一个二元二次函数，可以使用偏导数的方法求解出a和b，进而得到目标函数的最小值。（这里可能你会有疑问：为什么求出a和b，就能得到目标函数的最小值？——这是因为这里我们已知x和y，自变量其实是a和b，J是因变量，所以要求J的最小值，自然需要对a和b求偏导。对于损失函数、代价函数、目标函数的理解参见博客：[理解代价函数](https://www.cnblogs.com/geaozhang/p/11442343.html#commentform)）
推到过程如下图：

![](LineRegression.assets/20190913141935781.png?40)

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913141935781.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_9,color_FFFFFF,t_70" width="20%">
</div>
<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913142114758.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="20%">
</div>
<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913142126409.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="20%">
</div>

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913142151545.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="20%">
</div>

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913142203519.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="20%">
</div>

> Python计算得到模型的回归参数，有第三方模块statsmodels，它是专门用于统计建模的第三方模块，可以调用子模块中的ols函数计算a和b。

​		

#####  简单线性回归的实现

```python
import numpy as np

class SimpleLinearRegression1:

    def __init__(self):
        """初始化Simple Linear Regression 模型"""
        self.a_=None
        self.b_=None

    def fit(self,x_train,y_train):
        """根据训练数据集x_train,y_train训练Simple Linear Regression 模型"""
        assert x_train.ndim==1,\
        "Simple Linear Regressor can only solve single feature training data"
        assert len(x_train)==len(y_train),\
        "the size of x_train must be equal to the size of y_train"

        x_mean=np.mean(x_train)#计算均值
        y_mean=np.mean(y_train)#计算均值

        num=0.0
        d=0.0
        for x_i,y_i in zip(x_train,y_train):
            num+=(x_i-x_mean)*(y_i-y_mean)#计算a的分子
            d+=(x_i-x_mean)**2#计算a的分母

        self.a_=num/d#得到a
        self.b_=y_mean-self.a_*x_mean#得到b

    def predict(self,x_predict):
        """给定待预测数据集x_predict,返回表示x_predict的结果向量"""
        assert x_predict.ndim==1,\
        "Simple Linear Regressor can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None,\
        "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self,x_signle):
        """给定单个待预测数据x_signle,返回x_signle的预测结果值"""
        return self.a_*x_signle+self.b_#根据上面计算的a和b，构建线性模型

    def __repr__(self):

        return "SimpleLinearRegression1()"
```



#####  向量化

​		从上面的简单线性回归实现中，我们可以看到，对于回归系数的计算，我们是通过for循环+数学公式计算得到的，在这里我再次贴出那段实现代码：

```python
#for循环计算
for x_i,y_i in zip(x_train,y_train):
    num+=(x_i-x_mean)*(y_i-y_mean)#计算a的分子
    d+=(x_i-x_mean)**2#计算a的分母
```

​		如果数据量比较大，那么这个过程是很缓慢的，所以需要优化。

​		首先，从数学表达式上来看，
<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913142312680.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="20%">
</div>


​		a的分子和分母都可以用下面的向量的点乘表示。向量的运算速度高于for循环。

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913142332988.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="30%">
</div>
​		通过numpy的向量运算可以提高性能。那么我们将上面的for循环的代码修改一下：

```python
num=(x_i-x_mean).dot(y_i-y_mean)
d=(x_i-x_mean).dot(x_i-x_mean)
```



####  多元线性回归

#####  定义

​		上面讨论的是一元线性回归模型，相对来说比较简单。实际上，我们的数据集的属性（即自变量）不止一个。对于含有多个属性的数据构建线性回归模型就是多元线性回归模型。如下图：

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913142354427.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="30%">
</div>
​		从上图可以看出，X是一组向量，具有多个特征。

​		**线性回归模型预测**：
$$
\hat y^{(i)}=\theta_0+\theta_1X_1^{(1)}+\theta_2X_2^{(2)}+…\theta_nX_n^{(n)}
$$
​		**线性回归模型预测（向量化）**：
$$
\hat y=X_b\cdot {\theta}
$$
​		
##### 目标函数

​		对于多元线性回归模型，目标函数和一元线性回归模型基本一致：
<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913142417725.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>
<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913142430433.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>

​		上面我们添加X_0参数，使其恒等于1，这样我们就可以使用向量的方式表示预测模型了。

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913142451839.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="50%">
</div>

​		所以，目标函数的求解变成了：
<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913142506369.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="50%">
</div>



​		闭式解法——就是一个直接得出结果的数学方程（即多元线性回归的正规方程解）：
$$
\theta =(X_b^TX_b)^{-1}X_b^Ty
$$
​		求解出来的$\theta$如下：
<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913143759539.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="30%">
</div>



#####  多元线性回归的实现

```python
import numpy as np
#from .metrics import r2_score

class LinearRegression:

    def __init__(self):
        """"初始化Linear Regression模型"""
        self.coef_=None
        self.interception_=None
        self._theta=None

    def fit_normal(self,X_train,y_train):
        """"根据训练数据集X_train,y_train训练Linear Regression模型"""
        assert X_train.shape[0]==y_train.shape[0],\
            "the size of X_train must be equal to the size of y_train"

        X_b=np.hstack([np.ones((len(X_train),1)),X_train])
        self._theta=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.interception_=self._theta[0]
        self.coef_=self._theta[1:]

        return self

    def predict(self,X_predict):
        """给定待预测数据集X_predict，返回表示X——predict的结果向量"""
        assert self.interception_ is not None and self.coef_ is not None,\
            "must fit before predict!"
        assert  X_predict.shape[1]==len(self.coef_),\
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self,X_test,y_test):
        """根据测试数据集X_test和y_test确定当前模型的准确度"""

        y_predict=self.predict(X_test)
        #return r2_score(y_test,y_predict)

    def __repr__(self):
        return "LinearRegression()"
```



#####  计算复杂度

​		标准方程求逆的矩阵$X^T\cdot X$，是一个$n*n$的矩阵（n是特征数量）。对这种矩阵求逆的计算复杂度通常为$O(n^{2.4})到O(n^{3})$之间（取决于实现）。因此当特征数量特别大时，标准方程的计算是很缓慢的。好的一面，线性模型一经训练完成，预测就非常迅速。





####  多项式回归

##### 定义

​		其实可以用线性模型拟合**非线性数据**。一个简单的方法就是将每个特征的幂次方添加为一个新特征，然后在这个拓展过的特征集上训练线性模型。这种方法被称为**多项式回归**。

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913143847432.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="30%">
</div>

​		将$X^2$和$X$分别看作两个特征，那么这个多项式回归依然可以看成线性回归。只不过对于x来说，是一个2次方程。

​		【注意】：当存在多个特征时，多项式回归能够发现特征和特征之间的关系（纯线性模型做不到这一点）。这是因为PolynomialFeatures会在给定的多项式阶数下，**添加所有特征组合**。如下：

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913143904702.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="30%">
</div>
​		要小心特征组合的数量爆炸！！！



#####  验证数据集与交叉验证

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913143918265.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>
​		测试数据集不参与模型的创建。

​		仍然存在一个问题：**随机**？

​		由于我们的验证数据集都是随机的从数据集中切出来的，那么训练出来的模型可能对于这份验证数据集过拟合，但是我们只有这一份数据集，一旦这个数据集中相应的有比较极端的数据，就可能导致这个模型不准确。于是就有了**交叉验证**。


<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913143936274.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913143953799.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="30%">
</div>

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144005942.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="30%">
</div>







#####  过拟合和欠拟合

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144025319.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>
​		我们由已知的训练数据得到的曲线，在面对新的数据的能力非常弱，即**泛化能力差**。例如，我们在训练数据集上模型的准确率很好，但是在测试数据集上模型准确率却很差。


<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144041506.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>
​		因此，我们需要寻找泛化能力最好的地方。

​	

​		使用交叉验证来评估模型的泛化性能，如果模型在训练集上表现良好，但是交叉验证的泛化表现非常糟糕，那么模型就是**过拟合**。如果在二者上的表现都不佳，那就是**欠拟合**。这就是判读模型太简单还是太复杂的一种方法。如下图：分别是欠拟合和过拟合。

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144056722.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="50%">
</div>

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144106164.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="50%">
</div>

​		高阶多项式回归模型就可能过度拟合训练数据，而线性模型则是拟合不足。



#####  学习曲线

​		另外一种方法是观察学习曲线：这个曲线绘制的是模型在训练集和验证集上，关于”训练集大小“的性能函数。要生成这个曲线，只要在不同大小的训练子集上多次训练模型即可。随着训练样本的逐渐增多，算法训练出的模型的表现能力的变化。

​		

####  偏差方差权衡

$$
模型误差=偏差+方差+不可避免的误差
$$

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144127783.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="30%">
</div>

​		偏差——原因在于错误的假设。比如假设数据是线性的，而实际上是二次的。高偏差模型最有可能对训练数据拟合不足。


<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144142892.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>
​		方差——原因在于模型对于训练数据的微小变化过度敏感。具有高自由度的模型很可能有高方差，所以很容易对训练数据过拟合。

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144155791.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>

​		不可避免的误差——因为数据本身的噪声。清理数据，减少这部分误差。


<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144222271.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144234155.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="20%">
</div>

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144244242.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="20%">
</div>

####  正则线性模型

​		减少过拟合的一个好办法就是**对模型正则化**：它拥有的自由度越低，就越不容易过度拟合数据。

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/2019091314430022.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>


​		
前面根据线性回归模型的参数估计公式：$\theta=(X^TX)^{-1}X^Ty$可知，得到$\theta$的前提是矩阵$X^TX$可逆。但在实际应用中，可能会出现**自变量个数多于样本量**（即，矩阵不是n\*n的，而是n\*m的）或自变量存在**多重共线性**（比如列方向上存在某一列是另外一列的倍数）的情况，此时无法根据公式计算回归系数的估计值$\theta$。为解决这类问题，基于线性回归模型的另外两种扩展的回归模型，分别是**岭回归**和**LASSO回归**。

#####  岭回归

​		岭回归是线性回归的正则化版：在成本函数中添加一个等于$a\sum_{i=1}^{n}\theta_i^2$的正则项。

​		【注意】：正则项只能在**训练**的时候添加到成本函数中，一旦训练完成，你需要使用**未正则化的性能指标**来评估模型性能。

​		训练阶段使用的成本函数与测试时使用的成本函数不同是非常常见的现象。除了正则化以外，还有一个导致这种不同的原因是，训练时的成本函数通常都可以使用优化过的衍生函数，而测试用的性能指标需要尽可能接近最终目标。

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144314473.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>
也就是，

$$
J(\theta)=\sum_{i=1}^n (y-X_b\theta)^2+\alpha \frac{1}{2}\sum_{i=1}^n\theta^2
$$
为求解目标函数$J(\theta)$的最小值，需要对其求导，并令导函数为0。这里不再推导，只说一下大致步骤：
1）根据线性代数知识，展开目标函数中的平方项；
2）对展开的目标函数求导；
3）令导数为0，计算回归系数$\theta$。
求得结果：
$$
\theta=(X_b^TX_b+\alpha \frac{1}{2}E)^{-1}X_b^Ty
$$

这里可以看出来，和之前没有正则项的回归系数相比，仅仅多了正则项的系数。
$\alpha$是L2正则项平方的系数，用来平衡模型的方差（回归系数的方差）和偏差（真实值和预测值之间的差异）。
对于岭回归来说，随着$\alpha$的增大，模型方差会减小而偏差会增大。


>在执行岭回归之前，必须对数据进行缩放，因为它对输入特征的大小非常敏感。



#####  LASSO回归

​		与岭回归一样，它也是向成本函数添加一个正则项，但是它增加的是权重向量的L1范数，而不是L2范数的平方的一半。

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144439445.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>
​		LASSO回归的一个重要特点是它倾向于完全消除掉最不重要特征的权重（也就是将它们设置为0）。换句话说，LASSO回归会自动执行特征选择并输出一个稀疏模型（即只有很少的特征有非零权重）。

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144451848.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>

#####  弹性网络

​		弹性网络是岭回归与LASSO回归之间的中间地带。其正则项就是岭回归和LASSO回归的正则项的混合，混合比例通过r来控制。


<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144505887.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>

#####  L1正则，L2正则

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/2019091314451888.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>

####   梯度下降法

#####  定义

​		前面我们求解目标函数都是通过”闭式“方程解，第二种方法是使用优化迭代的方法，即梯度下降。

​		梯度下降是一种通用的优化算法，其核心思想就是**迭代调整参数，从而使成本函数最小化**。



​		梯度就是**分别对每个变量进行微分**，然后用逗号分隔开，梯度是用<>包括起来的，说明梯度其实是一个**向量**。
$$
J(\Theta)=0.55-(5\theta_1+2\theta_2+12\theta_3)
$$

$$
\nabla J(\Theta)=<\frac {\partial J } {\partial \theta_1 },\frac {\partial J } {\partial \theta_2 },\frac {\partial J } {\partial \theta_3 }>
=<-5,-2,12>
$$

​		**梯度的意义**：

- 在单变量的函数中，梯度其实就是函数的微分，代表函数在某个给定点的切线的斜率；

- 在多变量函数中，梯度就是一个向量，向量有方向，梯度的方向就指出了函数在给定点的上升最快的方向。




​		**梯度下降法**，是一种基于搜索的最优化方法；（不是一个机器学习算法）

​		**作用**：最小化一个损失函数；

​		**梯度上升法**：最大化一个效用函数。



​		导数可以代表方向，对应J增大的方向。
$$
-\eta \frac{dJ}{d\theta}
$$

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144606596.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="30%">
</div>
​		并不是所有函数都有唯一的极值点；

​		**解决方案**：

- 多次运行，随机化初始点；

- 梯度下降法的初始点也是一个超参数。

  

  

##### 模拟实现梯度下降法

```python
def gradient_descent(initial_theta,eta,epsilon=1e-8):
    theta=initial_theta
    theta_history.append(initial_theta)
    
    while True:
        gradient=dJ(theta)
        last_theta=theta
        theta=theta-eta*gradient
        theta_history.append(theta)
        
        if(abs(J(theta)-J(last_theta))<epsilon):
            break
            
def plot_theta_history():
    plt.plot(plot_x,J(plot_x))
    plt.plot(np.array(theta_history),J(np.array(theta_history)),'ro-')
    
    
eta=0.9
theta_history=[]
gradient_descent(0.,eta)
plot_theta_history()     
```

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144622730.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="30%">
</div>

#####  线性回归中使用梯度下降法

​		应用梯度下降法，需要保证所有特征值的大小比例都差不多，否则收敛时间会长很多。

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144643825.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>


​		在线性回归中，我们需要求解目标函数最小，现在使用梯度下降法试试：

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144656119.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>
​		求解梯度：

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/2019091314470669.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="50%">
</div>

​		**实现梯度下降法**：

```python
def fit_gd(self,X_train,y_train,eta=0.01,n_iters=1e4):
    """根据训练数据集X_train,y_train,使用梯度下降法训练Linear Regression模型"""
    assert X_train.shape[0]==y_train.shape[0],\
        "the size of X_train must be euqal to the size of y_train"

    def J(theta,X_b,y):#目标函数
        try:
            return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
        except:
            return float('inf')

    def dJ(theta, X_b, y):#求解偏导数
        res = np.empty(len(theta))
        res[0] = np.sum(X_b.dot(theta) - y)

        for i in range(1, len(theta)):
            res[i] = (X_b.dot(theta) - y).dot(X_b[:, i])
        return res * 2 / len(X_b)

    def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
        theta = initial_theta
        i_iter = 0

        while i_iter < n_iters:
            gradient = dJ(theta, X_b, y)
            last_theta = theta
            theta = theta - eta * gradient

            if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                break

            i_iter += 1

        return theta

    X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
    initial_theta = np.zeros(X_b.shape[1])  # theta向量的行数=X_b向量的列数
    self._theta=gradient_descent(X_b,y_train,initial_theta,eta,n_iters)

    self.interception_=self._theta[0]
    self.coef_=self._theta[1:]

    return  self
```



​		之前我们在目标函数中使用向量化对求解过程进行优化，这里我们也可以使用向量化。

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144720109.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="50%">
</div>

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144733346.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>

​		通过向量化的方式，我们程序在求解计算时就会快很多。



#####  随机梯度下降法

​		**批量梯度下降法（Batch Gradient Descent）**

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144748579.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>

​		这是之前的向量化公式，我们在求解梯度时，每一项都要对**所有的样本**进行计算。每一步都使用整批训练数据。因此面对非常庞大的训练集时，算法变得极慢。但是梯度下降法随特征数量扩展的表现比较好：如果要训练的线性模型拥有几十万个特征，使用梯度下降法比标准方程快得多。


<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144805345.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144816388.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>

​		随机梯度下降法的学习率不能是一个固定值，需要是递减的。随机性的好处在于可以逃离局部最优，但缺点是永远定位不出最小值。要解决这个困境，有一个办法是逐步降低学习率。【**模拟退火的思想**】

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144829990.png" width="18%">
</div>
​		

​		**SGD算法实现**：

```python
def fit_sgd(self,X_train,y_train,n_iters=5,t0=5,t1=50):
    """根据训练数据集X_train,y_train,使用随机梯度下降法训练Linear Regression模型"""
    assert X_train.shape[0]==y_train.shape[0],\
        "the size of X_train must be euqal to the size of y_train"
    assert n_iters>=1

    def dJ_sgd(theta, X_b_i, y_i):
        return X_b_i*(X_b_i.dot(theta)-y_i)*2.

    def sgd(X_b, y, initial_theta, n_iters, t0=5,t1=50):

        def learning_rate(t):
            return t0/(t+t1)

        theta=initial_theta
        m=len(X_b)

        for cur_iter in range(n_iters):
            indexes=np.random.permutation(m)
            X_b_new=X_b[indexes]
            y_new=y[indexes]
            for i in range(m):
                gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                theta = theta - learning_rate(cur_iter*m+i) * gradient

        return theta

    X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
    initial_theta = np.zeros(X_b.shape[1])  # theta向量的行数=X_b向量的列数
    self._theta=sgd(X_b,y_train,initial_theta,n_iters,t0,t1)

    self.interception_=self._theta[0]
    self.coef_=self._theta[1:]

    return  self
```



#####  关于梯度的调试

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/2019091314484895.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144905314.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>





####  衡量线性回归的指标


<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913145027264.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="30%">
</div>

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913145017458.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="30%">
</div>

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913145005383.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="20%">
</div>
​		**最好的衡量线性回归法的指标**：


<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144951792.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/2019091314494256.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>

​		**R Squared**：

- R^2^<=1;
- R^2^越大越好。当我们的预测模型不犯任何错误时，R^2^得到最大值1；
- 当我们的模型等于基准模型时，R^2^为0；
- 如果R^2^<0，说明我们学习到的模型还不如基准模型。此时，很有可能我们的数据不存在任何线性关系。

<div  align="center">    
 <img src="https://img-blog.csdnimg.cn/20190913144924195.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzM0MzgwMw==,size_16,color_FFFFFF,t_70" width="40%">
</div>



####  线性回归算法总结

​		1、评价线性回归算法：R Squared

​		2、典型的参数学习，对比KNN：非参数学习

​		3、只能解决回归问题，对比KNN：既可以解决分类问题，又可以解决线性问题	

​		4、对数据有假设：线性，对比KNN对数据没有假设

​		5、优点：对数据具有强解释性
