## 神经网络课程笔记
### 梯度下降算法&优化器
- GD(gradient descent)：计算目标函数关于参数的梯度 $g_t = \nabla_{\theta}J(\theta)$, 之后更新模型参数：$$\theta_{t +1} = \theta_{t} - \alpha \nabla_{\theta}J(\theta)$$在此基础上衍生了**BDG**（批量梯度下降）,一次考虑多对样本，考虑平均梯度:$$\theta_{t+1} = \theta_{t} - \alpha_{t}\cdot\frac{1}{n}\cdot\sum_{i= 1}^n \nabla_{\theta}J_{i}(\theta,x^i,y^i)$$同样衍生出了**SDG**（随机梯度下降）,每次随机选择一个样本计算梯度，训练速度快，但梯度下降的波动较大，容易从一个局部最优跳到另一个局部最优:$$\theta_{t+1} = \theta_{t} - \alpha\cdot \nabla_{\theta}J_{i}(\theta,x^i,y^i)$$

- Momentum：参数更新时在一定程度上保留之前更新的方向，同时又利用了GD算法中的梯度，公式形如：
  $$m_{t +1} = \mu \cdot m_t + \alpha \cdot \nabla_{\theta}J(\theta) \\ 
  \theta_{t+1} = \theta_{t} - m_{t+1}\text{ (where $\mu$ usually be 0.9)}$$**可以看出在梯度反向改变时，momentum能够降低参数更新速度，从而减少震荡；在梯度方向相同时，momentum可以加速参数更新，从而加速收敛。**

- AdaGrad: SGD和Momentum均以相同的学习率更新$\theta$的各个分量，深度学习模型中涉及大量参数，我们希望不同的参数的更新速度能够有所区别。AdaGrad便对此做出了实现，具体地：
  $$g \leftarrow \nabla_{\theta} J(\theta)\\
  r \leftarrow r + g^2\\
  \Delta \theta \leftarrow \frac{\delta}{\sqrt{r + \epsilon}}\cdot g\\
  \theta \leftarrow \theta -\Delta \theta$$其中$\delta$是全局学习率，可以看出$r = \sum_{i = 1}^t g_{i}^2$即梯度平方和，在前期梯度平方和较小，约束项$\frac{\delta}{\sqrt{r + \epsilon}}$较大，梯度更新较快，后期更新量逐渐减小。
  **算法的缺点在于需要手工设置一个全局学习率，中后期梯度平方和越来越大，容易造成训练提前结束。**

- RMSProp：其实RMSprop依然依赖于全局学习率，RMSprop算是Adagrad的一种发展，和Adadelta的变体，效果趋于二者之间，适合处理非平稳目标(包括季节性和周期性)——对于RNN效果很好。

- Adam：是RMSProp的动量版。公式如下（非必要）：
  $$g\leftarrow \nabla_{\theta} J(\theta)\\
  m_t \leftarrow \beta_1 \cdot m_{t-1} + (1-\beta)\cdot g_t\\
  v_t \leftarrow \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot g_t^2\\
  \hat{m_t}\leftarrow \frac{m_t}{1-\beta_1^t}\\
  \hat{v_t}\leftarrow \frac{v_t}{1-\beta^t_2}\\
  \theta_{t+1} = \theta_{t} - \frac{\delta}{\epsilon + \sqrt{\hat{v_t}}}\cdot \hat{m_t}$$

**Adam可能不收敛，二阶动量是固定时间窗口内的累积，随着时间窗口的变化，遇到的数据可能发生巨变。深度神经网络往往包含大量的参数，在这样一个维度极高的空间内，非凸的目标函数往往起起伏伏，拥有无数个高地和洼地。有的是高峰，通过引入动量可能很容易越过；但有些是高原，可能探索很多次都出不来，于是停止了训练。**
**SGD没有用到二阶动量，因此学习率是恒定的（实际使用过程中会采用学习率衰减策略，因此学习率递减）。AdaGrad的二阶动量不断累积，单调递增，因此学习率是单调递减的。因此，这两类算法会使得学习率不断递减，最终收敛到0，模型也得以收敛。**
### 反向传播算法(Back Propagation)
### 损失函数
- mean square error
- NLL (negative log-likelyhood)
- cross entropy
- KL 散度(KL divergence):a type of statistical distance: a measure of how one probability distribution P is different from a second, reference probability distribution Q.即衡量两个概率分布的差异。（在强化学习中很有用）。
  设定义在采样空间$\mathcal{X}$的两个概率分布P,Q,则从Q到P的KL散度定义为:$$D_{KL}(P || Q) = \sum_{x\in \mathcal{X}} P(x)\log\left(\frac{P(x)}{Q(x)}\right)$$对于连续变量的概率分布，定义为：
  $$ D_{KL}(P||Q) = \int_{-\infty}^\infty p(x) \log\left(\frac{p(x)}{q(x)}\right)dx $$

