#首先从 sklearn 库中导入线性模型中的线性回归算法
# from sklearn import linear_model
#其次训练线性回归模型。使用  fit() 喂入训练数据
# model = linear_model.LinearRegression()
# model.fit(x,y)
#最后一步就是对训练好的模型进行预测。调用 predict() 预测输出结果
# model.predict(x_)


#使用matplotlib绘制图像，使用numpy准备数据集
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

#1、准备数据:手动生成一个数据集
#使用numpy准备数据集
import numpy as np
#准备自变量x，-3到3的区间均分隔30份数
x = np.linspace(3,6,40)
# print(x)
#准备因变量y，这一个关于x的假设函数
y = 3 * x + 2

x = x+np.random.rand(40)

#由于fit需要传入二维矩阵数据，因此需要处理x，y的数据格式，将每个样本信息单独作为矩阵的一行
x = [[i]for i in x]
y = [[i]for i in y]
#构建线性回归模型
model = linear_model.LinearRegression()
#训练模型，‘喂入’数据
model.fit(x,y)
#准备测试数据x_,这里准备了三组，如下：
x_ =[[4],[5],[6]]
#打印预测结果
y_ = model.predict(x_)
print("y_:",y_)

#查看w和b的
print("w的值：",model.coef_)
print("b截距值为：",model.intercept_)

plt.rc('font',family='SimHei',size=14)
#数据集绘制，散点图，图像满足函数图像
plt.scatter(x,y)
#绘制最佳拟合直线
plt.plot(x_,y_,color='r',linewidth =3.0,linestyle = '-')
plt.title('线性回归预测(梯度下降)')
plt.legend(['拟合函数','数据'],loc=0)
plt.show()

