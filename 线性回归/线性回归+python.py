#线性回归+python+梯度下降法
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1, 5.56], [2, 5.70], [3, 5.91], [4, 6.40],[5, 6.80],
              [6, 7.05], [7, 8.90], [8, 8.70],[9, 9.00], [10, 9.05]])
m,n = np.shape(x)
# print(m,n)
x_data = np.ones((m,n))
# print(x_data)
x_data[:,:-1] = x[:,:-1]
# print(x_data)
y_data = x[:,-1]
# print(y_data)
m,n = np.shape(x_data)
# print(n)
theta = np.ones(n)
# print(theta)

def gradientDescent(iter,x,y,w,alpha):
    x_train = x.transpose() #转置
    for i in range(0,iter):
        pre = np.dot(x,w)   #矩阵x和w相乘
        loss = (pre - y)
        gradient = np.dot(x_train,loss)/m
        w =w-alpha*gradient
        cost =1.0/2*m*np.sum(np.square(np.dot(x,np.transpose(w))-y))
        print("第{}次梯度下降损失为: {}".format(i,round(cost,2)))
    return w

result = gradientDescent(1000,x_data,y_data,theta,0.01)
y_pre = np.dot(x_data,result)
print("线性回归模型w：",result)

plt.rc('font',family='SimHei',size=14)
plt.scatter(x[:,0],x[:,1],color='b',label='训练数据')
plt.plot(x[:,0],y_pre,color='r',label='预测数据')
plt.xlabel('x')
plt.ylabel('y')
plt.title('线性回归预测(梯度下降)')
plt.legend()
plt.show()
plt.close()
