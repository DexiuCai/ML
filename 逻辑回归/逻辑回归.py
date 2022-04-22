import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import csv

def loadDataset():
    data = []
    labels = []
    with open('logisticDataset.txt','r')as f:
        reader = csv.reader(f,delimiter='\t')
        for row in reader:
            data.append([1.0,float(row[0]),float(row[1])])
            labels.append(int(row[2]))
    return data,labels

def plotBestfit(w):
    #把训练集数据用坐标的形式画出来
    dataMat,labelMat=loadDataset()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')

    #把分类边界画出来
    x = np.arange(-3.0,3.0,0.1)
    y = (-w[0]-w[1]*x)/w[2]
    ax.plot(x,y)
    plt.show()

def plotloss(loss_list):
    x = np.arange(0,30,0.01)
    plt.plot(x,np.array(loss_list),label = 'linear')

    plt.xlabel('time')  #梯度下降的次数
    plt.ylabel('loss')  #损失值
    plt.title('loss trend')     #损失值随着w不断更新，不断 变化的趋势
    plt.legend()        #图像图例
    plt.show()

def main():
    #读取训练集（txt文件）中的数据
    data,labels = loadDataset()
    #将数据转换矩阵的形式，便于后面进行计算
    #构建特征矩阵X
    x = np.array(data)
    #构建标签矩阵y
    y = np.array(labels).reshape(-1,1)
    #随机生成一个w参数（权重）矩阵  .reshape((-1,1))的作用，不知道有多少行，只想变成一列
    w = 0.001*np.random.rand(3,1).reshape((-1,1))
    #m表示一共有多少组训练数据
    m = len(x)
    #定义梯度下降的学习率0.03
    learn_rate = 0.03

    loss_list = []
    #实现梯度下降法，不断跟新w，获得最优解，使损失函数的损失值最小
    for i in range(3000):
        #最重要的就是这里用numpy矩阵计算，完成假设函数计算，损失函数计算，梯度下降计算
        #计算假设函数h(w)x
        g_x = np.dot(x,w)
        h_x = 1/(1+np.exp(-g_x))

        #计算损失函数Cost Function 的损失值loss
        loss = np.log(h_x)*y+(1-y)*np.log(1-h_x)
        loss = -np.sum(loss)/m
        loss_list.append(loss)

        #梯度下降函数更新w权重
        dw = x.T.dot(h_x-y)/m
        w +=-learn_rate*dw

    #得到更新后的w，可视化
    print('w最优解：')
    print(w)
    print('最终得到的分类边界：')
    plotBestfit(w)
    print('损失值随着w不断更新，不断变化的趋势：')
    plotloss(loss_list)

    #定义一个测试数据，计算他属于哪一类别
    test_x = np.array([1,-1.395634,4.662541])
    test_y = 1/(1+np.exp(-np.dot(test_x,w)))
    print(test_y)
    print('最后的损失值loss_list[-1]:',loss_list[-1])

#print(data_arr)
if __name__=='__main__':
    main()

