import csv
import numpy as np
import matplotlib.pyplot as plt
import copy
from time import sleep
import random
import types


def loadDataset(filename):
    with open(filename, 'r') as f:
        lines = csv.reader(f)
        data_set = list(lines)
    if filename != 'titanic.csv':
        for i in range(len(data_set)):
            del(data_set[i][0])
    # 整理数据
    for i in range(len(data_set)):
        del(data_set[i][0])
        del(data_set[i][2])
        data_set[i][4] += data_set[i][5]
        del(data_set[i][5])
        del(data_set[i][5])
        del(data_set[i][6])
        del(data_set[i][-1])

    category = data_set[0]

    del (data_set[0])
    # 转换数据格式
    for data in data_set:
        data[0] = int(data[0])
        data[1] = int(data[1])
        if data[3] != '':
            data[3] = float(data[3])
        else:
            data[3] = None
        data[4] = float(data[4])
        data[5] = float(data[5])
    # 补全缺失值 转换记录方式 分类
    for data in data_set:
        if data[3] is None:
            data[3] = 28
        # male : 1, female : 0
        if data[2] == 'male':
            data[2] = 1
        else:
            data[2] = 0
# 经过测试，如果不将数据进行以下处理，分布会过于密集，处理后，数据的分布变得稀疏了
        # age <25 为0, 25<=age<31为1，age>=31为2
        if data[3] < 25:
            data[3] = 0
        elif data[3] >= 21 and data[3] < 60: # 但是测试得60分界准确率最高？？？！！！
            data[3] = 1
        else:
            data[3] = 2
        # sibsp&parcg以2为界限，小于为0，大于为1
        if data[4] < 2:
            data[4] = 0
        else:
            data[4] = 1
        # fare以64为界限
        if data[-1] < 64:
            data[-1] = 0
        else:
            data[-1] = 1


    return data_set, category

def split_data(data):

    data_set = copy.deepcopy(data)

    data_mat = []
    label_mat = []
    for i in range(len(data_set)):
        if data_set[i][0] == 0:
            data_set[i][0] = -1

        label_mat.append(data_set[i][0])
        del(data_set[i][0])
        data_mat.append(data_set[i])

    print(data_mat)
    print(label_mat)

    return data_mat, label_mat


def select_j_rand(i ,m):
    # 选取alpha
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def clip_alptha(aj, H, L):
    # 修剪alpha
    if aj > H:
        aj = H
    if L > aj:
        aj = L

    return aj

def smo(data_mat_In, class_label, C, toler, max_iter):
    # 转化为numpy的mat存储
    data_matrix = np.mat(data_mat_In)
    label_mat = np.mat(class_label).transpose()
    # data_matrix = data_mat_In
    # label_mat = class_label
    # 初始化b，统计data_matrix的纬度
    b = 0
    m, n = np.shape(data_matrix)
    # 初始化alpha，设为0
    alphas = np.mat(np.zeros((m, 1)))
    # 初始化迭代次数
    iter_num = 0
    # 最多迭代max_iter次
    while iter_num < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            # 计算误差Ei
            fxi = float(np.multiply(alphas, label_mat).T*(data_matrix*data_matrix[i, :].T)) + b
            Ei = fxi - float(label_mat[i])
            # 优化alpha，松弛向量
            if (label_mat[i]*Ei < -toler and alphas[i] < C) or (label_mat[i]*Ei > toler and alphas[i] > 0):
                # 随机选取另一个与alpha_j成对优化的alpha_j
                j = select_j_rand(i, m)
                # 1.计算误差Ej
                fxj = float(np.multiply(alphas, label_mat).T*(data_matrix*data_matrix[j, :].T)) + b
                Ej = fxj - float(label_mat[j])
                # 保存更新前的alpha，deepcopy
                alpha_i_old = copy.deepcopy(alphas[i])
                alpha_j_old = copy.deepcopy(alphas[j])
                # 2.计算上下界L和H
                if label_mat[i] != label_mat[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue
                # 3.计算eta
                eta = 2.0 * data_matrix[i, :]*data_matrix[j, :].T - data_matrix[i, :]*data_matrix[i, :].T - data_matrix[j, :]*data_matrix[j, :].T
                if eta >= 0:
                    print("eta >= 0")
                    continue
                # 4.更新alpha_j
                alphas[j] -= label_mat[j]*(Ei - Ej)/eta
                # 5.修剪alpha_j
                alphas[j] = clip_alptha(alphas[j], H, L)
                if abs(alphas[j] - alphas[i]) < 0.001:
                    print("alpha_j变化太小")
                    continue
                # 6.更新alpha_i
                alphas[i] += label_mat[j]*label_mat[i]*(alpha_j_old - alphas[j])
                # 7.更新b_1和b_2
                b_1 = b - Ei - label_mat[i]*(alphas[i] - alpha_i_old)*data_matrix[i, :]*data_matrix[i, :].T - label_mat[j]*(alphas[j] - alpha_j_old)*data_matrix[i, :]*data_matrix[j, :].T
                b_2 = b - Ej - label_mat[i]*(alphas[i] - alpha_i_old)*data_matrix[i, :]*data_matrix[j, :].T - label_mat[j]*(alphas[j] - alpha_j_old)*data_matrix[j, :] * data_matrix[j, :].T
                # 8.根据b_1和b_2更新b
                if 0 < alphas[i] and C > alphas[i]:
                    b = b_1
                elif 0 < alphas[j] and C > alphas[j]:
                    b = b_2
                else:
                    b = (b_1 + b_2)/2
                # 统计优化次数
                alpha_pairs_changed += 1
                # 打印统计信息
                print("第%d次迭代 样本：%d , alpha优化次数：%d" % (iter_num, i, alpha_pairs_changed))
        # 更新迭代次数
        if alpha_pairs_changed == 0:
            iter_num += 1
        else:
            iter_num = 0
        print("迭代次数：%d" % iter_num)

    return b, alphas


def caluelate_w(data_mat, label_mat, alphas):
    # 计算w
    alphas = np.array(alphas)
    data_mat = np.array(data_mat)
    label_mat = np.array(label_mat)

    # numpy.tile(A, reps):通过重复A给出的次数来构造数组。

    # numpy中reshape函数的三种常见相关用法
    # reshape(1, -1)转化成1行：
    # reshape(2, -1)转换成两行：
    # reshape(-1, 1)转换成1列：
    # reshape(-1, 2)转化成两列

    w = np.dot((np.tile(label_mat.reshape(1, -1).T, (1, 5))*data_mat).T, alphas)
    return w.tolist()


def prediction(test, w, b):
    test = np.mat(test)
    result = []

    for i in test:
        if i*w+b > 0:
            result.append(1)
        else:
            result.append(-1)

    print(result)

    return result


if __name__ == "__main__":
    test_set, category = loadDataset('titanic.csv')
    data_set, category = loadDataset('titanic.csv')

    test_mat, test_label = split_data(test_set)
    data_mat, label_mat = split_data(data_set)

    b, alphas = smo(data_mat, list(label_mat), 0.6, 0.001, 40)
    print(b)
    print(alphas)
    w = caluelate_w(data_mat, label_mat, alphas)
    print(w)

    print(test_mat)
    print(test_label)
    result = prediction(test_mat, w, b)

    count = 0
    for i in range(len(result)):
        if result[i] == test_label[i]:
            count += 1

    print(count)

    print(count/len(result))