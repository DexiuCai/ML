# # TensorFlow 2.0 + Linear Regression
import tensorflow as tf
import numpy as np

x =np.float32(np.random.rand(100,1))
# print(x.shape)

# y = a*x+b
y = np.dot(x,0.8)+0.2

w = tf.Variable(np.float32())
b = tf.Variable(np.float32())

def model(x):
    return w*x+b
def loss(predicted_y,desired_y):
    return tf.reduce_sum(tf.square(predicted_y - desired_y)) #计算一个张量的各个维度上元素的总和.

optimizer  = tf.optimizers.Adam(0.1)

for step in range(0,100):
    with tf.GradientTape() as t:    #tf.GradientTape () 是一个自动求导的记录器
        outputs = model(x)
        current_loss = loss(outputs,y)
        grads = t.gradient(current_loss,[w,b])
        optimizer.apply_gradients(zip(grads,[w,b]))
    if step %10 ==0:
        print("Step:%d,loss:%2.5f,weight:%2.5f,bias:%2.5f "%(step,current_loss.numpy(),w.numpy(),b.numpy()))

x_ = 2
y_ =w*x_+b
print(y_.numpy())