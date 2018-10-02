import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
x_data = np.linspace(-4, 4, 100)[:, np.newaxis]
noise = np.random.randint(0, 2, x_data.shape)
constant=25*np.ones(x_data.shape)
y_data = np.square(x_data) + noise-constant

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#定义神经网络中间层
Weight_L1 = tf.Variable(tf.random_normal([1,10]))#得到随机数，形状一行十列【1，10】1代表输入
biases_L1 = tf.Variable(tf.zeros([1,10]))#偏置值初始化为0，10个神经元，10个偏置值
Wx_plus_b_L1 = tf.matmul(x,Weight_L1) + biases_L1#信号的总和，矩阵的乘法+偏置值
L1 = tf.nn.tanh(Wx_plus_b_L1)#中间层的输出，激活函数#定义神经网络输出层

Weight_L2 = tf.Variable(tf.random_normal([10,1]))#定义权值10行一列
biases_L2 = tf.Variable(tf.zeros([1,1]))#输出层只有一列，只有一个偏置值
Wx_plus_b_L2 = tf.matmul(L1,Weight_L2) + biases_L2#输出层信号总和
#prediction = tf.nn.tanh(Wx_plus_b_L2)#输出层信道总和
prediction =Wx_plus_b_L2#输出层信道总和

#代价函数和训练方法：二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)#定义最小会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)#样本点
    plt.plot(x_data,prediction_value,'r-',lw=5)#红色，实线，线宽5    
    plt.show()
