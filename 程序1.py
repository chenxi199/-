'''import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点
x_data = np.linspace(-4, 4, 100)[:, np.newaxis]
noise = np.random.randint(0, 2, x_data.shape)
#noise = np.random.normal(0, 0.02, x_data.shape)
constant=25*np.ones(x_data.shape)
y_data = np.square(x_data) + noise-constant
''''''
x_data = np.linspace(-4, 4, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise''''''
# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络的中间层
weights_l1 = tf.Variable(tf.random_normal([1, 10]))
biases_l1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_l1 = tf.matmul(x, weights_l1) + biases_l1
l1 = tf.nn.tanh(Wx_plus_b_l1)

# 定义神经网络输出层
weights_l2 = tf.Variable(tf.random_normal([10, 1]))
biases_l2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_l2 = tf.matmul(l1, weights_l2) + biases_l2
prediction = tf.nn.elu(Wx_plus_b_l2)


# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#定义会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(600):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    #    print ('loss ',loss)
      #  print ('train_step ',train_step)
        
    # 获取预测值
    plt.figure()
    plt.scatter(x_data, y_data)
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 画图  
    #print ('prediction_value ',prediction_value)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
'''
#encoding:utf-8
#encoding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 神经层函数参数， 输入值， 输入的大小， 输出的大小，激励函数（默认为空）
def add_layer(inputs, in_size, out_size, activation_function=None):
    # 生成初始参数时，随机变量回比全部为0要好很多，所以weights为一个in_size
    # out_size列的随机变量矩阵

    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 在机器学习中biases的推荐值不为0，所以在0的基础上加了0.1
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # Wx_plus_b,即神经网络未激活的值。其中tf.matmul()是矩阵的乘法
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # activation_function 激励函数为None，输出就是当前的预测值Wx_plus_b
    # 不为空时就把Wx_plus_b传到activation_function()函数中得到
    if activation_function == None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs


# 导入数据
x_data = np.linspace(-4, 4, 100)[:, np.newaxis]
noise = np.random.randint(0, 2, x_data.shape)
constant=25*np.ones(x_data.shape)
y_data = np.square(x_data) + noise-constant
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

'''# matplotlib 可视化
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()#plt.ion()用于连续显示
plt.show()'''
# 搭建网络
# 隐藏层 10个神经元
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.tanh)  # tf自带激励函数tf.nn.relu
# 输出层
prediction = add_layer(l1, 10, 1, activation_function=None)

# 误差 二者差的平方和再取平均
loss = tf.reduce_mean(tf.square(ys - prediction))
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

# 梯度下降优化器  学习率为0.1 最小话损失函数
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        # 训练模型
        sess.run(train_op, feed_dict={xs: x_data, ys: y_data})   
    prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
    plt. scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
           
