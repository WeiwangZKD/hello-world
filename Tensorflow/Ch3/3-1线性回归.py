#线性回归
#准备数据
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
train_X=np.linspace(-1,1,100)
train_Y=2*train_X+np.random.randn(*train_X.shape)*0.3
plt.plot(train_X,train_Y,'ro',label='Original data')
plt.legend()
plt.show()


plotdata = { "batchsize":[], "loss":[] }
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

#创建模型
#占位符
X=tf.placeholder("float")
Y=tf.placeholder("float")
#模型参数
W=tf.Variable(tf.random_normal([1]),name="weight")
b=tf.Variable(tf.zeros([1]),name="bias")
#前向结构
z=tf.multiply(X,W)+b


#方向搭建模型
#反向优化
cost=tf.reduce_mean(tf.square(Y-z))
learning_rate=0.01
#梯度下降
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


#迭代训练模型
#1.训练模型
#初始化所有变量
init=tf.global_variables_initializer()
#定义参数
train_epochs=20
display_step=2

#启动session
with tf.Session() as sess:
    sess.run(init)
    plotdata={"batchsize":[],"loss":[]}   #存放批次值和损失值
    #向模型输入数据
    for epoch in range(train_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})

        #显示训练中的详细信息
        if epoch%display_step==0:
            loss=sess.run(cost,feed_dict={X:train_X,Y:train_Y})
            print("epoch:",epoch+1,"cost=",loss,"W=",sess.run(W),"b=",sess.run(b))
            if not (loss=="NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)

    print("Finished")
    print("cost=",sess.run(cost,feed_dict={X:train_X,Y:train_Y},),"W=",sess.run(W),"b=",sess.run(b))
    #2.训练模型可视化
    #图形显示
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X,sess.run(W)*train_X+sess.run(b),label='Fitted Line')
    plt.legend()
    plt.show()

    plotdata["avgloss"] = moving_average(plotdata["loss"])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')

    plt.show()

    #使用模型
    print("x=0.2,z=",sess.run(z,feed_dict={X:0.2}))
