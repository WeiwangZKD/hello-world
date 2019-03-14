
#本例使用全连接网络，将MNIST图像用多层神经网络分类

import tensorflow as tf  # 导入tensorflow库
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#定义参数
learning_rate=0.001
training_epochs=25
batch_size=100
display_step=1

#设置网络模型参数
n_hidden_1=256  #第一个隐藏层节点个数
n_hidden_2=256  #第二个隐藏层节点个数
n_input=784    #MNIST共784(28x28)维
n_classes=10   #MNISt共10个类别(0-9)

#定义占位符
x=tf.placeholder("float",[None,n_input])
y=tf.placeholder("float",[None,n_classes])

#创建model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

#学习参数
weights={
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}

biases={
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}


#输出值
pred=multilayer_perceptron(x,weights,biases)

#定义loss和优化器
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#初始化变量
init=tf.global_variables_initializer()

#启动session
with tf.Session() as sess:
    sess.run(init)

    #启动循环
    for epoch in range(training_epochs):
        avg_cost=0
        total_batch=int(mnist.train.num_examples/batch_size)
        #遍历全部数据集
        for i in range(total_batch):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})

            # Compute average loss
            avg_cost+=c/total_batch

        if epoch % display_step ==0:
            print("Epoch:",'%04d'%(epoch+1),"cost=",\
                  "{:.9f}".format((avg_cost)))

    print(" Finished! ")

    #测试model
    # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))










































