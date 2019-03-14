import tensorflow as tf
import numpy as np

learning_rate=1e-4
n_input=2
n_label=2  #1
n_hidden=100


x=tf.placeholder(tf.float32,[None,n_input])
y=tf.placeholder(tf.float32,[None,n_label])
weight={
    'h1':tf.Variable(tf.truncated_normal([n_input,n_hidden],stddev=0.1)),
    'h2':tf.Variable(tf.truncated_normal([n_hidden,n_label],stddev=0.1))
}
biases={
    'h1':tf.Variable(tf.zeros([n_hidden])),
    'h2':tf.Variable(tf.zeros([n_label]))
}

layer_1=tf.nn.relu(tf.add(tf.matmul(x,weight['h1']),biases['h1']))
y_pred=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weight['h2']),biases['h2']))
loss=tf.reduce_mean((y_pred-y)**2)

train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss)
#生成数据
X=np.array([[0,0],[0,1],[1,0],[1,1]])
Y=np.array([[1,0],[0,1],[0,1],[1,0]])
X=np.array(X).astype('float32')
Y=np.array(Y).astype('int16')

#加载
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#训练
for i in range(10000):
    sess.run(train_step,feed_dict={x:X,y:Y})


#计算预测值
print(sess.run(y_pred,feed_dict={x:X,y:Y}))

#查看隐含层的输出
print(sess.run(layer_1,feed_dict={x:X}))

























