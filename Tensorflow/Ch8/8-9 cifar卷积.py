

#通过一个带有全局平局池化层的卷积神经网络对CIFAR数据集分类

import cifar10_input
import tensorflow as tf
import numpy as np


batch_size = 128
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'
print("begin")
images_train, labels_train = cifar10_input.inputs(eval_data = False,data_dir = data_dir, batch_size = batch_size)
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir, batch_size = batch_size)
print("begin data")


#权重W和b
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def avg_pool_6x6(x):
    return tf.nn.avg_pool(x,ksize=[1,6,6,1],strides=[1,6,6,1],padding='SAME')

#tf Graph Input
x=tf.placeholder(tf.float32,[None,24,24,3])
y=tf.placeholder(tf.float32,[None,10])


W_conv1=weight_variable([5,5,3,64])
b_conv1=bias_variable([64])    #64个卷积核

x_image=tf.reshape(x,[-1,24,24,3])

h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

W_conv2=weight_variable([5,5,64,64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 10])
b_conv3 = bias_variable([10])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

#倒数第二层是没有最大池化的卷积层，因为共有10类，所以卷积输出的是10个通道，并使其全局平均池化为10个节点
nt_hpool3=avg_pool_6x6(h_conv3)
nt_hpool3_flat=tf.reshape(nt_hpool3,[-1,10])
y_conv=tf.nn.softmax(nt_hpool3_flat)


cross_entropy=-tf.reduce_mean(y*tf.log(y_conv))

train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))


#运行session进行训练
sess=tf.Session()
sess.run(tf.global_variables_initializer())
tf.train.start_queue_runners(sess=sess)
for i in range(15000):  # 20000
    image_batch, label_batch = sess.run([images_train, labels_train])
    label_b = np.eye(10, dtype=float)[label_batch]  # one hot

    train_step.run(feed_dict={x: image_batch, y: label_b}, session=sess)

    if i % 200 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: image_batch, y: label_b}, session=sess)
        print("step %d, training accuracy %g" % (i, train_accuracy))

image_batch, label_batch = sess.run([images_test, labels_test])
label_b = np.eye(10, dtype=float)[label_batch]  # one hot
print("finished！ test accuracy %g" % accuracy.eval(feed_dict={
    x: image_batch, y: label_b}, session=sess))

































































