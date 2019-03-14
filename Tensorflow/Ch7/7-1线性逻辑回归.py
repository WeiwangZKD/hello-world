#线性逻辑回归
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


#模拟数据点
#按照指定的均值和方差生成固定数量的样本
def generate(sample_size,mean,cov,diff,regression):
    num_classes = 2  # len(diff)
    samples_per_class = int(sample_size / 2)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        Y1 = (ci + 1) * np.ones(samples_per_class)

        X0 = np.concatenate((X0, X1))
        Y0 = np.concatenate((Y0, Y1))

    if regression == False:  # one-hot  0 into the vector "1 0
        class_ind = [Y == class_number for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    X, Y = shuffle(X0, Y0)

    return X, Y

input_dim = 2
np.random.seed(10)
num_classes =2
mean = np.random.randn(num_classes)
cov = np.eye(num_classes)
X, Y = generate(1000, mean, cov, [3.0],True)
colors = ['r' if l == 0 else 'b' for l in Y[:]]
plt.scatter(X[:,0], X[:,1], c=colors)
plt.xlabel("Scaled age (in yrs)")
plt.ylabel("Tumor size (in cm)")
plt.show()
lab_dim = 1
print(X.shape)


#构建网络结构
input_features=tf.placeholder(tf.float32,[None,input_dim])
input_labels=tf.placeholder(tf.float32,[None,lab_dim])

#定义学习参数
W=tf.Variable(tf.random_normal([input_dim,lab_dim]),name="weight")
b=tf.Variable(tf.zeros([lab_dim]),name="bias")

output=tf.nn.sigmoid(tf.matmul(input_features,W)+b)
cross_entropy=-(input_labels * tf.log(output)+(1-input_labels) * tf.log(1-output))
ser=tf.square(input_labels - output)
loss=tf.reduce_mean(cross_entropy)
err=tf.reduce_mean(ser) #平方差函数，用来评估模型的错误率
optimizer=tf.train.AdamOptimizer(0.04)  #尽量用这个，因为收敛快，会动态调节梯度
train=optimizer.minimize(loss)


maxEpochs=20
minibatchSize=25
#启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(maxEpochs):
        sumerr = 0
        for i in range(np.int32(len(Y) / minibatchSize)):
            x1 = X[i * minibatchSize:(i + 1) * minibatchSize, :]
            y1 = np.reshape(Y[i * minibatchSize:(i + 1) * minibatchSize], [-1, 1])
            tf.reshape(y1, [-1, 1])
            _, lossval, outputval, errval = sess.run([train, loss, output, err],
                                                     feed_dict={input_features: x1, input_labels: y1})
            sumerr = sumerr + errval

        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(lossval), "err=",
              sumerr / np.int32(len(Y) / minibatchSize))


    #图形显示
    train_X,train_Y=generate(100,mean,cov,[3.0],True)
    colors=['r' if l==0 else 'b' for l in train_Y[:]]
    plt.scatter(train_X[:,0],train_X[:,1],c=colors)
    x=np.linspace(-1,8,200)
    y=-x*(sess.run(W)[0]/sess.run(W)[1])-sess.run(b)/sess.run(W)[1]
    plt.plot(x,y,label='Fitted line')
    plt.legend()
    plt.show()






















