#softmax的应用
import tensorflow as tf
labels=[[0,0,1],[0,1,0]]  #标签labels
logits=[[2,0.5,6],[0.1,0,3]]   #网络输出值

logits_scaled=tf.nn.softmax(logits)
logits_scaled2=tf.nn.softmax(logits_scaled)

result11=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
result12=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits_scaled)
result13=-tf.reduce_sum(labels*tf.log(logits_scaled),1)

with tf.Session() as sess:
    print("scaled=",sess.run(logits_scaled))
    print("scaled2=",sess.run(logits_scaled2))

    print("rel1=",sess.run(result11),"\n")  #正确的方式
    print("rel2=",sess.run(result12),"\n")
    print("rel3=",sess.run(result13))


#one-hot实验
#对非one-hot编码为标签的数据进行交叉熵的计算,比较其与one-hot编码的交叉熵之间的差别
labels=[[0.4,0.1,0.5],[0.3,0.6,0.1]]  #标签总和为1
result14=tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
with tf.Session() as sess:
    print("rel4=",sess.run(result14),"\n")


#sparse交叉熵的使用
labels=[2,1]  #表明labels中总共3个类:0,1,2. [2,1]等价于onehot编码中001与010
result15=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
with tf.Session() as sess:
    print("rel5=",sess.run(result15),"\n")   #rel5与rel1结果一样

#计算loss值
loss=tf.reduce_mean(result11)
with tf.Session() as sess:
    print("loss=",sess.run(loss))

labels=[[0,0,1],[0,1,0]]
loss2=tf.reduce_mean(-tf.reduce_sum(labels*tf.log(logits_scaled),1))
with tf.Session() as sess:
    print("loss2=",sess.run(loss2))

    


