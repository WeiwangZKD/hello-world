#通过卷积操作来实现本章所讲的sobel算子,将彩色的图片生成带有边缘化的信息

import tensorflow as tf
import matplotlib.image as mping  #mping用于读取图片
import matplotlib.pyplot as plt   #plt用于显示图片
import numpy as np


mying=mping.imread('img.jpg')  #读取和代码处于同一目录下的图片
plt.imshow(mying)  #显示图片
plt.axis('off')  #不显示坐标轴
plt.show()
print(mying.shape)  #(3264,2448,3)




full=np.reshape(mying,[1,3264,2448,3])
inputfull = tf.Variable(tf.constant(1.0,shape = [1, 3264, 2448, 3]))

filter =  tf.Variable(tf.constant([[-1.0,-1.0,-1.0],  [0,0,0],  [1.0,1.0,1.0],
                                    [-2.0,-2.0,-2.0], [0,0,0],  [2.0,2.0,2.0],
                                    [-1.0,-1.0,-1.0], [0,0,0],  [1.0,1.0,1.0]],shape = [3, 3, 3, 1]))


op=tf.nn.conv2d(inputfull,filter,strides=[1,1,1,1],padding='SAME') #3个通道输入,生成1个feature map
#归一化处理
o=tf.cast(((op-tf.reduce_min(op))/(tf.reduce_max(op)-tf.reduce_min(op)))*255,tf.uint8)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    t, f = sess.run([o, filter], feed_dict={inputfull: full})
    # print(f)
    t = np.reshape(t, [3264, 2448])

    plt.imshow(t, cmap='Greys_r')  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()





















