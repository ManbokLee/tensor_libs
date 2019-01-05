import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# MNIST 문자 인식 프로그램
'''
MNIST란 28*28 크기의 0 ~ 9 사이의 숫자 이미지와 이에 해당하는 레이블로 구성된 
데이터베이스이다. MNIST 이미지중에 하나가 독립변수(x) 에 들어오면
그 이미지가 무슨 숫자인지를 해석해서 y로 출력해주는 가장 기본적인 이미지 인식 프로그램이다.
'''

# MINST 이미지를 다운로드 한다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot= True)
mnist
#one_hot: 인코딩방식 - 숫자화 시켜서 불러옴

# 변수들을 설정한다.
x = tf.placeholder(tf.float32, [None, 784]) # 28 * 28 = 784
W = tf.Variable(tf.zeros([784,10])) # zeros 는 0으로 각 간을 초기화
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

'''
softman(소프트맥스): softmax Regression 은 어떤 input X 가 주어졌을 때, 그것이 class i 일 것이라고
확신하는 정보(evidence)를 식으로 나타낸다.
'''

# cross-entropy 모델을 설정한다ㅏ.
y_ = tf.placeholder(tf.float32, [None, 10]) # y 다음에 _ 있음 !! 확정된 값이 아니라는 의미
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
'''
cross-entropy 는 예측값(prediction)이 실제값(truth) 을 설명하는데
얼마나 비효율적인지(inefficient) 를 나타낸다. 따라서 cross-entropy 가 낮을수록
좋은 모델이다. 즉 우리는 cross-entropy 를 최소화 하는 방향으로 학습을 진행한다.
'''

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


'''
cross-entropy 를 최소화 하는 방법으로 경사하강법을 사용하여 학습을 진행함
'''
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print(int(i/100) + 1, end="")
    if i%100 == 0:
        print('\n')


# MNIST: Mixed National Institue of Standards and Technology Database

# 학습된 모델이 얼마나 정확한지 출력한다.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))







