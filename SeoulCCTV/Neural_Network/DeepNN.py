"""
은닉층(신경망층)을 둘 이상으로 구성한 심층 신경망이 곧 딥러닝입니다.
다층신경망(=심층신경망)을 만드는 작업은 단일신경망에 가중치와 편향을
추가하기만 하면 완성됨
입력층과 출력층은 각각 feature와 분류갯수로 맞추고 중간의
연결 부분은 맞닿은 층의 유런 수와 같도록 맞추면 됨.
중간의 연결부분을 은닉층이라고 하며, 은닉층의 뉴러노수는 하이퍼파라이터이니
실험을 통해 가장 적절한 수를 정하면 됨
따라서, 아래의 10은 뉴런수를 증가시키면서 100%에 도달하여 얻은 결과임

AdamOptimizer 를 최적화 함수로 사용하였으며, 이는 앞서 사용한 GradingDescentOptiomizer 보다
보편적으로 성능이 좋은 평가임. 즉 필요한 함수를 개인이 사용하는 것도
성능을 끌어올리는 요건이 됨.
"""


import tensorflow as tf
import numpy as np

x_data = np.array(
    [[0,0],
     [1,0],
     [1,1],
     [0,0],
     [0,0],
     [0,1]])
# [기타, 포유류, 조류]

y_data = np.array([
    [1,0,0], # 기타
    [0,1,0], # 포유류
    [0,0,1], # 조류
    [1,0,0],
    [1,0,0],
    [0,0,1]
    ])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# **********
# 신경망 모델 구성
# **********
nn_cnt = 10

# 첫번째 가중치 차원을 [특성, 히든레이어 뉴런갯수] -> [2,10] 
W1 = tf.Variable(tf.random_uniform([2,nn_cnt], -1, 1.))
# 두번째 가중치 차원을 [히든레이어 뉴런갯수,분류갯수] -> [10,3] 
W2 = tf.Variable(tf.random_uniform([nn_cnt,3], -1, 1.))

# 편향을 각 레이어 아웃풋 갯수
# b1 은 히든레이어 뉴런 갯수 10
# b2 는 최종결과물인 분류 갯수 3
b1 = tf.Variable(tf.zeros([nn_cnt]))
b2 = tf.Variable(tf.zeros([3]))
#신경망의 히든레이어에 가중치 W1 과 편향b1 을 적용
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)
# 최종 아웃풋 계산
# 히든레이어에 W2 와 b2 를 적용하여 3개의 출력값 생성
model = tf.add(tf.matmul(L1, W2), b2)

# 텐서플로우 cross_entropy 함수를 이용하여 softmax 계산
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# **********
# 신경망 모델 학습
# **********

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if(step + 1) % 10 == 0:
        print(step +1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))



# *****
# 결과 확인
# 0 : 기타, 1: 포유류, 2: 조류
# *****

prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값: ', sess.run(prediction, feed_dict={X: x_data}))
print('실제값: ', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))

print('W1', sess.run(W1))

# 정확도: 100