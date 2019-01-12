
import tensorflow as tf
import numpy as np

# [털, 날개]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]]
)

y_data = np.array([
    [1, 0, 0], # 기타
    [0, 1, 0], # 포유류
    [0, 0, 1], # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1],
])

# [털, 날개] 털:0 날개:1
# [0, 0] -> [1, 0, 0] : 기타
# [1, 0] -> [0, 1, 0] : 포유류

# **************
# 신경망 모델 구성
# ************** 

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 신경망은 [입력층, 출력층] 으로 구성됨
# 입력츠은 feature, 출력층은 label
# 예를들어 입력층에는 털, 날개 ... 출력층에는 포유류, 조류가 출력됨
W = tf.Variable(tf.random_uniform([2, 3], -1.0, 1.0))

# 편향은 각 레이어의 아웃풀 갯수
# 현제 예제에서는 최종 결과값 분류종인 3으로 설정
b = tf.Variable(tf.zeros([3]))

# 신경망에서 가충치 W 와 편향 b 를 적용
L = tf.add(tf.matmul(X, W), b)

# tensorflow 에서 기본적으로 제공하는 활성화 함수인 ReLU 함수 적용
L = tf.nn.relu(L)

# softmax 함수를 이용하여 출력값을 사용하기 쉽게 전환
# sofrmax 함수는 결과값을 전체값이 1인 확률로 만드는 함수
model = tf.nn.softmax(L)
model
'''
<tf.Tensor 'Softmax:0' shape=(?, 3) dtype=float32>
'''

# 신경망을 최적화하기 위한 비용 함수 작성
# 각 개별 결과에 대한합을 구한 뒤 평균을 내는 방식 사용
# 전체 합이 아닌, 개별 결과를 구한 두 ㅣ평균을 내는 방식을 사용하기 때문에
# axis 옵션을 사용함, axis 옵션이 없으면 총합인 스칼라 값이 출력됨
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1)) # 크로스 엔트로피 적용한 식
# 크로스 엔트로피란 예측값과 실젝밧 사이의 확률 분포의 차이를 비용으로 계산한 값
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)
train_op
'''
<tf.Operation 'GradientDescent' type=NoOp>
'''


# *******************
# 신경망 모델 학습
# *******************
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
     sess.run(train_op, feed_dict={X: x_data, Y: y_data})
     if (step + 1) % 10 == 0:
            print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# *************
# 결과 확인
# 0: 기타, 1: 포유류, 2: 조류
# *************

prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
target
x_data
y_data
print('예측값: ', sess.run(prediction, feed_dict={X: x_data}))
print('실제값: ', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
'''
정확도: 16.67
'''

