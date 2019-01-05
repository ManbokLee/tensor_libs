import mglearn
import matplotlib.pyplot as plt

mglearn.plots.plot_knn_regression(n_neighbors = 1)
plt.show()

mglearn.plots.plot_knn_regression(n_neighbors = 3)
plt.show()


from mglearn.datasets import make_wave
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

x, y = make_wave(n_samples = 40)
x, y
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size=0.3)
x_train
x_test
y_train
y_test

knn_reg = KNeighborsRegressor(n_neighbors = 3, n_jobs = -1)
knn_reg
# n_jobs 사용할 코어의 수
# -1 이면 모든 코어 사용

knn_reg.fit(x_train, y_train)
print('{:.3f}'.format(knn_reg.score(x_test, y_test)))


import numpy as np
_, axes = plt.subplots(1,3) 

line = np.linspace(-5, 5, num = 1000)
line = line.reshape(-1, 1)

for i, ax in zip([1,3,9], axes.ravel()):
    knn_reg = KNeighborsRegressor(n_neighbors=i, n_jobs= -1)
    knn_reg.fit(x_train, y_train)

    prediction = knn_reg.predict(line)
    ax.plot(line, prediction, label = 'model predict', c= 'k')
    ax.scatter(x_train, y_train, marker= '^', c= 'darkred', label='train target')
    ax.scatter(x_test, y_test, marker= '^', c= 'darkblue', label='test target')

    train_score = knn_reg.score(x_train, y_train)
    test_score = knn_reg.score(x_test, y_test)
    ax.set_title('k={}\ntest score={:.3f}\ntrain score={:.3f}'.format(i,train_score,test_score))
    ax.set_xlabel('feature')
    ax.set_ylabel('target')

axes[0].legend(loc=2)
plt.show()
'''
n_neighbors 의 값에 따라 최근접 이웃 회귀로 만들어진 예측 회귀 비교
이웃을 하나만 사용할 때는 훈련세트의 각 데이터 포인트가 모든 선을 통과합니다.
이것은 너무 복잡하게 모델이 자 있는 것을 말하며 실제로 예측할 때는 그 결과가 좋지 못합니다.
반대로 이웃을 많이 사용할 수록 훈련 데이터에는 잘 안 맞을 수 있지만 높은 정확도를 얻게 됩니다.

KNN 은 이해하기 쉬운 모델입니다. 그러나 수백개 이상의 많은 특성을 가진 데이터 셋에는 잘 동작하지 않고
특성값이 ㅇ이 많은 데이터셋에는 잘 작동하지 않습니다.
또한 예측이 느리고, 많은 특성(feature)을 처리하는 능력이 부족하여 현업에서는 잘 쓰이지 않습니다.
'''
