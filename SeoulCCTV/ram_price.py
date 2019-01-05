import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import mglearn 
import numpy as np
path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=path).get_name()
rc('font', family=font_name)

ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, 'ram_price.csv'))
ram_prices


plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel('년')
plt.ylabel('가격')
plt.show()




'''
DecisionTreeRegressor 는 train set 범위 박의 데이터에 대해서는
예측을 할 수 없습니다.
'''
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

data_train = ram_prices[ram_prices['date'] < 2000]
data_test = ram_prices[ram_prices['date'] >= 2000]


x_train = data_train['date'][:,np.newaxis] # train data 를 1열로 만듭니다.
y_train = np.log(data_train['price'])
x_train
y_train


tree = DecisionTreeRegressor().fit(x_train, y_train)
lr = LinearRegression().fit(x_train, y_train)

# test 는 모든 데이터에 대해 적용합니다.

x_all = ram_prices['date'].values.reshape(-1, 1) # x_all 을 1열로 만듭니다.
x_all

pred_tree = tree.predict(x_all)
price_tree = np.exp(pred_tree) # log 값 되돌리기

pred_lr = lr.predict(x_all)
price_lr = np.exp(pred_lr)

plt.semilogy(ram_prices['date'], price_tree, label='tree predict', ls='--', dashes = (2,1))
plt.semilogy(ram_prices['date'], price_lr, label='linear reg predict', ls=':')
plt.semilogy(data_train['date'], data_train['price'], label='train data', alpha=0.4)
plt.semilogy(data_test['date'], data_test['price'], label='test data', alpha=0.4)


plt.legend(loc = 1)
plt.xlabel('year', size=15)
plt.ylabel('price', size=15)
plt.show()
'''
램가격 데이터로 만든 리니어 모델과 회귀트리 예측값 비교

리니어 모델은 직선으로 데이터를 근사합니다.
트리모델은 트레인셋을 완벽하게 핏팅했습니다.
그러나 트리모델은 트레인 데이터를 넘어가 버리면 마지막 포인트를 이용해서
예측하는 것이 전부입니다. 
결정트리의 주요 단점은 가지치기를 함에도 불구하고, 오버피팅되는 경향이 있어
일반화 성능이 좋지 않습니다.

이에 대한으로 ensemble(앙상블) 방법을 많이 사용합니다.

'''
















