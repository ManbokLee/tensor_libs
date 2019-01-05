# ************************************
# 랜덤 포레스트 알고리즘의 앙상블 기법
# 사이킷런에 내장된 아이리스 데이터셋 활용
# ************************************

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np

np.random.seed(0)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df.head()
df.columns
'''
Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)'],
      dtype='object')
'''
iris.target
iris.target_names
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
'''
Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)', 'species'],
      dtype='object')
'''


# learning 전에 지도학습 준비를 하기 위해 dataset 분류
df['is_train'] = np.random.uniform(0,1,len(df)) <= .75 # 75%
'''
Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)', 'species', 'is_train'],
      dtype='object')
'''

train, test = df[df['is_train'] == True], df[df['is_train'] == False]
len(train) # 118 개
len(test) # 32 개

features = df.columns[:4]
features
'''
Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)'],
      dtype='object')
'''

y = pd.factorize(train['species'])[0]
y
'''
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2], dtype=int64)
       # one-hot encoding
'''

# *************************
# 랜덤 포레스트 분류기를 이용한 학습 단계
# *************************

clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(train[features], y)

# 테스트셋에 분류기 적용
clf.predict(test[features])
'''
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 1, 1, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=int64)
'''

clf.predict_proba(test[features])[0:10]
'''
array([[1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.],
       [1., 0., 0.]])
'''


# **********************
# 분류기에 대한 평가
# **********************

preds = iris.target_names[clf.predict(test[features])]
preds[0:5]
test['species'].head()


pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species'])
'''
Predicted Species  setosa  versicolor  virginica
Actual Species                                  
setosa                 13           0          0
versicolor              0           5          2
virginica               0           0         12
'''


# ***********************************
# Feature importance: 판단(예측)하는데 있어 요소의 중요성을 수치화한 값
# ***********************************
list(zip(train[features], clf.feature_importances_))
'''
[
    ('sepal length (cm)', 0.11185992930506346), 
    ('sepal width (cm)', 0.016341813006098178), 
    ('petal length (cm)', 0.36439533040889194), 
    ('petal width (cm)', 0.5074029272799464)
]
'''




