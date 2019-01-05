from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# 학습데이터
aset = [
    [1,0,10],
    [1,0,10],
    [1,0,10],
    [0,1,1],
    [0,1,1],
    [0,1,1],
    [0,1,1],
    [1,1,11],
    [1,1,11],
    [1,1,11],
    [1,1,11],
    [1,1,11],
    [1,0,10],
    [0,1,1],
    [1,1,11]
    ]

# 검증할 데이터
bset = [
    [1,0],
    [1,0],
    [1,0],
    [1,1],
    [0,1],
    [1,1],
    [0,1],
    [1,1],
    [0,1],
    [1,1]
    ]

len(aset)
aset
bset

# 데이타 프레임화
df = pd.DataFrame(aset,columns=['a','b','r']) # a,b: 조건 / r: 결과

# 학습, 테스트 데이터 분리
m_train, m_test = df[0:10], df[11:14]

# 학습 데이터 분리
features = df.columns[:2] # train 데이터의 조건부분 분리
features

m_label, m_unique = pd.factorize(m_train['r']) # train 데이터의 결과값만 분리 (r 이 결과값) [기존에 문자를 인트형으로 변환한것과 같은 결과]
m_label # 결과값이 int 형으로 변환된 값 array index 로 활용
m_unique # m_lebel 에 매칭된 실제 값

# 학습
clf = RandomForestClassifier(n_jobs=2, random_state=0) # 램덤포레스트 정의
m_train[features]
clf.fit(m_train[features], m_label) # 모델을 학습시킴 


# 학습한 모델을 테스트 데이터에 적용
m_test[features]
r_y = clf.predict(m_test[features])


# 결과 확인 (수작업)
m_test_result = m_test[features].copy() # test feature 를 복사
m_test_result['r'] = m_unique[r_y] # 나온 결과 인텍스를 유니크값으로 환산해서 r 컬럼으로 머지

m_test # 테스트 데이터
m_test_result # 학습 모델을 적용한 결과 데이터


# 테스트한 결과로 모델을 평가 (수작업 한것과 같은 결과)
m_test[features]
preds = m_unique[clf.predict(m_test[features])]
preds[0:5]
m_test['r'].head()

pd.crosstab(m_test['r'], preds, rownames=['real r'], colnames=['Predicted r'])
list(zip(m_train[features], clf.feature_importances_)) # 가중치 확인


# 만들어진 모델로 검증할 데이터를 적용
n_df = pd.DataFrame(bset,columns=['a','b']) # 검증할데이터를 데이터프레임으로 변환
nr_y = clf.predict(n_df) # 이부분이 예측 
n_df['r'] = m_unique[nr_y] # 예측한결과를 검증할 데이터에 머지
n_df # 결과 데이터
