from sklearn.model_selection import train_test_split

t_text = [
    '스팸 전화',
    '마진 긴급',
    '마진 긴급',
    '마진 긴급',
    '마진 긴급',
    '마진 긴급',
    '마진 긴급',
    '형 오빠',
    '형 오빠',
    '형 오빠',
    '형 오빠',
    '형 오빠',
    '형 오빠',
    '형 오빠',
    '형 오빠',
    '삼촌 동생'
    ]
t_label = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1
    ]
len(t_text)
len(t_label)


X_train, X_test, Y_train, Y_test = train_test_split(t_text, t_label, test_size=0.33, random_state=42)
# 테스트 세트는 1/3, 랜덤값 시드 42

len(X_train),len(Y_train)
len(X_test),len(Y_test)

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB(alpha=1.0, fit_prior= True)
# 설정값이 1인 스무딩 파라미터를 사이킷 런에서는 alpha 라고 지칭함
# 학습데이터세트를 가지고 학습된 사전확률 prior 를 사이킷런에서는 fit_prior 라고 지칭함
term_docs_train = cv.fit_transform(X_train)
term_docs_test = cv.transform(X_test)
clf.fit(term_docs_train, Y_train) # fit 함수를 이용한 분류기 학습

prediction_prob = clf.predict_proba(term_docs_test) # 예측결과 계산
prediction_prob[0:10]
'''
array([[1.00000000e+00, 2.12716600e-10],
       [1.00000000e+00, 2.72887131e-75],
       [6.34671963e-01, 3.65328037e-01],
       [1.00000000e+00, 1.67181666e-12],
       [1.00000000e+00, 4.15341124e-12],
       [1.37860327e-04, 9.99862140e-01],
       [0.00000000e+00, 1.00000000e+00],
       [1.00000000e+00, 1.07066506e-18],
       [1.00000000e+00, 2.02235745e-13],
       [3.03193335e-01, 6.96806665e-01]])
'''
# 0.5는 기본 임계치, 만약 클래스의 예측확률이 0.5보다 크면
# 클래스1에 할당됨, 그렇지 않으면 클래스0에 할당됨


prediction = clf.predict(term_docs_test)
prediction[0:10]
'''
array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1])
'''

# score 함수를 이요한 성능 정확도 측정
accuracy = clf.score(term_docs_test, Y_test)
accuracy

print("The accuracy in {0} 나이브 베이즈 is: {1:.1f}% ".format(len(prediction), accuracy * 100))



