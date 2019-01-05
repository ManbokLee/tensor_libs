from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import names # 자연어 문장을 처리하는 라이브러리
from nltk.stem import WordNetLemmatizer
import nltk

context = 'C:\\Users\\ezen\\Documents\\tensorflow\\studio\\SeoulCCTV\\SeoulCCTV'
file_path = context + '\\ham/0007.1999-12-14.farmer.ham.txt'
with open(file_path, 'r') as infile:
    ham_sample = infile.read()

print(ham_sample)


file_path = context + '\\spam\\0058.2003-12-21.GP.spam.txt'
with open(file_path, 'r') as infile:
    spam_sample = infile.read()

print(spam_sample)

cv = CountVectorizer(stop_words='english', max_features=500)
# stop_words 불용어(사전에 등록되지않은 단어) 제거
# max_features 는 가장 출현빈도가 높은 상위 500개의 단어만 고려함
# 벡터화(Vectorizer)는 문서의 행렬을 "팀 도큐먼트 행렬" 형태로 변환

emails, labels = [],[]

import os
import glob
import numpy as np

file_path = context + "\\spam"
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding='ISO-8859-1') as infile:
        emails.append(infile.read())
        labels.append(1) # 스팸메일: 1


file_path = context + "\\ham"
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding='ISO-8859-1') as infile:
        emails.append(infile.read())
        labels.append(0)


def letters_only(astr):
    return astr.isalpha() # 숫자와 구두점 표기 제거 (알파벳만 남김)

len(emails)
len(labels)

emails[5000]
labels[5000]


# nltk.download('all') # 자연어 툴킷 전부 다운받음

all_names = set(names.words()) # 사람이름제거(옵션)
lemmatizer = WordNetLemmatizer()

def clean_text(docs):
    cleaned_docs = []
    for doc in docs:
         cleaned_docs.append(' '.join([lemmatizer.lemmatize(word.lower())
                                                    for word in doc.split()
                                                    if letters_only(word)
                                                    and word not in all_names
            ]) )
    return cleaned_docs


cleaned_emails = clean_text(emails)
term_docs = cv.fit_transform(cleaned_emails)
print(term_docs[0])
'''
  (0, 481)	1
  (0, 357)	1
  (0, 69)	1
  (0, 285)	1
  (0, 424)	1
  (0, 250)	1
  (0, 345)	1
  (0, 445)	1
  (0, 231)	1
  (0, 497)	1
  (0, 47)	1
  (0, 178)	2
  (0, 125)	2
  희소벡터(sparse vector): "팀 도큐먼트 행렬" 의 형태로 각 행이 문서와 메일의 용어의
  출현빈도를 나타냄
  (row index, feature/term index)
'''

# 용어 feature 를 key로 feature index(481)를 value로 어휘 사전 이용 가능
feature_mapping = cv.vocabulary
feature_names = cv.get_feature_names()
print(feature_names[481]) # websie
print(feature_names[357]) # read
print(feature_names[125]) # energy

# 사전확률(prior)를 구하기 위해 레이블을 기준으로 데이터를 그룹화하기
def get_label_index(labels):
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index

'''
선행기반 훈령샘플
return 값은 dictionary
key 값은 클래스 라벨
'''
# 사전확률(prior)를 구하기
def get_prior(label_index):
    prior = {label:len(index) for label, index in label_index.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= float(total_count)
    return prior

'''
선행기반 likelihood 유사가능도
return 값은 dictionary
key 값은 클래스 라벨
'''
def get_likelihood(term_document_matrix, label_index, smoothing=0):
    likelihood = {}
    for label, index in label_index.items():
        likelihood[label] = term_document_matrix[index,:].sum(axis=0) + smoothing
        likelihood[label] = np.asarray(likelihood[label])[0]
        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label] / float(total_count)
    return likelihood


feature_names[:5]

# 사후확률 계산하기 
'''
리턴값은 딕셔너리 데이터 타입
키값은 글래스레이블, 벨류는 관련한 사후확률값
'''
def get_posterior(term_document_matrix, prior, likelihood):
    num_docs = term_document_matrix.shape[0]
    posteriors = []
    for i in range(num_docs):
        # 사후 확률은 사전확률 * 유가 사능도에 비례
        # = exp(log(사전확률 * 유사가능도))
        # = exp(log(사전확률 * log(유사가능도)))
        posterior = {key: np.log(prior_label) 
                     for key, prior_label in prior.items()}
        for label, likelihood_label in likelihood.items():
            term_document_vector = term_document_matrix.getrow(i)
            counts = term_document_vector.data
            indices = term_document_vector.indices
            for count, index in zip(counts, indices):
                posterior[label] += np.log(likelihood_label[index]) * count
                # exp(-1000):exp(-999) 는 문보가 0이 되는 문제를 유발한다.
                # 하지만 이것은 exp(0):exp(1) 과 값이 같다.
        min_log_posterior = min(posterior.values())
        for label in posterior:
            try:
                posterior[label] = np.exp(posterior[label] - min_log_posterior)
            except:
                # 어떤 값의 로그 치환값이 지나치게 클 경우
                # 이 값에는 무한대를 의미하는 'inf'를 할당한다.
                posterior[label] = float('inf')
        # 전체 합이 1이 되도록 정규화 한다
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors


label_index = get_label_index(labels)
prior = get_prior(label_index)


# 높은 분류 성능을 얻고자 하면 1로 둔다. 값은 1 or 0
# 1: 라플라스 정규화 
# 0: 리드스톤 정규화
smoothing = 1
likelihood = get_likelihood(term_docs, label_index, smoothing)

email_test = [
     '''Subject: flat screens
    hello ,
    please call or contact regarding the other flat screens requested .
    trisha tlapek - eb 3132 b
    michael sergeev - eb 3132 a
    also the sun blocker that was taken away from eb 3131 a .
    trisha should two monitors also michael .
    thanks
    kevin moore''',
    '''Subject: having problems in bed ? we can help !
    cialis allows men to enjoy a fully normal sex life without having to plan the sexual act .
    if we let things terrify us , life will not be worth living .
    brevity is the soul of lingerie .
    suspicion always haunts the guilty mind .'''
    ]

clean_test = clean_text(email_test)
term_doxs_text = cv.transform(clean_test)
posterior = get_posterior(term_doxs_text, prior, likelihood)
print(posterior)
'''
[
    {1: 0.004531124550707277, 0: 0.9954688754492927}, 
    {1: 0.9996384394815188, 0: 0.0003615605184812136}
]

'''


# ******************
# 학습 learning
# ******************

from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split(cleaned_emails, labels, test_size=0.33, random_state=42)
# 테스트 세트는 1/3, 랜덤값 시드 42

len(X_train),len(Y_train)
len(X_test),len(Y_test)


term_docs_train = cv.fit_transform(X_train)
label_index = get_label_index(Y_train)
prior = get_prior(label_index)
likelihood = get_likelihood(term_docs_train, label_index, smoothing)

term_docs_test = cv.transform(X_test)
a = term_docs_test.getrow(1)

posterior = get_posterior(term_docs_test, prior, likelihood)

correct = 0.0
for pred, actual in zip(posterior, Y_test):
    if actual == 1:
        if pred[1] >= 0.5:
            correct += 1
    elif pred[0] >= 0.5:
        correct += 1

# 정확도 accuracy
print("The accuracy in {0} testing samples is: {1:.1f}% ".format(len(Y_test), correct/len(Y_test) * 100))
'''
The accuracy in 1707 testing samples is: 92.0% 
'''









from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB(alpha=1.0, fit_prior= True)
# 설정값이 1인 스무딩 파라미터를 사이킷 런에서는 alpha 라고 지칭함
# 학습데이터세트를 가지고 학습된 사전확률 prior 를 사이킷런에서는 fit_prior 라고 지칭함

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



