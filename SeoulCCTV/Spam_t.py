
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer

context = 'C:\\Users\\ezen\\Documents\\tensorflow\\studio\\SeoulCCTV\\SeoulCCTV'
file_path = context + '\\ham/0007.1999-12-14.farmer.ham.txt'
with open(file_path, 'r') as infile:
    ham_sample = infile.read()

print(ham_sample)


file_path = context + '\\spam\\0058.2003-12-21.GP.spam.txt'
with open(file_path, 'r') as infile:
    spam_sample = infile.read()

print(spam_sample)

cv = CountVectorizer(stop_words="english", max_features=500)
# stop_words 불용어 제거
# max_features 는 가장 출현빈도가 높은 상위 500개의 단어만 고려함
# 벡터화(Vectorizer) 는 문서의 행렬을 "팀 도큐먼트 행렬" 형태로 변환

emails, labels = [], []
import os
import glob
import numpy as np

file_path = context + "/spam/"
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding="ISO-8859-1") as infile:  
        emails.append(infile.read())
        labels.append(1)  #스팸메일 : 1
file_path = context + "/ham/"
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename, 'r', encoding="ISO-8859-1") as infile:  
        emails.append(infile.read())
        labels.append(0)   #정상메일 : 0

def letters_only(astr):
    return astr.isalpha()   # 숫자와 구두점 표기 제거(알파벳만 남김)

import nltk
# nltk.download('all') # 자연어 툴킷 전부 다운받음

all_names = set(names.words())  # 사람이름제거(옵션)
lemmatizer = WordNetLemmatizer()

def clean_text(docs):
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(' '.join([lemmatizer.lemmatize(word.lower())  # 스페이스 
                                                    for word in doc.split()
                                                    if letters_only(word)
                                                    and word not in all_names
            ]) )
    return cleaned_docs

cleaned_emails = clean_text(emails)
term_docs = cv.fit_transform(cleaned_emails)
print(term_docs[0])
"""  (0, 481)	1
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
  희소벡터(sparse vector) : "팀 도큐먼트 행렬" 의 형태로 각 행이 문서와 메일의 용어 
  출현 빈도를 나타냄
  (row index, feature/term index) 
"""

# 용어 feature 를 key 로 , feature index(481) 을 value 로 어휘 사전 이용 가능
feature_mapping = cv.vocabulary 
feature_names = cv.get_feature_names()

print(feature_names[481])  # website
print(feature_names[357])  # read
print(feature_names[125])  # energy

# 사전확률(prior) 를 구하기 위해 레이블을 기준으로 데이터를 그룹화하기
def get_label_index(labels):
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index

"""선행기반 훈련샘플
return 값은 dictionary
key 값은 클래스 라벨
"""
# 사전확률(prior) 를 구하기
def get_prior(label_index):
    prior = {label: len(index) for label, index in label_index.items()} # { } 조심 !!
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= float(total_count)
    return prior

"""선행기반 likelihood 유사가능도
return 값은 dictionary
key 값은 클래스 라벨
"""

def get_likelihood(term_document_matrix, label_index, smoothing=0):
    likelihood = {}
    for label, index in label_index.items():
        likelihood[label] = term_document_matrix[index, :].sum(axis=0) + smoothing
        likelihood[label] = np.asarray(likelihood[label])[0]
        total_count = likelihood[label].sum() # 고쳐짐
        total_count = likelihood[label] / float(total_count)
    return likelihood

feature_names[:5]
"""['able', 'access', 'account', 'accounting', 'act']"""
"""후행기반 테스트샘플
return 값은 dictionary
key 값은 클래스 라벨
"""
# 사후확률 계산하기
"""
리턴값은 딕셔너리 데이터 타입
키값은 클래스 레이블, 밸류는 관련한 사후확률값
"""

def get_posterior(term_document_matrix, prior, likelihood):
    num_docs = term_document_matrix.shape[0]
    posteriors =[]
    for i in range(num_docs):
        # 사후 확률은 사전 확률 * 유사 가능도에 비례
        # = exp(log(사전확률 * 유사가능도))
        # = exp(log(사전확률 + log(유사가능도))

        posterior = {key: np.log(prior_label) for key, prior_label in prior.items()}  # 고침
        for label, likelihood_label in likelihood.items():
            term_document_vector = term_document_matrix.getrow(i)
            counts = term_document_vector.data
            indices = term_document_vector.indices
            for count, index in zip(counts, indices):
                posterior[label] += np.log(likelihood_label[index]) * count
                # exp(-1000):exp(-999) 는 분모가 0이 되는 문제를 유발한다
                # 하지만 이것은 exp(0):exp(1) 과 값이 같다
            min_log_posterior = min(posterior.values())
            for label in posterior:
                try:
                    posterior[label] = np.exp(posterior[label] - min_log_posterior)
                except:
                    # 어떤 값의 로그 치환값이 지나치게 클 경우
                    # 이 값에는 무한대를 의미하는 'inf' 를 할당한다
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

"""나이브 베이즈의 이러한 정규화 방법은 가짜 수(pseudocount) 1일 경우 
라플라스 정규화(Laplace smoothing)라고 불리고,
일반적으로 리드스톤 정규화(Lidstone smoothing) = 0라고 불린다.
높은 분류 성능을 얻고자 하면 1로 둔다.
"""

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

cleanded_test = clean_text(email_test)
term_docs_test = cv.transform(cleanded_test)
posterior = get_posterior(term_docs_test, prior, likelihood)
print(posterior)
"""
1: 0.9999999999593137, 
0: 4.068630618433851e-11},  
첫번째 이메일은 99.5% 가 정상 메일
{1: 1.4314819702572765e-16, 
0: 0.9999999999999999}]
두번째 이메일은 거의 100% 스팸메일로 나왔음
두 결과 모두 올바른 예측
"""