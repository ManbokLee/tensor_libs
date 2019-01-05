from konlpy.tag import Okt
import NaiveBayesClassFier

'''
원핫 인코딩 one-hot-encoding
원핫 인코딩은 단어 집합의 크기를벡터의 크기로 잡고
표현하고 싶은 단어의 인덱스에 1의 값을 부여하고
다른 인덱스에는 0을 부여하는 단어의 표현방식
이렇게 표현된 벡터를 원-핫 벡터라고함

두 단계로 진행
1. 각 단어에 고유한 인덱스를 부여함
2. 표현하고 싶은 단어의 인덱스의 위치에는 1을 부여하고 
다른 단어의 인덱스 위치에는 0을 부여함
'''


okt = Okt() # 형태소 분석기
token = okt.morphs('나는 자연어 처리를 배운다.')
print(token)

word2index = {}
for voca in token:
    if voca not in word2index.keys():
        word2index[voca] = len(word2index)
print(word2index)

# 각 단어에 고유한 인덱스 부여

def one_hot_encoding(word, word2index):
    one_hot_vector = [0] * (len(word2index))
    index = word2index[word]
    one_hot_vector[index] = 1
    return one_hot_vector

# 단어를 입력하면 원-학 벡터를 만들어 내는 함수를 생성

one_hot_encoding("자연어", word2index)
'''
[0, 0, 1, 0, 0, 0, 0]
함수에 자연어라는 단어를 넣으면 자연어 인덱스 값인
2자리에 1이 나오고 나머지는 0으로 표기됨
'''

# one_hot_encoding("향유고래", word2index) # 없는 인덱스를 호출할 수 는 없다.







