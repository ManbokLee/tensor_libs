from NaiveBayesClassFier import NaiveBayesClassfier

model = NaiveBayesClassfier()
model.train(trainfile_path='./data/review_train.csv')
model.classify('우와 짱짱맨이다.')
model.classify('이건 뭐 영화도 아니구 망작이구먼')
model.classify('헐 이런영화가 한국에 나오다니 대박이다.')
model.classify('최악이네')
model.classify('내 인생영화다')
model.classify('니 인생영화일거같지 천만의 말씀 만만의 콩떡')
model.classify('이것은 좋은 평이다. 영화 재미있다 ')
model.classify('애인이랑 같이봐라 두번봐라')
model.classify('히야.. 잘잡네 꼬라지하고는 다신 안본다.')
model.classify('없는 유형은 못잡네')
model.classify('눈누난나 랄라라')
model.classify('다크나이트')
model.classify('내 인생 최고의 영화')
model.classify('똥 똥 똥 똥 똥 똥 똥 똥 똥 똥 ')
model.classify('내 인생 최고의 망작 영화')


