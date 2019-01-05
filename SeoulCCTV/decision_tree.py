import sklearn
from sklearn.tree import DecisionTreeClassifier
import sklearn.datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz

cancer = load_breast_cancer()
cancer

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify= cancer.target, random_state=42)
# train_test_split(data, data2, test_size, train_size, random_state)
'''
data: 독립 변수 데이터 배열 또는 pandas 데이터프레임
data2: 종속변수 데이터, data 인수에 종속 변수 데이터가 같이 있으면 생략할 수 있다.
test_size: 검증용 데이터 갯수, 1보다 작은 실수이면 비율을 나타낸다.
train_size: 학습용 데이터 개수, 1보다 작은 실수이면 비율을 나타낸다. test_size 와 train_size는 하나만 있어도 된다.
random_state: 난수 시드(계속 동일한 난수가 생성되도록 함)
'''
tree = DecisionTreeClassifier(max_depth=4, random_state = 42)
tree
tree.fit(X_train, y_train)
print('훈련 세트 정확도: {:.3f}'.format(tree.score(X_train, y_train)))
print('테스트 세트 정확도: {:.3f}'.format(tree.score(X_test, y_test)))

'''
표본내 성능검증(in-sample testing): 회귀분석 성능은 학습데이터 집합의 종속변수(y) 갑의 예측 정확도를 
            결정계수(coefficient of datermination)등을 이용하여 따지는 검증
표본외 성능검증(out-of-sample testing) 혹은 교차검증(cross validation): 
    회귀분석 모형을 만드는 목적 중 하나는 종속변수의 값을 아직 알지 못하고 따라서 학습에
    사용하지 않은 표본에 대한 종속 변수의 값을 알아내고자 하는 것, 즉 예측(prediction)이다.
    이렇게 학습에 쓰이지 않는 표본 데이터 집합의 종속변수값을 얼마나 잘 예측하는가를 검사하는 것
                                    
'''
export_graphviz(tree, out_file="tree.dot", class_names=["악성", "양성"],
                feature_names=cancer.feature_names,
                impurity=False, filled=True)
with open("tree.dot") as f:
    dot_graph = f.read()
display(graphviz.Source(dot_graph))


