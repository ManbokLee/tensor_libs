'''
타이타닉호의 침물 당시으 ㅣ승객 명단 데이터를 통해 생존자의 이름, 성별, 나이, 티켓요금,
생사여부의 정보를 획득합니다. 이를 분석하여 각각의 데이터들간의 연관성을 분석하여
생존에 영향을 미치는 요소를 찾아내는것
데이터는 train.csv 와 test.csv 두개가 제공됩니다.
목적데이터는 훈련데이터에서 Suvived 즉 생존여부에 대한 정보가 빠져있습니다.

즉 훈련데이터에 있는 정보를 통해서 적합한 분석 model을 구성한 뒤
이를 목적데이터에 반영하여 생존여부를 추측하는 과정을 수행하고자 합니다.
'''
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
import platform
import os
import pandas as pd
import numpy as np
import seaborn as sns

path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=path).get_name()
rc('font', family=font_name)

train = pd.read_csv(os.getcwd() + '/data/train.csv')
'''
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
'''
test = pd.read_csv(os.getcwd() + '/data/test.csv')
'''
Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
       'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')
'''
'''
'PassengerId':승객번호
'Survived': 생존여부 0: 사망 / 1: 생존
'Pclass': 티켓클래스 1 = 1st, 2 = 2nd, 3 = 3rd
'Name':
'Sex':male / female
'Age':
'SibSp': 동반한 형재, 자매, 배우자 수
'Parch': 동반한 부모, 자식 수
'Ticket': 티켓 고유넘버
'Fare': 티켓 요금
'Cabin': 객실번호
'Embarked': 승선한 항구명C = Cherbourg, Q = Queenstown, S = Southampton
'''


f, ax = plt.subplots(1,2, figsize = (18,8))
train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow = True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived', data = train, ax = ax[1])
ax[1].set_title('Survived')
plt.show()
'''
탐승객의 60% 이상이 사망했슴 (0 사망, 1 생존)
'''

f ,ax = plt.subplots(1,2, figsize = (18, 8))
train['Survived'][train['Sex'] == 'male'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow = True)
train['Survived'][train['Sex'] == 'female'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[1], shadow = True)
ax[0].set_title('남성 생존자')
ax[1].set_title('여성 생존자')
plt.show()
'''
남자의 사망율은 80%, 여자의 사망율은 25%
'''

# 성별과 객실 클래스와의 관계 시트 생성 하려고 할 때 , crosstab 사용함
df_1 = [train['Sex'], train['Survived']]
df_2 = train['Pclass']

pd.crosstab(df_1, df_2, margins = True)
'''
Pclass             1    2    3  All
Sex    Survived                    
female 0           3    6   72   81
       1          91   70   72  233
male   0          77   91  300  468
       1          45   17   47  109
All              216  184  491  891
'''
# 1등객실 여성의 생존률은 91/94 = 97%
# 3등객실 여성의 생존률은 50%
# 1등객실 남성의 생존율은 37%
# 3등객실 남성의 생존률은 13%


# 배를 탄 항구와의 연관성 추출
f, ax = plt.subplots(2,2, figsize = (20, 15))
sns.countplot('Embarked', data=train, ax = ax[0,0])
ax[0,0].set_title('승선한 인원')
sns.countplot('Embarked', hue='Sex', data=train, ax = ax[0,1])
ax[0,1].set_title('승선한 성별')
sns.countplot('Embarked', hue='Survived', data=train, ax = ax[1,0])
ax[1,0].set_title('승선한 항구 VS 생존자')
sns.countplot('Embarked', hue='Pclass', data=train, ax = ax[1,1])
ax[1,1].set_title('승선한 항구 VS 객실등급')
plt.show()
'''
절만 이상의 승객이 사우스햄프턴에서 배를 탔으며, 이기에 탑승한 승객의 70% 가량이 남성이었습니다.
남성의 사망률이 어성보다 훨씬 높았으므로 사우스 햄프턴에서 
탑승한 승객의 사망률이 높게 나왔습니다.
갬브릿지에서 탑승한 승객들은 1등ㅊ 객실 승객의 비중 및 생존률이 높은 것을 ㅗ보아
이 동네는 부자동네임을 짐작하게 합니다.
'''

# **********************************************
# 결과 도출을 위한 전처리 (Pre-Processing)
# **********************************************
'''
모든 요소들의 상관관계를 고려할 때, 성별, 객실등급, 탑승항구 세가지 정보를 가지고 생존률을 비교하는
모들엘 만드는 것이 가장 합리적이다.
'''
# 모델을 만들 때 가장 우선하는 프로세스는 결측값(NaN) 제거이다.
train.info
train.isnull().sum()
'''
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177 -> 나이는 생존률에 민감하므로 임의의 데이터로 채운다.
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687 -> 객실번호는 임의의 데이터로 산정하기 어렵고, 결측지가 너무 많아 제거하기로 한다.
Embarked         2 -> 승선한 항구 2개의 결측치는 수가 적으므로 임의의 값으로 대체한다.
dtype: int64
'''
sns.set()




def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    ded = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, ded])
    df.index = ['Survived','Ded']
    df.plot(kind='bar', stacked= True, figsize=(10, 5))
    plt.show()

bar_chart('Sex')
bar_chart('Pclass') # 사망자는 3등석, 생존자는 1등석
bar_chart('SibSp') # 동반한 형제자매, 배우자수 
bar_chart('Parch') # 동반한 부모, 자식수 
bar_chart('Embarked') # 승선한 항구
# S, Q 에 탑승한 사람이 더 많이 사망했고, C는 덜 사망했다.

'''
Feature Engineering 은 머신러닝 알고리즘을 작동하기 위해
데이터에 특징을 만드는 과정.
모델의 성능을 높이기 위해 모델에 입력할 데이터를 만들기 ㅜ이해
주어진 초기 데이터로부터 특징을 가공하고
생성하는 전체 과정을 의미합니다.
'''
'''
위 정보에서 얻을 수 있는 사실은 아래와 같습니다.
1. Age의 약 20프의 데이터가 Null로 되어있다.
2. Cabin의 대부분 값은 Null이다.
3. Name, Sex, Ticket, Cabin, Embarked는 숫자가 아닌 문자 값이다.
   - 연관성 없는 데이터는 삭제하거나 숫자로 바꿀 예정입니다.
     (머신러닝은 숫자를 인식하기 때문입니다.)
그리고 이를 바탕으로 이렇게 데이터를 가공해 보겠습니다.
1. Cabin과 Ticket 두 값은 삭제한다.(값이 비어있고 연관성이 없다는 판단하에)
2. Embarked, Name, Sex 값은 숫자로 변경할 것 입니다.
3. Age의 Null 데이터를 채워 넣을 것입니다.
4. Age의 값의 범위를 줄일 것입니다.(큰 범위는 머신러닝 분석시 좋지 않습니다.)
5. Fare의 값도 범위를 줄일 것입니다.
'''
# Cabin, Ticket 값 섹제
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)
train.head()
test.head()

# Embarked 값 가공
s_city = train[train['Embarked'] == 'S'].shape[0]
print("S : ", s_city) # S :  644
c_city = train[train['Embarked'] == 'C'].shape[0]
print("C : ", c_city) # C :  168
q_city = train[train['Embarked'] == 'Q'].shape[0]
print("Q : ", q_city) # Q :  77
'''
대부분의 값이 S 이므로 결측값 2개도 S로 채우는 것으로 결정 
'''
train = train.fillna({'Embarked' : 'S'})
'''
S-1
C-2
Q-3
-> 변경, 머신러닝은 숫자만 인식함
'''
city_mapping = {"S": 1, "C": 2, "Q": 3, 1: 1, 2: 2, 3: 3}
train['Embarked'] = train['Embarked'].map(city_mapping)
test['Embarked'] = test['Embarked'].map(city_mapping)

# Name 값 가공하기
combine = [train, test]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand= False) 
    # ([A-Za-z]+)W. 은 정규식, [] 은 글자단위인데 알파벳만 허용함, +는 한글자, 이상은. \.은
    # 글자 뒤에 반드시 점(.) 이용
pd.crosstab(train['Title'], train['Sex'])


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col','Dr','Major','Rev','Jonkheer','Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess','Sit','Lady'],'Royal')
    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'],'Miss')
    dataset['Title'] = dataset['Title'].replace(['Mme'],'Mrs')

train[['Title','Survived']].groupby(['Title'], as_index= False).mean()



title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()

# Name 과 PassengerId 삭제
train = train.drop(['Name', 'PassengerId'], axis = 1)
test = test.drop(['Name', 'PassengerId'], axis = 1)
train = train.drop(['Age', 'Fare'], axis = 1)
test = test.drop(['Age', 'Fare'], axis = 1)
combine = [train, test]
combine
train.head()

sex_mapping = {'male': 0, 'female': 1, 0: 0, 1: 1}

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
train.head()


train_data = train.drop('Survived', axis = 1)
target = train['Survived']
train_data.shape, target.shape


train.info
target

# 현재 train dml wjdqhrk chlwhd ahepfdml ahtmq
# NaN 이 없음, 전부 숫자으로 매핑된 상황

# ***************************
# 예측 모델 생성 및 결과 제출
# ***************************

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



train.isnull().sum()
test.isnull().sum()


train['Sex']


