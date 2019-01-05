from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
import platform
import os
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing

path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=path).get_name()
rc('font', family=font_name)

# *************
# 데이터 마이닝
# *************
titanic_df = pd.read_csv(os.getcwd() + '/data/train.csv')
titanic_df.head()

# ***************
# 데이터 프로세싱
# 파이선의 함수로 정의
# ***************

titanic_df['Sex']
titanic_df['Embarked']


def get_titles(df):
    df = df.copy()
    df['title'] = df.Name.str.extract(' ([A-z]+?)\.', expand=True)
    df['title'].replace(
        ['Lady', 'Countess','Capt', 'Col','Don',
         'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
        'rare',
        inplace=True)
    df['title'].replace('Mme', 'Mrs', inplace=True)
    df['title'].replace('Ms', 'Miss', inplace=True)
    df['title'].replace('Mlle', 'Miss', inplace=True)
    return df

def preprocess_df(df):
    df = df.copy()
    df['name_len'] = df['Name'].apply(len)
    df['has_cabin'] = df['Cabin'].apply(
    lambda x: 0 if isinstance(x, float) else 1)
    df['not_alone'] = df['SibSp'] | df['Parch']
    df.drop(['Ticket', 'Cabin', 'Name'], axis=1, inplace=True)
    df['Age'].fillna(np.median(df['Age'].dropna()), inplace=True)
    df['Fare'].fillna(np.median(df['Fare'].dropna()), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    df = pd.get_dummies(df)
    return df



training_data = preprocess_df(
    pd.read_csv(os.getcwd() + '/data/train.csv', index_col='PassengerId'))
y = training_data.pop('Survived')
X = training_data.values
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=10200)

params = dict(max_features=range(5, len(X_train.T), 2),
              max_depth=(None, 20, 10, 5))
rf = RandomForestClassifier(random_state=101, n_estimators=100)
gs_rf = GridSearchCV(estimator=rf, param_grid=params, cv=5).fit(X_train, y_train)
print(gs_rf.best_params_)
print(gs_rf.score(X_test, y_test))

rf_all = RandomForestClassifier(random_state=2101)
gs_rf_all = GridSearchCV(estimator=rf, param_grid=params, cv=5).fit(X, y)
print(gs_rf.best_params_)

test_data = preprocess_df(
    pd.read_csv(os.getcwd() + '/data/test.csv', index_col='PassengerId'))
X_test_proper = test_data.values
test_data.head()
test_data.index


submission = pd.DataFrame({
        "PassengerId": test_data.index,
        "Survived": gs_rf_all.predict(X_test_proper)
    })
submission.to_csv(os.getcwd() + '/data/submission.csv', index=False)
submission.head()


# 데이터 전처리
titanic_df['Survived'].mean()
titanic_df.groupby('Pclass').mean()
class_sex_grouping = titanic_df.groupby(['Pclass','Sec']).mean()
group_by_age = pd.cut(titanic_df['Age', np.arrange(0,90,10)]) # 0~90 10 단위
age_grouping = titanic_df.groupby(group_by_age).mean()

titanic_df.count()
titanic_df = titanic_df.dropna() # NA 값 삭제
titanic_df.count()

preprocess_df = preprocess_titanic_df(titanic_df)
preprocess_df



