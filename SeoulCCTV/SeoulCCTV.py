# *******************************************************
# 서울시 각 구별 CCTV 수를 파악하고, 인구대비 CCTV 비율을
# 파악해서 순위를 비교한 후 관련 그래프를 작성하라.
# *******************************************************

#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
import platform
import os
import pandas as pd
import xlrd



# CCTV 현황 데이터 출처 : 
seoul_cctv = pd.read_csv(os.getcwd() + '/seCCTV.csv', encoding='UTF-8')

seoul_cctv_index = seoul_cctv.columns
'''
print(seoul_cctv_index)
Index(['기관명', '소계', '2013년도 이전', '2014년', '2015년', '2016년'], dtype='object')
'''

seoul_cctv.rename(columns={seoul_cctv.columns[0]: '구별'}, inplace = True) # inplace = True : 실제 변수의 내용



# 서울시 구별 인구 데이터 출처 : https://data.seoul.go.kr/dataList/datasetView.do?serviceKind=2&infId=419&srvType=S&stcSrl=419
# excel 파일은 xlrd 모듈이 필요함.
seoul_pop = pd.read_excel(os.getcwd() + '/seoul_pop.xls', encoding='UTF-8'
                          , header=2, usecols='B,D,G,J,N')

seoul_pop.rename(columns={
    seoul_pop.columns[0]: '구별', 
    seoul_pop.columns[1]: '인구수',
    seoul_pop.columns[2]: '한국인',
    seoul_pop.columns[3]: '외국인',
    seoul_pop.columns[4]: '고령자'
    }, inplace = True)

seoul_pop.head()

import numpy as np
seoul_cctv.sort_values(by='소계', ascending = True).head()
'''
CCTV 의 전체 갯수가 가장 적은구는 도봉구, 강북구, 광진구, 강서구, 중랑구
'''

seoul_cctv.sort_values(by='소계', ascending = False).head()
'''
CCTV 의 전체 갯수가 가장 많은구는 강남구, 양천구, 서초구 , 안악구 은평구
'''

# 서울 인규표의 0번재 합계는 필요없는 값 -> 제거
seoul_pop.drop([0], inplace = True)
seoul_pop.head()

#전체 구의 목록을 출력
seoul_pop['구별'].unique()
'''
array(['종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구', '강북구',
       '도봉구', '노원구', '은평구', '서대문구', '마포구', '양천구', '강서구', '구로구', '금천구',
       '영등포구', '동작구', '관악구', '서초구', '강남구', '송파구', '강동구'], dtype=object)
'''

# NaN 값을 제거 
seoul_pop[seoul_pop['구별'].isnull()]
# TODO 제거

# *******************************************
# 외국인 비율과 고령자 비율 계산
# *******************************************
seoul_pop['외국인비율'] = seoul_pop['외국인'] / seoul_pop['인구수'] * 100
seoul_pop['고령자비율'] = seoul_pop['고령자'] / seoul_pop['인구수'] * 100
seoul_pop.head()

# *******************************************
# CCTV 데이터와 인구 현황 데이터 합치고 분석하기
# *******************************************


data_result = pd.merge(seoul_cctv, seoul_pop, on='구별')
data_result.head()


# 그래프로 그리기 위해서는 구이름을 인덱스로 설정
data_result.set_index('구별', inplace = True)
data_result.head()



'''
상관계수의 절대값이 
0.1 이하면 거의 무시
0.3 이하면 약한 상관관계
0.7 이하면 뚜렷한 상관관계
상관관계를 계산하는 명령이 numpy 에 있는 corrocoef 명령
이 명령의 결과는 행렬로 나타남
주 대학선을 기준으로 대칭인 행렬이고,
대각서을 빼고 다른 값을 읽도록 함
'''

np.corrcoef(data_result['고령자비율'], data_result['소계'])
'''
array([[ 1.        , -0.27533083],
       [-0.27533083,  1.        ]])
'''

np.corrcoef(data_result['외국인비율'], data_result['소계'])
'''
array([[ 1.        , -0.04796912],
       [-0.04796912,  1.        ]])
'''

np.corrcoef(data_result['인구수'], data_result['소계'])
'''
array([[1.       , 0.2242953],
       [0.2242953, 1.       ]])
'''

# CCTV와 고령자 비율은 약한 음의 상관관계
# 외국인 비율은 상고나관계 없음
# 인구수와의 약산 양의 상관관계를 가진다

# 한글 깨짐 방지  'C:/Windows/Fonts/malgun.ttf'
path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=path).get_name()
print(font_name)
rc('font', family=font_name)


data_result['CCTV비율'] = data_result['소계'] / data_result['인구수'] * 100
data_result['CCTV비율'].sort_values().plot(kind='barh', grid = True, figsize = (10,10))

# barh 수평 바 차드
plt.show()


plt.figure(figsize=(6,6))
plt.scatter(data_result['인구수'], data_result['소계'], s=50)
plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()

# CCTV 와 인구수는 양의 상관관계이므로 직선을 그릴 수 있다.
fp1 = np.polyfit(data_result['인구수'], data_result['소계'], 1)
# polyfit 은 직선구하기 명령
'''
array([1.08180882e-03, 1.07963746e+03])
'''

f1 = np.poly1d(fp1) # y 축 데이터
fx = np.linspace(100000, 700000, 100) # x 축 데이터

plt.figure(figsize=(10,10))
plt.scatter(data_result['인구수'], data_result['소계'], s= 50)
plt.plot(fx, f1(fx), ls = 'dashed', lw = 3, color = 'g')
plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()
# 이 데이터에서 직선이 전체 데이터의 대표값 역할을 한다면
# 인구수가 300000 일때 CCTV 는 1100 대 정도여야 한다는 결론을 내리게 된다.
# 오차를 계산할 수 있는 코드를 만들고, 오차가 큰 순으로 데이터를 정렬

fp1 = np.polyfit(data_result['인구수'], data_result['소계'], 1)

f1 = np.poly1d(fp1)
fx = np.linspace(100000,700000,100)

data_result['오차'] = np.abs(data_result['소계'] - f1(data_result['인구수']))
df_sort = data_result.sort_values(by='오차', ascending = False)
df_sort.head()

plt.figure(figsize=(14,10))
plt.scatter(data_result['인구수'], data_result['소계'],c= data_result['오차'], s= 50)
plt.plot(fx, f1(fx), ls = 'dashed', lw = 3, color = 'g')

for n in range(df_sort.shape[0]):
    plt.text(df_sort['인구수'][n] * 1.02, df_sort['소계'][n] * 0.99, df_sort.index[n], fontsize = 15)

plt.xlabel('인구수')
plt.ylabel('인구당비율')

plt.colorbar()
plt.grid()
plt.show()


data_result.to_csv(os.getcwd() + '/data_result.csv')


