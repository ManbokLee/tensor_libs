'''
서울시 범죄 현황 분석
googleMapKey = AIzaSyCUgvJXU3Iqv7VPyr-k86a7JscjT1hpK5Q

'''

from matplotlib import pyplot as plt
from matplotlib import font_manager, rc
import platform
import os
import pandas as pd
import xlrd
import numpy as np
import googlemaps 
import time
from sklearn import preprocessing
import seaborn as sns
import folium
import platform
import json

path = 'C:/Windows/Fonts/malgun.ttf'
font_name = font_manager.FontProperties(fname=path).get_name()
print(font_name)
rc('font', family=font_name)

crime_anal_police = pd.read_csv(os.getcwd() + '/crime_in_Seoul.csv',thousands=',', encoding='euc-kr')

crime_anal_police.columns
'''
Index(['관서명', '살인 발생', '살인 검거', '강도 발생', '강도 검거', '강간 발생', '강간 검거', '절도 발생',
       '절도 검거', '폭력 발생', '폭력 검거'],
      dtype='object')
'''


gmaps_key = "AIzaSyCUgvJXU3Iqv7VPyr-k86a7JscjT1hpK5Q"

gmaps = googlemaps.Client(key = gmaps_key)
gmaps.geocode('서울중부경찰서', language='ko')
'''
[{'address_components': [{'long_name': '２７', 'short_name': '２７', 'types': ['premise']}, {'long_name': '수표로', 'short_name': '수표로', 'types': ['political', 'sublocality', 'sublocality_level_4']}, {'long_name': '을지로동', 'short_name': '을지로동', 'types': ['political', 'sublocality', 'sublocality_level_2']}, {'long_name': '중구', 'short_name': '중구', 'types': ['political', 'sublocality', 'sublocality_level_1']}, {'long_name': '서울특별시', 'short_name': '서울특별시', 'types': ['administrative_area_level_1', 'political']}, {'long_name': '대한민국', 'short_name': 'KR', 'types': ['country', 'political']}, {'long_name': '100-032', 'short_name': '100-032', 'types': ['postal_code']}], 'formatted_address': '대한민국 서울특별시 중구 을지로동 수표로 27', 'geometry': {'location': {'lat': 37.5636465, 'lng': 126.9895796}, 'location_type': 'ROOFTOP', 'viewport': {'northeast': {'lat': 37.56499548029149, 'lng': 126.9909285802915}, 'southwest': {'lat': 37.56229751970849, 'lng': 126.9882306197085}}}, 'place_id': 'ChIJc-9q5uSifDURLhQmr5wkXmc', 'plus_code': {'compound_code': 'HX7Q+FR 대한민국 서울특별시', 'global_code': '8Q98HX7Q+FR'}, 'types': ['establishment', 'point_of_interest', 'police']}]
'''
station_name = []
for name in crime_anal_police['관서명']:
    station_name.append('서울' + str(name[:-1] + '경찰서'))

station_name


station_address = []
station_lat = []
station_lng = []
for name in station_name:
    tmp = gmaps.geocode(name, language='ko')
    station_address.append(tmp[0].get('formatted_address'))
    tmp_loc = tmp[0].get('geometry')
    station_lat.append(tmp_loc['location']['lat'])
    station_lng.append(tmp_loc['location']['lng'])
    time.sleep(0.5)
    print(name + '---->' + tmp[0].get('formatted_address'))

gu_name = []
for name in station_address:
    tmp = name.split()
    tmp_gu = [gu for gu in tmp if gu[-1] == '구'][0]
    print('*****' + tmp_gu)
    gu_name.append(tmp_gu)
gu_name


type(crime_anal_police) # <class 'pandas.core.frame.DataFrame'>
len(crime_anal_police['관서명']) #31
type(gu_name) # <class 'list'>
len(gu_name) # 31

crime_anal_police['구별'] = gu_name
crime_anal_police.head()

# 금천결찰서는 관악구 위치에 있어서 금천서는 예외처리 -> 금천구로 처리
crime_anal_police.loc[crime_anal_police['관서명'] == '금천서', ['구별']] = '금천구'
crime_anal_police[crime_anal_police['관서명'] == '금천서']

# 중간에 에러가 나서 계속 데이터를 제작하는 것을 방지하기 위시
# 2번 데이터로 저장

crime_anal_police.to_csv(os.getcwd() + '/crime_in_Seoul_conv.csv', encoding='UTF-8')

crime_anal_police2 = pd.read_csv(os.getcwd() + '/crime_in_Seoul_conv.csv', encoding='UTF-8')
crime_anal_police2

# 관서별로 되어있는 것을 구별로 바꾸는 작업

crime_anal_police3 = pd.pivot_table(crime_anal_police2, index='구별', aggfunc = np.sum)
crime_anal_police3

police = crime_anal_police3

police.columns
'''
Index(['Unnamed: 0', '강간 검거', '강간 발생', '강도 검거', '강도 발생', '살인 검거', '살인 발생',
       '절도 검거', '절도 발생', '폭력 검거', '폭력 발생'],
'''
police['강간검거율'] = police['강간 검거'] / police['강간 발생'] * 100
police['강도검거율'] = police['강도 검거'] / police['강도 발생'] * 100
police['살인검거율'] = police['살인 검거'] / police['살인 발생'] * 100
police['절도검거율'] = police['절도 검거'] / police['절도 발생'] * 100
police['폭력검거율'] = police['폭력 검거'] / police['폭력 발생'] * 100

del police['강간 검거']
del police['강도 검거']
del police['살인 검거']
del police['절도 검거']
del police['폭력 검거']

con_list = ['강간검거율','강도검거율','살인검거율','절도검거율','폭력검거율']

for i in con_list:
    police.loc[police[i] > 100, i] = 100
    # 검거율이 100이 넘는 값이보임, 1년이상의 기간이 포함된 데이터 오류
    # 비율이 100을 넘을 수 없으니 100 오버는 그냥 100으로 처리

police.rename(columns = {
    '강간 발생': '강간', 
    '강도 발생': '강도', 
    '살인 발생': '살인', 
    '절도 발생': '절도', 
    '폭력 발생': '폭력'
    }, inplace = True)


# 숫자값으로 모델링화 
col = ['강간','강도','살인','절도','폭력']

x = police[col].values
# min_max_scale(X): 최대/최소값이 각각 1, 0이 되도록 스케일링
min_max_scalar = preprocessing.MinMaxScaler()
'''
스케일링은 자료 집합에 적용되는 전처리 과정으로 모든 자료에
선형 변환을 적용하여 전체 자료의 분포를
평균 0, 분산 1이 되도록 만드는 과정이다.
'''


x_scaled = min_max_scalar.fit_transform(x.astype(float))
police_norm = pd.DataFrame(x_scaled, columns=col, index = police.index)
# 각 컬럼별로 정규화 하기

col2 = ['강간검거율','강도검거율','살인검거율','절도검거율','폭력검거율']
police_norm[col2] = police[col2]
# 발생건수를 정규화 시켰다.
type(police_norm)




# *********************
# 데이터 시각화 하기
# *********************

sns.pairplot(police_norm,
             vars = ['강도','살인','폭력'],
             kind = 'reg',
             size = 3)

plt.show()
# 강도와 폭력, 살인과 폭력, 강도와 살인 모두 양의 강관관계가 보입니다.




data_result = pd.read_csv(os.getcwd() + '/data_result.csv', index_col="구별")
data_result
# 두개 데이터 병합 해서 하나로 처리 
data_result2 = data_result.set_index('구별') # 인구수 CCTV 관련 데이터 
data_result2
police_norm
mergeData = pd.merge(police_norm, data_result2, on='구별')

mergeData.columns
'''
Index(['강간', '강도', '살인', '절도', '폭력', '강간검거율', '강도검거율', '살인검거율', '절도검거율',
       '폭력검거율', '범죄', '검거', '소계', '2013년도 이전', '2014년', '2015년', '2016년',
       '인구수', '한국인', '외국인', '고령자', '외국인비율', '고령자비율', 'CCTV비율', '오차'],
      dtype='object')
'''

sns.pairplot(mergeData,
             x_vars = ['인구수','소계'],
             y_vars = ['살인검거율','폭력검거율'],
             kind = 'reg',
             size = 3)

plt.show()


# 한개 데이터에 다른 데이터를 이동해와서 처리
data_result2 = data_result.set_index('구별') # 인구수 CCTV 관련 데이터 
police_norm[['인구수','CCTV']] = data_result2[['인구수','소계']]

police_norm['범죄'] = np.sum(police_norm[col], axis = 1)
police_norm['검거'] = np.sum(police_norm[col2], axis = 1)

police_norm.columns
'''
Index(['강간', '강도', '살인', '절도', '폭력', '강간검거율', '강도검거율', '살인검거율', '절도검거율',
       '폭력검거율', '범죄', '검거', '인구수', 'CCTV'],
      dtype='object')
'''
sns.pairplot(police_norm,
             x_vars = ['인구수','CCTV'],
             y_vars = ['살인','강도'],
             kind = 'reg',
             height = 3)

plt.show()
# 인구수와 CCTV 개수, 그리고 살인과 강도에 대한 상관관계 분석
# 전체적인 상관계수는 CCTV 와 살인의 관계가 낮다
# 단, CCTV 가 없을 때 살인이 많이 일어나는 구간이 있습니다.
# 즉 CCTV 개수를 기준으로 좌측면에 살인과 강도의 높음수를 
# 갖는 데이터가 보인다.



sns.pairplot(police_norm,
             x_vars = ['인구수','CCTV'],
             y_vars = ['살인검거율','강도검거율'],
             kind = 'reg',
             height = 3)

plt.show()
# 인구수와 CCTV와 검거율간의 관계가 음의 관계이거나
# 별다른 관계가 없는 것으로 파악된다.


tmp_max = police_norm['검거'].max()

police_norm['검거'] = police_norm['검거'] / tmp_max * 100
police_nmorm_sort = police_norm.sort_values(by='검거', ascending = False)
police_nmorm_sort
# 검거율 합계인 검거항목 최고값을 100으로 한정하고, 그 값으로 정렬한다음 

target_col = col2


# 히트맵 그리기
plt.figure(figsize = (10,10))
sns.heatmap(police_nmorm_sort[target_col], annot = True, fmt = 'f', linewidth=5)
plt.title('범죄 검거 비율(정규화된 검거의 합으로 정렬)')
plt.show()

'''
위 결과를 보면 절도 검거율은 다른 검거율이 비해 낮다.
그래프 하단으로 갈수록 검거율이 낮다. 그 중에서 강남3구 중에서 서초구가 보임.
전반적으로 검거율이 우수한 구는 도봉구, 광진구, 성동구로 보임
'''

# 발생건수의 합으로 정렬해서 heatmap 으로 관찰
target_col = ['강간','강도','살인','절도','폭력','범죄']
police_norm['범죄'] = police_norm['범죄'] / 5 # 발생건수 평균으로 정리
police_nmorm_sort = police_norm.sort_values(by='범죄', ascending = False)

plt.figure(figsize = (10,10))
sns.heatmap(police_nmorm_sort[target_col], annot = True, fmt = 'f', linewidth=5)
plt.title('범죄 발생 비율(정규화된 발생건수로 정렬)')
plt.show()
'''
발생건수로 보니 강남구, 양천구, 영등포구가 범죄발생 건수가 높습니다.
송파구와 서초구도 낮다고 볼 수 없습니다.
정말 강남3구가 안전한가에 대한 의문이 생김
'''


# ***********************************
# 지도 시각화 도구 folium
# ***********************************

# 서울시청 주변 지도보기 테스트 
map_osm = folium.Map(location = [37.566235, 126.977828])
map_osm
map_osm.save(os.getcwd() + '/map.html')

# ***********************************
# 서울시 지도 시각화
# ***********************************

# 범죄율 지도
geo_path = os.getcwd() + '/seoul_map.json'
geo_path
geo_str = json.load(open(geo_path))
geo_str

map_1 = folium.Map(location = [37.566235, 126.977828], zoom_start=12, tiles='Stamen Toner')
map_1.save(os.getcwd() + '/seoul_map.html')




# 인구대비 범죄 발생 비율 
map_1 = folium.Map(location = [37.566235, 126.977828], zoom_start=12, tiles='Stamen Toner')
tmp_criminal = police_norm['범죄'] / police_norm['인구수'] * 1000000
tmp_criminal
map_1.choropleth(geo_data = geo_str,
                 data=tmp_criminal,
                 columns=[police_norm.index,tmp_criminal],
                 fill_color = 'PuRd',
                 key_on= 'feature.id'
                 )
map_1.save(os.getcwd() + '/choropleth.html')




# 경찰서 지도 기존 arr 로

map = folium.Map(location = [37.566235, 126.977828], zoom_start=12)
for i in range(len(station_address)):
    print(station_address[i])
    print(station_lat[i])
    print(station_lng[i])
    folium.Marker([station_lat[i], station_lng[i]], popup=station_address[i]).add_to(map)
map.save(os.getcwd() + '/seoul_map_with_police.html')



# 경찰서 지도 변경된 값으로 
crime_anal_police
police_position = crime_anal_police
police_position
police_position['lat'] = station_lat
police_position['lng'] = station_lng
police_position.columns
police_position
'''
Index(['관서명', '살인 발생', '살인 검거', '강도 발생', '강도 검거', '강간 발생', '강간 검거', '절도 발생',
       '절도 검거', '폭력 발생', '폭력 검거', '구별', 'lat', 'lng'],
      dtype='object')
'''
col = ['살인 검거','강도 검거','강간 검거','절도 검거','폭력 검거']
tmp = police_position[col] / police_position[col].max()
police_position['검거'] = np.sum(tmp, axis = 1)

map_2 = folium.Map(location = [37.566235, 126.977828], zoom_start=12)
for i in police_position.index:
    folium.CircleMarker(location = [police_position.loc[i, 'lat'],police_position.loc[i, 'lng']],
                        radius = police_position.loc[i, '검거'] * 10, 
                        color = '#3186cc',
                        fill_color = '#3186cc',
                        popup=police_position.loc[i, '관서명']).add_to(map_2)
  
map_2.save(os.getcwd() + '/seoul_map_with_police_name.html')

















