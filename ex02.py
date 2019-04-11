
# 어떤 feature가 그것을 결정하는 가장 중요한 요인인가를 파악중요하다.
#       그것을 결정하는데 필요한 데이터를 수집하는것이 중요

import  numpy as np
import pandas as pd
from sklearn import linear_model, model_selection

names = ['age','workclass','fnlwgt','education','education-num','marital-status',
         'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week',
         'native-country','income']
df = pd.read_csv("../Data/adult.data.txt", header=None, names=names)
df = df[['age','workclass','education','occupation',
         'sex','race','hours-per-week','income'] ]


print(df['workclass']==' State-gov')


# new_df = pd.get_dummies(df)
#
# print(new_df.head())
# print(new_df.columns)
#
#
# x = new_df.iloc[:,:-2]
# y = new_df.iloc[:,-1]
#
#
# #문제와 답의 차수를 확인해 봅시다.
# print(x.shape)      #(32561, 44)        2차원
# print(y.shape)      #(32561,)           1차원
#
#
#
# train_x, test_x, train_y, test_y = model_selection.train_test_split(x,y)
#
#
# lr  = linear_model.LogisticRegression()
# lr.fit(train_x,train_y)     #훈련용 데이터와 답을 갖고 학습을 시킨다.
# r = lr.predict(test_x)      #검증용 문제를 갖고 훈련이 잘 되었는지 검증합니다.
#
#
# result = r == test_y        #예측한결과 r과 검증을 위한 진짜답 test_y를 서로 비교합니다.
# a = result.values           #검증한 결과가 Series라서 value만 뽑아옵니다.
# b = a[a ==True]             #True인것만 추출합니다.
# print(len(b))               #8141      6638
#
# print('정답률:',len(b)/len(test_y)*100)
#
# print("정답률:",lr.score(test_x,test_y))


#실 데이터를 적용시켜 봅시다.
#47, Private, 51835, Prof-school, 15, Married-civ-spouse,
# Prof-specialty, Wife, White, Female, 0, 1902, 60, Honduras, >50K
#연습) 위데이터를 예측시켜 결과를 확인해 봅니다.








