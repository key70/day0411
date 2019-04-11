
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

new_df = pd.get_dummies(df)

print(new_df.head())
print(new_df.columns)


x = new_df.iloc[:,:-2]
y = new_df.iloc[:,-1]


#문제와 답의 차수를 확인해 봅시다.
print(x.shape)      #(32561, 44)        2차원
print(y.shape)      #(32561,)           1차원



train_x, test_x, train_y, test_y = model_selection.train_test_split(x,y)


lr  = linear_model.LogisticRegression()
lr.fit(train_x,train_y)     #훈련용 데이터와 답을 갖고 학습을 시킨다.


n = [[47, ' Private', ' Prof-school',' Prof-specialty', ' Female',' White',60, ' <=50K']]
n_df = pd.DataFrame(n, columns=['age','workclass','education','occupation',
         'sex','race','hours-per-week','income'])

df2 = df.append(n_df)

#알고자하는 데이터를 훈련시킨 feature의 수와 동일하게 하기 위하여
#원래 원본데이터의 맨마지막에 추가시키고
#one-hot Encoding을 합시다.

one_hot = pd.get_dummies(df2)
print(len(one_hot.columns))         #51
print(len(new_df.columns))          #51


pred_x = np.array( one_hot.iloc[-1, :-2]).reshape(1,-1)
pred_y = lr.predict(pred_x)
print(pred_y)

# n_df = pd.DataFrame(n, columns=['age','workclass','education','occupation',
#          'sex','race','hours-per-week','income'])

#연습) 고객의 나이, 직업분류, 학력, 직업, 성별, 인종, 주당근무시간을
#           입력받아 연봉이 50000달러 이상이면 "대출가능"
#           그렇지 않으면 "대출불가능"을 출력하는 웹어프리케션을 구현합니다.
#               단, 직업분류, 학력, 직업, 성별, 인종은
#                   우리가 훈련시킬 데이터 adult.data.txt의 내용으로 제한하도록 합니다.

















