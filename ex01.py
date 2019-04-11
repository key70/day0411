
#  one-hot Encoding
#       ==> 기계학습을 위해서는  값이 범위가 큰것보다는 범위가 적은것이
#               훨씬 효율성이 높아요
#               값을 0,1의 상태로 만들어요

#  one-hot Encoding을 해 주는 함수
#    pandas의                get_dummies         1차원,2차원,컬럼이름도생성
#   ==> 만약 일차원배열인 경우 숫자이던 문자이던 one-hot Encoding을 만들어 줘요
#           그런데 2차원(데이터프레임)을 one-hot Encoding을 요구하면
#            기본적으로 숫자인 feature는 one-hot encoding에서 제외됨
#            만약 숫자의 feature도 one-hot Encoding을 원한다면
#               형변환을 먼저 수행해야 합니다.
#               데이터프레임[속성명] = str(데이터프레임[속성명])

#   sklearn->preprocess      Binarizer          2차원배열상대
#   sklearn->preprocess     LabelBinarizer      문자도가능, 1차원배열상대






#파일의 내용
# 39,               age
# State-gov,        workclass
# 77516,            fnlwgt
# Bachelors,        education
# 13,               education-num
# Never-married,    marital-status
# Adm-clerical,     occupation
# Not-in-family,    relationship
# White,            race
# Male,             sex
# 2174,             capital-gain
# 0,                capital-loss
# 40,               hours-per-week
# United-States,    native-country
# <=50K             income

#연습) adult.data.txt를 읽어 들여
#       연봉을 결정하는 중요한 7개의 속성으로만 추립니다.
#       숫자속성을 제외하고 one-hot Encoding으로 변경하여
#       생성된 컬럼을 확인해 봅니다.
#       7개의 속성 ==>나이,직업분류,학력,성별,주당일하는시간,직업,수입

import  numpy as np
import pandas as pd

names = ['age','workclass','fnlwgt','education','education-num','marital-status',
         'occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week',
         'native-country','income']
df = pd.read_csv("../Data/adult.data.txt", header=None, names=names)
df = df[['age','workclass','education','occupation',
         'sex','hours-per-week','income'] ]

#위의 데이터프레임을 one-hot을 하면 몇개의 feature가 생성될까요?
#workclass  의 값의 종류의 수
#education  의 값의 종류의 수
#occupation 의 값의 종류의 수
#sex의 값의 종류의 수
#income의 값의 종류의 수


# print(len(df['workclass'].unique()))        #9
# print(len(df['education'].unique()))        #16
# print(len(df['occupation'].unique()))        #15
# print(len(df['sex'].unique()))        #2
# print(len(df['income'].unique()))        #2
#
# print(9+16+15+2+2+2)        #46


new_df = pd.get_dummies(df)
#원래 갖고 있는 데이터프레임(7개의 feature)을 one-hot encoding하여
#새로운 데이터프레임을 생성함. 새로운 데이터프레임의 feature의 수는 46
#숫자를 제외한 문자속성인 칼럼은 값의 종류의 수 만큼 칼럼이 생성됩니다.

print(new_df.head())
print(new_df.columns)


#학습을 시키려면 갖고있는 데이터로 부터
#       문제와 답을 분리해요.

#연습)위의 데이터로 부터 문제는 x에
#답은 y에 담아 보세요
#뒤에서 부터 2개을 제외한 모든 속성을 문제로 하고
#맨마지막의 속성를 답으로 하려고 합니다.

#fancy Indexing
#2차원 배열인 경우 원하는 데이터를 추출하기 위하여 행열을 ,분리하여 범위를 지정하여
#       slicing할 수있어요
#   데이터프레임[행 , 열]
# pandas에 데이터프레임의   slicing을 위해서
#       loc         문자로 된 label을 지정
#       iloc        숫자index로 접근

# x = new_df.loc[:,'age':'sex_ Male']
# y = new_df.loc[:,'income_ >50K']


x = new_df.iloc[:,:-2]
y = new_df.iloc[:,-1]
# print(x.columns)
# print(y.head())

#문제와 답의 차수를 확인해 봅시다.
print(x.shape)      #(32561, 44)        2차원
print(y.shape)      #(32561,)           1차원


#문제와 답의 차수가 동일한가요?
#동일하지 않은가요?
# 둘이 차수가 달라요
#  기계학습을 시키기위한 어떤 메소드는 문제와 답의 차수가 동일하기를 기대할 수 있어요.
#   그 메세지를 파악해서 차수를 변경해야할 상황일 수 있어요.
#           ==> 차수를 바꾸는 명령은 reshape


#기계학습을 시키기 위하여(회귀분석을 해 주는 LogisticRegression을 이용해 봅시다.)
#Regression==> 회귀 분석
#sklearn의 linear_model의 LogisticRegression()

from sklearn import linear_model, model_selection

#문제x와 답y을 훈련에 참여시킬 데이타와 검증을 위한 데이터로 분리해요
train_x, test_x, train_y, test_y = model_selection.train_test_split(x,y)


# print(len(train_x), len(train_y))     #24420 24420
# print(len(test_x), len(test_y))       #8141 8141


lr  = linear_model.LogisticRegression()
lr.fit(train_x,train_y)     #훈련용 데이터와 답을 갖고 학습을 시킨다.
r = lr.predict(test_x)      #검증용 문제를 갖고 훈련이 잘 되었는지 검증합니다.

# print(len(r))
# print(len(test_y))

result = r == test_y        #예측한결과 r과 검증을 위한 진짜답 test_y를 서로 비교합니다.
a = result.values           #검증한 결과가 Series라서 value만 뽑아옵니다.
b = a[a ==True]             #True인것만 추출합니다.
print(len(b))               #8141      6638

print('정답률:',len(b)/len(test_y)*100)

#정답률을 알기위하여 LogisticRegression의 score함수를 이용해 봅시다.
print("정답률:",lr.score(test_x,test_y))









