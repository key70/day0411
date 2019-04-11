import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model, model_selection


#문자열 데이터의 각각의 값의 범위(도메인)을 반환하는 함수를 정의
def getDomain():
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
             'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
             'income']
    df = pd.read_csv('../Data/adult.data.txt', header=None, names=names)
    df = df[['age', 'workclass', 'education', 'occupation', 'race', 'sex', 'hours-per-week', 'income']]
    workclass = df['workclass'].unique()
    education = df['education'].unique()
    occupation = df['occupation'].unique()
    race = df['race'].unique()
    sex = df['sex'].unique()

    return workclass, education, occupation, race, sex


def adult_d(age,workclass,education,occupation,race,sex,hoursperweek):
    names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation',
             'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
    df = pd.read_csv('../Data/adult.data.txt',header=None,names=names)
    df = df[['age','workclass','education','occupation','race','sex','hours-per-week','income']]

    new_df = pd.get_dummies(df)

    x = new_df.iloc[:,:-2]
    y = new_df.iloc[:,-1]

    train_x, test_x, train_y, test_y = model_selection.train_test_split(x,y)

    lr = linear_model.LogisticRegression()
    lr.fit(train_x,train_y)         # 훈련은 위한 데이터 fit..

    #내가 알고 싶은 데이터


    n = [[int(age),workclass,education,occupation,race,sex,int(hoursperweek),' <=50K']]
    n_df = pd.DataFrame(n,columns=['age','workclass','education','occupation','race','sex','hours-per-week','income'])

    df2 = df.append(n_df)

    # print(df2.iloc[-1,:])
    one_hot= pd.get_dummies(df2)
    # print(one_hot.tail(1))
    pred_x = np.array(one_hot.iloc[-1,:-2]).reshape(1,-1)
    pred_y = lr.predict(pred_x)
    print(pred_y)
    return pred_y
    # return "a"





