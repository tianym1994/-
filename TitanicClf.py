import pandas as pd
#数据加载
test_data=pd.read_csv('TitanicTest.csv')
train_data=pd.read_csv('TitanicTrain.csv')
#数据探索
print(train_data.info())
print('-'*30)
print(train_data.describe())
print('-'*30)
print(train_data.describe(include=['O']))
print('-'*30)
print(train_data.head())
print('-'*30)
print(train_data.tail())
#使用平均年龄来填充年龄中的NAN值
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
#使用票价的均值填充票价中的NAN值
train_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

#打印Embarked字段的取值
print(train_data['Embarked'].value_counts())
train_data['Embarked'].fillna('S',inplace=True)
test_data['Embarked'].fillna('S',inplace=True)
