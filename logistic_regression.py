import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

credit_data = pd.read_csv('credit_data.csv')
print(credit_data.head())
print(credit_data.describe())
print(credit_data.corr())

features = credit_data[['income', 'age', 'loan']]
print(features.head())
target = credit_data.default

# 30% of the data-set is for testing and 70% of data-set is for training
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

model = LogisticRegression()
model.fit(feature_train, target_train)

predictions = model.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))
print(model.coef_)
print(model.intercept_)

data={'income':[5000],'age':[59], 'loan':[3000]}
input_predict = pd.DataFrame(data)

print(input_predict)
print(model.predict(input_predict))