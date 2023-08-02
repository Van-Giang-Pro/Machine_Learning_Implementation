import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

credit_data = pd.read_csv('credit_data.csv')
feature = credit_data[['income', 'age', 'loan']]
target = credit_data.default

# Machine learning handle arrays not dataframes

X = np.array(feature).reshape(-1, 3)
y = np.array(target)

print(feature.corr())

feature_train, feature_test,  target_train, target_test = train_test_split(X, y, test_size=0.3)

model = GaussianNB()
fitted_model = model.fit(feature_train, target_train)
predictions = fitted_model.predict(feature_test)

print(confusion_matrix(target_test, predictions))
print(accuracy_score(target_test, predictions))


