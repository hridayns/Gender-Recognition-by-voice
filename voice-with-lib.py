import numpy as np
import pandas as pd
import operator
from sklearn import preprocessing, neighbors
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('voice.csv')

for col in dataset.drop(['label'],1):
	mu = np.mean(np.array(dataset[col]))
	sd = np.std(np.array(dataset[col]))
	dataset[col] = np.divide(np.subtract(np.array(dataset[col]),mu),sd)

dataset = dataset.sample(frac=1,random_state=43)

train_size = int(dataset.shape[0] * 0.7)
test_size = int(dataset.shape[0] * 0.3)

[train_data,test_data] = [dataset[:train_size],dataset[train_size:]]

# X = train_data.loc[:,dataset.columns != 'label'].as_matrix()
X = np.array(train_data.drop(['label'],1))
# y = train_data.loc[:,dataset.columns == 'label'].as_matrix()
y = np.array(train_data[['label']])
# X_test = test_data.loc[:,dataset.columns != 'label'].as_matrix()
X_test = np.array(test_data.drop(['label'],1))
# y_test = test_data.loc[:,dataset.columns == 'label'].as_matrix()
y_test = np.array(test_data[['label']])

model = neighbors.KNeighborsClassifier()
model.fit(X, y)

acc_sc = accuracy_score(y_test,model.predict(X_test)) * 100
print('Accuracy: ',acc_sc)
