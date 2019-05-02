# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 21:49:17 2019

@author: Utkarsh Kumar
"""

import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold

"""Run below code once to create finadata.csv"""
path = 'F:/Machine Learning/jpg_to_mnist' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

data = pd.concat(li, axis=0, ignore_index=True)
data = data.drop('Index',axis=1)
data.to_csv('finaldata.csv')
""" Upto here """

data = pd.read_csv('finaldata.csv')
data = data.sample(frac=1).reset_index(drop=True)

data = data.drop(['Unnamed: 0'],axis=1)
data.iloc[:,1:] = data.iloc[:,1:].astype(float)

data.head(10)


type(data.iloc[5,5])

data.value_counts()

image = pd.DataFrame(columns=range(28))
image = data.iloc[1,1:].values.reshape(28,28)
val = data.iloc[1,0]
image = pd.DataFrame(image)
image.iloc[:,:] = image.iloc[:,:].astype(float)
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()


test_data = pd.read_csv('test.csv')
test_data = test_data.drop('Index',axis=1)


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(data['Character'])


X = data.drop(['Character'],axis=1)
Y = pd.DataFrame(data.iloc[:,0].values.reshape(-1,1))


X_test1 = test_data.drop(['Character'],axis=1)
Y_test1 = pd.DataFrame(test_data.iloc[:,0].values.reshape(-1,1))

X_test2=X_test1.iloc[:,:].values.reshape(28,28)
plt.figure()
plt.imshow(X_test2, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
Y_test[0].value_counts()
Y_train[0].value_counts().min()


"""Different models"""
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()#criterion='entropy')

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, criterion = 'entropy', min_samples_split = 10, n_jobs = 4)

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100,learning_rate=2)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class = 'auto', solver = 'lbfgs', n_jobs = 4, verbose = 1)

from sklearn.svm import SVC
model = SVC(gamma='auto')

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()

#Neural Network Model

from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
"""
parameter_space = {
    'hidden_layer_sizes': [(10,10,10),(35,35,35)],
    'alpha': [0.0001,0.1,0.001,0.01],
    'learning_rate': ['constant'],
    'batch_size':[50,100,150]
}
"""
para_space = {
    'C': [10 ** x for x in range(-2,2)]
}
gb_para = {
        'n_estimators':[10,20,30],'learning_rate':[1,2,5]
}
dt_para = { 'criterion' : ['entropy', 'gini']}
rf_para = {'n_estimators' : [10, 20, 30, 40, 50]}
param_distributions = {
        'n_neighbors': [5,10,15,20,25]
}

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(model, dt_para, n_jobs=-1, cv=10, scoring = 'accuracy')
clf.fit(X_train,Y_train)
clf.cv_results_
print('Best parameters found:\n', clf.best_params_)
Avg_score = clf.cv_results_['mean_test_score']

param_distributions = {
    'hidden_layer_sizes': [(10,10,10),(35,35,35)],
    'alpha': [0.0001,0.1,0.001,0.01],
    'learning_rate': ['constant','adaptive'],
    'batch_size':[100,150,200]
}
from sklearn.model_selection import RandomizedSearchCV
clf = RandomizedSearchCV(model,param_distributions=dt_para,n_jobs=4,n_iter=100,verbose=1, scoring = 'accuracy')
clf.fit(X,Y)
print('Best parameters found:\n', clf.best_params_)
clf.best_score_

model = DecisionTreeClassifier(criterion='entropy')


kfld = KFold(5)
Avg_score = []
Preff_C = [(10 ** x) for x in range(-1,5)]
for c in Preff_C:
    scores = []
    for train, test in kfld.split(X, Y):
        model = LogisticRegression(multi_class = 'auto', solver = 'lbfgs', n_jobs = 4, C = c)
        model.fit(X.iloc[train], Y.iloc[train])
        scores.append(model.score(X.iloc[train], Y.iloc[train]))
    print(c, scores)
    Avg_score.append(np.average(scores))


Avg_error = [(1 - x) for x in Avg_score]
Log_Preff_C = [np.log10(x) for x in Preff_C]
Avg_error
plt.plot(Log_Preff_C, Avg_error)
plt.xlabel('log10(C)')
plt.ylabel('Error')

Best_C = Preff_C[np.argmin(Avg_error)]
model = LogisticRegression(multi_class = 'auto', solver = 'lbfgs', n_jobs = 4, C = Best_C)
"""Models upto here"""

model.fit(X_train, Y_train)
print(model)
# make predictions
expected = Y_test
predicted = pd.DataFrame(model.predict(X_test))

expected = Y_test1
predicted = pd.DataFrame(model.predict(X_test1))





model.score(X,Y)
model.score(X_train,Y_train)
model.score(X_test,Y_test)
model.score(X_test1,Y_test1)


from sklearn import metrics
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
cf  = metrics.confusion_matrix(expected, predicted)
sns.heatmap(cf, annot=True, fmt="d")