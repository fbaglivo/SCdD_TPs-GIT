# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:38:05 2017

@author: Fabri
"""

from sklearn.datasets import samples_generator
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt 


 # generate some data to play with
X, y = samples_generator.make_classification(n_informative=5, n_redundant=0, random_state=42) 


# Split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0) 

# Stratified K-FOLD split
 
#test_tags = ['S'] *10 + ['P'] * 10
#data = {'alpha': np.random.rand(20), 'beta' : np.random.rand(20)}
#test_df = pd.DataFrame(data = data, index = test_tags)

clf =   svm.SVC()
clf.fit(X_train,y_train)
clf.score(X_test, y_test)
scores = cross_val_score(clf, X, y, cv=10)
predicted = cross_val_predict(clf, X, y, cv=10) #For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
print roc_auc_score(y, predicted) 

false_positive_rate, recall, thresholds = roc_curve(y, predicted)
roc_auc = auc(false_positive_rate, recall)
plt.plot(false_positive_rate, recall, 'g', label = 'AUC %s = %0.2f' % ('SVM', roc_auc))
plt.plot([0,1], [0,1], 'r--')
plt.legend(loc = 'lower right')
plt.ylabel('Predicted')
plt.xlabel('Real')
plt.title('ROC Curve')



