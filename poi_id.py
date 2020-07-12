#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
sys.path.append("../tools/")
import matplotlib.pyplot as plt

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

def replace(group, stds):
    group[np.abs(group - group.mean()) > stds * group.std()] = (group.mean()+stds*group.std())
    return group

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list_old = ['poi','salary','deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees','shared_receipt_with_poi' ,'percentage_to_poi','percentage_from_poi']
features_list = ['poi',  'total_payments', 'exercised_stock_options', 'shared_receipt_with_poi' ,'percentage_to_poi']
initial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
print(len(data_dict))
### Task 2: Remove outliers

identified_outliers = ["TOTAL"]

for outlier in identified_outliers:
    data_dict.pop(outlier)
print(len(data_dict))
keys = data_dict.keys()

from sklearn import preprocessing
for f in initial_features:
    a = [data_dict[k][f] for k in keys]
    for i in range(0,len(a)):
        if a[i] == "NaN":
            a[i] = 0 #remove NaN values
        if type(a[i]) == str:
        	a[i] = 0
    a = np.array(a).astype(np.float)
    a = replace(a, 4) #4 Std dev
    ta_scaled = preprocessing.minmax_scale(a)
    i = 0
    for key in keys:
        data_dict[key][f] = ta_scaled[i]
        i = i + 1
    
### Task 3: Create new feature(s)

for key in keys:
    if data_dict[key]['from_poi_to_this_person']!=0:
        data_dict[key]['percentage_from_poi'] = (data_dict[key]['from_poi_to_this_person'])/float(data_dict[key]['to_messages'])
    else:
        data_dict[key]['percentage_from_poi'] = 0
        
    if data_dict[key]['from_this_person_to_poi'] == 0:
        data_dict[key]['percentage_to_poi'] = 0
    else:
        data_dict[key]['percentage_to_poi'] = (data_dict[key]['from_this_person_to_poi'])/float(data_dict[key]['from_messages'])
        
    data_dict[key]['ctc'] = data_dict[key]['salary'] + data_dict[key]['bonus'] + data_dict[key]['exercised_stock_options']


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clfnb = GaussianNB()
from sklearn.svm import SVC
clfsv = SVC(kernel = 'linear', C = 10, gamma = 'auto')
from sklearn.tree import DecisionTreeClassifier
clft = DecisionTreeClassifier(min_samples_split = 4)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clfnb.fit(features_train, labels_train)
prednb = clfnb.predict(features_test)
clfsv.fit(features_train, labels_train)
predsv = clfsv.predict(features_test)
clft.fit(features_train, labels_train)
predt = clft.predict(features_test)

from sklearn.metrics import classification_report #selecting the best algorithm
names = ['Not PoI', 'PoI']
print('Gaussian NB')
print(classification_report(labels_test, prednb, target_names = names))
print('Support Vector Classifier')
print(classification_report(labels_test, predsv, target_names = names))
print('Decision Tree Classifier')
print(classification_report(labels_test, predt, target_names = names))

from sklearn.model_selection import GridSearchCV #parameter tuning
param_grid = {'min_samples_split': np.arange(2, 10)}
tree = GridSearchCV(DecisionTreeClassifier(), param_grid) #since decision tree is the best classifier
tree.fit(features_train, labels_train)
print(tree.best_params_)
clf = tree

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
