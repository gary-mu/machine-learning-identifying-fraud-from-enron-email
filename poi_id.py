#!/usr/bin/python

from __future__ import division
import sys
import pickle
sys.path.append("../../tools/")
import numpy as np
import pandas as pd
import matplotlib.pyplot
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


#Find number of features and POIs
poi_count = 0
for key in data_dict:
    if data_dict[key]['poi'] ==1:
        poi_count +=1 
poi_count

'Number of POIs are: ', poi_count
'Number of POIs are ', len(data_dict.itervalues().next())
for key in (data_dict.itervalues().next()): print key


### Task 2: Remove outliers
#Plot out data to identify outliers
data = featureFormat(data_dict, features_list)
for point in data:
    poi = point[0]
    salary = point[1]
    bonus = point[2]
    color = 'red' if poi == 1 else 'blue'
    matplotlib.pyplot.scatter( salary, bonus,  s= 500 , c = color)

matplotlib.pyplot.xlabel("salary") 
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.gray()

matplotlib.pyplot.show()

data_dict.pop('TOTAL', 0)

#Export data to csv for data Exploration in R
df = pd.DataFrame.from_dict(data_dict, orient='index', dtype=None)
df.to_csv('enron.csv')

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
for key in data_dict:
    if data_dict[key]['from_poi_to_this_person'] == "NaN":
        data_dict[key]['from_poi_to_this_person'] = 0
    if data_dict[key]['from_messages'] == "NaN":
        data_dict[key]['from_messages'] = 0
    if data_dict[key]['from_messages'] ==0:
        data_dict[key]['fraction_from_poi'] = 0
    else:
        data_dict[key]['fraction_from_poi'] = int(data_dict[key]['from_poi_to_this_person'])/int(data_dict[key]['from_messages'])
######Modify to poi msg to fractions
for key in data_dict:
    if data_dict[key]['from_this_person_to_poi'] == "NaN":
        data_dict[key]['from_this_person_to_poi'] = 0
    if data_dict[key]['to_messages'] == "NaN":
        data_dict[key]['to_messages'] = 0
    if data_dict[key]['to_messages'] ==0:
        data_dict[key]['fraction_to_poi'] = 0
    else:
        data_dict[key]['fraction_to_poi'] = int(data_dict[key]['from_this_person_to_poi'])/int(data_dict[key]['to_messages'])

my_dataset = data_dict

### Extract features and labels from dataset for local testing
features_list = ['poi','salary', 'bonus', 'exercised_stock_options',
                 'total_payments','restricted_stock', 'total_stock_value', 
                 'expenses', 'fraction_from_poi', 'fraction_to_poi'
                 ]
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Scale features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

#Reduce data dimensionality with PCA and conduct feature selection with SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
pca = PCA()
selection = SelectKBest()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

######fit Gaussian NB#########
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

########fit SVM###############
# from sklearn.svm import SVC
# clf = SVC()

#######fit RF ###############
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier()


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


#Partition data into testing and training set
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
	features_scaled, labels, test_size = 0.3, random_state = 42)

##Use Pipeline to tansform data
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

combined_features = FeatureUnion([("pca", pca), ("selection", selection)])
pipeline = Pipeline(
    [
     ('features', combined_features),
     ('clf', clf),
     ])

#Pipeline parameter for GNB
params_gnb = dict(features__selection__k=[3,4,5,6,7,8],
              features__pca__n_components = [1,2,3,4],
              features__pca__whiten = [True])

#Pipeline parameter for SVC
# params_svc = dict(
# 			features__selection__k=[1,2,3,4,5,6,7,8],
#             features__pca__n_components = [1,2,3,4],
#             clf__C=[1,10,100,1000,10000],
#             clf__kernel = ['linear', 'rbf'])

#Pipeline parameter for RF
# params_rf = dict(
# 			features__selection__k=[1,2,3,4,5,6,7,8],
#             features__pca__n_components = [1,2,3,4],
#             clf__n_estimators=[50, 200,500],
#             clf__min_samples_split=[2, 10,100])


#Train data with GridSearchCV to find the best combination of PCA components and number of features
cv = GridSearchCV(pipeline, param_grid=params_gnb, verbose=1)
cv.fit(features_train, labels_train)

#Model accuracy evaluation
prediction = cv.predict(features_test)

report = classification_report( labels_test, prediction )
print report
print cv.best_estimator_,'\n' , cv.best_score_, '\n',cv.best_params_

from sklearn.metrics import accuracy_score
print accuracy_score(prediction, labels_test)

select_k_bool = cv.best_estimator_.named_steps['features'].transformer_list[1][1].get_support()
print "The Selected Features Are: " ,[x for x, y in zip(features_list[1:], select_k_bool) if y]

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)