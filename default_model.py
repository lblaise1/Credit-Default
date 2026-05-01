'''Importing libraries'''
import pandas as pd
import numpy as np
import datetime
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
#import pandas_datareader.data as web
import random
import matplotlib as mplt
import matplotlib as mpl
%matplotlib inline
mpl.style.use('ggplot')
figsize = (20, 8)
#pd.set_option('display.max_columns',None)
import folium
#to install new libraries in anaconda: pip install package_name



import seaborn as sns
import pandas as pd
import numpy as np
import datetime as dt
import os
import gzip
pd.set_option('display.max_rows', 500)


lc1=pd.read_csv('model_data.csv')

# Importing the splitter, classification model, and the metric
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


#numerical variables'''
full_model = lc1[['purpose', 'loan_amnt', 'inq_last_6mths', 'revol_util', 'pub_rec_bankruptcies', 'ln_annual_inc']]

final_mod_NumVar = lc1[['loan_amnt', 'inq_last_6mths', 'revol_util', 'pub_rec_bankruptcies', 'ln_annual_inc']]

#Catergorical Variables'''
final_cat_features = lc1[['purpose']]


#Creating a numpy array for the label value
labels = np.array(lc1['loan_status_1'])

'''Preprocessing: Creating the model matrix'''

from sklearn import preprocessing
def encode_string(cat_features):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_features)
    enc_cat_features = enc.transform(cat_features)
    ## Now, apply one hot encoding
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_features.reshape(-1,1))
    return encoded.transform(enc_cat_features.reshape(-1,1)).toarray()





#The columns of the categorical features 
final_cat_features =['purpose']

Features = encode_string(lc1['purpose'])

for col in final_cat_features:
    temp = encode_string(lc1[col])
    Features = np.concatenate([Features, temp], axis = 1)
    
    
# Numerical features'''
# Next the numeric features must be concatenated to the numpy array by executing the code in the cell below.'''
Features = np.concatenate([Features, np.array(lc1[['loan_amnt', 'inq_last_6mths', 'revol_util', 'pub_rec_bankruptcies', 'ln_annual_inc']])], axis = 1)
# print(Features.shape)
# print(Features[:2, :])

import sklearn.model_selection as ms
import numpy.random as nr

nr.seed(9988)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 1000)

#Training and Test Features
X_train = Features[indx[0],:]
X_test = Features[indx[1],:]

#Training and test labels
y_train = np.ravel(labels[indx[0]])
y_test = np.ravel(labels[indx[1]])



#Import the SMOTE-NC
from imblearn.over_sampling import SMOTENC
#Create the oversampler. 
#For SMOTE-NC we need to pinpoint the location of the categorical features.

smotenc = SMOTENC([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],random_state = 101)
x_oversample, y_oversample = smotenc.fit_resample(X_train, y_train)


#With Oversampling
scaler = preprocessing.StandardScaler().fit(x_oversample[:,28:])
x_oversample[:,28:] = scaler.transform(x_oversample[:,28:])
X_test[:,28:] = scaler.transform(X_test[:,28:])
#x_oversample[:2,]


# With Oversampling'''
from sklearn import linear_model
logistic_mod = linear_model.LogisticRegression() 
logistic_mod.fit(x_oversample, y_oversample)


#Calculating probabilities
probabilities = logistic_mod.predict_proba(X_test)


#Scoring the model
def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:,1]])
scores = score_model(probabilities, 0.25)



#Confusion Matrix
import sklearn.metrics as sklm

def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])
    
print_metrics(y_test, scores)  


#ROC and AUC

def plot_auc(labels, probs):
    ## Compute the false positive rate, true positive rate
    ## and threshold along with the AUC
    fpr, tpr, threshold = sklm.roc_curve(labels, probs[:,1])
    auc = sklm.auc(fpr, tpr)
    
    ## Plot the result
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
plot_auc(y_test, probabilities) 

'''Class weight'''
logistic_mod = linear_model.LogisticRegression(class_weight = {0:0.50, 1:0.5}) 
logistic_mod.fit(x_oversample, y_oversample)


'''Predicting probabilities'''
#compute and display the class probabilities for each case.'''
probabilities = logistic_mod.predict_proba(X_test)
