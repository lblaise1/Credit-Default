#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Full Stack Loan Default Modeling


# The goal of the project is to develop a specialized analysis for subprime borrowers to predict loan default.
# Generally, we think of subprime borrowers as a homogenous group of high risk applicants with similar credit profiles. 
# However, analysis of data from subprime borrowers on a peer-2-peer lending platform, LendingClub, paints a different picture.
# The data show us that these borrowers are a diverse group of customers and most importantly, only 15% of the loans from the platform defaulted from 2007 to 2012. 
# This tells us that there are many potential borrowers who are considered subprime or high risk by traditional standards, 
# yet they do indeed repay their loans. These data suggest loan applicants are being rejected and lending institutions are losing
# potential customers.
# 
# Below, I have developed a data pipeline which analyzed over 42,000 loans with 144 variables from the LendingClub Platform. 
# After data wrangling, cleaning, feature engineering and selection I identified 6 predictive variables that were included in the credit risk model, which uses logistic regression. 
# 

# ## Importing Libraries

# In[1]:


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
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use('ggplot')
figsize = (20, 8)
#pd.set_option('display.max_columns',None)
import folium
#to install new libraries in anaconda: pip install package_name


# In[2]:


import seaborn as sns
import pandas as pd
import numpy as np
import datetime as dt
import os
import gzip
pd.set_option('display.max_rows', 500)


# ## Setting the Directory and Import the Dataset

# In[3]:


path = '/home/jovyan/datacourse/Capstone Project'


# In[4]:


os.chdir(path)


# In[5]:


lc=pd.read_csv('LoanStats3a.csv', skiprows=1, low_memory=False)


# ## Data Investigation

# In[6]:


'''Let's investigate the dataset'''
print("Number of Rows   : ", lc1.shape[0])
print("Number of Columns   : ", lc1.shape[1])
print("\nName of the Features : \n", lc1.columns.tolist())
print("\nContain Missing Values? (True/False): ", lc1.isnull().any())
print("\nNumber of Unique values :  \n", lc1.nunique())


# In[7]:


'''tail investigation'''
lc.tail(3)


# # Data Cleaning/Wrangling

# ### 1. Missing Values

# In[8]:


#Droping the last 2 rows: they are useful to the analysis
#Their contents: Total amount funded in policy code 1: 46029615
lc.drop(lc.tail(2).index,inplace=True)


# In[9]:


#Removing columns with all with more than 94% missing values
#Subsetting the data by removing all columns with more than 30% missing values
lc1=lc[lc.columns[lc.isnull().mean() < 0.06175]]
lc1.isnull().sum()


# In[10]:


#After this stage, we are left with 51 features and 42,536 loans (out of the original 144 variables and 42,538 rows)
lc1.shape


# In[11]:


#Removing missing values
#pub_rec_bankruptcies has the highest number of missing values: 1,366
#I will drop all missing values from the dataset
lc1 = lc1.dropna(axis = 0, how ='any') 


# In[12]:


#After this stage, we are left with 51 features and 39,913 loans (out of the original 144 variables and 42,538 rows)
#2,625 rows/loans were removed due to missing values
lc1.shape


# In[13]:


lc1.isnull().sum()


# ### Cleaning/Transformation

# In[14]:


#Variable Cleaning
'''i) revol_util: removing % signs and convert to float'''
lc1['revol_util'] = lc1['revol_util'].str.rstrip('%').astype('float')

'''ii) removing % signs from the interest rate column'''
lc1['int_rate'] = lc1['int_rate'].str.rstrip('%').astype('float')


# In[15]:


#Variable Transformation
'''i) loan terms'''
#Creating a dictionary:
months = {' 60 months':60, ' 36 months': 36}
#building a list comprehension:
lc1['term_1'] = [months[x] for x in lc1['term']]

'''ii) emp_length: employment history variable:looking into employment length'''
#Recoding the columns to numeric type
years = {'10+ years':10, '6 years': 6, '< 1 year': 0.8, '4 years': 4, '3 years':3,
         '2 years': 2, '1 year': 1, '9 years': 9, '5 years': 5, '7 years': 7, 
         '8 years': 8}
lc1['emp_length_1'] = lc1['emp_length'].replace(years)


'''iii) Verification Status'''
#cleaning the verification status variable: Verified=1, not verified=0
verification = {'Source Verified':'Verified', 'Not Verified': 'Not Verified', 'Verified': 'Verified'}
lc1['verification_status'] = [verification[x] for x in lc1['verification_status']]

'''iv) Loan Status 1: I need this variable to be of types both numeric and object: So I created 2 versions'''
#Cleaning the loan status variable: Fully paid/paid off=1; Charged Off/default=0
status = {'Does not meet the credit policy. Status:Charged Off':1, 
          'Does not meet the credit policy. Status:Fully Paid':0,
          'Fully Paid': 0, 
          'Charged Off':1}
lc1['loan_status_1'] = [status[x] for x in lc1['loan_status']]

'''v) Loan Status 2'''
#Cleaning the loan status variable: Fully paid/paid off=1; Charged Off/default=0
status = {'Does not meet the credit policy. Status:Charged Off':'Charged Off', 
          'Does not meet the credit policy. Status:Fully Paid':'Paid Off',
          'Fully Paid': 'Paid Off', 
          'Charged Off':'Charged Off'}
lc1['loan_status_2'] = [status[x] for x in lc1['loan_status']]

'''v) Home Ownership'''
#cleaning the home ownership variable: categorical
status = {'MORTGAGE':'Owner', 'NONE':'Non-Owner', 'OTHER': 'Non-Owner', 'RENT':'Non-Owner', 'OWN':'Owner'} #1=Yes, 0=No
#building a list comprehension:
lc1['home_ownership'] = [status[x] for x in lc1['home_ownership']]


# In[16]:


#Type coversion: Converting the follwing variables to datetime format
lc1['issue_d']=pd.to_datetime(lc1['issue_d'])
lc1['last_pymnt_d']=pd.to_datetime(lc1['last_pymnt_d'])
lc1['earliest_cr_line']=pd.to_datetime(lc1['earliest_cr_line'])
lc1['last_credit_pull_d']=pd.to_datetime(lc1['last_credit_pull_d'])


# # Feature Engineering and Selection

# ## Feature Engineering

# In[17]:


#Creating and Cleaning variables'''
#1. Credit history'''
lc1['cred_hist']=lc1['issue_d']-lc1['earliest_cr_line']
lc1['cred_hist']=lc1['cred_hist'].astype('str')
#lc1['cred_hist']=lc1['cred_hist'].str.strip(' days 00:00:00.000000000').astype('float')
lc1['cred_hist']=lc1['cred_hist'].str[0:3].astype('float')


#Creating the credit history variable 
lc1['cred_hist']=round(lc1['cred_hist']/365.25)
lc1['cred_hist'].value_counts()


'''Numerical Variables'''
#Some variable creation
lc1['ln_annual_inc'] = np.log(lc1['annual_inc'])
lc1['dti_square'] = lc1['dti']**2
lc1['dti_new'] = np.log(lc1['dti']+1)
lc1['ln_loan_amnt'] = np.log(lc1['loan_amnt'])
lc1['pub_rec_bankruptcies_'] = lc1['pub_rec']*lc1['pub_rec_bankruptcies']
lc1['ln_revol_bal'] = np.log(lc1['revol_bal']+1)
lc1['ln_revol_util'] = np.log(lc1['revol_util']+1)


# ## Feature Selection

# Based on the goal of this model, I want to retain variables that would be associated with an applicant at the time of the loan application. I aim to remove variables in this dataset that were generated after the loan was issued.

# ### 1. Judgement Based

# In[18]:


#List of columns to drop: not needed for analysis or feature engineering
#application_type': Individual; they are all the same type; can be removed
to_drop = lc1[['title', 'zip_code', 'addr_state', 'initial_list_status', 'policy_code']]

#Dropping those variables'''
lc1.drop(to_drop, axis=1, inplace=True)


# In[19]:


#Collecting columns that have too many missing values and and those that are  not useful to the analysis.
'''I am dropping these the following variables because they have the same values for all the loans'''
to_drop2 = (lc1[['acc_now_delinq', 'tax_liens', 'delinq_amnt', 'pymnt_plan', 'out_prncp','out_prncp_inv', 
               'collections_12_mths_ex_med', 'chargeoff_within_12_mths','hardship_flag', 'application_type', 'debt_settlement_flag']])
#Dropping those variables'''
lc1.drop(to_drop2, axis=1, inplace=True)


# ### 2. Algorithmic Based (Collinearity)

# In[20]:


#The numerical variables
num_var = (lc1[['loan_amnt','funded_amnt','funded_amnt_inv','int_rate','installment','annual_inc','dti','delinq_2yrs',
                'inq_last_6mths','open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'total_pymnt', 
                'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int','total_rec_late_fee','recoveries', 
                'collection_recovery_fee', 'last_pymnt_amnt', 'pub_rec_bankruptcies','term_1','emp_length_1', 
                'loan_status_1']])


# In[21]:


date_var = lc1[['earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d']]


# In[22]:


#Categorical Variables
cat_var = lc1[['term','grade','sub_grade','emp_length','home_ownership','verification_status','purpose','loan_status_2']]


# In[23]:


'''Correlation among the numerical variables'''
num_var.corr()


# In[24]:


#Collinear Variables
collinear_var = lc1[['funded_amnt', 'funded_amnt_inv', 'installment','total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int']]
    
 #Removing the collinear variables'''
lc1.drop(collinear_var, axis=1, inplace=True)   


# In[25]:


#Remaining numerical variables: 19 numerical variables
final_num_var = (lc1[['loan_amnt','int_rate','annual_inc','dti','delinq_2yrs', 'inq_last_6mths','open_acc', 'pub_rec', 
                'revol_bal', 'revol_util', 'total_acc', 'total_rec_late_fee','recoveries', 
                'collection_recovery_fee', 'last_pymnt_amnt', 'pub_rec_bankruptcies','term_1','emp_length_1', 'loan_status_1']])
# #Categorical Variables
# cat_var = (lc1[['term','grade','sub_grade','emp_length','home_ownership','verification_status','purpose',
#                'debt_settlement_flag','loan_status_2']])


# In[26]:


final_num_var.shape


# # Visualization of the Features

# Through vizualizations, I am seeking to identify differences in profile of variables between loans that defaulted versus those that were paid off. 

# ## 1. Numerical Features

# ### Boxplots

# In[27]:


num_cols = (['loan_amnt','int_rate','annual_inc','dti','delinq_2yrs', 'inq_last_6mths','open_acc', 'pub_rec', 
                'revol_bal', 'revol_util', 'total_acc', 'total_rec_late_fee','recoveries', 
                'collection_recovery_fee', 'last_pymnt_amnt', 'pub_rec_bankruptcies','term_1','emp_length_1'])


# In[28]:


'''Visualize class separation by numeric features'''
import seaborn as sns
def plot_box(data, cols, col_x = 'loan_status_1'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.boxplot(col_x, col, data=data)
        plt.xlabel(col_x) # Set text for the x axis
        plt.ylabel(col)# Set text for y axis
        plt.show()
        


# In[29]:


plot_box(final_num_var , num_cols)


# ## 2. Categorical Features

# In[30]:


import numpy as np
#Categorical Variables
cat_cols = (lc1[['term','grade','sub_grade','emp_length','home_ownership','verification_status','purpose']])

lc1['dummy'] = np.ones(shape = lc1.shape[0])
for col in cat_cols:
    print(col)
    counts = lc1[['dummy', 'loan_status_1', col]].groupby(['loan_status_1', col],
                   as_index = False).count()
    temp = counts[counts['loan_status_1'] == 0][[col, 'dummy']]
    _ = plt.figure(figsize = (10,4))
    plt.subplot(1, 2, 1)
    temp = counts[counts['loan_status_1'] == 0][[col, 'dummy']]
    plt.bar(temp[col], temp.dummy)
    plt.xticks(rotation=90)
    plt.title('Counts for ' + col + '\n Paid Off Loans')
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    temp = counts[counts['loan_status_1'] == 1][[col, 'dummy']]
    plt.bar(temp[col], temp.dummy)
    plt.xticks(rotation=90)
    plt.title('Counts for ' + col + '\n Defaulted Loans')
    plt.ylabel('count')
    plt.show()


# ** Observations **:
#     
#     
#     1. Grade: Internally determined by the company
#         
#     2. Subgrade:  This variable was determined internally and LendingClub has discontinued its use.
#         
#     3. Employment Length: This shows no obvious difference between defaulted loans and paid off loans.
#         
#     4. Verification: This shows no obvious difference between defaulted loans and paid off loans.
#         
#     5. Home Ownership: Paid off loans have a higher proportion of home owners whereas there were similar numbers of owners 
#        and non-owners that defaulted.
#         
# ** Conclusion **: 
#     
#     
#     1. Loan purpose: This variable will be used in model training.
#         
#     2. Home Ownership: This variable could have some predictive power, thus it will be kept for model training.

# In[31]:


final_cat_var = (lc1[['term','home_ownership', 'purpose']])

final_num_var = (lc1[['loan_amnt','int_rate','annual_inc','dti','delinq_2yrs', 'inq_last_6mths','open_acc', 'pub_rec', 
                'revol_bal', 'revol_util', 'total_acc', 'total_rec_late_fee','recoveries', 
                'collection_recovery_fee', 'last_pymnt_amnt', 'pub_rec_bankruptcies','term_1','emp_length_1', 'loan_status_1']])


# # Modelling

# ### 1. LOGISTIC REGRESSION WITH STATTSMODELS: Categorical Variables

# Constructing potential models to predict default with different variables

# In[32]:


import statsmodels.api as sm
import statsmodels.formula.api as smf 


# In[33]:


#1. CATEGORICAL VARIABLES
lreg01=smf.logit(formula ='loan_status_1 ~ home_ownership + purpose + term', data=lc1).fit()
print(lreg01.summary())


# In[34]:


#2. CATEGORICAL VARIABLES
lreg02=smf.logit(formula ='loan_status_1 ~ home_ownership + term', data=lc1).fit()
print(lreg02.summary())


# In[35]:


#3. CATEGORICAL VARIABLES
lreg03=smf.logit(formula ='loan_status_1 ~ term', data=lc1).fit()
print(lreg03.summary())


# In[36]:


#3. CATEGORICAL VARIABLES
lreg04=smf.logit(formula ='loan_status_1 ~ purpose + term', data=lc1).fit()
print(lreg04.summary())


# Observations: 
#     1. Home Ownership: does not contribute any prediction power of default
#     2. Purpose: has some predictive power
#     3. Term: provides additional predictive power
#    
#   Conclusions:
#       1. Homeownership will not be retained in the final model
#       2. Term: given the motivation of this analysis, term will not be used in the first stage of the modelling process as          it is a post loan approval variable. 
#       3. Purpose: this variable will be retained for additional modelling/analysis

# In[37]:


retained_cat_var = lc1[['term','purpose']]


# ### 2. LOGISTIC REGRESSION WITH STATTSMODELS: Categorical Variables

# In[38]:


final_num_var = (lc1[['loan_amnt','int_rate','annual_inc','dti','delinq_2yrs', 'inq_last_6mths','open_acc', 'pub_rec', 
                'revol_bal', 'revol_util', 'total_acc', 'total_rec_late_fee','recoveries', 
                'collection_recovery_fee', 'last_pymnt_amnt', 'pub_rec_bankruptcies','term_1','emp_length_1', 'loan_status_1']])


# In[39]:


post_approval_vars = [['int_rate', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 'term_1']]


# In[40]:


lreg1 = smf.logit(formula = 'loan_status_1 ~ loan_amnt +  annual_inc + dti + delinq_2yrs + inq_last_6mths + open_acc + pub_rec + revol_bal + revol_util + total_acc + pub_rec_bankruptcies', data = lc1).fit()
print(lreg1.summary())


# In[41]:


#These are the variables that are retained for pre-modeling
#They are variables that are determined before loan approval
pre_app_num_var = lc1[['loan_amnt', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'pub_rec_bankruptcies']]


# In[42]:


'''Correlation among the numerical variables'''
pre_app_num_var.corr()


# In[43]:


#numeric variables heatmap
plt.figure(figsize = (10,10))
sns.heatmap(pre_app_num_var.corr(), annot=True)


# Observations:
#     1. Open Account and Total Account are highly correlated
#     2. Public Record Bankcruptcies and Public Record are highly correlated
# 
# Conclusions:
#     Based on univariate analysis, I will retain:
#         1. Pub Record Bankruptcies
#         2. Total Account
# They both have higher coefficients that are statistically significant when compared to the variable with which they are correlated   

# In[44]:


# interim1 = smf.logit(formula = 'loan_status_1 ~ open_acc + total_acc', data = lc1).fit()
# print(interim1.summary())


# In[45]:


# interim2 = smf.logit(formula = 'loan_status_1 ~ pub_rec + pub_rec_bankruptcies', data = lc1).fit()
# print(interim2.summary())


# In[46]:


# interim3 = smf.logit(formula = 'loan_status_1 ~ pub_rec', data = lc1).fit()
# print(interim3.summary())


# In[47]:


# interim4 = smf.logit(formula = 'loan_status_1 ~ pub_rec_bankruptcies', data = lc1).fit()
# print(interim4.summary())


# In[48]:


# interim5 = smf.logit(formula = 'loan_status_1 ~ open_acc', data = lc1).fit()
# print(interim5.summary())


# In[49]:


# interim6 = smf.logit(formula = 'loan_status_1 ~ total_acc', data = lc1).fit()
# print(interim6.summary())


# In[50]:


lreg2 = smf.logit(formula = 'loan_status_1 ~ loan_amnt +  annual_inc + dti + delinq_2yrs + inq_last_6mths + revol_bal + revol_util + total_acc + pub_rec_bankruptcies', data = lc1).fit()
print(lreg2.summary())


# In[51]:


transformed_var= lc1[['ln_annual_inc', 'dti_square', 'dti_new', 'ln_loan_amnt', 'pub_rec_bankruptcies_', 'ln_revol_bal', 'ln_revol_util']]


# In[52]:


lreg3 = smf.logit(formula = 'loan_status_1 ~ ln_annual_inc + dti_square + ln_loan_amnt + pub_rec_bankruptcies_ + ln_revol_bal + ln_revol_util', data = lc1).fit()
print(lreg3.summary())


# In[53]:


# loan_amnt +  annual_inc + dti + delinq_2yrs + inq_last_6mths + revol_bal + revol_util + total_acc + pub_rec_bankruptcies


# In[54]:


lreg4 = smf.logit(formula = 'loan_status_1 ~  loan_amnt +  annual_inc + dti + delinq_2yrs + inq_last_6mths + revol_bal + revol_util + total_acc + pub_rec_bankruptcies+ ln_annual_inc + dti_square + ln_loan_amnt + pub_rec_bankruptcies_ + ln_revol_bal + ln_revol_util', data = lc1).fit()
print(lreg4.summary())


# In[55]:


lreg4 = smf.logit(formula = 'loan_status_1 ~  ln_loan_amnt + delinq_2yrs + inq_last_6mths + revol_util + pub_rec_bankruptcies  + ln_annual_inc + ln_revol_util', data = lc1).fit()
print(lreg4.summary())


# ### Mixed Model: categorical and numerical variables

# In[56]:


mixed_mod_1 = smf.logit(formula = 'loan_status_1 ~   purpose + loan_amnt + inq_last_6mths + revol_util + pub_rec_bankruptcies  + ln_annual_inc', data = lc1).fit()
print(mixed_mod_1.summary())


# In[57]:


# odds ratios
print ("Odds Ratios")
print (round(np.exp(mixed_mod_1.params),2))


# In[58]:


final_mod_var = lc1[['loan_status_1', 'loan_amnt', 'inq_last_6mths', 'revol_util', 'pub_rec_bankruptcies', 'ln_annual_inc']]


# In[81]:


model_data = lc1[['loan_status_1', 'purpose', 'loan_amnt', 'inq_last_6mths', 'revol_util', 'pub_rec_bankruptcies', 'ln_annual_inc']]


# In[82]:


model_data.to_csv('model_data.csv')


# In[59]:


#numeric variables heatmap
plt.figure(figsize = (10,10))
sns.heatmap(final_mod_var.corr(), annot=True)


# The pairplots illustrate that there is considerable overlap for defaulted and paid-off loans for nearly all variables. 
# While we can see that the variables are not highly correlated, we can also see that the loan status is broadly distributed across the continum for most variables.  

# In[60]:


#Paiplotting all the variables
sns.pairplot(final_mod_var, hue='loan_status_1', height=5)
plt.show()


# ## LOGISTIC REGRESSION WITH SCIKIT LEARN

# In[61]:


# Importing the splitter, classification model, and the metric
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[62]:


#numerical variables'''
full_model = lc1[['purpose', 'loan_amnt', 'inq_last_6mths', 'revol_util', 'pub_rec_bankruptcies', 'ln_annual_inc']]

final_mod_NumVar = lc1[['loan_amnt', 'inq_last_6mths', 'revol_util', 'pub_rec_bankruptcies', 'ln_annual_inc']]

#Catergorical Variables'''
final_cat_features = lc1[['purpose']]


# In[63]:


#Creating a numpy array for the label value
labels = np.array(lc1['loan_status_1'])


# In[80]:


lc1['purpose'].value_counts()


# ### Preprocessing: Creating the model matrix

# In[64]:


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


# In[65]:


#The columns of the categorical features 
final_cat_features =['purpose']

Features = encode_string(lc1['purpose'])

for col in final_cat_features:
    temp = encode_string(lc1[col])
    Features = np.concatenate([Features, temp], axis = 1)
    print(Features)

print(Features.shape)
print(Features[:2, :]) 


# In[66]:


# Numerical features'''
# Next the numeric features must be concatenated to the numpy array by executing the code in the cell below.'''
Features = np.concatenate([Features, np.array(lc1[['loan_amnt', 'inq_last_6mths', 'revol_util', 'pub_rec_bankruptcies', 'ln_annual_inc']])], axis = 1)
print(Features.shape)
print(Features[:2, :])  


# This step is critical. If machine learning models are tested on the training data, 
# the results will be both biased and artificially predictive.
# 
# The code in the cell below performs the following processing:
#     
#     1. An index vector is Bernoulli sampled using the train_test_split function from 
#        the model_selection package of scikit-learn.
# 
#     2. The first column of the resulting index array contains the indices of the 
#        samples for the training cases.
# 
#     3. The second column of the resulting index array contains the indices of the 
#        samples for the test cases.
#    
# I randomly sampled cases to create independent training and test data

# In[67]:


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


# ### Addressing Imbalance in the Dataset

# In[68]:


#Import the SMOTE-NC
from imblearn.over_sampling import SMOTENC
#Create the oversampler. 
#For SMOTE-NC we need to pinpoint the location of the categorical features.

smotenc = SMOTENC([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],random_state = 101)
x_oversample, y_oversample = smotenc.fit_resample(X_train, y_train)


# Now, we must rescale the numeric features so they have a similar range of values. 
# Rescaling prevents features from having an undue influence on model training 
# simply because they have a larger range of numeric variables.
# 
# The code below uses the StanardScaler function from the Scikit Learn 
# preprocessing package to Zscore scale the numeric features. 
# 
# Notice that the scaler is fit only on the training data. 
# The trained scaler is then applied to the test data. 
# Test data should always be scaled using the parameters from the training data.
# 

# In[69]:


#With Oversampling
scaler = preprocessing.StandardScaler().fit(x_oversample[:,28:])
x_oversample[:,28:] = scaler.transform(x_oversample[:,28:])
X_test[:,28:] = scaler.transform(X_test[:,28:])
x_oversample[:2,]


# '''***Construct the logistic regression model
# 
# Now, we can construct the logistic regression model as follows:
# 
# 1. Define a logistic regression model object using the LogisticRegression method 
#    from the scikit-learn linear_model package.
# 
# 2. Fit the linear model using the numpy arrays of the features and the labels 
#    for the training dataset.

# In[70]:


# With Oversampling'''
from sklearn import linear_model
logistic_mod = linear_model.LogisticRegression() 
logistic_mod.fit(x_oversample, y_oversample)


# In[71]:


# ***Now, print and examine the model coefficients'''
print(logistic_mod.intercept_)
print(logistic_mod.coef_)


# ***** Calculating the probabilities*****
# 
# 
# Recall that the logistic regression model outputs probabilities for each class. 
# The class with the highest probability is taken as the score (prediction).
# 
# Note-
# 
# The first column is the probability of a score of  0  and the second column is 
# the probability of a score of 1.

# In[72]:


probabilities = logistic_mod.predict_proba(X_test)
print(probabilities[:15,:])


# '''*****Score and evaluate the classification model********
# Now that the class probabilities have been computed these values must be 
# transformed into actual class scores. 
# Recall that the log likelihoods for two-class logistic regression are computed 
# by applying the sigmoid or logistic transformation to the output of the linear 
# model.
# 
# We can set the threshold between the two likelihoods at  0.5. 
# We apply this initial threshold to the probability of a score of 0 for the 
# test data.'''

# In[73]:


def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:,1]])
scores = score_model(probabilities, 0.5)
print(np.array(scores[:10]))
print(y_test[:10])


# ### Confusion Matrix

# In[74]:


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


# ### Plotting the Receiver Operating Characteristic (ROC) Curve and the AUC

# In[75]:


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


# ********1. Adding Class Weight*****************'''
# 
# One approach to these problems is to weight the classes when computing the 
# logistic regression model. '''

# In[76]:


logistic_mod = linear_model.LogisticRegression(class_weight = {0:0.50, 1:0.5}) 
logistic_mod.fit(x_oversample, y_oversample)


# In[77]:


#compute and display the class probabilities for each case.'''
probabilities = logistic_mod.predict_proba(X_test)
print(probabilities[:15,:])


# ****Comparing the weighted model to the unweighted model****
# To find if there is any significant difference with the unweighted model, 
# compute the scores and the metrics and display the metrics.
# The Accuracy of the model may change. So could the precision, recall and F1 score
# 
# '''

# In[78]:


scores = score_model(probabilities, 0.5)
print_metrics(y_test, scores)  
plot_auc(y_test, probabilities)  


# ********2. Finding a better threshold********
# Adjusting the scoring threshold
# 
# Recall that the score is determined by setting the threshold along the 
# sigmoidal or logistic function. 
# It is possible to favor either positive or negative cases by changing the 
# threshold along this curve.

# In[79]:


def test_threshold(probs, labels, threshold):
    scores = score_model(probs, threshold)
    print('')
    print('For threshold = ' + str(threshold))
    print_metrics(labels, scores)

thresholds = [0.45, 0.40, 0.35, 0.3, 0.25, .1, 0]
for t in thresholds:
    test_threshold(probabilities, y_test, t)


# ***Preliminary Conclusion*** 
# 
# The goal of this model to predict loan default. Therefore I aim to select a model with optimal recall and thresholding for default cases.
# 
# Based on a test set of 1000 cases, with 15%  defaulted loans:
# For a 25% threshold, the recall Recall for Postive and Negative cases is nearly balanced, (recall = 0.54 (positive) and 0.66 (negative), respectively.
# However, since the goal of the model is to predtict default, and not repayment, reducing the threshold to 25% improves the Recall for Negative cases to 76%, while maintaining similar precision and F1 scores to the 25% threshold model.
# Therefore, the final model will use a threshold of 25% to optimize predicting loan default.                                                                                         

# 
