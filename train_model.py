"""
Training script for Credit Default Logistic Regression model.

Adapted from default_model.py (Jupyter notebook export).
Trains the model and pickles it along with all preprocessing objects
so that app.py can load them and make predictions on new user input.

Run this once: `python train_model.py`
Output: model_artifacts.pkl
"""

# Importing libraries
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
import sklearn.model_selection as ms
import sklearn.metrics as sklm
import numpy.random as nr
from imblearn.over_sampling import SMOTENC


# Load the data
lc1 = pd.read_csv('model_data.csv')

# Numerical and categorical variables (matching original model)
final_cat_features = ['purpose']
numerical_features = ['loan_amnt', 'inq_last_6mths', 'revol_util',
                      'pub_rec_bankruptcies', 'ln_annual_inc']

# Label
labels = np.array(lc1['loan_status_1'])


# Preprocessing: encode categorical strings -> one-hot
# We keep the LabelEncoder and OneHotEncoder so we can apply the
# same transformation to user input later in the Flask app.
def fit_encoders(cat_series):
    """Fit a LabelEncoder + OneHotEncoder pair on a categorical column."""
    label_enc = preprocessing.LabelEncoder()
    label_enc.fit(cat_series)
    encoded_labels = label_enc.transform(cat_series)

    onehot_enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
    onehot_enc.fit(encoded_labels.reshape(-1, 1))

    return label_enc, onehot_enc


def transform_with_encoders(cat_series, label_enc, onehot_enc):
    """Apply already-fit encoders to a categorical column."""
    encoded_labels = label_enc.transform(cat_series)
    return onehot_enc.transform(encoded_labels.reshape(-1, 1)).toarray()


# Fit encoders for 'purpose' and build the feature matrix
purpose_label_enc, purpose_onehot_enc = fit_encoders(lc1['purpose'])
Features = transform_with_encoders(lc1['purpose'],
                                   purpose_label_enc,
                                   purpose_onehot_enc)

# Number of one-hot columns produced by 'purpose' encoding
# (needed to know where numerical columns start when scaling)
n_cat_cols = Features.shape[1]

# Concatenate numerical features
Features = np.concatenate(
    [Features, np.array(lc1[numerical_features])],
    axis=1
)

# Train / test split
nr.seed(9988)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size=1000)

X_train = Features[indx[0], :]
X_test = Features[indx[1], :]
y_train = np.ravel(labels[indx[0]])
y_test = np.ravel(labels[indx[1]])


# SMOTE-NC oversampling
# We tell SMOTE-NC which columns are categorical (the one-hot columns).
cat_indices = list(range(n_cat_cols))
smotenc = SMOTENC(categorical_features=cat_indices, random_state=101)
x_oversample, y_oversample = smotenc.fit_resample(X_train, y_train)


# Scale numerical features (fit on oversampled training data only)
scaler = preprocessing.StandardScaler().fit(x_oversample[:, n_cat_cols:])
x_oversample[:, n_cat_cols:] = scaler.transform(x_oversample[:, n_cat_cols:])
X_test[:, n_cat_cols:] = scaler.transform(X_test[:, n_cat_cols:])


# Fit Logistic Regression with class weights (matches original)
logistic_mod = linear_model.LogisticRegression(
    class_weight={0: 0.50, 1: 0.5},
    max_iter=1000
)
logistic_mod.fit(x_oversample, y_oversample)


# Quick evaluation so we know the model is sane
probabilities = logistic_mod.predict_proba(X_test)


def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:, 1]])


scores = score_model(probabilities, 0.25)


def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0, 0] + '             %5d' % conf[0, 1])
    print('Actual negative    %6d' % conf[1, 0] + '             %5d' % conf[1, 1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])


print_metrics(y_test, scores)


# Save everything Flask will need
artifacts = {
    'model': logistic_mod,
    'purpose_label_enc': purpose_label_enc,
    'purpose_onehot_enc': purpose_onehot_enc,
    'scaler': scaler,
    'numerical_features': numerical_features,
    'purpose_categories': list(purpose_label_enc.classes_),
    'n_cat_cols': n_cat_cols,
    'threshold': 0.25,
}

with open('model_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print('\n[OK] Model + preprocessing saved to model_artifacts.pkl')
print('Purpose categories:', artifacts['purpose_categories'])
