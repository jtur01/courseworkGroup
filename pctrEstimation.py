import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

############### IMPORTING FILES ################
train = pd.read_csv("train.csv")
validation = pd.read_csv("validation.csv")
################################################
validationBidid = validation['bidid'].tolist()

############## FEATURE ENGINEERING #############
# Negative downsample to try and balance click/non-clicks better in the training data
print("negative downsampling")
from sklearn.utils import resample
majority = train[train.click == 0]
minority = train[train.click == 1]
majorityResampled = resample(majority, replace=False, n_samples=400000, random_state=12)
train = pd.concat([minority, majorityResampled])

fullData = pd.concat([train, validation]).reset_index(drop=True)

# Need to know for the index to split the validation/train data after feature engineering
trainLength = len(train)
del train
del validation

# Split useragent into os and broswer
newData = pd.DataFrame(fullData.useragent.str.split('_').tolist(), columns=['os', 'browser'])
fullData = pd.concat([fullData, newData], axis=1)

# Columns not wanted for the CTR prediction
columnsToDrop = ['bidid', 'userid', 'IP', 'url', 'urlid', 'slotid', 'keypage', 'bidprice',
                 'payprice', 'domain', 'useragent']
fullData = fullData.drop(columnsToDrop, axis=1)

# Create dummy variables for these columns
columnsForDummies = ['weekday', 'hour', 'city', 'region', 'slotwidth', 'slotheight', 'advertiser', 'creative',
                     'slotprice', 'adexchange', 'slotformat', 'slotvisibility', 'os', 'browser']

print("creating fullData dummmies")
for i in columnsForDummies:
    print("completing: " + i)
    dummies = pd.get_dummies(fullData[i], prefix=i)
    joined = pd.concat([fullData, dummies], axis=1)
    fullData = joined.drop(i, axis=1)

print("completing: usertag")
fullData['usertag'].fillna(value="null", inplace=True)
usertags = fullData.usertag.str.split(',').tolist()
usertagDf = pd.DataFrame(usertags)
usetags = 0
usertagDummies = pd.get_dummies(usertagDf, prefix='usertag')
usertagDf = 0
usertagDummies = usertagDummies.groupby(usertagDummies.columns, axis=1).sum()
fullData = pd.concat([fullData, usertagDummies], axis=1)
fullData = fullData.drop('usertag', axis=1)
print("finished dummies")
usertagDummies = 0

# Split the train and validation data
train = fullData[0:trainLength]
validation = fullData[trainLength:]
fullData = 0

# Split the features and target variable for training and validation sets
print("X/y preparation")
X_train = train.drop('click', axis=1)
y_train = train['click']
X_validation = validation.drop('click', axis=1)
y_validation = validation['click']
train = 0
validation = 0
################################################


###### MODEL FITTING ###########################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# first try logistic regression, tuning the inverse regularisation strength
strengths = [10, 1, 0.1, 0.01, 0.001, 0.0001]
aucScores = []
for strengthC in strengths:
    logisticModel = LogisticRegression(class_weight="balanced", C=strengthC)
    print("fitting logistic regression: " + str(strengthC))
    logisticModel.fit(X_train, y_train)
    predictions = logisticModel.predict(X_validation)
    print("finished fitting")
    aucScores.append([strengthC, roc_auc_score(y_validation, predictions)])

for score in aucScores:
    print("Strength: " + str(score[0]) +"\tAUC: " + str(score[1]))


#Optimum logistic regression model
print("fitting model")
logisticModel = LogisticRegression(class_weight="balanced", C=0.001)
logisticModel.fit(X_train, y_train)
prediction = logisticModel.predict(X_validation)
predictionProba = logisticModel.predict_proba(X_validation)
print(roc_auc_score(y_validation, prediction))
print("finished fitting")


#Recalibrate the results
def recalibrate(pr):
    return pr/(pr+((1-pr)/0.16))


predictList = predictionProba[:, 1].tolist()
recaList = []
for i in predictList:
    new = recalibrate(i)
    recaList.append(new)

# Save to disk
predictionData = pd.DataFrame()
predictionData['predictions'] = recaList
predictionData['bidid'] = validationBidid
predictionData.to_csv("validation_predictions.csv", index=False)