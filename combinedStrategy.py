import pandas as pd
import numpy as np

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

trainBidid = train['bidid'].tolist()

fullData = pd.concat([train, validation]).reset_index(drop=True)

# Need to know for the index to split the validation/train data after feature engineering
trainLength = len(train)
del train
del validation

# Split useragent into os and browser
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

print("X/y preparation")
# Split the train and validation data
train = fullData[0:trainLength]
validation = fullData[trainLength:]
fullData = 0

X_train = train.drop('click', axis=1)
y_train = train['click']
X_validation = validation.drop('click', axis=1)
y_validation = validation['click']
train = 0
validation = 0

################################################
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Define optimal individual classifiers
logisticModel = LogisticRegression(class_weight="balanced", C=0.001)
svrModel = LinearSVC(class_weight="balanced", C=0.0001)
treeModel = DecisionTreeClassifier(max_depth=10, class_weight="balanced")
rfModel = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, n_jobs=-1, class_weight="balanced")

# Base model performances
print("fitting base models")
logisticModel.fit(X_train, y_train)
logiPredict = logisticModel.predict(X_validation)
svrModel.fit(X_train, y_train)
svrPredict = svrModel.predict(X_validation)
treeModel.fit(X_train, y_train)
treePredict = treeModel.predict(X_validation)
rfModel.fit(X_train, y_train)
forestPredict = rfModel.predict(X_validation)

print("base model scores")
print("logistic AUC: " + str(roc_auc_score(y_validation, logisticModel.predict(X_validation))))
print("SVM AUC: " + str(roc_auc_score(y_validation, svrModel.predict(X_validation))))
print("decision tree AUC: " + str(roc_auc_score(y_validation, treeModel.predict(X_validation))))
print("random forest AUC: " + str(roc_auc_score(y_validation, rfModel.predict(X_validation))))

# Ensemble Voting
print("fitting ensemble voting")
votesHard = VotingClassifier(estimators=[('lr', logisticModel), ('sv', svrModel), ('tr', treeModel), ('rf', rfModel)],
                             voting="hard")
votesSoft = VotingClassifier(estimators=[('lr', logisticModel), ('tr', treeModel), ('rf', rfModel)],
                             voting="soft")
# Ensmble voting scores
votesHard.fit(X_train, y_train)
votesSoft.fit(X_train, y_train)
print("Hard AUC: " + str(roc_auc_score(y_validation, votesHard.predict(X_validation))))
print("Hard f1: " + str(f1_score(y_validation, votesHard.predict(X_validation))))
print("Soft AUC: " + str(roc_auc_score(y_validation, votesSoft.predict(X_validation))))
print("Soft f1: " + str(f1_score(y_validation, votesSoft.predict(X_validation))))

# Save predictions to disk
predictionProba = votesSoft.predict_proba(X_validation)
predictionData = pd.DataFrame()
predictionData['predictions'] = predictionProba[:,1].tolist()
predictionData.to_csv("validation_predictions.csv", index=False)
print("finished")


