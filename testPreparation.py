import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
idList = test['bidid'].tolist()

# Negative downsample to try and balance click/non-clicks better in the training data
print("negative downsampling")
from sklearn.utils import resample
majority = train[train.click == 0]
minority = train[train.click == 1]
# print(train.click.value_counts())
majorityResampled = resample(majority, replace=False, n_samples=400000)
train = pd.concat([minority, majorityResampled])
# print(train.click.value_counts())

fullData = pd.concat([train, test]).reset_index(drop=True)

# Need to know for the index to split the validation/train data after feature engineering
trainLength = len(train)

del train
del test

# Columns not wanted for the CTR prediction
columnsToDrop = ['bidid', 'userid', 'IP', 'url', 'urlid', 'slotid', 'keypage', 'bidprice', 'domain']

fullData = fullData.drop(columnsToDrop, axis=1)

# Create dummy variables for these columns
columnsForDummies = ['weekday', 'city', 'hour', 'region', 'slotwidth', 'slotheight', 'advertiser', 'creative',
                     'slotprice', 'adexchange', 'slotformat', 'slotvisibility', 'useragent']

print("creating fullData dummmies")
for i in columnsForDummies:
    print("completing: " + i)
    dummies = pd.get_dummies(fullData[i], prefix=i)
    joined = pd.concat([fullData, dummies], axis=1)
    fullData = joined.drop(i, axis=1)

print("usertag dummies")
fullData['usertag'].fillna(value="null", inplace=True)
usertags = fullData.usertag.str.split(',').tolist()
usertagDf = pd.DataFrame(usertags)
del usertags
usertagDummies = pd.get_dummies(usertagDf,prefix='usertag')
del usertagDf
usertagDummies = usertagDummies.groupby(usertagDummies.columns, axis=1).sum()
fullData = pd.concat([fullData, usertagDummies], axis=1)
fullData = fullData.drop('usertag', axis=1)
print("finished dummies")
del usertagDummies

# Split the train and validation data
train = fullData[0:trainLength]
test = fullData[trainLength:]
fullData = 0

# Split the features and target variable for training and validation sets
print("X/y preparation")
X_train = train.drop(['click', 'payprice'], axis=1)
X_test = test.drop(['click', 'payprice'], axis=1)
y_click_train = train['click']
y_pay_train = train['payprice']
train = 0
validation = 0


# Logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

# # Model of choice here
logisticModel = LogisticRegression(class_weight="balanced", C=0.001)
treeModel = DecisionTreeClassifier(max_depth=10, class_weight="balanced")
rfModel = RandomForestClassifier(n_estimators=100, min_samples_leaf=20, n_jobs=-1, class_weight="balanced")

votesSoft = VotingClassifier(estimators=[('lr', logisticModel), ('tr', treeModel), ('rf', rfModel)],
                             voting="soft")

votesSoft.fit(X_train, y_click_train)
predictionProb = votesSoft.predict_proba(X_test)

predictionData = pd.DataFrame()
predictionData['ctrPrediction'] = predictionProb[:,1].tolist()
# predictionData.to_csv("testPredictions.csv", index=False)

from sklearn.linear_model import Lasso
lassoModel = Lasso(precompute=True)
lassoModel.fit(X_train, y_pay_train)
predictions = lassoModel.predict(X_test)

predictionData['payPredictions'] = predictions
# predictionData.to_csv("testPredictions.csv", index=False)
# predictionData = pd.read_csv("testPredictions.csv")

payList = predictionData['payPredictions'].tolist()
clickList = predictionData['ctrPrediction'].tolist()

print(len(idList))
print(len(payList))
print(len(clickList))

# Calculate bids and crate file
payClickRatio = []
for index in range(0, len(clickList)):
    ratio = clickList[index]/payList[index]
    payClickRatio.append(ratio)

bidList = []
bidPrice=300
for index in range(0,len(payClickRatio)):
    id = idList[index]
    if payClickRatio[index]>=0.0042:
        bid = bidPrice
    else:
        bid = 0

    bidList.append([id, bid])

print(len(bidList))

df = pd.DataFrame(bidList, columns=['bidid', 'bidprice'])
df.to_csv("bids.csv", index=False)