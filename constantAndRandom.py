import pandas as pd
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import shuffle, randint

# Results graphing function
def graphResults(xLab, y1Lab, y2Lab, frame, save=False):
    fig, ax1 = plt.subplots()
    ax1.plot(frame[xLab], frame[y1Lab], 'b-')
    ax1.set_xlabel(xLab)
    ax1.set_ylabel(y1Lab, color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(frame[xLab], frame[y2Lab], 'g-')
    ax2.set_ylabel(y2Lab, color='g')
    ax2.tick_params('y', colors='g')
    fig.tight_layout()
    if save == True:
        plt.savefig(str(xLab)+ str(y1Lab) + "Graph.png")
    plt.show()

###### DATA LOADING AND PREP #####
# # Load dataset
df = pd.read_csv("train.csv")

clickList = df['click'].tolist()
payList = df['payprice'].tolist()

# Put click and payprice into a list for ease of use later
clickCostList = []
for index in range(0, len(clickList)):
    click = clickList[index]
    pay = payList[index]
    tempTuple = (click, pay)

    clickCostList.append(tempTuple)
#
# Create and shuffle copies of the click cost list
clickCostListOne = clickCostList.copy()
clickCostListTwo = clickCostList.copy()
clickCostListThree = clickCostList.copy()
clickCostListFour = clickCostList.copy()
shuffle(clickCostListOne)
shuffle(clickCostListTwo)
shuffle(clickCostListThree)
shuffle(clickCostListFour)
turnList = [clickCostList, clickCostListOne, clickCostListTwo, clickCostListThree, clickCostListFour]
##################################

# Constant bidding function
# Set the budget
budget = 6250
constantList = []
# Dictionaires to collect values over the shuffle iterations, can be used to get averages later
for bidPrice in range(df['payprice'].min()+1, df['payprice'].max()+1):
    print(bidPrice)
    impList = []
    clickList = []
    totalList = []
    CTRList = []
    CPMList = []
    CPCList = []
    for clickCost in turnList:
        impressions = 0
        clicks = 0
        totalPrice = 0
        # Initialise the dictionaries
        for clickPayPair in clickCost:
            if bidPrice > clickPayPair[1]:
                # Check if we have reached the budget, if so break the loop
                if totalPrice+(clickPayPair[1]/1000) <= budget:
                    impressions = impressions + 1
                    totalPrice = totalPrice + (clickPayPair[1]/1000)
                    clicks = clicks + clickPayPair[0]
                else:
                    continue
        # Calculate metrics
        CTR = clicks/impressions*100
        CPM = (totalPrice/impressions)*1000
        # avoid division by zero error
        if clicks == 0:
            CPC = 0
        else:
            CPC = totalPrice/clicks
        # for each shuffled list add the metrics
        impList.append(impressions)
        clickList.append(clicks)
        totalList.append(totalPrice)
        CTRList.append(CTR)
        CPMList.append(CPM)
        CPCList.append(CPC)
    # need to average the metrics over the shuffled lists
    constantList.append([bidPrice, np.mean(impList), np.mean(clickList), np.mean(CTRList), np.mean(totalList),
                         np.mean(CPMList), np.mean(CPCList)])

constantDataB = pd.DataFrame(constantList, columns=['bidPrice', 'impressions', 'clicks', 'CTR', 'totalPrice', 'CPM', 'CPC'])
# constantDataB.to_csv("constantOutput.csv", index=False)
# constantDataB = pd.read_csv("constantOutput.csv")
graphResults('bidPrice', 'clicks', 'CTR', constantDataB)
graphResults('bidPrice', 'impressions', 'CPC', constantDataB)

#
# Top ten bids by clicks
constantDataB.sort_values('clicks', ascending=False, inplace=True)
print("top 10 by clicks:")
print(constantDataB.head(n=5))

# Only use three shuffled versions as random process is much slower
# # Random bidding function
budget = 6250
boundList = []
# Random bid between two constants (upper and lower bounds). Use steps of 10 otherwise it is much too slow
for lowerBound in range(df['payprice'].min()+10,df['payprice'].max(),10):
    print(lowerBound)
    for upperBound in range(lowerBound+10, (df['payprice'].max()),10):
        print(upperBound)
        impList = []
        clickList = []
        totalList = []
        CTRList = []
        CPMList = []
        CPCList = []
        for clickCost in turnList:
            impressions = 0
            clicks = 0
            totalPrice = 0
            for clickPayPair in clickCostList:
                bidPrice = randint(lowerBound,upperBound)
                if bidPrice > clickPayPair[1]:
                    if totalPrice+(clickPayPair[1]/1000) <= budget:
                        impressions = impressions+1
                        totalPrice = totalPrice + (clickPayPair[1]/1000)
                        clicks = clicks + clickPayPair[0]
                    else:
                        continue

            # Calculate metrics
            CTR = clicks / impressions * 100
            CPM = (totalPrice / impressions) * 1000
            # avoid division by zero error
            if clicks == 0:
                CPC = 0
            else:
                CPC = totalPrice / clicks
            # for each shuffled list add the metrics
            impList.append(impressions)
            clickList.append(clicks)
            totalList.append(totalPrice)
            CTRList.append(CTR)
            CPMList.append(CPM)
            CPCList.append(CPC)

        boundList.append([lowerBound, upperBound, np.mean(impList), np.mean(clickList), np.mean(CTRList),
                          np.mean(totalList), np.mean(CPMList), np.mean(CPCList)])

randomData = pd.DataFrame(boundList, columns=['lower', 'upper', 'impressions', 'clicks', 'CTR', 'totalPrice', 'CPM',
                                              'CPC'])
# print(randomData.head())
# Save to disk for quicker future use
randomData.to_csv("randomOutputAv.csv", sep=",", index=False)
randomData = pd.read_csv("randomOutputAv.csv")

# # # #
# # # # Plot graph for lower and upper bound clicks
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(randomData['lower'], randomData['upper'], randomData['clicks'], cmap=plt.cm.plasma, linewidth=0.2)
# ax.view_init(40, 320)
plt.xlabel("lower")
plt.ylabel("upper")
ax.set_zlabel('Clicks')
plt.show()
#
randomData.sort_values('clicks', ascending=False, inplace=True)
print("top 10 by clicks:")
print(randomData.head(n=10))

# Testing optimal constant and random values from the training data on the validation data

df = pd.read_csv('validation.csv')

clickList = df['click'].tolist()
payList = df['payprice'].tolist()

# Put click and payprice into a list for ease of use later
clickCostList = []
for index in range(0, len(clickList)):
    click = clickList[index]
    pay = payList[index]
    tempTuple = (click, pay)

    clickCostList.append(tempTuple)

bidPrice = 31
impressions = 0
clicks = 0
totalPrice = 0
budget = 6250
for clickPayPair in clickCostList:
    if bidPrice > clickPayPair[1]:
        if totalPrice + (clickPayPair[1] / 1000) <= budget:
            impressions = impressions + 1
            totalPrice = totalPrice + (clickPayPair[1] / 1000)
            clicks = clicks + clickPayPair[0]
        else:
            continue

print("constant bidding price at 31:")
print("clicks: " + str(clicks))
print("impressions: " + str(impressions))
print("total spent: " + str(totalPrice))
print("CTR: " + str(clicks/impressions*100))
print("CPC: " + str(totalPrice/clicks))
print("CPM: " + str(totalPrice/impressions*1000))

#  Actual optimal constant bidding price on the validation set
budget = 6250
resultList = []
for bidPrice in range(1,301):
    print(bidPrice)
    clicks = 0
    impressions = 0
    totalPrice = 0
    for clickPayPair in clickCostList:
        if bidPrice > clickPayPair[1]/1000:
            if totalPrice+(clickPayPair[1]/1000) <= budget:
                impressions = impressions+1
                totalPrice = totalPrice + (clickPayPair[1]/1000)
                clicks = clicks + clickPayPair[0]
            else:
                continue

    CTR = clicks / impressions * 100
    CPM = (totalPrice / impressions) * 1000
    if clicks == 0:
        CPC = 0
    else:
        CPC = totalPrice / clicks
    resultList.append([bidPrice, impressions, clicks, CTR, CPM, totalPrice, CPC])


# Save results to dataframe
constantValidation = pd.DataFrame(resultList, columns=['bidPrice', 'impressions', 'clicks', 'CTR', 'CPM', 'totalPrice',
                                                       'CPC'])
# Save dataframe to csv so results can be laoded without needing to rerun bidding process
# constantValidation.to_csv("constValResults.csv", index=False)
# constantValidation = pd.read_csv("constValResults.csv")

# Graph constant bidding on validation set
graphResults('bidPrice', 'clicks', 'CTR', constantValidation, save=False)
graphResults('bidPrice', 'impressions', 'CPM', constantValidation, save=False)

# constantValidation.sort_values('clicks', ascending=False, inplace=True)
constantValidation.sort_values('clicks', ascending=False, inplace=True)
print("top 10 by clicks:")
print(constantValidation.head(n=3))
#

# Testing optimal random on validation set
lower = 20
upper = 30
impressions = 0
clicks = 0
totalPrice = 0
budget = 6250
for clickPayPair in clickCostList:
    bidPrice = randint(lower,upper)
    if bidPrice > clickPayPair[1]:
        if totalPrice + (clickPayPair[1] / 1000) <= budget:
            impressions = impressions + 1
            totalPrice = totalPrice + (clickPayPair[1] / 1000)
            clicks = clicks + clickPayPair[0]
        else:
            continue

print("constant bidding price at 31:")
print("clicks: " + str(clicks))
print("impressions: " + str(impressions))
print("total spent: " + str(totalPrice))
print("CTR: " + str(clicks/impressions*100))
print("CPC: " + str(totalPrice/clicks))
print("CPM: " + str(totalPrice/impressions*1000))

# Actual optimal random bidding parameters
boundList = []
budget = 6250
# Random bid between two constants (upper and lower bounds). Use steps of 10 otherwise it is much too slow
for lowerBound in range(df['payprice'].min()+10,df['payprice'].max(),10):
    print(lowerBound)
    for upperBound in range(lowerBound+10, (df['payprice'].max()),10):
        print(upperBound)
        impressions = 0
        clicks = 0
        totalPrice = 0
        for clickPayPair in clickCostList:
            bidPrice = randint(lowerBound,upperBound)
            if bidPrice > clickPayPair[1]:
                if totalPrice+(clickPayPair[1]/1000)<=budget:
                    impressions = impressions+1
                    totalPrice = totalPrice + (clickPayPair[1]/1000)
                    clicks = clicks + clickPayPair[0]
                else:
                    continue

        if clicks == 0:
            CPC = 0
        else:
            CPC = totalPrice / clicks

        CTR = clicks/impressions*100
        boundList.append([lowerBound, upperBound, impressions, clicks, CTR, totalPrice, CPC])

randomData = pd.DataFrame(boundList, columns=['lower', 'upper', 'impressions', 'clicks', 'CTR', 'totalPrice', 'CPC'])
# save results to disk for quicker access if future evaluation is needed
# randomData.to_csv("randomOutput2.csv", sep=",", index=False)
# randomData = pd.read_csv("randomOutput2.csv")

# Plot graph for lower and upper bound clicks
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(randomData['lower'], randomData['upper'], randomData['clicks'], cmap=plt.cm.plasma, linewidth=0.2)
ax.view_init(45, 300)
plt.xlabel("lower")
plt.ylabel("upper")
ax.set_zlabel('Clicks')
plt.show()

randomData.sort_values('clicks', ascending=False, inplace=True)
print(randomData.head())

