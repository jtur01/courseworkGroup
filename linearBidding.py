import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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

df = pd.read_csv("validation.csv")
preds = pd.read_csv("validation_predictions.csv")
predList = preds['predictions'].tolist()
clickList = df['click'].tolist()
payList = df['payprice'].tolist()

# Put the data to a list
clickCostPredList = []
for i in range(0, len(preds)-1):
    click = clickList[i]
    pay = payList[i]
    pred = predList[i]
    tempTuple = (click, pay, pred)
    clickCostPredList.append(tempTuple)

# Finding the optimum base bid
avgCTR = df['click'].sum()/len(df)*100
budget = 6250
linearBidList = []
for constant in range(1,300):
    print(constant)
    impressions = 0
    clicks = 0
    totalPrice = 0
    for imp in clickCostPredList:
        bidprice = constant * (imp[2]/avgCTR)
        if bidprice > imp[1]:
            if totalPrice+(imp[1]/1000)<=6250:
                impressions = impressions + 1
                totalPrice = totalPrice + (imp[1]/1000)
                clicks = clicks + imp[0]
            else:
                continue
    CTR = clicks/impressions*100
    CPM = (totalPrice/impressions)*1000
    if clicks == 0:
        CPC = 0
    else:
        CPC = totalPrice/clicks
    linearBidList.append([constant, impressions, clicks, CTR, totalPrice, CPM, CPC])

linearData = pd.DataFrame(linearBidList, columns=['constant', 'impressions', 'clicks', 'CTR', 'totalPrice', 'CPM', 'CPC'])

# linearData.to_csv("linearBidOutput.csv", index=False)
# linearData = pd.read_csv("linearBidOutput.csv")

graphResults('constant', 'clicks', 'CTR', linearData, save=False)
graphResults('constant', 'impressions', 'CPM', linearData, save=False)

linearData.sort_values('clicks', ascending=False, inplace=True)
print(linearData.head())

