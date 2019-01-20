import pandas as pd
import numpy as np
from sklearn.metrics import recall_score,precision_score,confusion_matrix,precision_recall_curve
import matplotlib.pyplot as plt



def pre_processing(df,value_for_na = -1):
    df["class"] = df["class"].map({"neg":0,"pos":1})
    df = df.replace("na",value_for_na)
    for col in df.columns:
        if col != "origin":
            df[col] = pd.to_numeric(df[col])
    return df

def get_tag(name):
    return name.split("_")[0]

def show_results(y_true,y_pred,treshold=0.5):
    print("precision",precision_score(y_true,y_pred>treshold))
    print("recall",recall_score(y_true,y_pred>treshold))
    confusionMatrix = confusion_matrix(y_true,y_pred>treshold)
    print(confusionMatrix)
    precision, recall = graph_curve(y_true,y_pred)
    return (precision, recall)

def graph_curve(y_true,y_pred,label=None):
    precision, recall, thresholds = precision_recall_curve(y_true,y_pred)
    plt.step(recall,precision,where="post",label=label)
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    if label is  not None:
        plt.legend()
    return (precision, recall)

def the_money_equivalent(dtrain,preds,treshold):
    labels = dtrain
    # return a pair metric_name, result. The metric name must not contain a colon (:) or a space
    # since preds are margin(before logistic transformation, cutoff at 0)
    error = sum(np.where((preds<treshold) & (labels==1),500,0))
    error += sum(np.where((preds>=treshold) & (labels==0),10,0))
    return error

def create_evalmoney(treshold):
    def evalerror_money(preds, dtrain):
        #print(preds[0])
        labels = dtrain.get_label()
            # return a pair metric_name, result. The metric name must not contain a colon (:) or a space
            # since preds are margin(before logistic transformation, cutoff at 0)
        preds = 1. / (1. + np.exp(-preds))
        error = sum(np.where((preds<treshold) & (labels==1),500,0))
        error += sum(np.where((preds>=treshold) & (labels==0),10,0))
        return 'money_equivalent', error, False
    return evalerror_money

def logregobjspecial(weight):
    def custom(preds, train_data):
        labels = train_data.get_label()
        weights = np.where(labels == 1.0, weight, 1.0)
        preds = 1. / (1. + np.exp(-preds))
        grad = preds - labels
        hess = preds * (1. - preds)
        return grad * weights, hess * weights
    return custom