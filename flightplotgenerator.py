# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 18:44:57 2017

@author: schaa
"""
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression





X_train=pd.read_pickle('xtrain.pkl')
X_test=pd.read_pickle('xtest.pkl')
y_train=pd.read_pickle('ytrain.pkl')
y_test=pd.read_pickle('ytest.pkl')
mod=pickle.load( open( "logregmod.p", "rb" ) )
print(mod.predict_proba(X_train.iloc[4]))


def createplots(xr,xs,yr,ys,tr,clf):
    pro = clf.predict_proba(xs)
    pre = [ 0 if x < tr else 1 for x in pro[:,1]]
    protr = clf.predict_proba(xr)
    print(metrics.accuracy_score(ys, pre))
    print(metrics.roc_auc_score(ys, pro[:, 1]))
    print(clf.score( xs,ys))
    cm=metrics.confusion_matrix(ys, pre)
    print(cm)
    negref=(cm[0,0]+cm[0,1])
    posref=(cm[1,0]+cm[1,1])
    print(metrics.classification_report(ys, pre))
    fpr, tpr, thresholds=metrics.roc_curve(ys, pro[:,1], pos_label=1)
    fprtr, tprtr, thresholdstr=metrics.roc_curve(yr, protr[:,1], pos_label=1)
    fp, tp =fpr*negref, tpr*posref
    tn, fn = negref-fp, posref- tp
    NPV= tn/(tn+fn)
    PPV= tp/(tp+fp)
    
    thresholds[0]=1    
    fig,p1=plt.subplots()
    p1.plot(thresholds,NPV,label='NPV')
    p1.set_ylabel('NPV')
    p1.set_xlabel('Cutoff')
    plt.legend(loc='center')
    p2=p1.twinx()
    p2.set_ylabel('Flights predicted ontime')
    p2.plot(thresholds,tn+fn,'r',label='Predicted ontime')
    fig.tight_layout()
    plt.legend(loc='center right')
    fig.savefig('Plots/NPV.png')

    fig,p1=plt.subplots()
    p1.plot(thresholds,PPV,label='PPV')
    p1.set_ylabel('PPV')
    p1.set_xlabel('Cutoff')
    plt.legend(loc='center right')
    p2=p1.twinx()
    p2.plot(thresholds,tp+fp,'r',label='Predicted delayed')
    p2.set_ylabel('Flights predicted delayed')
    fig.tight_layout()
    plt.legend()
    fig.savefig('Plots/PPV.png')
    
    fig=plt.figure()
    plt.plot(thresholds,tp/posref,'r',label='Sensitivity')
    plt.plot(thresholds,tn/negref,label='Specificity')
    plt.xlabel('Cutoff')
    plt.ylabel('Rate')
    plt.legend()
    fig.savefig('Plots/sensspec.png')
        
    fig=plt.figure()
    plt.plot(fprtr,tprtr,label='Training set')
    plt.plot(fpr,tpr,'r',label='Test set')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    fig.savefig('Plots/AUC.png')        
        
    fig=plt.figure()
    plt.plot([0,1],[negref/(posref+negref),negref/(posref+negref)],'k',label='NULL')
    plt.plot(thresholds,(tp+tn)/(posref+negref),label='Accuracy')
    plt.xlabel('Cutoff')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    fig.savefig('Plots/accuracy.png')
    
    
    
createplots(X_train, X_test, y_train, y_test,0.17,mod)    