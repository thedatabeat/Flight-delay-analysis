# -*- coding: utf-8 -*-
"""
Created on Sat May 13 14:43:50 2017

@author: schaa
"""

import pandas as pd
import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import itertools
import pickle 
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.neural_network import MLPClassifier

from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

kf=KFold(n_splits=200, random_state=None, shuffle=False)



data=pd.read_pickle('pickdat.pkl')
dumdata=pd.read_pickle('pickdatwdum.pkl')

X=dumdata.drop('ARR_DELAY',axis=1)
y=dumdata['ARR_DELAY']
y=dumdata['ARR_DELAY']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,
                                                    random_state=None)

y_trainsw=1-y_train
y_testsw=1-y_test

pickle.dump(list(data['ORIGIN'].unique()),open( "origins.p", "wb" ),protocol=2)
pickle.dump(list(data['DEST'].unique()),open( "destinations.p", "wb" ),protocol=2)
def logreg(xr,xs,yr,ys):
    clf = LogisticRegression(class_weight={0: 1, 1: 4.7}, 
                             verbose=5,penalty= 'l2')
    clf = clf.fit(xr, yr)
    pre = clf.predict(xs)
    pro = clf.predict_proba(xs)
    print(metrics.accuracy_score(ys, pre))
    print(metrics.roc_auc_score(ys, pro[:, 1]))
    print(clf.score( xs,ys))
    print(metrics.confusion_matrix(ys, pre))
    print(metrics.classification_report(ys, pre))
    fpr, tpr, thresholds=metrics.roc_curve(ys, pro[:,1], pos_label=1)
    plt.plot(fpr,tpr)
#logreg(X_train, X_test, y_train, y_test)

def logregR(xr,xs,yr,ys):
    formula='ARR_DELAY ~ CARRIER_AA'
    for l in range(1,len(list(xr))): formula+='+'+list(xr)[l]
    dat=pd.concat([xr,yr],axis=1)
    clf = smf.glm(formula=formula, data=dat, family=sm.families.Binomial())
    result = clf.fit()
    print(result.summary())
#logregR(X_train, X_test, y_train, y_test)


def logregnoweightstr(xr,xs,yr,ys,tr):
    clf = LogisticRegression(class_weight=None, 
                             verbose=5,penalty= 'l2')
    clf = clf.fit(xr, yr)
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
    plt.legend()
    p2=p1.twinx()
    p2.plot(thresholds,tn+fn,'r',label='Predicted ontime')
    fig.tight_layout()
    plt.legend()

    fig,p1=plt.subplots()
    p1.plot(thresholds,PPV,label='PPV')
    plt.legend()
    p2=p1.twinx()
    p2.plot(thresholds,tp+fp,'r',label='Predicted delayed')
    fig.tight_layout()
    plt.legend()
    
    plt.figure()
    plt.plot(thresholds,tp/posref,'r',label='Sensitivity')
    plt.plot(thresholds,tn/negref,label='Specificity')
    plt.legend()
        
    plt.figure()
    plt.plot(fprtr,tprtr,label='Training set')
    plt.plot(fpr,tpr,'r',label='Test set')
    plt.legend()        
        
    plt.figure()
    plt.plot([0,1],[negref/(posref+negref),negref/(posref+negref)],'k',label='NULL')
    plt.plot(thresholds,(tp+tn)/(posref+negref),label='Accuracy')
    plt.legend()
    plt.show()
    
    
    #plt.plot(fpr,tpr)
    return  clf

def generatemodel():
    mod=logregnoweightstr(X_train, X_test, y_train, y_test,0.17)
    pickle.dump(mod,open( "logregmod.p", "wb" ),protocol=2)

    X_train.to_pickle('xtrain.pkl')
    X_test.to_pickle('xtest.pkl')
    y_train.to_pickle('ytrain.pkl')
    y_test.to_pickle('ytest.pkl')
    return mod
    



#logregnoweightstr(X_train, X_test, y_trainsw, y_testsw,1-0.17)
def randomforest(xr,xs,yr,ys):
    clf = RandomForestClassifier(n_estimators=50,class_weight={0: 1, 1: 1000} )
    clf = clf.fit(xr, yr)
    pre = clf.predict(xs)
    pro = clf.predict_proba(xs)
    print(metrics.accuracy_score(ys, pre))
    print(metrics.roc_auc_score(ys, pro[:, 1]))
    print(clf.score( xs,ys))
    print(metrics.confusion_matrix(ys, pre))
    print(metrics.classification_report(ys, pre))

def neural(xr,xs,yr,ys):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-10, verbose = True,
                        hidden_layer_sizes=(5, 2), random_state=None)
    clf = clf.fit(xr, yr)
    pre = clf.predict(xs)
    pro = clf.predict_proba(xs)
    print(metrics.accuracy_score(ys, pre))
    print(metrics.roc_auc_score(ys, pro[:, 1]))
    print(clf.score( xs,ys))
    print(metrics.confusion_matrix(ys, pre))
    print(metrics.classification_report(ys, pre))   
#randomforest(X_train, X_test, y_train, y_test)

def adaboost(xr,xs,yr,ys):
    clf = AdaBoostClassifier(base_estimator=LogisticRegression(class_weight=None, 
                                                               verbose=5,penalty= 'l2'),
                             n_estimators=3,
                             learning_rate=1.0, algorithm='SAMME.R',
                             random_state=None)
    clf = clf.fit(xr, yr)
    pre = clf.predict(xs)
    pro = clf.predict_proba(xs)
    print(metrics.accuracy_score(ys, pre))
    print(metrics.roc_auc_score(ys, pro[:, 1]))
    print(clf.score( xs,ys))
    print(metrics.confusion_matrix(ys, pre))
    print(metrics.classification_report(ys, pre))   
    
def gradboost(xr,xs,yr,ys):
    clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, verbose=5,
                                     max_depth=3, random_state=None)
    clf = clf.fit(xr, yr)
    pre = clf.predict(xs)
    pro = clf.predict_proba(xs)
    print(metrics.accuracy_score(ys, pre))
    print(metrics.roc_auc_score(ys, pro[:, 1]))
    print(clf.score( xs,ys))
    print(metrics.confusion_matrix(ys, pre))
    print(metrics.classification_report(ys, pre))  
#gradboost(X_train, X_test, y_train, y_test)    
#adaboost(X_train, X_test, y_train, y_test)    
#neural(X_train, X_test, y_train, y_test)
#print(model2.score( X_test,y_test))
#print(metrics.confusion_matrix(y_test, predicted))
#print(metrics.classification_report(y_test, predicted))
def testgro(a,b):
    return abs(a.mean()-b.mean())/np.sqrt(a.std()**2/a.shape[0]+b.std()**2/b.shape[0])


def testsignificance(d):
    out=pd.DataFrame(columns=list(d),index=list(d))
    for x in itertools.product(list(d), repeat=2):
        out[x[0]].loc[x[1]]=testgro(d[x[0]],d[x[1]])
    return out
#res=testsignificance(total)
    
#sig=res[res>3]

def testsigall(labels,spl):
    kf=KFold(n_splits=spl, random_state=None, shuffle=False)
    maximums=pd.DataFrame(columns=labels)
    for lab in labels:
        total=pd.DataFrame(columns=data[lab].unique())
        i=0
        for x,y in kf.split(range(data.shape[0])):
            i+=1
            if i%2==0:
                continue
            delbylab=data.iloc[y].groupby(lab)['ARR_DELAY'].mean()
            total=total.append(pd.DataFrame(delbylab).T)
        res=testsignificance(total)
        maximums.set_value(0,lab,res.max().max())
        #print(maximums)
    return maximums
labs=   ['MONTH',
 'DAY_OF_MONTH',
 'DAY_OF_WEEK',
 'CARRIER',
 'ORIGIN',
 'DEST',
 'CRS_DEP_TIME']
#maximumss=testsigall(labs,5)


def splittest(n,labels):
    maxi=pd.DataFrame(columns=labels)
    for i in n:
        print(testsigall(labels,i))
        maxi=maxi.append(testsigall(labels,i))
    return maxi
rounds=[100,500,700,1000,1300]
#supertest=splittest(rounds,labs)



    
#print(testgro(total['NK'],total['AA']))
    
    
#delbycarr=data.groupby('CARRIER')['ARR_DELAY'].sum()
#delbycarrm=data.groupby('CARRIER')['ARR_DELAY'].mean()

#delbydestrm=data.groupby('DEST')['ARR_DELAY'].mean()

#delbydesstatrm=data.groupby('DEST_STATE_ABR')['ARR_DELAY'].mean()
#delmondaytrm=data.groupby('DAY_OF_WEEK')['ARR_DELAY'].mean()

#delbycarr=data.iloc[y].groupby('CARRIER')['ARR_DELAY'].mean()
# keys ['YEAR',
# 'QUARTER',
# 'MONTH',
# 'DAY_OF_MONTH',
# 'DAY_OF_WEEK',
# 'CARRIER',
# 'ORIGIN',
# 'ORIGIN_STATE_ABR',
# 'DEST',
# 'DEST_STATE_ABR',
# 'ARR_DELAY',
# 'DISTANCE']

#datadum['DISTANCE'].apply(lambda x: int(500*round(x/500.0)))
