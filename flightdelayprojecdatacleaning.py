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
from sklearn.cross_validation import train_test_split

rawdata=[]
#rawdata=pd.read_csv('On_Time_On_Time_Performance_2017_1.csv')
for i in range(9):
    rawdata.append(pd.read_csv('t20160'+str(i+1) +'.csv'))
for i in range(9,12):    
    rawdata.append(pd.read_csv('t2016'+str(i+1) +'.csv'))
    
rawdata=pd.concat(rawdata)

#airports=['JFK','LGA','DCA','IAD']
#conraw=conraw.loc[conraw['ORIGIN'].isin(airports)]
#conraw=conraw.loc[conraw['DEST'].isin(airports)]
todrop=[ 'FL_DATE', 'UNIQUE_CARRIER', 'AIRLINE_ID',
        'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID','Unnamed: 20','YEAR','QUARTER', 
  'ORIGIN_STATE_ABR', 'DEST_STATE_ABR','CRS_ELAPSED_TIME','DISTANCE','CRS_ARR_TIME']
notcat=rawdata.drop(todrop,axis=1)
notcat.dropna(inplace=True)
notcat.loc[(notcat['ARR_DELAY'] <= 15),'ARR_DELAY'] =0
notcat.loc[(notcat['ARR_DELAY'] > 15),'ARR_DELAY'] =1

#dumdata['DISTANCE']=((dumdata['DISTANCE']+250)/500).round()*500
#notcat['CRS_ARR_TIME']=(notcat['CRS_ARR_TIME']/100).astype(int)
notcat['CRS_DEP_TIME']=(notcat['CRS_DEP_TIME']/100).astype(int)
cathegoricaldata=[ 'CARRIER','ORIGIN', 'DEST','DAY_OF_WEEK',
                  'CRS_DEP_TIME','DAY_OF_MONTH','MONTH']
dumdata=pd.get_dummies(notcat,columns=cathegoricaldata,drop_first=False)

dumdata=dumdata.reindex_axis(sorted(dumdata.columns), axis=1)

rem, sdumdata = train_test_split(dumdata, test_size=0.01,
                                                    random_state=0)
rem, snotcat = train_test_split(dumdata, test_size=0.01,
                                                    random_state=0)
#,'DAY_OF_WEEK'

dumdata.to_pickle('pickdatwdum.pkl')
sdumdata.to_pickle('picksdatwdum.pkl')
notcat.to_pickle('pickdat.pkl')
snotcat.to_pickle('notcat.pkl')
#datawithdum.to_csv('preprocesseddata.csv', index=False)

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
# 'CRS_DEP_TIME',
# 'CRS_ARR_TIME',
# 'ARR_DELAY',
# 'CRS_ELAPSED_TIME',
# 'DISTANCE']




