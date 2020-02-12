
import sklearn
import pandas as pd
import numpy as np
import pyodbc
import sys
import time
import hashlib
import seaborn as sns
import fonctions
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier



train=pd.read_csv(r"C:\Users\StéphaneDEHLAVICENOV\OneDrive - CENOVA\Projets\Uplfit\ope10_40_v2_export_train.csv",encoding="utf-8",sep=",",engine='python')
holdout=pd.read_csv(r"C:\Users\StéphaneDEHLAVICENOV\OneDrive - CENOVA\Projets\Uplfit\ope10_40_v2_export_test.csv",encoding="utf-8",sep=",",engine='python')


def process_df(df):
    df=fonctions.process_missing(df,train)
    df=fonctions.process_age(df)
    df=fonctions.process_fare(df)
    df=fonctions.process_cabin(df)
    df=fonctions.process_titles(df)
    
    columns=[ "Age_categories", "Fare_categories","Title", "Cabin_type","Sex"]
    for col in columns:
        df=fonctions.create_dummies(df,col)
    
    return df


train=process_df(train)