import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier



def process_age(df):
    """Process the Age column into pre-defined 'bins' 

    Usage
    ------

    train = process_age(train)
    """
    df["age"] = df["age"].fillna(0)
    cut_points = [10,15,18,30,40,50,60]
    label_names = ["10-15","15-18","18-30","30-40","40-50","50-60"]
    df["age_categories"] = pd.cut(df["age"],cut_points,labels=label_names)
    return df

def process_anciennete(df):
    """Process the anciennete_jour column into pre-defined 'bins' 

    Usage
    ------

    train = process_anciennete(train)
    """
    
    df["anciennete_jour"] = df["anciennete_jour"].fillna(0)
    anciennete_max=df["anciennete_jour"].fillna(0).astype('int64').max()
    cut_points = [i for i in range(0,1825,30)]
    label_names=[str(i)+" mois" for i in range(1,len(cut_points))]
    df["anciennete_mois"] = pd.cut(df["anciennete_jour"],cut_points,labels=label_names)
    return df

def process_recence_mois(df):
    """Process the recence_jours column into pre-defined 'bins' 

    Usage
    ------

    train = process_recence(train)
    """
    df["recence_jours"] = df["recence_jours"].fillna(0)
    cut_points = [i for i in range(0,365,30)]
    label_names=[str(i)+" mois" for i in range(1,len(cut_points))]
    df["recence_mois"] = pd.cut(df["recence_jours"],cut_points,labels=label_names)
    return df

def process_ca_12mois(df):
    """Process the ca_12mois column into pre-defined 'bins' 

    Usage
    ------

    train = process_ca_12mois(train)
    """
    df["ca_12mois"] = df["ca_12mois"].fillna(0)
    cut_points = [0,20,40,60,80,100]
    label_names = ["0-20","20-40","40-60","60-80","80-100"]
    df["ca_12mois_categories"] = pd.cut(df["ca_12mois"],cut_points,labels=label_names)
    return df

def process_Emails_envoyes(df):
    """Process the Emails_envoyes column into pre-defined 'bins' 

    Usage
    ------

    train = process_Emails_envoyes(train)
    """
    df["Emails_envoyes"] = df["Emails_envoyes"].fillna(0)
    cut_points = [10,15,18,30,40,50,60]
    label_names = ["10-15","15-18","18-30","30-40","40-50","50-60"]
    df["Emails_envoyes_categories"] = pd.cut(df["Emails_envoyes"],cut_points,labels=label_names)
    return df

def process_emails_recus_mois(df):
    """Process the emails_recus_mois column into pre-defined 'bins' 

    Usage
    ------

    train = process_emails_recus_mois(train)
    """
    df["emails_recus_mois"] = df["emails_recus_mois"].fillna(0)
    cut_points = [10,15,18,30,40,50,60]
    label_names = ["10-15","15-18","18-30","30-40","40-50","50-60"]
    df["emails_recus_mois_categories"] = pd.cut(df["emails_recus_mois"],cut_points,labels=label_names)
    return df


def create_dummies(df,column_name):
    """Create Dummy Columns (One Hot Encoding) from a single Column

    Usage
    ------

    train = create_dummies(train,"Age")
    """
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

def select_features(df):
    # Remove non-numeric columns, columns that have null values
    df = df.select_dtypes([np.number]).dropna(axis=1)
    all_X = df.drop(["Survived","PassengerId"],axis=1)
    all_y = df["Survived"]
    
    clf = RandomForestClassifier(random_state=1)
    selector = RFECV(clf,cv=10)
    selector.fit(all_X,all_y)
    
    best_columns = list(all_X.columns[selector.support_])
    print("Best Columns \n"+"-"*12+"\n{}\n".format(best_columns))
    
    return best_columns

def select_model(df,features):
    all_X=df[features]
    all_y=df["Survived"]
    
    model_list=[{
    "name": "KNeighborsClassifier",
    "estimator": KNeighborsClassifier(),
    "hyperparameters":
        {
            "n_neighbors": range(1,20,2),
            "weights": ["distance", "uniform"],
            "algorithm": ["ball_tree", "kd_tree", "brute"],
            "p": [1,2]
        }
       },
        {
    "name": "RandomForest",
    "estimator": RandomForestClassifier(),
    "hyperparameters":
        {
            "n_estimators": [4, 6, 9],
            "criterion": ["entropy", "gini"],
            "max_depth": [2, 5, 10],
            "max_features": ["log2", "sqrt"],
            "min_samples_leaf": [1, 5, 8],
            "min_samples_split": [2, 3, 5]
        }
       },
        {
    "name": "LogisticRegression",
    "estimator": LogisticRegression(),
    "hyperparameters":
        {
            "solver": ["newton-cg", "lbfgs", "liblinear"]
        }
       }]
        
        