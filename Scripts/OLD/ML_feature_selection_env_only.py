#ML Feature Selection Script (environmental features only)
# %%
####
print('Importing modules...', end='')
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', 'Solver terminated early.*')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from collections import Counter
import os
import time
import joblib
from pickle import dump, load
from scipy.stats import randint
print('done')
# %%
####
# Import data....
def data():
    print("Loading data...", end="")
    X_train = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Env_Only_Uniform_Model/X_train_env_data.csv')
    y_train = np.load("/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Env_Only_Uniform_Model/y_train_env_data.npy", allow_pickle= True)
    #y_train = y_train['group'].to_numpy()
    X_test = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Env_Only_Uniform_Model/X_test_env_data.csv')
    y_test = np.load("/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Env_Only_Uniform_Model/y_test_env_data.npy", allow_pickle= True)
    #y_test = y_test['group'].to_numpy()
    print("done!")
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = data()
#display(y_train)
# %%
####
#Import Modules
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# %%
####
##Sequential Forward Feature Selection
#Logistic Regression
sffs_LR = SequentialFeatureSelector(LogisticRegression(random_state=42),
                                    k_features = (5, 35), forward=True,
                                    floating=True, scoring='roc_auc',
                                    cv=3)
sffs_LR = sffs_LR.fit(X_train, y_train)

selected_features = X_train.columns[list(sffs_LR.k_feature_idx_)]
print(selected_features)
print(sffs_LR.k_score_)

###Output
Index(['ple_y_ss_affected_bad_sum', 'peq_ss_relational_aggs',
       'peq_ss_overt_victim', 'total_substances_used', 'weeks_preterm',
       'injury_totals', 'p_edu_1', 'income_3', 'motor_1', 'motor_3', 'motor_4',
       'motor_5', 'speech_3', 'speech_4', 'speech_5'],
      dtype='object')
0.8029779851361658
# %%
####
#Random Forest Classifier
sffs_RF = SequentialFeatureSelector(RandomForestClassifier(),
                                    k_features = (5, 45), forward=True,
                                    floating=True, scoring='roc_auc',
                                    cv=3)
sffs_RF = sffs_RF.fit(X_train, y_train)

selected_features = X_train.columns[list(sffs_RF.k_feature_idx_)]
print(selected_features)
print(sffs_RF.k_score_)
###Output
Index(['ple_y_ss_affected_bad_sum', 'peq_ss_relational_aggs',
       'peq_ss_overt_victim', 'weeks_preterm', 'injury_totals', 'p_edu_1',
       'p_edu_3', 'income_2', 'income_3', 'motor_3', 'motor_5', 'speech_1',
       'speech_3', 'speech_4', 'speech_5'],
      dtype='object')
0.7855150738553198
# %%
####
###Sequential Backwards Feature Selection
#Logistic Regression
sbfs_LR = SequentialFeatureSelector(LogisticRegression(random_state=42),
                                    k_features = 10, forward=False,
                                    floating=True, scoring='roc_auc',
                                    cv=3)
sbfs_LR = sbfs_LR.fit(X_train, y_train)

selected_features = X_train.columns[list(sbfs_LR.k_feature_idx_)]
print(selected_features)
print(sbfs_LR.k_score_)

###Output
Index(['ple_y_ss_affected_bad_sum', 'peq_ss_relational_aggs',
       'peq_ss_overt_victim', 'weeks_preterm', 'p_edu_1', 'income_3',
       'motor_5', 'speech_1', 'speech_2', 'speech_3'],
      dtype='object')
0.8005386464840126

# %%
#Random Forest Classifier
sbfs_RF = SequentialFeatureSelector(RandomForestClassifier(),
                                    k_features = 10, forward= False,
                                    floating=True, scoring='roc_auc',
                                    cv=3, verbose= 2)
sbfs_RF = sbfs_RF.fit(X_train, y_train)

selected_features = X_train.columns[list(sbfs_RF.k_feature_idx_)]
print(selected_features)
print(sbfs_RF.k_score_)

####Output
Index(['ple_y_ss_affected_bad_sum', 'peq_ss_relational_aggs',
       'peq_ss_overt_victim', 'injury_totals', 'p_edu_1', 'income_1',
       'income_2', 'motor_3', 'speech_1', 'speech_3'],
      dtype='object')
0.7744769440493124
# %%
#####
#Recursive Feature Selection
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.model_selection import RFECV
from sklearn.model_selection import StratifiedKFold
# %%
cv = StratifiedKFold(5)
visualizer = RFECV(LogisticRegression(), cv=cv, scoring='f1_weighted')
visualizer.fit(X_train, y_train)        # Fit the data to the visualizer
visualizer.show()
# %%
# Get list of chosen variables
cols = []
for i in range(len(visualizer.ranking_)):
    if visualizer.ranking_[i] == 1:
        cols.append(X_train.columns[i])

print(cols)
['ple_y_ss_total_bad', 'ple_y_ss_affected_bad_sum', 'peq_ss_relational_aggs', 'peq_ss_overt_victim', 
 'weeks_preterm', 'injury_totals', 'p_edu_1', 'p_edu_3', 'income_1', 'income_2', 'income_3', 'motor_1',
   'motor_2', 'motor_4', 'motor_5', 'speech_1', 'speech_2', 'speech_4', 'speech_5']
# %%
ok_cols = ['ple_y_ss_affected_bad_sum', 'peq_ss_relational_aggs',
       'peq_ss_overt_victim', 'total_substances_used', 'weeks_preterm',
       'injury_totals', 'p_edu_1', 'income_3', 'motor_1', 'motor_3', 'motor_4',
       'motor_5', 'speech_3', 'speech_4', 'speech_5', 'ple_y_ss_affected_bad_sum', 'peq_ss_relational_aggs',
       'peq_ss_overt_victim', 'weeks_preterm', 'injury_totals', 'p_edu_1',
       'p_edu_3', 'income_2', 'income_3', 'motor_3', 'motor_5', 'speech_1',
       'speech_3', 'speech_4', 'speech_5','ple_y_ss_affected_bad_sum', 'peq_ss_relational_aggs',
       'peq_ss_overt_victim', 'weeks_preterm', 'p_edu_1', 'income_3',
       'motor_5', 'speech_1', 'speech_2', 'speech_3', 'ple_y_ss_affected_bad_sum', 'peq_ss_relational_aggs',
       'peq_ss_overt_victim', 'injury_totals', 'p_edu_1', 'income_1',
       'income_2', 'motor_3', 'speech_1', 'speech_3', 'ple_y_ss_total_bad', 'ple_y_ss_affected_bad_sum', 'peq_ss_relational_aggs', 'peq_ss_overt_victim', 
 'weeks_preterm', 'injury_totals', 'p_edu_1', 'p_edu_3', 'income_1', 'income_2', 'income_3', 'motor_1',
   'motor_2', 'motor_4', 'motor_5', 'speech_1', 'speech_2', 'speech_4', 'speech_5']
no_doubles = []
for l in ok_cols:
    if l in no_doubles:
        pass
    else:
        no_doubles.append(l)
# %%
X_train_reduced = X_train[no_doubles]
X_test_reduced = X_test[no_doubles]
# %%
X_test_reduced.to_csv('/Users/madeleineseitz/Desktop/X_test_uniform_env_reduced.csv')
X_train_reduced.to_csv('/Users/madeleineseitz/Desktop/X_train_uniform_env_reduced.csv')
# %%
print(no_doubles)
# %%
['ple_y_ss_affected_bad_sum', 'peq_ss_relational_aggs', 'peq_ss_overt_victim', 'total_substances_used', 
'weeks_preterm', 'injury_totals', 'p_edu_1', 'income_3', 'motor_1', 'motor_3', 'motor_4', 'motor_5',
 'speech_3', 'speech_4', 'speech_5', 'p_edu_3', 'income_2', 'speech_1', 'speech_2', 'income_1',
   'ple_y_ss_total_bad', 'motor_2']
# %%
len(ok_cols)
# %%
len(no_doubles)
# %%
