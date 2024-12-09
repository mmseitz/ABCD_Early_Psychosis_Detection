#ML Feature Selection Script (anatomical data only)
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
    X_train = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Anat_Only_Uniform_Model/X_train_anat_only_data.csv')
    y_train = np.load("/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Anat_Only_Uniform_Model/y_train_anat_data.npy", allow_pickle= True)
    #y_train = y_train['group'].to_numpy()
    X_test = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Anat_Only_Uniform_Model/X_test_anat_only_data.csv')
    y_test = np.load("/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Anat_Only_Uniform_Model/y_test_anat_only_data.npy", allow_pickle= True)
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
# Index(['smri_thick_cdk_cdmdfrlh', 'smri_thick_cdk_cuneuslh',
#        'smri_thick_cdk_fusiformlh', 'smri_thick_cdk_ifpllh',
#        'smri_thick_cdk_ihcatelh', 'smri_thick_cdk_parahpallh',
#        'smri_thick_cdk_pericclh', 'smri_thick_cdk_postcnlh',
#        'smri_thick_cdk_ptcatelh', 'smri_thick_cdk_precnlh',
#        'smri_thick_cdk_rracatelh', 'smri_thick_cdk_sufrlh',
#        'smri_thick_cdk_supllh', 'smri_thick_cdk_banksstsrh',
#        'smri_thick_cdk_fusiformrh', 'smri_thick_cdk_iftmrh',
#        'smri_thick_cdk_lobfrrh', 'smri_thick_cdk_paracnrh',
#        'smri_thick_cdk_smrh', 'smri_thick_cdk_insularh',
#        'smri_vol_cdk_cdmdfrlh', 'smri_vol_cdk_iftmlh', 'smri_vol_cdk_mobfrlh',
#        'smri_vol_cdk_paracnlh', 'smri_vol_cdk_precnlh', 'smri_vol_cdk_smlh',
#        'smri_vol_cdk_banksstsrh', 'smri_vol_cdk_lobfrrh',
#        'smri_vol_cdk_mdtmrh', 'smri_vol_cdk_parahpalrh',
#        'smri_vol_cdk_precnrh', 'smri_vol_cdk_sufrrh', 'smri_vol_cdk_tmpolerh',
#        'smri_vol_cdk_insularh'],
#       dtype='object')
# 0.6455031511086325
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
# Index(['smri_thick_cdk_ehinalrh', 'smri_thick_cdk_lobfrrh',
#        'smri_vol_cdk_iftmlh', 'smri_vol_cdk_linguallh',
#        'smri_vol_cdk_pericclh', 'smri_vol_cdk_precnlh', 'smri_vol_cdk_sufrlh',
#        'smri_vol_cdk_banksstsrh', 'smri_vol_cdk_lobfrrh',
#        'smri_vol_cdk_totallh', 'smri_vol_cdk_total'],
#       dtype='object')
# 0.6286062669754521
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
# Index(['smri_thick_cdk_cuneuslh', 'smri_thick_cdk_ihcatelh',
#        'smri_thick_cdk_pericclh', 'smri_thick_cdk_precnlh',
#        'smri_thick_cdk_rracatelh', 'smri_thick_cdk_sufrlh',
#        'smri_thick_cdk_fusiformrh', 'smri_vol_cdk_mobfrlh',
#        'smri_vol_cdk_paracnlh', 'smri_vol_cdk_mdtmrh'],
#       dtype='object')
# 0.6129985038365813

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
# Index(['smri_thick_cdk_fusiformlh', 'smri_thick_cdk_mobfrlh',
#        'smri_thick_cdk_precnlh', 'smri_thick_cdk_fusiformrh',
#        'smri_vol_cdk_mobfrlh', 'smri_vol_cdk_paracnlh',
#        'smri_vol_cdk_parsopclh', 'smri_vol_cdk_paracnrh',
#        'smri_vol_cdk_sufrrh', 'smri_vol_cdk_tmpolerh'],
#       dtype='object')
# 0.6229473580258388
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
# ['smri_thick_cdk_cdacatelh', 'smri_thick_cdk_cuneuslh', 'smri_thick_cdk_ihcatelh', 'smri_thick_cdk_paracnlh',
# 'smri_thick_cdk_postcnlh', 'smri_thick_cdk_precnlh', 'smri_thick_cdk_rracatelh', 'smri_thick_cdk_sufrlh',
# 'smri_thick_cdk_ehinalrh', 'smri_thick_cdk_iftmrh', 'smri_thick_cdk_loccrh', 'smri_thick_cdk_paracnrh',
# 'smri_thick_cdk_periccrh', 'smri_thick_cdk_rrmdfrrh', 'smri_thick_cdk_sutmrh', 'smri_thick_cdk_mean',
# 'smri_vol_cdk_banksstslh', 'smri_vol_cdk_cdacatelh', 'smri_vol_cdk_cdmdfrlh', 'smri_vol_cdk_fusiformlh',
# 'smri_vol_cdk_iftmlh', 'smri_vol_cdk_mobfrlh', 'smri_vol_cdk_paracnlh', 'smri_vol_cdk_postcnlh',
# 'smri_vol_cdk_ptcatelh', 'smri_vol_cdk_precnlh', 'smri_vol_cdk_fusiformrh', 'smri_vol_cdk_lobfrrh', 
# 'smri_vol_cdk_lingualrh', 'smri_vol_cdk_mdtmrh', 'smri_vol_cdk_precnrh', 'smri_vol_cdk_pcrh', 
# 'smri_vol_cdk_suplrh', 'smri_vol_cdk_smrh', 'smri_vol_cdk_tmpolerh', 'smri_vol_cdk_trvtmrh', 
# 'smri_vol_cdk_totallh']
# %%
ok_cols = ['smri_thick_cdk_cdacatelh', 'smri_thick_cdk_cuneuslh', 'smri_thick_cdk_ihcatelh', 'smri_thick_cdk_paracnlh',
'smri_thick_cdk_postcnlh', 'smri_thick_cdk_precnlh', 'smri_thick_cdk_rracatelh', 'smri_thick_cdk_sufrlh',
'smri_thick_cdk_ehinalrh', 'smri_thick_cdk_iftmrh', 'smri_thick_cdk_loccrh', 'smri_thick_cdk_paracnrh',
'smri_thick_cdk_periccrh', 'smri_thick_cdk_rrmdfrrh', 'smri_thick_cdk_sutmrh', 'smri_thick_cdk_mean',
'smri_vol_cdk_banksstslh', 'smri_vol_cdk_cdacatelh', 'smri_vol_cdk_cdmdfrlh', 'smri_vol_cdk_fusiformlh',
'smri_vol_cdk_iftmlh', 'smri_vol_cdk_mobfrlh', 'smri_vol_cdk_paracnlh', 'smri_vol_cdk_postcnlh',
'smri_vol_cdk_ptcatelh', 'smri_vol_cdk_precnlh', 'smri_vol_cdk_fusiformrh', 'smri_vol_cdk_lobfrrh', 
'smri_vol_cdk_lingualrh', 'smri_vol_cdk_mdtmrh', 'smri_vol_cdk_precnrh', 'smri_vol_cdk_pcrh', 
'smri_vol_cdk_suplrh', 'smri_vol_cdk_smrh', 'smri_vol_cdk_tmpolerh', 'smri_vol_cdk_trvtmrh', 
'smri_vol_cdk_totallh', 'smri_thick_cdk_fusiformlh', 'smri_thick_cdk_mobfrlh',
       'smri_thick_cdk_precnlh', 'smri_thick_cdk_fusiformrh',
       'smri_vol_cdk_mobfrlh', 'smri_vol_cdk_paracnlh',
       'smri_vol_cdk_parsopclh', 'smri_vol_cdk_paracnrh',
       'smri_vol_cdk_sufrrh', 'smri_vol_cdk_tmpolerh', 'smri_thick_cdk_cuneuslh', 'smri_thick_cdk_ihcatelh',
       'smri_thick_cdk_pericclh', 'smri_thick_cdk_precnlh',
       'smri_thick_cdk_rracatelh', 'smri_thick_cdk_sufrlh',
       'smri_thick_cdk_fusiformrh', 'smri_vol_cdk_mobfrlh',
       'smri_vol_cdk_paracnlh', 'smri_vol_cdk_mdtmrh', 'smri_thick_cdk_ehinalrh', 'smri_thick_cdk_lobfrrh',
       'smri_vol_cdk_iftmlh', 'smri_vol_cdk_linguallh',
       'smri_vol_cdk_pericclh', 'smri_vol_cdk_precnlh', 'smri_vol_cdk_sufrlh',
       'smri_vol_cdk_banksstsrh', 'smri_vol_cdk_lobfrrh',
       'smri_vol_cdk_totallh', 'smri_vol_cdk_total', 'smri_thick_cdk_cdmdfrlh', 'smri_thick_cdk_cuneuslh',
       'smri_thick_cdk_fusiformlh', 'smri_thick_cdk_ifpllh',
       'smri_thick_cdk_ihcatelh', 'smri_thick_cdk_parahpallh',
       'smri_thick_cdk_pericclh', 'smri_thick_cdk_postcnlh',
       'smri_thick_cdk_ptcatelh', 'smri_thick_cdk_precnlh',
       'smri_thick_cdk_rracatelh', 'smri_thick_cdk_sufrlh',
       'smri_thick_cdk_supllh', 'smri_thick_cdk_banksstsrh',
       'smri_thick_cdk_fusiformrh', 'smri_thick_cdk_iftmrh',
       'smri_thick_cdk_lobfrrh', 'smri_thick_cdk_paracnrh',
       'smri_thick_cdk_smrh', 'smri_thick_cdk_insularh',
       'smri_vol_cdk_cdmdfrlh', 'smri_vol_cdk_iftmlh', 'smri_vol_cdk_mobfrlh',
       'smri_vol_cdk_paracnlh', 'smri_vol_cdk_precnlh', 'smri_vol_cdk_smlh',
       'smri_vol_cdk_banksstsrh', 'smri_vol_cdk_lobfrrh',
       'smri_vol_cdk_mdtmrh', 'smri_vol_cdk_parahpalrh',
       'smri_vol_cdk_precnrh', 'smri_vol_cdk_sufrrh', 'smri_vol_cdk_tmpolerh',
       'smri_vol_cdk_insularh']
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
X_test_reduced.to_csv('/Users/madeleineseitz/Desktop/X_test_uniform_anat_reduced.csv')
X_train_reduced.to_csv('/Users/madeleineseitz/Desktop/X_train_uniform_anat_reduced.csv')
# %%
print(no_doubles)
# %%
# ['smri_thick_cdk_cdacatelh', 'smri_thick_cdk_cuneuslh', 'smri_thick_cdk_ihcatelh', 'smri_thick_cdk_paracnlh',
#   'smri_thick_cdk_postcnlh', 'smri_thick_cdk_precnlh', 'smri_thick_cdk_rracatelh', 'smri_thick_cdk_sufrlh',
#  'smri_thick_cdk_ehinalrh', 'smri_thick_cdk_iftmrh', 'smri_thick_cdk_loccrh', 'smri_thick_cdk_paracnrh',
# 'smri_thick_cdk_periccrh', 'smri_thick_cdk_rrmdfrrh', 'smri_thick_cdk_sutmrh', 'smri_thick_cdk_mean',
#  'smri_vol_cdk_banksstslh', 'smri_vol_cdk_cdacatelh', 'smri_vol_cdk_cdmdfrlh', 'smri_vol_cdk_fusiformlh',
# 'smri_vol_cdk_iftmlh', 'smri_vol_cdk_mobfrlh', 'smri_vol_cdk_paracnlh', 'smri_vol_cdk_postcnlh', 'smri_vol_cdk_ptcatelh',
# 'smri_vol_cdk_precnlh', 'smri_vol_cdk_fusiformrh', 'smri_vol_cdk_lobfrrh', 'smri_vol_cdk_lingualrh', 'smri_vol_cdk_mdtmrh',
# 'smri_vol_cdk_precnrh', 'smri_vol_cdk_pcrh', 'smri_vol_cdk_suplrh', 'smri_vol_cdk_smrh', 'smri_vol_cdk_tmpolerh',
#  'smri_vol_cdk_trvtmrh', 'smri_vol_cdk_totallh', 'smri_thick_cdk_fusiformlh', 'smri_thick_cdk_mobfrlh',
#  'smri_thick_cdk_fusiformrh', 'smri_vol_cdk_parsopclh', 'smri_vol_cdk_paracnrh', 'smri_vol_cdk_sufrrh',
# 'smri_thick_cdk_pericclh', 'smri_thick_cdk_lobfrrh', 'smri_vol_cdk_linguallh', 'smri_vol_cdk_pericclh', 
# 'smri_vol_cdk_sufrlh', 'smri_vol_cdk_banksstsrh', 'smri_vol_cdk_total', 'smri_thick_cdk_cdmdfrlh', 'smri_thick_cdk_ifpllh',
#  'smri_thick_cdk_parahpallh', 'smri_thick_cdk_ptcatelh', 'smri_thick_cdk_supllh', 'smri_thick_cdk_banksstsrh', 
#  'smri_thick_cdk_smrh', 'smri_thick_cdk_insularh', 'smri_vol_cdk_smlh', 'smri_vol_cdk_parahpalrh', 'smri_vol_cdk_insularh']
# %%
len(ok_cols)
# %%
len(no_doubles)
# %%
