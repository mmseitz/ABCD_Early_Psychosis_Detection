#ML Feature Selection Script
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
    X_train = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/X_train_matched_data.csv')
    y_train = pd.read_csv("/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/y_train_matched_data.csv")
    y_train = y_train['group'].to_numpy()
    X_test = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/X_test_matched_data.csv')
    y_test = pd.read_csv("/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/y_test_matched_data.csv")
    y_test = y_test['group'].to_numpy()
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
# Index(['tfmri_ma_alrvn_b_scs_ayrh', 'tfmri_ma_rpvnfb_b_cds_clcgelh',
#        'tfmri_ma_rpvnfb_b_cds_roagelh', 'tfmri_sst_beh_performflag',
#        'tfmri_sst_all_beh_crgo_rt', 'tfmri_sst_all_beh_incrlg_nt',
#        'tfmri_sacgvf_bscs_crbwmlh', 'tfmri_sacgvf_bscs_crbwmrh',
#        'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_sacgvf_bscs_pdrh',
#        'tfmri_saigvcg_bscs_crbwmrh', 'tfmri_saigvcg_bscs_crbcxrh',
#        'tfmri_nback_all_73', 'tfmri_nback_all_83', 'tfmri_nback_all_178',
#        'tfmri_nback_all_214', 'tfmri_nback_all_223', 'tfmri_nback_all_232',
#        'tfmri_nback_all_262', 'tfmri_rec_all_beh_posf_dpr',
#        'tfmri_rec_all_beh_negface_pr', 'tfmri_rec_all_beh_negf_dp',
#        'smri_thick_cdk_cuneuslh', 'smri_thick_cdk_rracatelh',
#        'smri_thick_cdk_rrmdfrlh', 'smri_thick_cdk_sufrlh',
#        'smri_thick_cdk_supllh', 'smri_thick_cdk_trvtmlh',
#        'smri_thick_cdk_parsopcrh', 'smri_thick_cdk_sufrrh',
#        'smri_vol_cdk_trvtmlh', 'smri_vol_cdk_iftmrh', 'smri_vol_cdk_paracnrh',
#        'sit_scr_values_count3', 'lmt_scr_num_timed_out'],
#       dtype='object')
# sffs_LR K score: 0.7260974735538727
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
# Index(['tfmri_ma_lvnfb_b_cds_clatcgelh', 'tfmri_sst_nbeh_nruns',
#        'tfmri_sst_all_beh_crgo_nt', 'tfmri_sst_all_beh_crlg_nt',
#        'tfmri_sst_all_beh_incrlg_rt', 'tfmri_sacgvf_bscs_crbcxrh',
#        'tfmri_sacgvf_bscs_purh', 'tfmri_sacsvcg_bscs_crbcxlh',
#        'tfmri_saasvcg_bscs_tplh', 'tfmri_saasvcg_bscs_aalh',
#        'tfmri_sacsvis_bscs_aarh', 'tfmri_nback_all_148', 'tfmri_nback_all_154',
#        'tfmri_nback_all_185', 'tfmri_nback_all_216', 'tfmri_nback_all_224',
#        'tfmri_nback_all_237', 'tfmri_rec_all_beh_newnf_hr',
#        'tfmri_rec_all_beh_negface_pr', 'smri_thick_cdk_fusiformlh',
#        'smri_thick_cdk_iftmlh', 'smri_thick_cdk_parsobislh',
#        'smri_thick_cdk_pclh', 'smri_thick_cdk_smlh', 'smri_thick_cdk_frpolelh',
#        'smri_thick_cdk_trvtmlh', 'smri_thick_cdk_banksstsrh',
#        'smri_thick_cdk_ifplrh', 'smri_thick_cdk_insularh',
#        'smri_vol_cdk_rrmdfrrh', 'sit_scr_expr_mratdiff1', 'nihtbx_pattern_fc'],
#       dtype='object')
# sffs_RF K score: 0.6876291159451124
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
# Index(['tfmri_mid_all_beh_hrwnfb_nt', 'tfmri_mid_all_beh_hlnfb_nt',
#        'tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crgo_mrt',
#        'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_sacgvf_bscs_pulh',
#        'tfmri_saigvcg_bscs_pulh', 'tfmri_saigvcg_bscs_purh',
#        'tfmri_nback_all_215', 'smri_vol_cdk_paracnrh'],
#       dtype='object')
# sbfs_LR K score: 0.665980092808225

# %%
#Random Forest Classifier
sbfs_RF = SequentialFeatureSelector(RandomForestClassifier(),
                                    k_features = 10, forward=False,
                                    floating=True, scoring='roc_auc',
                                    cv=3)
sbfs_RF = sbfs_RF.fit(X_train, y_train)

selected_features = X_train.columns[list(sbfs_RF.k_feature_idx_)]
print(selected_features)
print(sbfs_RF.k_score_)

####Output
#Index(['tfmri_ma_alvcr_b_scs_tprh', 'tfmri_sst_all_beh_s_nt',
#       'tfmri_sacgvf_bscs_pulh', 'tfmri_nback_all_51', 'tfmri_nback_all_148',
#       'tfmri_nback_all_185', 'tfmri_nback_all_186', 'tfmri_nback_all_214',
#       'smri_thick_cdk_ptcaterh', 'nihtbx_flanker_agecorrected'],
#      dtype='object')
#0.6207487335184898
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
#cols: ['tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crgo_rt', 'tfmri_sst_all_beh_crgo_mrt', 'tfmri_sst_all_beh_crlg_nt', 
# 'tfmri_sacgvf_bscs_crbcxlh', 'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_saigvcg_bscs_pulh', 'tfmri_saigvcg_bscs_purh', 'tfmri_nback_all_208', 
# 'tfmri_nback_all_215', 'smri_vol_cdk_paracnrh', 'nihtbx_flanker_agecorrected', 'nihtbx_pattern_agecorrected', 'nihtbx_flanker_fc', 'nihtbx_pattern_fc',
#  'lmt_scr_avg_rt', 'lmt_scr_rt_correct']