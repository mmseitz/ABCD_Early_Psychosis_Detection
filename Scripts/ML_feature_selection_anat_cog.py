#ML Feature Selection Script (anatomical and cognitive data)
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
    X_train = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Anat_Cog_Uniform_Model/X_train_anat_cog_data.csv')
    y_train = np.load("/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Anat_Cog_Uniform_Model/y_train_anat_cog_data.npy", allow_pickle= True)
    #y_train = y_train['group'].to_numpy()
    X_test = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Anat_Cog_Uniform_Model/X_test_anat_cog_data.csv')
    y_test = np.load("/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Anat_Cog_Uniform_Model/y_test_anat_cog_data.npy", allow_pickle= True)
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
# Index(['tfmri_ma_acgml_b_scs_aylh', 'tfmri_ma_lvnfb_b_cds_clmdfrrh',
#        'tfmri_mid_all_beh_srwpfb_nt', 'tfmri_sacgvf_bscs_crbcxlh',
#        'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_saasvcg_bscs_crbwmrh',
#        'tfmri_sacsvis_bscs_crbwmlh', 'tfmri_sacsvis_bscs_cderh',
#        'tfmri_saigvcg_bscs_cdelh', 'tfmri_sst_beh_performflag',
#        'tfmri_sst_all_beh_crgo_mrt', 'tfmri_sst_all_beh_crgo_stdrt',
#        'tfmri_sst_all_beh_crlg_rt', 'tfmri_sst_all_beh_incrlg_nt',
#        'tfmri_nback_all_141', 'tfmri_nback_all_155', 'tfmri_nback_all_232',
#        'tfmri_rec_all_beh_newnf_fa', 'tfmri_rec_all_beh_negface_pr',
#        'tfmri_rec_all_beh_place_br', 'gdt_scr_values_single',
#        'sit_scr_values_count3', 'nihtbx_flanker_agecorrected',
#        'nihtbx_reading_agecorrected', 'nihtbx_flanker_fc',
#        'smri_thick_cdk_cdacatelh', 'smri_thick_cdk_ihcatelh',
#        'smri_thick_cdk_paracnlh', 'smri_thick_cdk_rrmdfrlh',
#        'smri_thick_cdk_tmpolelh', 'smri_thick_cdk_loccrh',
#        'smri_thick_cdk_parahpalrh', 'smri_vol_cdk_mobfrlh',
#        'smri_vol_cdk_banksstsrh', 'smri_vol_cdk_paracnrh'],
#       dtype='object')
# 0.7417848223748971
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
# Index(['tfmri_ma_acvn_b_scs_tplh', 'tfmri_ma_asrvn_b_scs_ayrh',
#        'tfmri_ma_lvnfb_b_cds_clatcgerh', 'tfmri_ma_lvnfb_b_cds_clmdfrrh',
#        'tfmri_sacgvf_bscs_aalh', 'tfmri_sacgvf_bscs_crbcxrh',
#        'tfmri_sacgvf_bscs_tprh', 'tfmri_saisvcg_bscs_pdlh',
#        'tfmri_saisvcg_bscs_cderh', 'tfmri_saisvcg_bscs_aarh',
#        'tfmri_saasvcg_bscs_cdelh', 'tfmri_saasvcg_bscs_crbwmrh',
#        'tfmri_sst_all_beh_crs_nt', 'tfmri_sst_all_beh_ssds_rt',
#        'sit_scr_values_count1', 'sit_scr_values_count3',
#        'smri_thick_cdk_parsopclh', 'smri_thick_cdk_tmpolerh',
#        'smri_vol_cdk_locclh', 'smri_vol_cdk_lobfrlh',
#        'smri_vol_cdk_cdacaterh'],
#       dtype='object')
# 0.724119103617418
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
# Index(['tfmri_ma_acgml_b_scs_aylh', 'tfmri_sacgvf_bscs_crbcxlh',
#        'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_sst_all_beh_crgo_mrt',
#        'tfmri_sst_all_beh_crlg_nt', 'tfmri_nback_all_232',
#        'sit_scr_values_count3', 'nihtbx_flanker_agecorrected',
#        'nihtbx_flanker_fc', 'smri_vol_cdk_paracnrh'],
#       dtype='object')
# 0.6743715424792164

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
# Index(['tfmri_sacsvcg_bscs_pdlh', 'tfmri_saisvcg_bscs_tplh',
#        'tfmri_sacsvis_bscs_crbwmlh', 'tfmri_sst_all_beh_nrgo_rt',
#        'tfmri_nback_all_155', 'tfmri_nback_all_186',
#        'tfmri_rec_all_beh_newpl_fa', 'gdt_scr_values_account_balance',
#        'sit_scr_values_count3', 'smri_thick_cdk_banksstslh'],
#       dtype='object')
# 0.6554556406419386
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
# ['tfmri_sacgvf_bscs_crbcxlh', 'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_saigvcg_bscs_cdelh', 'tfmri_saigvcg_bscs_cderh',
#  'tfmri_sst_all_beh_crgo_mrt', 'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_nback_all_215', 'tfmri_rec_all_beh_newpl_fa',
#  'tfmri_rec_all_beh_newposf_fa', 'tfmri_rec_all_beh_newnegf_fa', 'tfmri_rec_all_beh_posface_br',
#  'tfmri_rec_all_beh_negface_br', 'tfmri_rec_all_beh_place_pr', 'tfmri_rec_all_beh_place_br', 'sit_scr_values_count3',
#  'nihtbx_flanker_agecorrected', 'nihtbx_pattern_agecorrected', 'nihtbx_reading_agecorrected', 'nihtbx_flanker_fc',
#  'nihtbx_pattern_fc', 'nihtbx_cryst_fc', 'smri_thick_cdk_precnlh', 'smri_thick_cdk_loccrh', 'smri_thick_cdk_meanlh']
# %%
ok_cols = ['tfmri_sacgvf_bscs_crbcxlh', 'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_saigvcg_bscs_cdelh', 'tfmri_saigvcg_bscs_cderh',
 'tfmri_sst_all_beh_crgo_mrt', 'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_nback_all_215', 'tfmri_rec_all_beh_newpl_fa',
 'tfmri_rec_all_beh_newposf_fa', 'tfmri_rec_all_beh_newnegf_fa', 'tfmri_rec_all_beh_posface_br',
 'tfmri_rec_all_beh_negface_br', 'tfmri_rec_all_beh_place_pr', 'tfmri_rec_all_beh_place_br', 'sit_scr_values_count3',
 'nihtbx_flanker_agecorrected', 'nihtbx_pattern_agecorrected', 'nihtbx_reading_agecorrected', 'nihtbx_flanker_fc',
 'nihtbx_pattern_fc', 'nihtbx_cryst_fc', 'smri_thick_cdk_precnlh', 'smri_thick_cdk_loccrh', 'smri_thick_cdk_meanlh', 
 'tfmri_sacsvcg_bscs_pdlh', 'tfmri_saisvcg_bscs_tplh',
       'tfmri_sacsvis_bscs_crbwmlh', 'tfmri_sst_all_beh_nrgo_rt',
       'tfmri_nback_all_155', 'tfmri_nback_all_186',
       'tfmri_rec_all_beh_newpl_fa', 'gdt_scr_values_account_balance',
       'sit_scr_values_count3', 'smri_thick_cdk_banksstslh', 'tfmri_ma_acgml_b_scs_aylh', 'tfmri_sacgvf_bscs_crbcxlh',
       'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_sst_all_beh_crgo_mrt',
       'tfmri_sst_all_beh_crlg_nt', 'tfmri_nback_all_232',
       'sit_scr_values_count3', 'nihtbx_flanker_agecorrected',
       'nihtbx_flanker_fc', 'smri_vol_cdk_paracnrh','tfmri_ma_acvn_b_scs_tplh', 'tfmri_ma_asrvn_b_scs_ayrh',
       'tfmri_ma_lvnfb_b_cds_clatcgerh', 'tfmri_ma_lvnfb_b_cds_clmdfrrh',
       'tfmri_sacgvf_bscs_aalh', 'tfmri_sacgvf_bscs_crbcxrh',
       'tfmri_sacgvf_bscs_tprh', 'tfmri_saisvcg_bscs_pdlh',
       'tfmri_saisvcg_bscs_cderh', 'tfmri_saisvcg_bscs_aarh',
       'tfmri_saasvcg_bscs_cdelh', 'tfmri_saasvcg_bscs_crbwmrh',
       'tfmri_sst_all_beh_crs_nt', 'tfmri_sst_all_beh_ssds_rt',
       'sit_scr_values_count1', 'sit_scr_values_count3',
       'smri_thick_cdk_parsopclh', 'smri_thick_cdk_tmpolerh',
       'smri_vol_cdk_locclh', 'smri_vol_cdk_lobfrlh',
       'smri_vol_cdk_cdacaterh',  'tfmri_ma_acgml_b_scs_aylh', 'tfmri_ma_lvnfb_b_cds_clmdfrrh',
       'tfmri_mid_all_beh_srwpfb_nt', 'tfmri_sacgvf_bscs_crbcxlh',
       'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_saasvcg_bscs_crbwmrh',
       'tfmri_sacsvis_bscs_crbwmlh', 'tfmri_sacsvis_bscs_cderh',
       'tfmri_saigvcg_bscs_cdelh', 'tfmri_sst_beh_performflag',
       'tfmri_sst_all_beh_crgo_mrt', 'tfmri_sst_all_beh_crgo_stdrt',
       'tfmri_sst_all_beh_crlg_rt', 'tfmri_sst_all_beh_incrlg_nt',
       'tfmri_nback_all_141', 'tfmri_nback_all_155', 'tfmri_nback_all_232',
       'tfmri_rec_all_beh_newnf_fa', 'tfmri_rec_all_beh_negface_pr',
       'tfmri_rec_all_beh_place_br', 'gdt_scr_values_single',
       'sit_scr_values_count3', 'nihtbx_flanker_agecorrected',
       'nihtbx_reading_agecorrected', 'nihtbx_flanker_fc',
       'smri_thick_cdk_cdacatelh', 'smri_thick_cdk_ihcatelh',
       'smri_thick_cdk_paracnlh', 'smri_thick_cdk_rrmdfrlh',
       'smri_thick_cdk_tmpolelh', 'smri_thick_cdk_loccrh',
       'smri_thick_cdk_parahpalrh', 'smri_vol_cdk_mobfrlh',
       'smri_vol_cdk_banksstsrh', 'smri_vol_cdk_paracnrh']
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
X_test_reduced.to_csv('/Users/madeleineseitz/Desktop/X_test_uniform_anat_cog_reduced.csv')
X_train_reduced.to_csv('/Users/madeleineseitz/Desktop/X_train_uniform_anat_cog_reduced.csv')
# %%
print(no_doubles)
# %%
['tfmri_sacgvf_bscs_crbcxlh', 'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_saigvcg_bscs_cdelh', 'tfmri_saigvcg_bscs_cderh', 
 'tfmri_sst_all_beh_crgo_mrt', 'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_nback_all_215', 'tfmri_rec_all_beh_newpl_fa', 
 'tfmri_rec_all_beh_newposf_fa', 'tfmri_rec_all_beh_newnegf_fa', 'tfmri_rec_all_beh_posface_br', 'tfmri_rec_all_beh_negface_br', 
 'tfmri_rec_all_beh_place_pr', 'tfmri_rec_all_beh_place_br', 'sit_scr_values_count3', 'nihtbx_flanker_agecorrected', 
 'nihtbx_pattern_agecorrected', 'nihtbx_reading_agecorrected', 'nihtbx_flanker_fc', 'nihtbx_pattern_fc', 'nihtbx_cryst_fc',
 'smri_thick_cdk_precnlh', 'smri_thick_cdk_loccrh', 'smri_thick_cdk_meanlh', 'tfmri_sacsvcg_bscs_pdlh', 'tfmri_saisvcg_bscs_tplh',
 'tfmri_sacsvis_bscs_crbwmlh', 'tfmri_sst_all_beh_nrgo_rt', 'tfmri_nback_all_155', 'tfmri_nback_all_186',
 'gdt_scr_values_account_balance', 'smri_thick_cdk_banksstslh', 'tfmri_ma_acgml_b_scs_aylh', 'tfmri_sst_all_beh_crlg_nt',
 'tfmri_nback_all_232', 'smri_vol_cdk_paracnrh', 'tfmri_ma_acvn_b_scs_tplh', 'tfmri_ma_asrvn_b_scs_ayrh',
 'tfmri_ma_lvnfb_b_cds_clatcgerh', 'tfmri_ma_lvnfb_b_cds_clmdfrrh', 'tfmri_sacgvf_bscs_aalh', 'tfmri_sacgvf_bscs_tprh',
 'tfmri_saisvcg_bscs_pdlh', 'tfmri_saisvcg_bscs_cderh', 'tfmri_saisvcg_bscs_aarh', 'tfmri_saasvcg_bscs_cdelh',
 'tfmri_saasvcg_bscs_crbwmrh', 'tfmri_sst_all_beh_crs_nt', 'tfmri_sst_all_beh_ssds_rt', 'sit_scr_values_count1',
 'smri_thick_cdk_parsopclh', 'smri_thick_cdk_tmpolerh', 'smri_vol_cdk_locclh', 'smri_vol_cdk_lobfrlh', 'smri_vol_cdk_cdacaterh',
 'tfmri_mid_all_beh_srwpfb_nt', 'tfmri_sacsvis_bscs_cderh', 'tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crlg_rt',
 'tfmri_sst_all_beh_incrlg_nt', 'tfmri_nback_all_141', 'tfmri_rec_all_beh_newnf_fa', 'tfmri_rec_all_beh_negface_pr', 
 'gdt_scr_values_single', 'smri_thick_cdk_cdacatelh', 'smri_thick_cdk_ihcatelh', 'smri_thick_cdk_paracnlh',
 'smri_thick_cdk_rrmdfrlh', 'smri_thick_cdk_tmpolelh', 'smri_thick_cdk_parahpalrh', 'smri_vol_cdk_mobfrlh', 
 'smri_vol_cdk_banksstsrh']
# %%
len(ok_cols)
# %%
len(no_doubles)
# %%
