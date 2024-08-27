#ML Feature Selection Script (cognitive [behavioral and fMRI task data] data only)
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
    X_train = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Cog_Only_Uniform_Model/X_train_cog_only_data.csv')
    y_train = np.load("/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Cog_Only_Uniform_Model/y_train_cog_data.npy", allow_pickle= True)
    #y_train = y_train['group'].to_numpy()
    X_test = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Cog_Only_Uniform_Model/X_test_cog_only_data.csv')
    y_test = np.load("/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Cog_Only_Uniform_Model/y_test_cog_only_data.npy", allow_pickle= True)
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
# Index(['tfmri_ma_aclvn_b_scs_aylh', 'tfmri_ma_acmvn_b_scs_aylh',
#        'tfmri_ma_rpvnfb_b_cds_clmdfrlh', 'tfmri_ma_rpvnfb_b_cds_romdfrrh',
#        'tfmri_sacgvf_bscs_crbcxlh', 'tfmri_sacgvf_bscs_crbcxrh',
#        'tfmri_sacsvcg_bscs_crbwmrh', 'tfmri_sacsvcg_bscs_pdrh',
#        'tfmri_saigvcg_bscs_cdelh', 'tfmri_saigvcg_bscs_crbcxrh',
#        'tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crgo_nt',
#        'tfmri_sst_all_beh_crgo_mrt', 'tfmri_sst_all_beh_incrgo_nt',
#        'tfmri_sst_all_beh_incrlg_nt', 'tfmri_sst_all_beh_nrgo_rt',
#        'tfmri_sst_all_beh_incrs_nt', 'tfmri_sst_all_beh_total_meanrt',
#        'tfmri_nback_all_4', 'tfmri_nback_all_52', 'tfmri_nback_all_65',
#        'tfmri_nback_all_103', 'tfmri_nback_all_111', 'tfmri_nback_all_185',
#        'tfmri_nback_all_215', 'tfmri_nback_all_254', 'tfmri_nback_all_262',
#        'tfmri_rec_all_beh_neutface_br', 'tfmri_rec_all_beh_negface_pr',
#        'gdt_scr_values_account_balance', 'sit_scr_expr_mrt_finalrat',
#        'sit_scr_values_count3', 'nihtbx_flanker_agecorrected',
#        'nihtbx_flanker_fc', 'lmt_scr_num_timed_out'],
#       dtype='object')
# 0.7218706817308845
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
# Index(['tfmri_ma_acgml_b_scs_aylh', 'tfmri_ma_rpvnfb_b_cds_clcgelh',
#        'tfmri_ma_lvnfb_b_cds_clatcgelh', 'tfmri_ma_lvnfb_b_cds_roagelh',
#        'tfmri_mid_all_beh_srwnfb_nt', 'tfmri_mid_all_beh_ntpfb_rate',
#        'tfmri_sacgvf_bscs_pdrh', 'tfmri_sacsvcg_bscs_crbwmlh',
#        'tfmri_sacsvcg_bscs_tprh', 'tfmri_saisvcg_bscs_crbwmrh',
#        'tfmri_saisvcg_bscs_cderh', 'tfmri_sacsvis_bscs_crbwmlh',
#        'tfmri_sst_all_beh_s_nt', 'tfmri_sst_all_beh_crgo_stdrt',
#        'tfmri_sst_all_beh_incrgo_nt', 'tfmri_sst_all_beh_incrgo_mrt',
#        'tfmri_sst_all_beh_incrs_stdrt', 'tfmri_nback_all_14',
#        'tfmri_nback_all_23', 'tfmri_nback_all_43', 'tfmri_nback_all_82',
#        'tfmri_nback_all_88', 'tfmri_nback_all_163', 'tfmri_nback_all_203',
#        'sit_scr_values_count3', 'nihtbx_flanker_fc'],
#       dtype='object')
# 0.7066736563523679
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
# Index(['tfmri_ma_acvn_b_scs_aylh', 'tfmri_ma_aclvn_b_scs_aylh',
#        'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_saigvcg_bscs_cdelh',
#        'tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crgo_nt',
#        'tfmri_sst_all_beh_incrlg_nt', 'sit_scr_values_count3',
#        'nihtbx_flanker_agecorrected', 'nihtbx_flanker_fc'],
#       dtype='object')
# 0.6604448176995594

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
# Index(['tfmri_ma_acgml_b_scs_aylh', 'tfmri_mid_all_beh_srwnfb_nt',
#        'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_saisvcg_bscs_pdrh',
#        'tfmri_saasvcg_bscs_pulh', 'tfmri_saigvcg_bscs_pulh',
#        'tfmri_sst_all_beh_crs_nt', 'tfmri_nback_all_216',
#        'tfmri_rec_all_beh_newpl_fa', 'sit_scr_values_count3'],
#       dtype='object')
# 0.6678643319110945
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
# ['tfmri_ma_acvn_b_scs_aylh', 'tfmri_ma_alrvn_b_scs_aylh', 'tfmri_ma_alrvn_b_scs_ayrh', 'tfmri_ma_asrvn_b_scs_ayrh', 
#  'tfmri_ma_alvcr_b_scs_tplh', 'tfmri_ma_alvcr_b_scs_aylh', 'tfmri_ma_alvcr_b_scs_ayrh', 'tfmri_ma_aclvn_b_scs_aylh', 
#  'tfmri_ma_aclvn_b_scs_ayrh', 'tfmri_ma_acmvn_b_scs_ayrh', 'tfmri_ma_acgml_b_scs_aylh', 'tfmri_ma_acgml_b_scs_ayrh',
# 'tfmri_ma_rpvnfb_b_cds_clmdfrlh', 'tfmri_ma_rpvnfb_b_cds_roagelh', 'tfmri_ma_rpvnfb_b_cds_clmdfrrh',
#  'tfmri_ma_lvnfb_b_cds_roagelh', 'tfmri_ma_lvnfb_b_cds_clmdfrrh', 'tfmri_ma_lvnfb_b_cds_roatcgerh', 'tfmri_mid_all_beh_srwnfb_nt',
#    'tfmri_mid_all_beh_lrwpfb_nt', 'tfmri_mid_all_beh_lrwpfb_rate', 'tfmri_mid_all_beh_lrwnfb_nt', 'tfmri_mid_all_beh_slpfb_nt',
#  'tfmri_mid_all_beh_llpfb_nt', 'tfmri_mid_all_beh_llpfb_rate', 'tfmri_mid_all_beh_llnfb_nt', 'tfmri_mid_all_beh_hrwpfb_nt',
# 'tfmri_mid_all_beh_hrwnfb_rate', 'tfmri_mid_all_beh_hlpfb_nt', 'tfmri_mid_all_beh_hlnfb_nt', 'tfmri_mid_all_beh_hlnfb_rate', 
# 'tfmri_sacgvf_bscs_crbwmlh', 'tfmri_sacgvf_bscs_crbcxlh', 'tfmri_sacgvf_bscs_pulh', 'tfmri_sacgvf_bscs_crbcxrh',
#  'tfmri_sacgvf_bscs_tprh', 'tfmri_sacgvf_bscs_purh', 'tfmri_sacgvf_bscs_pdrh', 'tfmri_sacsvcg_bscs_crbwmrh',
# 'tfmri_saisvcg_bscs_crbwmlh', 'tfmri_saisvcg_bscs_cdelh', 'tfmri_saisvcg_bscs_purh', 'tfmri_saasvcg_bscs_crbwmlh', 
# 'tfmri_saasvcg_bscs_tprh', 'tfmri_sacsvis_bscs_crbwmlh', 'tfmri_sacsvis_bscs_crbwmrh', 'tfmri_saigvcg_bscs_crbwmlh',
#  'tfmri_saigvcg_bscs_cdelh', 'tfmri_saigvcg_bscs_pulh', 'tfmri_saigvcg_bscs_crbwmrh', 'tfmri_saigvcg_bscs_crbcxrh', 
#  'tfmri_saigvcg_bscs_tprh', 'tfmri_saigvcg_bscs_cderh', 'tfmri_saigvcg_bscs_purh', 'tfmri_saigvcg_bscs_pdrh', 
#  'tfmri_saigvis_bscs_crbwmlh', 'tfmri_saigvis_bscs_crbcxlh', 'tfmri_saigvis_bscs_cdelh', 'tfmri_saigvis_bscs_crbcxrh', 
#  'tfmri_saigvis_bscs_tprh', 'tfmri_saigvis_bscs_purh', 'tfmri_saigvis_bscs_pdrh', 'tfmri_sst_beh_performflag', 
#  'tfmri_sst_nbeh_nruns', 'tfmri_sst_all_beh_crgo_nt', 'tfmri_sst_all_beh_crgo_rt', 'tfmri_sst_all_beh_crgo_mrt',
#    'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_sst_all_beh_crlg_nt', 'tfmri_sst_all_beh_incrgo_rt', 'tfmri_sst_all_beh_incrgo_stdrt', 
#    'tfmri_sst_all_beh_incrlg_rt', 'tfmri_sst_all_beh_nrgo_rt', 'tfmri_sst_all_beh_crs_rt', 'tfmri_nback_all_4', 
#    'tfmri_nback_all_13', 'tfmri_nback_all_27', 'tfmri_nback_all_36', 'tfmri_nback_all_43', 'tfmri_nback_all_51', 
#    'tfmri_nback_all_52', 'tfmri_nback_all_53', 'tfmri_nback_all_65', 'tfmri_nback_all_73', 'tfmri_nback_all_81',
#      'tfmri_nback_all_82', 'tfmri_nback_all_83', 'tfmri_nback_all_87', 'tfmri_nback_all_88', 'tfmri_nback_all_103', 
#      'tfmri_nback_all_111', 'tfmri_nback_all_112', 'tfmri_nback_all_141', 'tfmri_nback_all_148', 'tfmri_nback_all_154',
#        'tfmri_nback_all_163', 'tfmri_nback_all_171', 'tfmri_nback_all_172', 'tfmri_nback_all_177', 'tfmri_nback_all_178', 
#        'tfmri_nback_all_184', 'tfmri_nback_all_186', 'tfmri_nback_all_215', 'tfmri_nback_all_223', 'tfmri_nback_all_231',
#     'tfmri_nback_all_244', 'tfmri_nback_all_245', 'tfmri_nback_all_262', 'tfmri_rec_all_beh_newnf_hr',
#     'tfmri_rec_all_beh_newnf_fa', 'tfmri_rec_all_beh_newpl_hr', 'tfmri_rec_all_beh_newpl_fa', 
#     'tfmri_rec_all_beh_newposf_fa', 'tfmri_rec_all_beh_newnegf_fa', 'tfmri_rec_all_beh_posface_pr', 
#     'tfmri_rec_all_beh_posface_br', 'tfmri_rec_all_beh_posf_dpr', 'tfmri_rec_all_beh_neutface_pr', 'tfmri_rec_all_beh_neutface_br',
#       'tfmri_rec_all_beh_negface_br', 'tfmri_rec_all_beh_place_br', 'tfmri_rec_all_beh_place_dp', 'gdt_scr_values_account_balance',
#         'gdt_scr_values_wins', 'sit_scr_expr_mfinalrat', 'sit_scr_values_count1', 'sit_scr_values_count2', 'sit_scr_values_count3',
#      'sit_scr_values_countflips', 'nihtbx_flanker_agecorrected', 'nihtbx_pattern_agecorrected', 'nihtbx_picture_agecorrected',
#      'nihtbx_reading_agecorrected', 'nihtbx_flanker_fc', 'nihtbx_pattern_fc', 'nihtbx_reading_fc', 'lmt_scr_perc_correct', 'lmt_scr_num_timed_out']
# %%
ok_cols = ['tfmri_ma_aclvn_b_scs_aylh', 'tfmri_ma_acmvn_b_scs_aylh',
       'tfmri_ma_rpvnfb_b_cds_clmdfrlh', 'tfmri_ma_rpvnfb_b_cds_romdfrrh',
       'tfmri_sacgvf_bscs_crbcxlh', 'tfmri_sacgvf_bscs_crbcxrh',
       'tfmri_sacsvcg_bscs_crbwmrh', 'tfmri_sacsvcg_bscs_pdrh',
       'tfmri_saigvcg_bscs_cdelh', 'tfmri_saigvcg_bscs_crbcxrh',
       'tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crgo_nt',
       'tfmri_sst_all_beh_crgo_mrt', 'tfmri_sst_all_beh_incrgo_nt',
       'tfmri_sst_all_beh_incrlg_nt', 'tfmri_sst_all_beh_nrgo_rt',
       'tfmri_sst_all_beh_incrs_nt', 'tfmri_sst_all_beh_total_meanrt',
       'tfmri_nback_all_4', 'tfmri_nback_all_52', 'tfmri_nback_all_65',
       'tfmri_nback_all_103', 'tfmri_nback_all_111', 'tfmri_nback_all_185',
       'tfmri_nback_all_215', 'tfmri_nback_all_254', 'tfmri_nback_all_262',
       'tfmri_rec_all_beh_neutface_br', 'tfmri_rec_all_beh_negface_pr',
       'gdt_scr_values_account_balance', 'sit_scr_expr_mrt_finalrat',
       'sit_scr_values_count3', 'nihtbx_flanker_agecorrected',
       'nihtbx_flanker_fc', 'lmt_scr_num_timed_out', 'tfmri_ma_acgml_b_scs_aylh', 'tfmri_ma_rpvnfb_b_cds_clcgelh',
       'tfmri_ma_lvnfb_b_cds_clatcgelh', 'tfmri_ma_lvnfb_b_cds_roagelh',
       'tfmri_mid_all_beh_srwnfb_nt', 'tfmri_mid_all_beh_ntpfb_rate',
       'tfmri_sacgvf_bscs_pdrh', 'tfmri_sacsvcg_bscs_crbwmlh',
       'tfmri_sacsvcg_bscs_tprh', 'tfmri_saisvcg_bscs_crbwmrh',
       'tfmri_saisvcg_bscs_cderh', 'tfmri_sacsvis_bscs_crbwmlh',
       'tfmri_sst_all_beh_s_nt', 'tfmri_sst_all_beh_crgo_stdrt',
       'tfmri_sst_all_beh_incrgo_nt', 'tfmri_sst_all_beh_incrgo_mrt',
       'tfmri_sst_all_beh_incrs_stdrt', 'tfmri_nback_all_14',
       'tfmri_nback_all_23', 'tfmri_nback_all_43', 'tfmri_nback_all_82',
       'tfmri_nback_all_88', 'tfmri_nback_all_163', 'tfmri_nback_all_203',
       'sit_scr_values_count3', 'nihtbx_flanker_fc', 'tfmri_ma_acvn_b_scs_aylh', 'tfmri_ma_aclvn_b_scs_aylh',
       'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_saigvcg_bscs_cdelh',
       'tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crgo_nt',
       'tfmri_sst_all_beh_incrlg_nt', 'sit_scr_values_count3',
       'nihtbx_flanker_agecorrected', 'nihtbx_flanker_fc', 'tfmri_ma_acgml_b_scs_aylh', 'tfmri_mid_all_beh_srwnfb_nt',
       'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_saisvcg_bscs_pdrh',
       'tfmri_saasvcg_bscs_pulh', 'tfmri_saigvcg_bscs_pulh',
       'tfmri_sst_all_beh_crs_nt', 'tfmri_nback_all_216',
       'tfmri_rec_all_beh_newpl_fa', 'sit_scr_values_count3', 
        'tfmri_ma_acvn_b_scs_aylh', 'tfmri_ma_alrvn_b_scs_aylh', 'tfmri_ma_alrvn_b_scs_ayrh', 'tfmri_ma_asrvn_b_scs_ayrh', 
 'tfmri_ma_alvcr_b_scs_tplh', 'tfmri_ma_alvcr_b_scs_aylh', 'tfmri_ma_alvcr_b_scs_ayrh', 'tfmri_ma_aclvn_b_scs_aylh', 
 'tfmri_ma_aclvn_b_scs_ayrh', 'tfmri_ma_acmvn_b_scs_ayrh', 'tfmri_ma_acgml_b_scs_aylh', 'tfmri_ma_acgml_b_scs_ayrh',
'tfmri_ma_rpvnfb_b_cds_clmdfrlh', 'tfmri_ma_rpvnfb_b_cds_roagelh', 'tfmri_ma_rpvnfb_b_cds_clmdfrrh',
 'tfmri_ma_lvnfb_b_cds_roagelh', 'tfmri_ma_lvnfb_b_cds_clmdfrrh', 'tfmri_ma_lvnfb_b_cds_roatcgerh', 'tfmri_mid_all_beh_srwnfb_nt',
   'tfmri_mid_all_beh_lrwpfb_nt', 'tfmri_mid_all_beh_lrwpfb_rate', 'tfmri_mid_all_beh_lrwnfb_nt', 'tfmri_mid_all_beh_slpfb_nt',
 'tfmri_mid_all_beh_llpfb_nt', 'tfmri_mid_all_beh_llpfb_rate', 'tfmri_mid_all_beh_llnfb_nt', 'tfmri_mid_all_beh_hrwpfb_nt',
'tfmri_mid_all_beh_hrwnfb_rate', 'tfmri_mid_all_beh_hlpfb_nt', 'tfmri_mid_all_beh_hlnfb_nt', 'tfmri_mid_all_beh_hlnfb_rate', 
'tfmri_sacgvf_bscs_crbwmlh', 'tfmri_sacgvf_bscs_crbcxlh', 'tfmri_sacgvf_bscs_pulh', 'tfmri_sacgvf_bscs_crbcxrh',
 'tfmri_sacgvf_bscs_tprh', 'tfmri_sacgvf_bscs_purh', 'tfmri_sacgvf_bscs_pdrh', 'tfmri_sacsvcg_bscs_crbwmrh',
'tfmri_saisvcg_bscs_crbwmlh', 'tfmri_saisvcg_bscs_cdelh', 'tfmri_saisvcg_bscs_purh', 'tfmri_saasvcg_bscs_crbwmlh', 
'tfmri_saasvcg_bscs_tprh', 'tfmri_sacsvis_bscs_crbwmlh', 'tfmri_sacsvis_bscs_crbwmrh', 'tfmri_saigvcg_bscs_crbwmlh',
 'tfmri_saigvcg_bscs_cdelh', 'tfmri_saigvcg_bscs_pulh', 'tfmri_saigvcg_bscs_crbwmrh', 'tfmri_saigvcg_bscs_crbcxrh', 
 'tfmri_saigvcg_bscs_tprh', 'tfmri_saigvcg_bscs_cderh', 'tfmri_saigvcg_bscs_purh', 'tfmri_saigvcg_bscs_pdrh', 
 'tfmri_saigvis_bscs_crbwmlh', 'tfmri_saigvis_bscs_crbcxlh', 'tfmri_saigvis_bscs_cdelh', 'tfmri_saigvis_bscs_crbcxrh', 
 'tfmri_saigvis_bscs_tprh', 'tfmri_saigvis_bscs_purh', 'tfmri_saigvis_bscs_pdrh', 'tfmri_sst_beh_performflag', 
 'tfmri_sst_nbeh_nruns', 'tfmri_sst_all_beh_crgo_nt', 'tfmri_sst_all_beh_crgo_rt', 'tfmri_sst_all_beh_crgo_mrt',
   'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_sst_all_beh_crlg_nt', 'tfmri_sst_all_beh_incrgo_rt', 'tfmri_sst_all_beh_incrgo_stdrt', 
   'tfmri_sst_all_beh_incrlg_rt', 'tfmri_sst_all_beh_nrgo_rt', 'tfmri_sst_all_beh_crs_rt', 'tfmri_nback_all_4', 
   'tfmri_nback_all_13', 'tfmri_nback_all_27', 'tfmri_nback_all_36', 'tfmri_nback_all_43', 'tfmri_nback_all_51', 
   'tfmri_nback_all_52', 'tfmri_nback_all_53', 'tfmri_nback_all_65', 'tfmri_nback_all_73', 'tfmri_nback_all_81',
     'tfmri_nback_all_82', 'tfmri_nback_all_83', 'tfmri_nback_all_87', 'tfmri_nback_all_88', 'tfmri_nback_all_103', 
     'tfmri_nback_all_111', 'tfmri_nback_all_112', 'tfmri_nback_all_141', 'tfmri_nback_all_148', 'tfmri_nback_all_154',
       'tfmri_nback_all_163', 'tfmri_nback_all_171', 'tfmri_nback_all_172', 'tfmri_nback_all_177', 'tfmri_nback_all_178', 
       'tfmri_nback_all_184', 'tfmri_nback_all_186', 'tfmri_nback_all_215', 'tfmri_nback_all_223', 'tfmri_nback_all_231',
    'tfmri_nback_all_244', 'tfmri_nback_all_245', 'tfmri_nback_all_262', 'tfmri_rec_all_beh_newnf_hr',
    'tfmri_rec_all_beh_newnf_fa', 'tfmri_rec_all_beh_newpl_hr', 'tfmri_rec_all_beh_newpl_fa', 
    'tfmri_rec_all_beh_newposf_fa', 'tfmri_rec_all_beh_newnegf_fa', 'tfmri_rec_all_beh_posface_pr', 
    'tfmri_rec_all_beh_posface_br', 'tfmri_rec_all_beh_posf_dpr', 'tfmri_rec_all_beh_neutface_pr', 'tfmri_rec_all_beh_neutface_br',
      'tfmri_rec_all_beh_negface_br', 'tfmri_rec_all_beh_place_br', 'tfmri_rec_all_beh_place_dp', 'gdt_scr_values_account_balance',
        'gdt_scr_values_wins', 'sit_scr_expr_mfinalrat', 'sit_scr_values_count1', 'sit_scr_values_count2', 'sit_scr_values_count3',
     'sit_scr_values_countflips', 'nihtbx_flanker_agecorrected', 'nihtbx_pattern_agecorrected', 'nihtbx_picture_agecorrected',
     'nihtbx_reading_agecorrected', 'nihtbx_flanker_fc', 'nihtbx_pattern_fc', 'nihtbx_reading_fc', 'lmt_scr_perc_correct', 'lmt_scr_num_timed_out' ]
no_doubles = []
for l in ok_cols:
    if l in no_doubles:
        pass
    else:
        no_doubles.append(l)

# %%
X_train = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Cog_Only_Uniform_Model/X_train_cog_only_data.csv')
X_test = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/ML/Cog_Only_Uniform_Model/X_test_cog_only_data.csv')
# %%
X_train_reduced = X_train[no_doubles]
X_test_reduced = X_test[no_doubles]
# %%
X_test_reduced.to_csv('/Users/madeleineseitz/Desktop/X_test_uniform_cog_only_reduced.csv')
X_train_reduced.to_csv('/Users/madeleineseitz/Desktop/X_train_uniform_cog_only_reduced.csv')
# %%
print(no_doubles)
# %%
['tfmri_ma_aclvn_b_scs_aylh', 'tfmri_ma_acmvn_b_scs_aylh', 'tfmri_ma_rpvnfb_b_cds_clmdfrlh', 'tfmri_ma_rpvnfb_b_cds_romdfrrh', 
 'tfmri_sacgvf_bscs_crbcxlh', 'tfmri_sacgvf_bscs_crbcxrh', 'tfmri_sacsvcg_bscs_crbwmrh', 'tfmri_sacsvcg_bscs_pdrh',
   'tfmri_saigvcg_bscs_cdelh', 'tfmri_saigvcg_bscs_crbcxrh', 'tfmri_sst_beh_performflag', 'tfmri_sst_all_beh_crgo_nt',
 'tfmri_sst_all_beh_crgo_mrt', 'tfmri_sst_all_beh_incrgo_nt', 'tfmri_sst_all_beh_incrlg_nt', 'tfmri_sst_all_beh_nrgo_rt', 
 'tfmri_sst_all_beh_incrs_nt', 'tfmri_sst_all_beh_total_meanrt', 'tfmri_nback_all_4', 'tfmri_nback_all_52', 'tfmri_nback_all_65',
 'tfmri_nback_all_103', 'tfmri_nback_all_111', 'tfmri_nback_all_185', 'tfmri_nback_all_215', 'tfmri_nback_all_254',
 'tfmri_nback_all_262', 'tfmri_rec_all_beh_neutface_br', 'tfmri_rec_all_beh_negface_pr', 'gdt_scr_values_account_balance', 
 'sit_scr_expr_mrt_finalrat', 'sit_scr_values_count3', 'nihtbx_flanker_agecorrected', 'nihtbx_flanker_fc', 'lmt_scr_num_timed_out',
 'tfmri_ma_acgml_b_scs_aylh', 'tfmri_ma_rpvnfb_b_cds_clcgelh', 'tfmri_ma_lvnfb_b_cds_clatcgelh', 'tfmri_ma_lvnfb_b_cds_roagelh',
 'tfmri_mid_all_beh_srwnfb_nt', 'tfmri_mid_all_beh_ntpfb_rate', 'tfmri_sacgvf_bscs_pdrh', 'tfmri_sacsvcg_bscs_crbwmlh', 
 'tfmri_sacsvcg_bscs_tprh', 'tfmri_saisvcg_bscs_crbwmrh', 'tfmri_saisvcg_bscs_cderh', 'tfmri_sacsvis_bscs_crbwmlh',
 'tfmri_sst_all_beh_s_nt', 'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_sst_all_beh_incrgo_mrt', 'tfmri_sst_all_beh_incrs_stdrt', 
 'tfmri_nback_all_14', 'tfmri_nback_all_23', 'tfmri_nback_all_43', 'tfmri_nback_all_82', 'tfmri_nback_all_88', 'tfmri_nback_all_163', 
 'tfmri_nback_all_203', 'tfmri_ma_acvn_b_scs_aylh', 'tfmri_saisvcg_bscs_pdrh', 'tfmri_saasvcg_bscs_pulh', 'tfmri_saigvcg_bscs_pulh',
 'tfmri_sst_all_beh_crs_nt', 'tfmri_nback_all_216', 'tfmri_rec_all_beh_newpl_fa', 'tfmri_ma_alrvn_b_scs_aylh', 'tfmri_ma_alrvn_b_scs_ayrh',
 'tfmri_ma_asrvn_b_scs_ayrh', 'tfmri_ma_alvcr_b_scs_tplh', 'tfmri_ma_alvcr_b_scs_aylh', 'tfmri_ma_alvcr_b_scs_ayrh', 'tfmri_ma_aclvn_b_scs_ayrh',
 'tfmri_ma_acmvn_b_scs_ayrh', 'tfmri_ma_acgml_b_scs_ayrh', 'tfmri_ma_rpvnfb_b_cds_roagelh', 'tfmri_ma_rpvnfb_b_cds_clmdfrrh', 
 'tfmri_ma_lvnfb_b_cds_clmdfrrh', 'tfmri_ma_lvnfb_b_cds_roatcgerh', 'tfmri_mid_all_beh_lrwpfb_nt', 'tfmri_mid_all_beh_lrwpfb_rate', 
 'tfmri_mid_all_beh_lrwnfb_nt', 'tfmri_mid_all_beh_slpfb_nt', 'tfmri_mid_all_beh_llpfb_nt', 'tfmri_mid_all_beh_llpfb_rate', 
 'tfmri_mid_all_beh_llnfb_nt', 'tfmri_mid_all_beh_hrwpfb_nt', 'tfmri_mid_all_beh_hrwnfb_rate', 'tfmri_mid_all_beh_hlpfb_nt', 
 'tfmri_mid_all_beh_hlnfb_nt', 'tfmri_mid_all_beh_hlnfb_rate', 'tfmri_sacgvf_bscs_crbwmlh', 'tfmri_sacgvf_bscs_pulh', 'tfmri_sacgvf_bscs_tprh', 
 'tfmri_sacgvf_bscs_purh', 'tfmri_saisvcg_bscs_crbwmlh', 'tfmri_saisvcg_bscs_cdelh', 'tfmri_saisvcg_bscs_purh', 'tfmri_saasvcg_bscs_crbwmlh',
 'tfmri_saasvcg_bscs_tprh', 'tfmri_sacsvis_bscs_crbwmrh', 'tfmri_saigvcg_bscs_crbwmlh', 'tfmri_saigvcg_bscs_crbwmrh', 'tfmri_saigvcg_bscs_tprh', 
 'tfmri_saigvcg_bscs_cderh', 'tfmri_saigvcg_bscs_purh', 'tfmri_saigvcg_bscs_pdrh', 'tfmri_saigvis_bscs_crbwmlh', 'tfmri_saigvis_bscs_crbcxlh', 
 'tfmri_saigvis_bscs_cdelh', 'tfmri_saigvis_bscs_crbcxrh', 'tfmri_saigvis_bscs_tprh', 'tfmri_saigvis_bscs_purh', 'tfmri_saigvis_bscs_pdrh', 
 'tfmri_sst_nbeh_nruns', 'tfmri_sst_all_beh_crgo_rt', 'tfmri_sst_all_beh_crlg_nt', 'tfmri_sst_all_beh_incrgo_rt', 'tfmri_sst_all_beh_incrgo_stdrt',
 'tfmri_sst_all_beh_incrlg_rt', 'tfmri_sst_all_beh_crs_rt', 'tfmri_nback_all_13', 'tfmri_nback_all_27', 'tfmri_nback_all_36', 'tfmri_nback_all_51',
 'tfmri_nback_all_53', 'tfmri_nback_all_73', 'tfmri_nback_all_81', 'tfmri_nback_all_83', 'tfmri_nback_all_87', 'tfmri_nback_all_112', 'tfmri_nback_all_141', 
 'tfmri_nback_all_148', 'tfmri_nback_all_154', 'tfmri_nback_all_171', 'tfmri_nback_all_172', 'tfmri_nback_all_177', 'tfmri_nback_all_178', 
 'tfmri_nback_all_184', 'tfmri_nback_all_186', 'tfmri_nback_all_223', 'tfmri_nback_all_231', 'tfmri_nback_all_244', 'tfmri_nback_all_245', 'tfmri_rec_all_beh_newnf_hr',
 'tfmri_rec_all_beh_newnf_fa', 'tfmri_rec_all_beh_newpl_hr', 'tfmri_rec_all_beh_newposf_fa', 'tfmri_rec_all_beh_newnegf_fa', 'tfmri_rec_all_beh_posface_pr', 
 'tfmri_rec_all_beh_posface_br', 'tfmri_rec_all_beh_posf_dpr', 'tfmri_rec_all_beh_neutface_pr', 'tfmri_rec_all_beh_negface_br', 'tfmri_rec_all_beh_place_br',
 'tfmri_rec_all_beh_place_dp', 'gdt_scr_values_wins', 'sit_scr_expr_mfinalrat', 'sit_scr_values_count1', 'sit_scr_values_count2', 'sit_scr_values_countflips',
 'nihtbx_pattern_agecorrected', 'nihtbx_picture_agecorrected', 'nihtbx_reading_agecorrected', 'nihtbx_pattern_fc', 'nihtbx_reading_fc', 'lmt_scr_perc_correct']
# %%
len(ok_cols)
# %%
len(no_doubles)
# %%
