#This script investigates statistical differences in environment, mental health, and demographics
#in LR and HR ABCD participants using pdem02, mhy02, and abcd_cbcls01
# %%
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats as stats
import numpy as np

# %%
#import csvs
#demographics (reduced--income/race/education/sex)
#code categorical variables numerically
dems = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/reduced_dem.csv')
dems = dems.drop(columns = ['Unnamed: 0'])
dems = dems.rename(columns= {'demo_sex_v2': 'sex'})
dems.loc[dems['p_edu'] == 'A. high school', 'p_edu'] = 1
dems.loc[dems['p_edu'] == 'B. college', 'p_edu'] = 2
dems.loc[dems['p_edu'] == 'C. post grad', 'p_edu'] = 3
dems.loc[dems['income'] == 'A.lower', 'income'] = 1
dems.loc[dems['income'] == 'B.middle', 'income'] = 2
dems.loc[dems['income'] == 'C.higher', 'income'] = 3
dems.loc[dems['race'] == 'A.white', 'race'] = 1
dems.loc[dems['race'] == 'B.black', 'race'] = 2
dems.loc[dems['race'] == 'C.hispanic', 'race'] = 3
dems.loc[dems['race'] == 'D.Other', 'race'] = 4
dems.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/num_demos.csv')

#ABCD mental health summary scores (youth survey)
mhss = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/abcd_mhy02.csv')
#mhss = mhss.drop(columns = ['Unnamed: 0'])

#PQ-BC HR/LR groups
pqbc_2 = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/PQ-BC/PPS_HR_T2_thresh_2.csv')
pqbc_2 = pqbc_2.drop(columns= ['Unnamed: 0'])

pqbc_3 = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/PQ-BC/PPS_HR_T2_thresh_3.csv')
pqbc_3 = pqbc_3.drop(columns= ['Unnamed: 0'])

pqbc_4 = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/PQ-BC/PPS_HR_T2_thresh_4.csv')
pqbc_4 = pqbc_4.drop(columns= ['Unnamed: 0'])

#Cognitive task behavioral data (aggregated, follow-up 2)
behav = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Master_Files/behav_master.csv')
#Cognitive task neuroimaging data (aggregated, abridged, follow-up 2)
brain = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Master_Files/brain_master.csv')
#brain = brain.drop(columns= ['Unnamed: 0'])
#ABCD Child Behavioral Checklist (CBCL)
cbcl = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/ABCD_CBCL_followup2.csv')
# %%
#function to check how many NaNs exist in each selected df column
def count_nans(df):
    l = list(df)
    for i in l:
        if df[i].isna().sum() >= 50:
            print(i, df[i].isna().sum())
# %%
#merge chosen PQ-BC df with other df
def df_merge(df1, pps_df, visit):
    pps_df = pps_df.loc[pps_df['visit'] == visit]
    pps_df = pps_df[['subjectkey', 'group']]
    new_df = df1.merge(pps_df, on = 'subjectkey')
    return new_df

# %%
#find statistical differences in demos df
def demos_dict(dems_df):
    dems_HR = dems_df.loc[dems_df['group'] == 1]
    dems_LR = dems_df.loc[dems_df['group'] == 0]
    dems_dict = {}
    dems_dict['sex_kruskal'] = stats.kruskal(dems_HR['sex'], dems_LR['sex'])
    dems_dict['income_kruskal'] = stats.kruskal(dems_HR['income'], dems_LR['income'])
    dems_dict['p_edu_kruskal'] = stats.kruskal(dems_HR['p_edu'], dems_LR['p_edu'])
    return dems_dict

# %%
#find statistical differences in mental health scores df
def mh_dict(mh_df):
    mh_df['ple_y_ss_total_bad'] = pd.to_numeric(mh_df['ple_y_ss_total_bad'])
    mh_df['ple_y_ss_affected_bad_sum'] = pd.to_numeric(mh_df['ple_y_ss_affected_bad_sum'])
    mh_df['peq_ss_overt_aggression'] = pd.to_numeric(mh_df['peq_ss_overt_aggression'])
    mh_df['peq_ss_relational_aggs'] = pd.to_numeric(mh_df['peq_ss_relational_aggs'])
    mh_df_HR = mh_df.loc[mh_df['group'] == 1]
    mh_df_LR = mh_df.loc[mh_df['group'] == 0]
    mh_dict = {}
    mh_dict['ACE_total_welch'] = stats.ttest_ind(mh_df_HR['ple_y_ss_total_bad'], mh_df_LR['ple_y_ss_total_bad'], equal_var = False)
    mh_dict['ACE_affected_welch'] = stats.ttest_ind(mh_df_HR['ple_y_ss_affected_bad_sum'], mh_df_LR['ple_y_ss_affected_bad_sum'], equal_var = False)
    mh_dict['overt_aggress_welch'] = stats.ttest_ind(mh_df_HR['peq_ss_overt_aggression'], mh_df_LR['peq_ss_overt_aggression'], equal_var = False)
    mh_dict['rel_aggress_welch'] = stats.ttest_ind(mh_df_HR['peq_ss_relational_aggs'], mh_df_LR['peq_ss_relational_aggs'], equal_var = False)
    return mh_dict

# %%
#find statistical differences in CBCL df
def cbcl_dict(cl_df):
    cl_df_HR = cl_df.loc[cl_df['group'] == 1]
    cl_df_LR = cl_df.loc[cl_df['group'] == 0]
    cbcl_dict = {}
    cbcl_dict['cbcl_thought_welch'] = stats.ttest_ind(cl_df_HR['cbcl_scr_syn_thought_t'], cl_df_LR['cbcl_scr_syn_thought_t'], equal_var = False)
    cbcl_dict['cbcl_total_welch'] = stats.ttest_ind(cl_df_HR['cbcl_scr_syn_totprob_t'], cl_df_LR['cbcl_scr_syn_totprob_t'], equal_var = False)
    return cbcl_dict

# %%
#Preprocess mental health summary scores df
#select salient columns (mean etc. columns omitted due to high NaN frequency)
#Columns pertaining to ACEs, UPPS-P, BIS/BAS, delinquency, 7UP, peer experiences (nts--read publications cited in NIH query)
mhss = mhss[['subjectkey', 'eventname', 'interview_age', 'ple_y_ss_total_number', 'ple_y_ss_total_good', 'ple_y_ss_total_bad', 'ple_y_ss_affect_sum', 
             'ple_y_ss_affected_good_sum', 'ple_y_ss_affected_bad_sum', 'upps_y_ss_negative_urgency', 'upps_y_ss_lack_of_planning', 
             'upps_y_ss_sensation_seeking', 'upps_y_ss_positive_urgency', 'upps_y_ss_lack_of_perseverance', 'bis_y_ss_bis_sum', 'bis_y_ss_bas_rr', 
             'bis_y_ss_bas_drive', 'bis_y_ss_bas_fs', 'delq_y_ss_sum', 'sup_y_ss_sum', 'gish_y_ss_m_sum', 'gish_y_ss_f_sum', 
              'peq_ss_relational_victim', 'peq_ss_reputation_aggs', 'peq_ss_reputation_victim', 'peq_ss_overt_aggression', 
              'peq_ss_overt_victim', 'peq_ss_relational_aggs']]
#['delq_y_ss_sum', 'sup_y_ss_sum'] dropped due to high NaN frequency
mhss = mhss.drop(columns= ['delq_y_ss_sum', 'sup_y_ss_sum'])
#Isolate follow-up 2 data
mhss = mhss.loc[mhss['eventname'] == '2_year_follow_up_y_arm_1']
#Count NaNs
count_nans(mhss) #max Nan 268 (not including sex-divided Gish q's ['gish_y_ss_m_sum', 'gish_y_ss_f_sum'])
#save CSV
mhss.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/mhss_reduced.csv')

# %%
#Preprocess CBCL df
#select salient columns (thought score and total score)--analyze t-score
cbcl = cbcl[['EID', 'Interview', 'cbcl_scr_syn_thought_r', 'cbcl_scr_syn_thought_t', 'cbcl_scr_syn_totprob_r', 'cbcl_scr_syn_totprob_t']]
#rename subject column
cbcl = cbcl.rename(columns = {'EID': 'subjectkey'})
#count NaNs
count_nans(cbcl)
# %%

#drop NaNs from master files to determine size of final dataset
master = behav.merge(brain, on = ['subjectkey', 'eventname']) #(shape: (6252, 360))
master_nonan = master.dropna().reset_index().drop(['index'], axis=1) #(shape: (4685, 360))
# %%

#combine demographics data and mental health scores with no NaN master df
dems_small = dems[['subjectkey', 'sex', 'income', 'p_edu']]
mhss_small = mhss[['subjectkey', 'ple_y_ss_total_bad', 'ple_y_ss_affected_bad_sum', 'peq_ss_overt_aggression', 'peq_ss_relational_aggs']]
cbcl_small = cbcl[['subjectkey', 'cbcl_scr_syn_thought_t', 'cbcl_scr_syn_totprob_t']]
master_dems = master_nonan.merge(dems_small, on = 'subjectkey')
master_mh = master_nonan.merge(mhss_small, on = 'subjectkey')
master_cb = master_nonan.merge(cbcl_small, on= 'subjectkey')
# %%

#drop NaNs from master dfs
master_dems = master_dems.dropna().reset_index().drop(['index'], axis=1)
master_mh = master_mh.dropna().reset_index().drop(['index'], axis=1)
master_cb = master_cb.dropna().reset_index().drop(['index'], axis=1)
# %%

#create master files with each pps grouping
#PPS bother threshold 4+
p_4 = pqbc_4.loc[pqbc_4['visit'] == 2] #isolate follow-up 2 data
p_4 = p_4[['subjectkey', 'group']]
print(p_4['group'].sum()) #493 HR
#PPS bother threshold 3+
p_3 = pqbc_3.loc[pqbc_3['visit'] == 2] #isolate follow-up 2 data
p_3 = p_3[['subjectkey', 'group']]
print(p_3['group'].sum()) #799 HR
p_2 = pqbc_2.loc[pqbc_2['visit'] == 2] #isolate follow-up 2 data
p_2 = p_2[['subjectkey', 'group']]
print(p_2['group'].sum()) #1265 HR
# %%
#PPS bother threshold 4+
#demographics
master_dems_4 = master_dems.merge(p_4, on= 'subjectkey')
print(master_dems_4['group'].sum()) #182 HR
#mental health df
master_mh_4 = master_mh.merge(p_4, on= 'subjectkey')
print(master_mh_4['group'].sum()) #202 HR
#cbcl 
master_cb_4 = master_cb.merge(p_4, on= 'subjectkey')
print(master_cb_4['group'].sum()) #174 HR

#PPS bother threshold 3+
#demographics
master_dems_3 = master_dems.merge(p_3, on= 'subjectkey')
print(master_dems_3['group'].sum()) #320 HR
#mental health df
master_mh_3 = master_mh.merge(p_3, on= 'subjectkey')
print(master_mh_3['group'].sum()) #353 HR
#cbcl 
master_cb_3 = master_cb.merge(p_3, on= 'subjectkey')
print(master_cb_3['group'].sum()) #295 HR

#PPS bother threshold 2+
#demographics
master_dems_2 = master_dems.merge(p_2, on= 'subjectkey')
print(master_dems_2['group'].sum()) #499 HR
#mental health df
master_mh_2 = master_mh.merge(p_2, on= 'subjectkey')
print(master_mh_2['group'].sum()) #555 HR
#cbcl 
master_cb_2 = master_cb.merge(p_2, on= 'subjectkey')
print(master_cb_2['group'].sum()) #466 HR
# %%

#Calculate Sig Dictionaries
#PQBC Bother Threshold 4+
demos_dict_4 = demos_dict(master_dems_4)
mh_dict_4 = mh_dict(master_mh_4)
cbcl_dict_4 = cbcl_dict(master_cb_4)
#PQBC Bother Threshold 3+
demos_dict_3 = demos_dict(master_dems_3)
mh_dict_3 = mh_dict(master_mh_3)
cbcl_dict_3 = cbcl_dict(master_cb_3)
#PQBC Bother Threshold 2+
demos_dict_2 = demos_dict(master_dems_2)
mh_dict_2 = mh_dict(master_mh_2)
cbcl_dict_2 = cbcl_dict(master_cb_2)