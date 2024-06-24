
#This script investigates statistical differences in environment, mental health, and demographics in LR and HR ABCD participants
# Script considers family history of psychosis, participant developmental milestones, medical and mental health history, and demographics 
#Preprocessing of csvs done in "demos_env_health_preprocessing.py"
# %%
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats as stats
import numpy as np
from scipy.stats import chi2_contingency

# %%
####
#Import demographics/health/environment preprocessed csvs
#participant demographics csv (preprocessed)
demos = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/num_demos.csv')

#ABCD Child Behavioral Checklist (preprocessed)
cbcl = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/cbcl_reduced.csv')

#ABCD mental health summary scores (preprocessing)
mhss = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/mhss_reduced.csv')

#ABCD Developmental History Questionnaire (preprocessed)
devhx = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/devhx_reduced.csv')

#ABCD Family History Questionnaire (preprocessed)
famhx = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/famhx_short.csv')

#ABCD follow-up 2 KSADS--parent (preprocessed)
ksads_p = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/ksads_parent_short.csv')

#ABCD Medical History Questionnaire (preprocessed)
medhx = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/med_hx_reduced.csv')

#PQ-BC (participants who endorsed 4+ PQ-BC questions labeled as 'High Risk')
pps = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/PQ-BC/PPS_T2_endorsed_4.csv')

# %%
####
#function to check how many NaNs exist in each selected df column
def count_nans(df):
    l = list(df)
    for i in l:
        if df[i].isna().sum() >= 50:
            print(i, df[i].isna().sum())
# %%
####
#merge chosen PQ-BC df with other df
def df_merge(df1, pps_df, visit):
    pps_df = pps_df.loc[pps_df['visit'] == visit]
    pps_df = pps_df[['subjectkey', 'group']]
    new_df = df1.merge(pps_df, on = 'subjectkey')
    return new_df
# %%
####
#Demographics statistical analyses between HR/LR groups
#Add HR/LR groupings to demongraphics df
demos_grouped = df_merge(demos, pps, 2)
#create dictionary of demographics statistical comparisons (all categorical variables; chi-square contingency tests used)
dems_dict = {}
contingency_table1 = pd.crosstab(demos_grouped['group'], demos_grouped['sex'])
dems_dict['sex_chi'] = chi2_contingency(contingency_table1)
contingency_table2 = pd.crosstab(demos_grouped['group'], demos_grouped['income'])
dems_dict['income_chi'] = chi2_contingency(contingency_table2)
contingency_table3 = pd.crosstab(demos_grouped['group'], demos_grouped['p_edu'])
dems_dict['p_edu_chi'] = chi2_contingency(contingency_table3)
contingency_table4 = pd.crosstab(demos_grouped['group'], demos_grouped['race'])
dems_dict['ethnicity_chi'] = chi2_contingency(contingency_table4)
print(dems_dict)

demos_grouped.shape
# %%
####
#Child Behavioral Checklist (CBCL) statistical analyses between HR/LR groups
#add HR/LR groupings to cbcl (follow-up 2) df
cbcl_grouped = df_merge(cbcl, pps, 2)
#perform welch's t-test on relevant CBCL summary scores (thought/internalizing/externalizing/total problems)
cbcl_dic = {}
cbcl_HR = cbcl_grouped.loc[cbcl_grouped['group'] == 1]
cbcl_LR = cbcl_grouped.loc[cbcl_grouped['group'] == 0]
cbcl_dic['cbcl_thought_welch'] = stats.ttest_ind(cbcl_HR['cbcl_scr_syn_thought_t'], cbcl_LR['cbcl_scr_syn_thought_t'], equal_var = False)
cbcl_dic['cbcl_internal_welch'] = stats.ttest_ind(cbcl_HR['cbcl_scr_syn_internal_t'], cbcl_LR['cbcl_scr_syn_internal_t'], equal_var = False)
cbcl_dic['cbcl_external_welch'] = stats.ttest_ind(cbcl_HR['cbcl_scr_syn_external_t'], cbcl_LR['cbcl_scr_syn_external_t'], equal_var = False)
cbcl_dic['cbcl_totals_welch'] = stats.ttest_ind(cbcl_HR['cbcl_scr_syn_totprob_t'], cbcl_LR['cbcl_scr_syn_totprob_t'], equal_var = False)
print(cbcl_dic)
cbcl_grouped.shape
# %%
####
#ABCD Mental Health Summary Scores (MHSS) statistical analyses between HR/LR groups
#add HR/LR groupings to mhss df
mhss_grouped = df_merge(mhss, pps, 2)
mhss_grouped['ple_y_ss_total_bad'] = pd.to_numeric(mhss_grouped['ple_y_ss_total_bad'])
mhss_grouped['ple_y_ss_affected_bad_sum'] = pd.to_numeric(mhss_grouped['ple_y_ss_affected_bad_sum'])
mhss_grouped['ple_y_ss_total_good'] = pd.to_numeric(mhss_grouped['ple_y_ss_total_good'])
mhss_grouped['ple_y_ss_affected_good_sum'] = pd.to_numeric(mhss_grouped['ple_y_ss_affected_good_sum'])
mhss_grouped['peq_ss_overt_aggression'] = pd.to_numeric(mhss_grouped['peq_ss_overt_aggression'])
mhss_grouped['peq_ss_relational_aggs'] = pd.to_numeric(mhss_grouped['peq_ss_relational_aggs'])
mhss_grouped['peq_ss_overt_victim'] = pd.to_numeric(mhss_grouped['peq_ss_overt_victim'])
mh_df_HR = mhss_grouped.loc[mhss_grouped['group'] == 1]
mh_df_LR = mhss_grouped.loc[mhss_grouped['group'] == 0]
mh_dict = {}
mh_dict['ACE_total_welch'] = stats.ttest_ind(mh_df_HR['ple_y_ss_total_bad'], mh_df_LR['ple_y_ss_total_bad'], equal_var = False)
mh_dict['ACE_affected_welch'] = stats.ttest_ind(mh_df_HR['ple_y_ss_affected_bad_sum'], mh_df_LR['ple_y_ss_affected_bad_sum'], equal_var = False)
mh_dict['total_good_welch'] = stats.ttest_ind(mh_df_HR['ple_y_ss_total_good'], mh_df_LR['ple_y_ss_total_good'], equal_var = False)
mh_dict['good_affected_welch'] = stats.ttest_ind(mh_df_HR['ple_y_ss_affected_good_sum'], mh_df_LR['ple_y_ss_affected_good_sum'], equal_var = False)
#dropping nans (following columns finicky)
aggs = mhss_grouped[['subjectkey', 'peq_ss_overt_aggression', 'peq_ss_relational_aggs', 'peq_ss_overt_victim', 'group']]
aggs = aggs.dropna()
aggs_HR = aggs.loc[aggs['group'] == 1]
aggs_LR = aggs.loc[aggs['group'] == 0]
mh_dict['overt_aggs'] = stats.ttest_ind(aggs_HR['peq_ss_overt_aggression'], aggs_LR['peq_ss_overt_aggression'], equal_var = False)
mh_dict['rel_aggs'] = stats.ttest_ind(aggs_HR['peq_ss_relational_aggs'], aggs_LR['peq_ss_relational_aggs'], equal_var = False)
mh_dict['overt_victim'] = stats.ttest_ind(aggs_HR['peq_ss_overt_victim'], aggs_LR['peq_ss_overt_victim'], equal_var = False)
print(mh_dict)
mhss_grouped.shape
# %%
####
#ABCD developmental questionnaire statistical analyses between HR and LR groups
#add HR/LR groupings to developmental df
devs_grouped = df_merge(devhx, pps, 2)
list(devs_grouped)
devhx_cols = ['total_substances_used', 'weeks_preterm', 'preterm', 'age_roll_over', 'age_sit', 'age_walk', 'age_first_word',
 'motor_dev_comparison', 'speech_dev_comparison']
cont_grouped = devs_grouped[['subjectkey', 'total_substances_used', 'weeks_preterm', 'age_roll_over', 'age_sit', 'age_walk', 'age_first_word', 'group']]
cat_grouped = devs_grouped[['subjectkey', 'preterm', 'motor_dev_comparison', 'speech_dev_comparison', 'group']]
devhx_dic = {}
dev_HR = cont_grouped.loc[cont_grouped['group'] == 1]
dev_LR = cont_grouped.loc[cont_grouped['group'] == 0]
#welch's t-test for columns containing continuous variables
devhx_dic['prenatal_substances'] = stats.ttest_ind(dev_HR['total_substances_used'], dev_LR['total_substances_used'], equal_var = False)
dev_HR = dev_HR.dropna()
dev_LR = dev_LR.dropna()
devhx_dic['weeks_preterm'] = stats.ttest_ind(dev_HR['weeks_preterm'], dev_LR['weeks_preterm'], equal_var = False)
devhx_dic['age_roll_over'] = stats.ttest_ind(dev_HR['age_roll_over'], dev_LR['age_roll_over'], equal_var = False)
devhx_dic['age_sit'] = stats.ttest_ind(dev_HR['age_sit'], dev_LR['age_sit'], equal_var = False)
devhx_dic['age_walk'] = stats.ttest_ind(dev_HR['age_walk'], dev_LR['age_walk'], equal_var = False)
devhx_dic['age_first_word'] = stats.ttest_ind(dev_HR['age_first_word'], dev_LR['age_first_word'], equal_var = False)
contingency_tableA = pd.crosstab(devs_grouped['group'], devs_grouped['preterm'])
devhx_dic['preterm_chi'] = chi2_contingency(contingency_tableA)
contingency_tableB = pd.crosstab(devs_grouped['group'], devs_grouped['motor_dev_comparison'])
devhx_dic['motor_dev_chi'] = chi2_contingency(contingency_tableB)
contingency_tableC = pd.crosstab(devs_grouped['group'], devs_grouped['speech_dev_comparison'])
devhx_dic['speech_dev_chi'] = chi2_contingency(contingency_tableC)
devhx_dic
# %%
####
#ABCD medical history quesitonnaire statistical analyses between HR and LR groups
#add HR/LR groups to medhx df
med_grouped = df_merge(medhx, pps, 2)
#Perform welch's t-test on medical history totaled injury incidence
med_HR = med_grouped.loc[med_grouped['group'] == 1]
med_LR = med_grouped.loc[med_grouped['group'] == 0]
stats.ttest_ind(med_HR['injury_totals'], med_LR['injury_totals'], equal_var = False)
# %%
####
#ABCD family history questionnaire statstical analsyes between HR and LR groups
#add HR/LR groups to famhx df
fam_grouped = df_merge(famhx, pps, 2)
#Perform welch's t-test on family history of psychosis items
fam_HR = fam_grouped.loc[fam_grouped['group'] == 1]
fam_LR = fam_grouped.loc[fam_grouped['group'] == 0]
fam_dic = {}
fam_dic['first_deg_fam_psychosis'] = stats.ttest_ind(fam_HR['first_deg_psychosis_hx'], fam_LR['first_deg_psychosis_hx'], equal_var = False)
fam_dic['grandparent_psychosis'] = stats.ttest_ind(fam_HR['grandparent_psychosis_hx'], fam_LR['grandparent_psychosis_hx'], equal_var = False)
print(fam_dic)
# %%
####
#KSADS-parent questionnaire statistical analyses between HR and LR groups
#add HR/LR groups to ksads-p df
ksads_grouped = df_merge(ksads_p, pps, 2)
#perform welch's t-test on KSADS sx and dx totals
k_HR = ksads_grouped.loc[ksads_grouped['group'] == 1]
k_LR = ksads_grouped.loc[ksads_grouped['group'] == 0]
ksads_dic = {}
ksads_dic['KSADS_psychosis_sx'] = stats.ttest_ind(k_HR['k_p_total_psychosis_sx'], k_LR['k_p_total_psychosis_sx'], equal_var = False)
ksads_dic['KSADS_psychosis_dx'] = stats.ttest_ind(k_HR['k_p_total_psychosis_dx'], k_LR['k_p_total_psychosis_dx'], equal_var = False)
print(ksads_dic)
# %%
