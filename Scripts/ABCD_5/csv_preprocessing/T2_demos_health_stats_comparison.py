
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
cbcl = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_cbcl_T2_reduced.csv')

#ABCD Developmental History Questionnaire (preprocessed)
devhx = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_baseline_devhx_reduced.csv')

#ABCD Family History Questionnaire (preprocessed)
famhx = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_famhx_short.csv')

#ABCD follow-up 2 KSADS--parent (preprocessed)
ksads_p = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd_5_ksads_parent_T2_short.csv')

#ABCD Medical History Questionnaire (preprocessed)
medhx = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_med_hx_reduced.csv')

#ABCD Adverse Life Experiences (youth) (preprocessed)
ale_y = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_ALE_y_T2_short.csv')

#ABCD Adverse Life Expereinces (parent) (preprocessed)
ale_p = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_ALE_p_T2_short.csv')

#Peer Experiences Questionnaire (preprocessed)
peq = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_PEQ_y_T2_ss.csv')

#Discrimination Questionnaire (youth) (preprocessed)
disc = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_disc_y_T2_short.csv')

#Neighborhood Safety Questionnaire (youth) (preprocessed)
nsc = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_NSC_T2_short.csv')

#neglect (T3) (youth) (preprocessed)
neglect = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_mnbs_y_T3_short.csv')
# %%
#community cohesion score (T2/T4) (parent) (preprocessed
ccs = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_preprocessed/abcd5_ccs_p_short.csv')

#PQ-BC (participants who endorsed 4+ PQ-BC questions labeled as 'High Risk')
pps = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/PQBC_prepped/PQBC_T2_grouped_ABCD_5.csv')

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
    pps_df = pps_df[['src_subject_id', 'group']]
    new_df = df1.merge(pps_df, on = 'src_subject_id')
    return new_df
# %%
####
#Demographics statistical analyses between HR/LR groups
#Add HR/LR groupings to demongraphics df
demos_grouped = df_merge(demos, pps, 2)
#create dictionary of demographics statistical comparisons (all categorical variables; chi-square contingency tests used)
dems_dict = {}
contingency_table1 = pd.crosstab(demos_grouped['group'], demos_grouped['sex'])
dems_dict['sex_chi'] = chi2_contingency(contingency_table1) # p = 0.1839
contingency_table2 = pd.crosstab(demos_grouped['group'], demos_grouped['income'])
dems_dict['income_chi'] = chi2_contingency(contingency_table2) # p = 6.6465E-62
contingency_table3 = pd.crosstab(demos_grouped['group'], demos_grouped['p_edu'])
dems_dict['p_edu_chi'] = chi2_contingency(contingency_table3) # p = 1.4621E-23
contingency_table4 = pd.crosstab(demos_grouped['group'], demos_grouped['race'])
dems_dict['ethnicity_chi'] = chi2_contingency(contingency_table4) # p = 1.4902E-68
print(dems_dict)

demos_grouped.shape
# %%
pps = pps.rename(columns= {'subjectkey': 'src_subject_id'})
# %%
####
#Child Behavioral Checklist (CBCL) statistical analyses between HR/LR groups
#add HR/LR groupings to cbcl (follow-up 2) df
cbcl_grouped = df_merge(cbcl, pps, 2)
cbcl_grouped = cbcl_grouped.dropna()
#perform welch's t-test on relevant CBCL summary scores (thought/internalizing/externalizing/total problems)
cbcl_dic = {}
cbcl_HR = cbcl_grouped.loc[cbcl_grouped['group'] == 1]
cbcl_LR = cbcl_grouped.loc[cbcl_grouped['group'] == 0]
cbcl_dic['cbcl_thought_welch'] = stats.ttest_ind(cbcl_HR['cbcl_scr_syn_thought_t'], cbcl_LR['cbcl_scr_syn_thought_t'], equal_var = False) # p = 2.0164E-24
cbcl_dic['cbcl_internal_welch'] = stats.ttest_ind(cbcl_HR['cbcl_scr_syn_internal_t'], cbcl_LR['cbcl_scr_syn_internal_t'], equal_var = False) # p = 3.6082E-26
cbcl_dic['cbcl_external_welch'] = stats.ttest_ind(cbcl_HR['cbcl_scr_syn_external_t'], cbcl_LR['cbcl_scr_syn_external_t'], equal_var = False) # p = 1.1279E-32
cbcl_dic['cbcl_totals_welch'] = stats.ttest_ind(cbcl_HR['cbcl_scr_syn_totprob_t'], cbcl_LR['cbcl_scr_syn_totprob_t'], equal_var = False) # p = 2.8478E-44
print(cbcl_dic)
cbcl_grouped.shape
# %%
####
#ABCD developmental questionnaire statistical analyses between HR and LR groups
#add HR/LR groupings to developmental df
devs_grouped = df_merge(devhx, pps, 2)
list(devs_grouped)
devhx_cols = ['total_substances_used', 'weeks_preterm', 'preterm', 'age_roll_over', 'age_sit', 'age_walk', 'age_first_word',
 'motor_dev_comparison', 'speech_dev_comparison']
cont_grouped = devs_grouped[['src_subject_id', 'total_substances_used', 'weeks_preterm', 'age_roll_over', 'age_sit', 'age_walk', 'age_first_word', 'group']]
cat_grouped = devs_grouped[['src_subject_id', 'preterm', 'motor_dev_comparison', 'speech_dev_comparison', 'group']]
devhx_dic = {}
dev_HR = cont_grouped.loc[cont_grouped['group'] == 1]
dev_LR = cont_grouped.loc[cont_grouped['group'] == 0]
#welch's t-test for columns containing continuous variables
devhx_dic['prenatal_substances'] = stats.ttest_ind(dev_HR['total_substances_used'], dev_LR['total_substances_used'], equal_var = False) # p = 2.0891E-11
dev_HR = dev_HR.dropna()
dev_LR = dev_LR.dropna()
devhx_dic['weeks_preterm'] = stats.ttest_ind(dev_HR['weeks_preterm'], dev_LR['weeks_preterm'], equal_var = False) # p = 0.3866
devhx_dic['age_roll_over'] = stats.ttest_ind(dev_HR['age_roll_over'], dev_LR['age_roll_over'], equal_var = False) # p = 0.3107
devhx_dic['age_sit'] = stats.ttest_ind(dev_HR['age_sit'], dev_LR['age_sit'], equal_var = False) # p = 0.6128
devhx_dic['age_walk'] = stats.ttest_ind(dev_HR['age_walk'], dev_LR['age_walk'], equal_var = False) # p = 0.285
devhx_dic['age_first_word'] = stats.ttest_ind(dev_HR['age_first_word'], dev_LR['age_first_word'], equal_var = False) # p = 0.6333
contingency_tableA = pd.crosstab(devs_grouped['group'], devs_grouped['preterm']) # p = 0.7884
devhx_dic['preterm_chi'] = chi2_contingency(contingency_tableA) 
contingency_tableB = pd.crosstab(devs_grouped['group'], devs_grouped['motor_dev_comparison']) # p = 6.8213E-08
devhx_dic['motor_dev_chi'] = chi2_contingency(contingency_tableB)
contingency_tableC = pd.crosstab(devs_grouped['group'], devs_grouped['speech_dev_comparison']) # p = 8.6029E-05
devhx_dic['speech_dev_chi'] = chi2_contingency(contingency_tableC)
devhx_dic
# %%
####
#ABCD medical history quesitonnaire statistical analyses between HR and LR groups
#add HR/LR groups to medhx df
med_grouped = df_merge(medhx, pps, 2)
med_grouped = med_grouped.dropna()
#Perform welch's t-test on medical history totaled injury incidence
med_HR = med_grouped.loc[med_grouped['group'] == 1]
med_LR = med_grouped.loc[med_grouped['group'] == 0]
stats.ttest_ind(med_HR['total_times_to_ER'], med_LR['total_times_to_ER'], equal_var = False) #p = 9.4165E-08
stats.ttest_ind(med_HR['times_to_ER_past_year'], med_LR['times_to_ER_past_year'], equal_var = False) # p = 1.0406E-06
stats.ttest_ind(med_HR['injury_totals'], med_LR['injury_totals'], equal_var = False) # p = 0.0899
# %%
####
#ABCD family history questionnaire statstical analsyes between HR and LR groups
#add HR/LR groups to famhx df
fam_grouped = df_merge(famhx, pps, 2)
#Perform welch's t-test on family history of psychosis items
fam_HR = fam_grouped.loc[fam_grouped['group'] == 1]
fam_LR = fam_grouped.loc[fam_grouped['group'] == 0]
fam_dic = {}
fam_dic['first_deg_fam_psychosis'] = stats.ttest_ind(fam_HR['first_deg_psychosis_hx'], fam_LR['first_deg_psychosis_hx'], equal_var = False) # p = 0.0041
fam_dic['grandparent_psychosis'] = stats.ttest_ind(fam_HR['grandparent_psychosis_hx'], fam_LR['grandparent_psychosis_hx'], equal_var = False) # p = 0.0357
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
ksads_dic['KSADS_psychosis_sx'] = stats.ttest_ind(k_HR['k_p_total_psychosis_sx'], k_LR['k_p_total_psychosis_sx'], equal_var = False) # p = 1.056E-11
ksads_dic['KSADS_psychosis_dx'] = stats.ttest_ind(k_HR['k_p_total_psychosis_dx'], k_LR['k_p_total_psychosis_dx'], equal_var = False) # p = 2.6904E-07
print(ksads_dic)

# %%
####
#Adverse Life Experiences (Youth) statistical analyses between HR/LR groups
#add HR/LR groupings to ALE_y df
ale_y_grouped = df_merge(ale_y, pps, 2)
ale_y_grouped = ale_y_grouped.dropna()

ale_y_grouped['ple_y_ss_total_number'] = pd.to_numeric(ale_y_grouped['ple_y_ss_total_number'])
ale_y_grouped['ple_y_ss_total_good'] = pd.to_numeric(ale_y_grouped['ple_y_ss_total_good'])
ale_y_grouped['ple_y_ss_total_bad'] = pd.to_numeric(ale_y_grouped['ple_y_ss_total_bad'])
ale_y_grouped['ple_y_ss_affected_good_sum'] = pd.to_numeric(ale_y_grouped['ple_y_ss_affected_good_sum'])
ale_y_grouped['ple_y_ss_affected_bad_sum'] = pd.to_numeric(ale_y_grouped['ple_y_ss_affected_bad_sum'])

ale_y_HR = ale_y_grouped.loc[ale_y_grouped['group'] == 1]
ale_y_LR = ale_y_grouped.loc[ale_y_grouped['group'] == 0]
ale_y_dict = {}
ale_y_dict['ALE_total_welch'] = stats.ttest_ind(ale_y_HR['ple_y_ss_total_number'], ale_y_LR['ple_y_ss_total_number'], equal_var = False) # p = 3.5785E-121
ale_y_dict['ALE_total_good'] = stats.ttest_ind(ale_y_HR['ple_y_ss_total_good'], ale_y_LR['ple_y_ss_total_good'], equal_var = False) # p = 2.6247E-08
ale_y_dict['ALE_total_bad'] = stats.ttest_ind(ale_y_HR['ple_y_ss_total_bad'], ale_y_LR['ple_y_ss_total_bad'], equal_var = False) # p = 3.2403E-105
ale_y_dict['ALE_affected_good'] = stats.ttest_ind(ale_y_HR['ple_y_ss_affected_good_sum'], ale_y_LR['ple_y_ss_affected_good_sum'], equal_var = False) #p = 0.7574
ale_y_dict['ALE_affected_bad'] = stats.ttest_ind(ale_y_HR['ple_y_ss_affected_bad_sum'], ale_y_LR['ple_y_ss_affected_bad_sum'], equal_var = False) # p = 2.0301E-100
print(ale_y_dict)
# %%
####
#Adverse Life Expereinces (Parent) statistical analyses between HR/LR groups
#add HR/LR groupings to ALE_p df
ale_p_grouped = df_merge(ale_p, pps, 2)
ale_p_grouped = ale_p_grouped.dropna()

ale_p_grouped['ple_p_ss_total_number'] = pd.to_numeric(ale_p_grouped['ple_p_ss_total_number'])
ale_p_grouped['ple_p_ss_total_good'] = pd.to_numeric(ale_p_grouped['ple_p_ss_total_good'])
ale_p_grouped['ple_p_ss_total_bad'] = pd.to_numeric(ale_p_grouped['ple_p_ss_total_bad'])
ale_p_grouped['ple_p_ss_affected_good_sum'] = pd.to_numeric(ale_p_grouped['ple_p_ss_affected_good_sum'])
ale_p_grouped['ple_p_ss_affected_bad_sum'] = pd.to_numeric(ale_p_grouped['ple_p_ss_affected_bad_sum'])

ale_p_HR = ale_p_grouped.loc[ale_p_grouped['group'] == 1]
ale_p_LR = ale_p_grouped.loc[ale_p_grouped['group'] == 0]
ale_p_dict = {}
ale_p_dict['ALE_total_welch'] = stats.ttest_ind(ale_p_HR['ple_p_ss_total_number'], ale_p_LR['ple_p_ss_total_number'], equal_var = False) # p = 8.5446E-11
ale_p_dict['ALE_total_good'] = stats.ttest_ind(ale_p_HR['ple_p_ss_total_good'], ale_p_LR['ple_p_ss_total_good'], equal_var = False) # p = 0.2947
ale_p_dict['ALE_total_bad'] = stats.ttest_ind(ale_p_HR['ple_p_ss_total_bad'], ale_p_LR['ple_p_ss_total_bad'], equal_var = False) # p = 5.8056E-18
ale_p_dict['ALE_affected_good'] = stats.ttest_ind(ale_p_HR['ple_p_ss_affected_good_sum'], ale_p_LR['ple_p_ss_affected_good_sum'], equal_var = False) #p = 0.6237
ale_p_dict['ALE_affected_bad'] = stats.ttest_ind(ale_p_HR['ple_p_ss_affected_bad_sum'], ale_p_LR['ple_p_ss_affected_bad_sum'], equal_var = False) # p = 1.9103E-19
print(ale_p_dict)

# %%
####
#Peer Experiences Questionnaire (Youth) statistical analyses between HR/LR groups
#add HR/LR groupings to peq df
peq_cols = ['peq_ss_relational_victim' , 'peq_ss_reputation_aggs', 'peq_ss_reputation_victim', 'peq_ss_overt_aggression', 'peq_ss_overt_victim', 'peq_ss_relational_aggs']
peq_grouped = df_merge(peq, pps, 2)
peq_grouped = peq_grouped.dropna()

for i in peq_cols:
    peq_grouped[i] = pd.to_numeric(peq_grouped[i])

peq_HR = peq_grouped.loc[peq_grouped['group'] == 1]
peq_LR = peq_grouped.loc[peq_grouped['group'] == 0]
peq_dict = {}

for i in peq_cols:
    peq_dict[i] = stats.ttest_ind(peq_HR[i], peq_LR[i], equal_var = False)
print(peq_dict)

#relational victim p = 1.7271
#reputational aggression p = 2.5198E-24
#reputation victim p = 7.4267E-90
#overt aggression p = 1.4208E-45
#overt victim p = 6.3868E-80
#relational aggression p = 2.4046E-31
# %%
#Discrimination measure (youth) Statistical analyses between HR/LR groups
disc_cols = ['dim_yesno_q1', 'dim_yesno_q2', 'dim_yesno_q3', 'dim_yesno_q4', 'dim_y_ss_mean']
disc_grouped = df_merge(disc, pps, 2)
disc_grouped = disc_grouped.dropna()
disc_grouped = disc_grouped.drop(columns= ['Unnamed: 0'])

for i in disc_cols:
    disc_grouped[i] = pd.to_numeric(disc_grouped[i])

disc_HR = disc_grouped.loc[disc_grouped['group'] == 1]
disc_LR = disc_grouped.loc[disc_grouped['group'] == 0]
disc_dict = {}

for i in disc_cols:
    disc_dict[i] = stats.ttest_ind(disc_HR[i], disc_LR[i], equal_var = False)
print(disc_dict)

#dm_q1 p = 1.3064E-27
#dm_q2 p = 7.6423E-12
#dm_q3 p = 2.4001E-35
#dm_q4 p = 4.4796E-31
#dm_mean p = 3.5863E-64
# %%
####
#Neighborhood Safety Questionnaire (youth) statistical analyses between HR/LR groups

nsc_grouped = df_merge(nsc, pps, 2)
nsc_grouped = nsc_grouped.dropna()
nsc_grouped = nsc_grouped.drop(columns= ['Unnamed: 0'])

nsc_HR = nsc_grouped.loc[nsc_grouped['group'] == 1]
nsc_LR = nsc_grouped.loc[nsc_grouped['group'] == 0]

stats.ttest_ind(nsc_HR['neighborhood_crime_y'], nsc_LR['neighborhood_crime_y'], equal_var = False) # p = 5.4626E-56

# %%
#Neglect measure (youth) Statistical analyses between HR/LR groups
neglect_cols = [ 'mnbs_bad', 'mnbs_doing', 'mnbs_friend', 'mnbs_homework', 'mnbs_play', 'mnbs_school', 'mnbs_trouble', 'mnbs_understand', 
                'mnbs_ss_monitor_supervision', 'mnbs_ss_mean_all', 'mnbs_ss_ed_support']
neglect_grouped = df_merge(neglect, pps, 2)
neglect_grouped = neglect_grouped.dropna()
neglect_grouped = neglect_grouped.drop(columns= ['Unnamed: 0'])

for i in neglect_cols:
    neglect_grouped[i] = pd.to_numeric(neglect_grouped[i])

neglect_HR = neglect_grouped.loc[neglect_grouped['group'] == 1]
neglect_LR = neglect_grouped.loc[neglect_grouped['group'] == 0]
neglect_dict = {}

for i in neglect_cols:
    neglect_dict[i] = stats.ttest_ind(neglect_HR[i], neglect_LR[i], equal_var = False)
print(neglect_dict)

# 'mnbs_bad' p = 0.0006
#  'mnbs_doing' p = 9.3442E-08
#  'mnbs_friend' p = 1.2905E-08
#  'mnbs_homework' p = 1.0848
#  'mnbs_play' p = 3.8188E-09
#  'mnbs_school' p = 0.0002
#  'mnbs_trouble' p = 9.0668E-08
#  'mnbs_understand' p = 4.5788E-19
#  'mnbs_ss_monitor_supervision' p = 1.07575E-15
#  'mnbs_ss_mean_all' p = 2.189E-25
#  'mnbs_ss_ed_support p = 1.0429E-18
# %%
#community cohesion score (parent) Statistical analyses between HR/LR groups
ccs_cols = ['comc_ss_cohesion_p', 'comc_ss_control_p', 'comc_ss_collective_capacity_p']
ccs_grouped = df_merge(ccs, pps, 2)
ccs_grouped = ccs_grouped.dropna()
ccs_grouped = ccs_grouped.drop(columns= ['Unnamed: 0'])

for i in ccs_cols:
    ccs_grouped[i] = pd.to_numeric(ccs_grouped[i])

ccs_HR = ccs_grouped.loc[ccs_grouped['group'] == 1]
ccs_LR = ccs_grouped.loc[ccs_grouped['group'] == 0]
ccs_dict = {}

for i in ccs_cols:
    ccs_dict[i] = stats.ttest_ind(ccs_HR[i], ccs_LR[i], equal_var = False)
print(ccs_dict)

# 'ss_cohesion_p' p = 0.2144
# 'comc_ss_control_p' p = 0.4403
# 'comc_ss_collective_capacity_p' p = 0.2977
# %%
