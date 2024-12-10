
#This script refines the PQ-BC Low-Risk group. Participants who endorsed any PQ-BC items are removed, 
#as are participants who have any family history of psychiatric disorders
#preprocessing to match groups on sex, ethnicity, and age (matching done in risk_group_demo_matching.r)
# %%
####
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
#import CSVs
#Prodromal Psychosis Scores (preprocessed)
pps = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/PQBC_prepped/PQBC_T4_thresh4_ABCD_5.csv')
#PPS scores (raw)
abcd_log = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/abcd_y_lt.csv')
#Family History Questionnaire parts 1 and 2
fhx = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/Env_files/mh_p_fhx.csv')

#participant demographics csv (preprocessed)
demos = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/num_demos.csv')
# %%
pps = pps.rename(columns= {'subjectkey': 'src_subject_id'})
demos = demos.rename(columns= {'subjectkey': 'src_subject_id'})
# %%
#Read back in data, now matched by R MatchIt
#Matched HR and LR sample
matched_sample = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/R_matchIt/T4_demos_matched.csv')
#Matched LR group
matched_LR = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/R_matchIt/T4_LR_matched.csv')
# %%
####
#function to check how many NaNs exist in each selected df column
def count_nans(df):
    l = list(df)
    for i in l:
        if df[i].isna().sum() >= 50:
            print(i, df[i].isna().sum())
# %%
#function to count how many of each ethnicity group in HR and LR groups
def count_instances_race(df):
    white = 0 
    black = 0
    hispanic = 0
    other = 0
    all = 0
    for row in df.index: 
        all += 1
        if df['race'][row] == 'white':
            white += 1
        if df['race'][row] == 'black':
            black += 1
        if df['race'][row] == 'hispanic': 
            hispanic += 1
        if df['race'][row] == 'other':
            other += 1
    print('white')
    print(white/all)
    print('black')
    print(black/all)
    print('hispanic')
    print(hispanic/all)
    print('other')
    print(other/all)
# %%
####
#function to count instances of male and female partiicpants in dfs
def count_instances_sex(df):
    m = 0
    f = 0 
    other = 0
    all = 0
    for row in df.index: 
        all += 1
        if df['sex'][row] == 1:
            m += 1
        if df['sex'][row] == 2:
            f += 1
        if df['sex'][row] == 3:
            other += 1
    print('male')
    print(m/all)
    print('female')
    print(f/all)
    print('other')
    print(other/all)
# %%
####
#Remove LR participants who endorsed any PQ-BC items at timepoint 2
#separate LR participants from HR particpants 
pps_HR = pps.loc[pps['group'] == 1]
pps_LR = pps.loc[pps['group'] == 0]
#Isolate T2 data (earlier endorsement of PQ-BC items may be due to common childhood PLEs)
pps_LR_t4 = pps_LR.loc[pps_LR['visit'] == 4]
#count how many pts endorsed 0 PQBC items at T2
pps_lr_count = 0 
for row in pps_LR_t4.index:
    if pps_LR_t4['PQBC_total_endorsed'][row] == 0:
        pps_lr_count += 1
print(pps_lr_count)
#create LR df containing only pts who endorsed 0 PQBC items at T2
no_PQBC = pps_LR_t4.loc[lambda pps_LR_t4: pps_LR_t4['PQBC_total_endorsed'] == 0] #LR group is 3020 pts
# %%
####
#Remove any LR participants with first-degree relative family history of mental illness
#depression, mania, psychosis symptoms, anxiety, sought professional help for behavioral/emotional/substance problems, hospitalized
#for emotional/mental problems, died by suicide
fam_hx_qs =['src_subject_id', 'fam_history_q6a_depression', 'fam_history_q6d_depression', 'q6k_full_sib_young1_depression', 'q6k_full_sib_young2_depression', 
 'q6k_full_sib_young3_depression', 'q6k_full_sib_young4_depression', 'q6k_full_sib_young5_depression', 'q6l_full_sib_old1_depression', 
 'q6l_full_sib_old2_depression', 'q6l_full_sib_old3_depression', 'q6l_full_sib_old4_depression', 'q6l_full_sib_old5_depression', 
 'q6m_full_sib_same1_depression', 'q6m_full_sib_same2_depression',
'fam_history_q7a_mania', 'fam_history_q7d_mania', 'q7k_full_sib_young1_mania', 'q7k_full_sib_young2_mania', 
 'q7k_full_sib_young3_mania', 'q7k_full_sib_young4_mania', 'q7k_full_sib_young5_mania', 'q7l_full_sib_old1_mania', 
 'q7l_full_sib_old2_mania', 'q7l_full_sib_old3_mania', 'q7l_full_sib_old4_mania', 'q7l_full_sib_old5_mania', 
 'q7m_full_sib_same1_mania', 'q7m_full_sib_same2_mania', 'fam_history_q8a_visions', 'fam_history_q8d_visions', 
 'q8k_full_sib_young1_visions', 'q8k_full_sib_young2_visions', 'q8k_full_sib_young3_visions', 'q8k_full_sib_young4_visions', 
 'q8k_full_sib_young5_visions', 'q8l_full_sib_old1_visions', 'q8l_full_sib_old2_visions', 'q8l_full_sib_old3_visions', 
 'q8l_full_sib_old4_visions', 'q8l_full_sib_old5_visions', 'q8m_full_sib_same1_visions', 'q8m_full_sib_same2_visions', 
 'fam_history_q10a_nerves', 'fam_history_q10d_nerves', 'q10k_full_sib_young1_nerves', 'q10k_full_sib_young2_nerves', 'q10k_full_sib_young3_nerves', 
 'q10k_full_sib_young4_nerves', 'q10k_full_sib_young5_nerves', 'q10l_full_sib_old1_nerves', 'q10l_full_sib_old2_nerves', 
 'q10l_full_sib_old3_nerves', 'q10l_full_sib_old4_nerves', 'q10l_full_sib_old5_nerves', 'q10m_full_sib_same1_nerves', 'q10m_full_sib_same2_nerves', 
 'fam_history_q11a_professional', 'fam_history_q11d_professional', 'q11k_full_sib_young1_prof', 'q11k_full_sib_young2_prof', 
 'q11k_full_sib_young3_prof', 'q11k_full_sib_young4_prof', 'q11k_full_sib_young5_prof', 'q11l_full_sib_old1_prof', 
 'q11l_full_sib_old2_prof', 'q11l_full_sib_old3_prof', 'q11l_full_sib_old4_prof', 'q11l_full_sib_old5_prof', 'q11m_full_sib_same1_prof', 
 'q11m_full_sib_same2_prof', 'fam_history_q12a_hospitalized', 'fam_history_q12d_hospitalized', 'q12k_full_sib_young1_hosp', 
 'q12k_full_sib_young2_hosp', 'q12k_full_sib_young3_hosp', 'q12k_full_sib_young4_hosp', 'q12k_full_sib_young5_hosp', 
 'q12l_full_sib_old1_hosp', 'q12l_full_sib_old2_hosp', 'q12l_full_sib_old3_hosp', 'q12l_full_sib_old4_hosp', 'q12l_full_sib_old5_hosp', 
 'q12m_full_sib_same1_hosp', 'q12m_full_sib_same2_hosp', 'fam_history_q13a_suicide', 'fam_history_q13d_suicide', 'q13k_full_sib_young1_suicide', 
 'q13k_full_sib_young2_suicide', 'q13k_full_sib_young3_suicide', 'q13k_full_sib_young4_suicide', 'q13k_full_sib_young5_suicide', 
 'q13l_full_sib_old1_suicide', 'q13l_full_sib_old2_suicide', 'q13l_full_sib_old3_suicide', 'q13l_full_sib_old4_suicide', 'q13l_full_sib_old5_suicide', 
 'q13m_full_sib_same1_suicide', 'q13m_full_sib_same2_suicide']
#pull columns pertaining to first degree relatives 
fhx_qs = fhx[fam_hx_qs]
fam_hx_qs.pop(0)
#make numeric and replace value 999 (don't know) with 0
for col in fam_hx_qs:
    fhx_qs[col] = pd.to_numeric(fhx_qs[col])
    fhx_qs.loc[fhx_qs[col] == 999, [col]] = 0
#sum first degree relative hx
fhx_qs['first_deg_fam_totals'] = fhx_qs[fam_hx_qs].sum(axis=1)
#count how many participants have no family history of mental illness
all_good = 0
for row in fhx_qs.index:
    if fhx_qs['first_deg_fam_totals'][row] == 0:
        all_good += 1
print(all_good) #5979 total
#merge family history df with LR group that endorsed 0 PQ-BC items
no_PQBC_small = no_PQBC[['src_subject_id', 'PQBC_total_endorsed']]
risk_hx = fhx_qs.merge(no_PQBC_small, on = 'src_subject_id')
#drop any participants with any immediate family history of mental illness
no_risk_hx = risk_hx.loc[lambda risk_hx: risk_hx['first_deg_fam_totals'] == 0] #df shape (1608, 101)
#make list of refined low-risk IDs
no_risk_list = no_risk_hx['src_subject_id'].tolist()
#remove higher risk LR pts from PPS df
pps_refined = pps
pps_refined['risky_LR'] = 0
for row in pps_refined.index: 
    if pps_refined['group'][row] == 0:
        if pps_refined['src_subject_id'][row] not in no_risk_list:
            pps_refined['risky_LR'][row] = 1
#remove any pt with family history of mental health problems
pps_refined = pps_refined.loc[lambda pps_refined: pps_refined['risky_LR'] == 0]
pps_refined = pps_refined.drop(columns= ['risky_LR'])
#save shortened pps df to file
pps_refined.to_csv('/Users/madeleineseitz/Desktop/pps_refined_LR_T4.csv')
# %%
####
#Match participant HR adn LR groups by age, sex, and ethnicity
#merge demos with refined pps df
pps_r_2 = pps_refined.loc[pps_refined['visit'] == 4]
pps_r_short = pps_r_2[['src_subject_id', 'group']]
demos_r_grouped = demos.merge(pps_r_short, on = 'src_subject_id')
#add interview_age column to demos_r_grouped
pps_age = abcd_log.loc[abcd_log['eventname'] == '4_year_follow_up_y_arm_1']
pps_age = pps_age[['src_subject_id', 'interview_age']]
demos_r_grouped = demos.merge(pps_age, on= 'src_subject_id')
demos_r_grouped = demos_r_grouped.merge(pps_r_short, on = 'src_subject_id')
#convert interview_age to numeric and separate HR and LR participants
demos_r_grouped['interview_age'] = pd.to_numeric(demos_r_grouped['interview_age'])
demos_r_HR = demos_r_grouped.loc[demos_r_grouped['group'] == 1]
demos_r_LR = demos_r_grouped.loc[demos_r_grouped['group'] == 0]
# %%
####
#test for significant differences in demographics 
dems_dict = {}
contingency_table1 = pd.crosstab(demos_r_grouped['group'], demos_r_grouped['sex'])
dems_dict['sex_chi'] = chi2_contingency(contingency_table1) # p = 5.3711E-15
contingency_table2 = pd.crosstab(demos_r_grouped['group'], demos_r_grouped['income'])
dems_dict['income_chi'] = chi2_contingency(contingency_table2) # p = 3.312E-19
contingency_table3 = pd.crosstab(demos_r_grouped['group'], demos_r_grouped['p_edu'])
dems_dict['p_edu_chi'] = chi2_contingency(contingency_table3) # p = 3.3493E-07
contingency_table4 = pd.crosstab(demos_r_grouped['group'], demos_r_grouped['race'])
dems_dict['ethnicity_chi'] = chi2_contingency(contingency_table4) # p = 1.9562E-09
dems_dict['age'] = stats.ttest_ind(demos_r_HR['interview_age'], demos_r_LR['interview_age'], equal_var= False) # p = 1.629E-06
print(dems_dict)
demos_r_LR.describe()
#results indicate that groups must be matched for sex, age, and ethnicity
# %%
####
#Prepare data for R MatchIt
#drop NaN values
#save demos_r_grouped to file
#demos_r_grouped = demos_r_grouped.drop(columns=['Unnamed: 0'])
demos_r_grouped_short = demos_r_grouped[['src_subject_id', 'interview_age', 'sex', 'race', 'group']]
demos_r_grouped_short = demos_r_grouped_short.dropna()
demos_r_grouped_short['race'] = pd.to_numeric(demos_r_grouped_short['race'])
demos_r_grouped_short.loc[demos_r_grouped_short['race'] == 1, ['race']] = 'white'
demos_r_grouped_short.loc[demos_r_grouped_short['race'] == 2, ['race']] = 'black'
demos_r_grouped_short.loc[demos_r_grouped_short['race'] == 3, ['race']] = 'hispanic'
demos_r_grouped_short.loc[demos_r_grouped_short['race'] == 4, ['race']] = 'other'
#demos_r_grouped.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/PQ-BC/pps_T2_demos.csv')
#split into HR and LR dfs
demos_r_HR = demos_r_grouped_short.loc[demos_r_grouped['group'] == 1]
demos_r_LR = demos_r_grouped_short.loc[demos_r_grouped['group'] == 0]
#save HR and LR dfs to file
demos_r_HR.to_csv('/Users/madeleineseitz/Desktop/pps_T4_HR_demos.csv')
demos_r_LR.to_csv('/Users/madeleineseitz/Desktop/pps_T4_LR_demos.csv')
# %%
####
#R MatchIt group breakdown sanity check
#isolate salient columns in matched df
matched_short = matched_sample[['src_subject_id', 'interview_age', 'sex', 'race', 'group.1']]
matched_short = matched_short.rename(columns= {'group.1': 'group'})
#isolate HR and LR groups into separate dfs
matched_HR = matched_short.loc[matched_short['group'] == 1]
matched_LR = matched_short.loc[matched_short['group'] == 0]
# %%
#save matched_short to file
matched_short.to_csv('/Users/madeleineseitz/Desktop/T4_matched_demos_short.csv')
# %%
#perform statistical analysis of demos in group breakdown
dems_dict = {}
contingency_table1 = pd.crosstab(matched_short['group'], matched_short['sex'])
dems_dict['sex_chi'] = chi2_contingency(contingency_table1) # p = 0.7935
contingency_table4 = pd.crosstab(matched_short['group'], matched_short['race'])
dems_dict['ethnicity_chi'] = chi2_contingency(contingency_table4) # p = 0.6198
dems_dict['age'] = stats.ttest_ind(matched_HR['interview_age'], matched_LR['interview_age']) # p = 0.8663
print(dems_dict)
#none of the tests returned significant results
# %%
count_instances_race(matched_HR)
# %%
count_instances_race(matched_LR)
# %%
count_instances_sex(matched_HR)
# %%
count_instances_sex(matched_LR)
# %%
####
#Update pps df with completed high and low risk groupings
low_risk_group = list(matched_LR['src_subject_id'])
high_risk_group = list(matched_HR['src_subject_id'])
pps_hr_lr = pps
pps_hr_lr['risky_LR'] = 0
for row in pps_hr_lr.index: 
    if pps_hr_lr['group'][row] == 0:
        if pps_hr_lr['src_subject_id'][row] not in low_risk_group:
            pps_hr_lr['risky_LR'][row] = 1
    else:
        if pps_hr_lr['src_subject_id'][row] not in high_risk_group:
            pps_hr_lr['risky_LR'][row] = 1
#remove any LR participant not in matched df
pps_hr_lr = pps_hr_lr.loc[lambda pps_refined: pps_refined['risky_LR'] == 0]
pps_hr_lr = pps_hr_lr.drop(columns= ['risky_LR'])
#save df to file
pps_hr_lr.to_csv('/Users/madeleineseitz/Desktop/T4_pps_hr_lr_all_fu.csv')
#isolate df containing HR and LR at follow-up 2 only
pps_hr_lr_4 = pps_hr_lr.loc[pps_hr_lr['visit'] == 4]
#save T2 df to file
pps_hr_lr_4.to_csv('/Users/madeleineseitz/Desktop/pps_hr_lr_t4.csv')
# %%
num_demos = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/num_demos.csv')
num_demos = num_demos.rename(columns= {'subjectkey': 'src_subject_id'})
age_demos = matched_short[['src_subject_id', 'interview_age', 'group']]
matched_demos = num_demos.merge(age_demos, on= 'src_subject_id')
# %%
#matched_demos = matched_demos.drop(['Unnamed: 0'])
matched_demos.to_csv('/Users/madeleineseitz/Desktop/T4_matched_demos.csv')
# %%
