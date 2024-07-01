
#This script refines the PQ-BC Low-Risk group. Participants who endorsed any PQ-BC items are removed, 
#as are participants who have any family history of psychiatric disorders
# %%
####
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats as stats
import numpy as np
# %%
####
#import CSVs
pps = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/PQ-BC/PQBC_4.csv')
fhx_1 = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/abcd_may2024/fhxp102.csv')
fhx_2 = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/abcd_may2024/fhxp201.csv')
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
#Remove LR participants who endorsed any PQ-BC items at timepoint 2
#separate LR participants from HR particpants 
pps_HR = pps.loc[pps['group'] == 1]
pps_LR = pps.loc[pps['group'] == 0]
#Isolate T2 data (earlier endorsement of PQ-BC items may be due to common childhood PLEs)
pps_LR_t2 = pps_LR.loc[pps_LR['visit'] == 2]
#count how many pts endorsed 0 PQBC items at T2
pps_lr_count = 0 
for row in pps_LR_t2.index:
    if pps_LR_t2['PQBC_total_endorsed'][row] == 0:
        pps_lr_count += 1
print(pps_lr_count)
#create LR df containing only pts who endorsed 0 PQBC items at T2
no_PQBC = pps_LR_t2.loc[lambda pps_LR_t2: pps_LR_t2['PQBC_total_endorsed'] == 0] #LR group is 5930 pts
# %%
# ####
# #Remove LR particpants with any history of family psychiatric illness
# fhx_1 = fhx_1.drop([0])
# fhx_2 = fhx_2.drop([0])
# fhx_2 = fhx_2.rename(columns= {'visit': 'eventname'})
# fhx = fhx_1.merge(fhx_2, on = ['subjectkey', 'eventname'])
# psych_cols = ['subjectkey', 'eventname','famhx_4_p', 'fam_history_5_yes_no', 'fam_history_6_yes_no', 'fam_history_7_yes_no', 'fam_history_8_yes_no', 
#                  'fam_history_9_yes_no', 'fam_history_10_yes_no', 'fam_history_11_yes_no', 'fam_history_12_yes_no', 'fam_history_13_yes_no']
# #create reduced df containing yes/no responses to family history questions 
# fhx_small = fhx[psych_cols]
# #drop NaN rows
# count_nans(fhx_small) #144 NaNs per column
# fhx_nonan = fhx_small.dropna()
# #replace value '999' (don't know) with 0
# num_cols = ['famhx_4_p', 'fam_history_5_yes_no', 'fam_history_6_yes_no', 'fam_history_7_yes_no', 'fam_history_8_yes_no', 
#                  'fam_history_9_yes_no', 'fam_history_10_yes_no', 'fam_history_11_yes_no', 'fam_history_12_yes_no', 'fam_history_13_yes_no']
# for col in num_cols:
#     fhx_nonan[col] = pd.to_numeric(fhx_nonan[col])
#     fhx_nonan.loc[fhx_nonan[col] == 999, [col]] = 0
# #add together total endorsed questions
# fhx_totals = fhx_nonan
# fhx_totals['fhx_total'] = fhx_totals[num_cols].sum(axis=1)
# #separate more ambiguous questions surrounding substance use and general "trouble" from other mental health history questions
# fhx_totals['subs_total'] = fhx_totals[['famhx_4_p', 'fam_history_5_yes_no', 'fam_history_9_yes_no']].sum(axis=1)
# fhx_totals['mh_totals'] = fhx_totals[['fam_history_6_yes_no', 'fam_history_7_yes_no', 'fam_history_8_yes_no', 'fam_history_10_yes_no', 'fam_history_11_yes_no', 'fam_history_12_yes_no', 'fam_history_13_yes_no']].sum(axis=1)
# #count how many participants fall into each group
# all_count = 0
# mh_count = 0
# for row in fhx_totals.index:
#     if fhx_totals['fhx_total'][row] == 0:
#         all_count += 1
#     if fhx_totals['mh_totals'][row] == 0:
#         mh_count += 1
# print('Fhx totals LR count =')
# print(all_count)
# print('fhx mh total LR count =')
# print(mh_count)
# #merge shortened PPS df with fhx df
# no_PQBC_small = no_PQBC[['subjectkey', 'PQBC_total_endorsed']]
# risk_df = fhx_totals.merge(no_PQBC_small, on = 'subjectkey')
# #remove any participants who had family members with mental health history
# no_risk_df = risk_df.loc[lambda risk_df: risk_df['mh_totals'] == 0]
# #make list of subjectkeys
# no_risk_list = no_risk_df['subjectkey'].tolist()
# #remove higher risk LR pts from PPS df
# pps_refined = pps
# pps_refined['risky_LR'] = 0
# for row in pps_refined.index: 
#     if pps_refined['group'][row] == 0:
#         if pps_refined['subjectkey'][row] not in no_risk_list:
#             pps_refined['risky_LR'][row] = 1
# #remove any pt with family history of mental health problems
# pps_refined = pps_refined.loc[lambda pps_refined: pps_refined['risky_LR'] == 0]
# #save shortened pps df to file
# pps_refined.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/PQ-BC/pps_refined_LR.csv')
# %%
####
#Remove any LR participants with first-degree relative family history of mental illness
#depression, mania, psychosis symptoms, anxiety, sought professional help for behavioral/emotional/substance problems, hospitalized
#for emotional/mental problems, died by suicide
fam_hx_qs =['subjectkey', 'fam_history_q6a_depression', 'fam_history_q6d_depression', 'q6k_full_sib_young1_depression', 'q6k_full_sib_young2_depression', 
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
fam_hx_qs = fam_hx_qs.pop[0]
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
print(all_good) #5985 total
#merge family history df with LR group that endorsed 0 PQ-BC items
no_PQBC_small = no_PQBC[['subjectkey', 'PQBC_total_endorsed']]
risk_hx = fhx_qs.merge(no_PQBC_small, on = 'subjectkey')
#drop any participants with any immediate family history of mental illness
no_risk_hx = risk_hx.loc[lambda risk_hx: risk_hx['first_deg_fam_totals'] == 0] #df shape (3119, 101)
#make list of refined low-risk IDs
no_risk_list = no_risk_hx['subjectkey'].tolist()
#remove higher risk LR pts from PPS df
pps_refined = pps
pps_refined['risky_LR'] = 0
for row in pps_refined.index: 
    if pps_refined['group'][row] == 0:
        if pps_refined['subjectkey'][row] not in no_risk_list:
            pps_refined['risky_LR'][row] = 1
#remove any pt with family history of mental health problems
pps_refined = pps_refined.loc[lambda pps_refined: pps_refined['risky_LR'] == 0]
#save shortened pps df to file
pps_refined.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/PQ-BC/pps_refined_LR.csv')
# %%
