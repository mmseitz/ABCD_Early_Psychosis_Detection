#%%
#This script performs some data cleaning on the ABCD Prodromal Psychosis Scale CSV, calculating how many items each participant endorsed and whether they were 
#bothered by each item in both the total PQ-BC and the shortened 7-item PQ-BC screener. Screener totals and PQ-BC totals are saved in CSVs at the end. 

import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats as stats
import numpy as np

#%%
#function takes a dataframe and time point (str) to calculate the number of endorsed screener items in
#the ABCD PQ-BC. Output is df with addtional column (screener endorsed count) and value_counts of df

def calc_endorsed_screen_total(df, timepoint_num):
    if timepoint_num == 'baseline':
        tp = 'baseline_year_1_arm_1'
    else:
        tp = str(timepoint_num + '_year_follow_up_y_arm_1') 
    print('returning totals for ' + tp)
    col = str('screen_endorsed_total_' + timepoint_num)
    df[col] = 'NaN'
    for row in df.index:
        if df['visit'][row] == tp:
            df[col][row] = 0
            if df['prodromal_5_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_10_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_14_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_16_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_19_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_20_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_21_y'][row] == 1:
                df[col][row] += 1
        else:
            pass
    return df[col].value_counts()

#%%
#function takes a dataframe and time point (str) to calculate the number of endorsed screener items in
#the ABCD PQ-BC that also bothered the participant. Output is df with addtional column (screener bother count)
#and value_counts of df

def calc_screen_bother_total(df, timepoint_num):
    if timepoint_num == 'baseline':
        tp = 'baseline_year_1_arm_1'
    else:
        tp = str(timepoint_num + '_year_follow_up_y_arm_1') 
    print('returning totals for ' + tp)
    col = str('screen_bother_total_' + timepoint_num)
    df[col] = 'NaN'
    for row in df.index: 
        if df['visit'][row] == tp:
            df[col][row] = 0
            if df['pps_5_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_10_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_14_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_16_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_19_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_20_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_21_bother_yn'][row] == 1:
                df[col][row] += 1
    return df[col].value_counts()
# %%
#function takes a dataframe and time point (str) to calculate the total number of endorsed items in
#the ABCD PQ-BC. Output is df with addtional column (pq-bc endorsed totals) and value_counts of df

def calc_pqbc_endorsed_total(df, timepoint_num):
    if timepoint_num == 'baseline':
        tp = 'baseline_year_1_arm_1'
    else:
        tp = str(timepoint_num + '_year_follow_up_y_arm_1') 
    print('returning totals for ' + tp)
    col = str('pqbc_endorsed_total_' + timepoint_num)
    df[col] = 'NaN'
    for row in df.index:
        if df['visit'][row] == tp:
            df[col][row] = 0
            if df['prodromal_1_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_2_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_3_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_4_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_5_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_6_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_7_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_8_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_9_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_10_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_11_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_12_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_13_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_14_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_15_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_16_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_17_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_18_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_19_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_20_y'][row] == 1:
                df[col][row] += 1
            if df['prodromal_21_y'][row] == 1:
                df[col][row] += 1
        else:
            pass
    return df[col].value_counts()

# %%
#function takes a dataframe and time point (str) to calculate the total number of PQ-BC items that participants
#indicated bothered them. Output is df with addtional column (pq-bc bother totals) and value_counts of df

def calc_pqbc_bother_total(df, timepoint_num):
    if timepoint_num == 'baseline':
        tp = 'baseline_year_1_arm_1'
    else:
        tp = str(timepoint_num + '_year_follow_up_y_arm_1') 
    print('returning totals for ' + tp)
    col = str('pqbc_bother_total_' + timepoint_num)
    df[col] = 'NaN'
    for row in df.index:
        if df['visit'][row] == tp:
            df[col][row] = 0
            if df['pps_1_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_2_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_3_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_4_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_5_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_6_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_7_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_8_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_9_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_10_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_11_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_12_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_13_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_14_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_15_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_16_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_17_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_18_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_19_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_20_bother_yn'][row] == 1:
                df[col][row] += 1
            if df['pps_21_bother_yn'][row] == 1:
                df[col][row] += 1
        else:
            pass
    return df[col].value_counts()

# %%
#function takes df with multiple timepoint columns and collapses columns into single "totals" column to promote easier plotting
#function also replaces longer timepoint names with numbers
def collapse_cols_plt(df, bl_col_name, T1_col_name, T2_col_name, T3_col_name, new_col_name):
    df.loc[df['visit'] == 'baseline_year_1_arm_1', 'visit'] = 0
    df.loc[df['visit'] == '1_year_follow_up_y_arm_1', 'visit'] = 1
    df.loc[df['visit'] == '2_year_follow_up_y_arm_1', 'visit'] = 2
    df.loc[df['visit'] == '3_year_follow_up_y_arm_1', 'visit'] = 3
    df[new_col_name] = 0
    for row in df.index: 
        if df['visit'][row] == 0:
            df[new_col_name][row] = df[bl_col_name][row]
        if df['visit'][row] == 1:
            df[new_col_name][row] = df[T1_col_name][row]
        if df['visit'][row] == 2:
            df[new_col_name][row] = df[T2_col_name][row]
        if df['visit'][row] == 3:
            df[new_col_name][row] = df[T3_col_name][row]
    df = df.drop(columns = [bl_col_name, T1_col_name, T2_col_name, T3_col_name])
    return df

# %%
#Read in PQ-BC CSV
pps = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/data/abcd_fu4/abcd_fu4_csv/psychosis_csvs/pps01.csv')

#Drop unnecessary columns
pps = pps.drop(columns = ['collection_id', 'pps01_id', 'dataset_id', 'src_subject_id', 'interview_date',
                         'collection_title'])
pps = pps.drop([0])
# %%
#create df for PQ-BC totals
pps_counts = pps

#Count all instances of endorsed PQ-BC questions at baseline

#All endorsed PQ-BC questions at baseline
calc_pqbc_endorsed_total(pps_counts, 'baseline')

#All endorsed PQ-BC questions that also bothered participants at baseline
calc_pqbc_bother_total(pps_counts, 'baseline')

#All endorsed PQ-BC questions within the 7-question sub-screener  at baseline
calc_endorsed_screen_total(pps_counts, 'baseline')

#All edorsed PQ-BC screeer questions that also bothered participants at baseline
calc_screen_bother_total(pps_counts, 'baseline')

#create df containing only PQ-BC totals at baseline
pps_counts_baseline = pps_counts[['subjectkey', 'interview_age', 'sex', 'visit', 'pqbc_endorsed_total_baseline',
                                  'pqbc_bother_total_baseline', 'screen_endorsed_total_baseline',
                                   'screen_bother_total_baseline']]
pps_counts_baseline = pps_counts_baseline[pps_counts_baseline['visit'] == 'baseline_year_1_arm_1']

# %%
#Count all instances of endorsed PQ-BC questions at T1

#All endorsed PQ-BC questions at T1
calc_pqbc_endorsed_total(pps_counts, '1')

#All endorsed PQ-BC questions that also bothered participants at T1
calc_pqbc_bother_total(pps_counts, '1')

#All endorsed PQ-BC questions within the 7-question sub-screener  at T1
calc_endorsed_screen_total(pps_counts, '1')

#All edorsed PQ-BC screeer questions that also bothered participants at T1
calc_screen_bother_total(pps_counts, '1')

#create df containing only PQ-BC totals at T1
pps_counts_T1 = pps_counts[['subjectkey', 'interview_age', 'sex', 'visit', 'pqbc_endorsed_total_1',
                                  'pqbc_bother_total_1', 'screen_endorsed_total_1', 'screen_bother_total_1']]
pps_counts_T1 = pps_counts_T1[pps_counts_T1['visit'] == '1_year_follow_up_y_arm_1']
# %%
#Count all instances of endorsed PQ-BC questions at T2

#All endorsed PQ-BC questions at T2
calc_pqbc_endorsed_total(pps_counts, '2')

#All endorsed PQ-BC questions that also bothered participants at T2
calc_pqbc_bother_total(pps_counts, '2')

#All endorsed PQ-BC questions within the 7-question sub-screener  at T2
calc_endorsed_screen_total(pps_counts, '2')

#All edorsed PQ-BC screeer questions that also bothered participants at T2
calc_screen_bother_total(pps_counts, '2')

#create df containing only PQ-BC totals at T2
pps_counts_T2 = pps_counts[['subjectkey', 'interview_age', 'sex', 'visit', 'pqbc_endorsed_total_2',
                                  'pqbc_bother_total_2', 'screen_endorsed_total_2', 'screen_bother_total_2']]
pps_counts_T2 = pps_counts_T2[pps_counts_T2['visit'] == '2_year_follow_up_y_arm_1']
# %%
#Count all instances of endorsed PQ-BC questions at T3

#All endorsed PQ-BC questions at T3
calc_pqbc_endorsed_total(pps_counts, '3')

#All endorsed PQ-BC questions that also bothered participants at T3
calc_pqbc_bother_total(pps_counts, '3')

#All endorsed PQ-BC questions within the 7-question sub-screener  at T3
calc_endorsed_screen_total(pps_counts, '3')

#All edorsed PQ-BC screeer questions that also bothered participants at T3
calc_screen_bother_total(pps_counts, '3')

#create df containing only PQ-BC totals at T2
pps_counts_T3 = pps_counts[['subjectkey', 'interview_age', 'sex', 'visit', 'pqbc_endorsed_total_3',
                                  'pqbc_bother_total_3', 'screen_endorsed_total_3', 'screen_bother_total_3']]
pps_counts_T3 = pps_counts_T3[pps_counts_T3['visit'] == '3_year_follow_up_y_arm_1']
# %%
#Create shortened df of pps_counts containing only totals across timepoints 
pps_counts_short = pps_counts[['subjectkey', 'interview_age', 'sex', 'visit', 'pqbc_endorsed_total_baseline',
                                  'pqbc_bother_total_baseline', 'screen_endorsed_total_baseline',
                                   'screen_bother_total_baseline', 'pqbc_endorsed_total_1',
                                  'pqbc_bother_total_1', 'screen_endorsed_total_1', 'screen_bother_total_1',
                                   'pqbc_endorsed_total_2', 'pqbc_bother_total_2', 'screen_endorsed_total_2', 
                                   'screen_bother_total_2', 'pqbc_endorsed_total_3',
                                  'pqbc_bother_total_3', 'screen_endorsed_total_3', 'screen_bother_total_3']]
# %%
#Create dfs divided by endorsement type across all four timepoints

#Df containig only PQ-BC endorsed totals across timepoints
pps_endorsed_totals = pps_counts_short[['subjectkey', 'interview_age', 'sex', 'visit', 'pqbc_endorsed_total_baseline',
                                        'pqbc_endorsed_total_1', 'pqbc_endorsed_total_2', 'pqbc_endorsed_total_3']]

#Df containing only PQ-BC edorsed totals in PQ-BC 7 question screener across timepoints
pps_screener_totals = pps_counts_short[['subjectkey', 'interview_age', 'sex', 'visit', 'screen_endorsed_total_baseline',
                                       'screen_endorsed_total_1', 'screen_endorsed_total_2', 'screen_endorsed_total_3']]

#Df containing only PQ-BC endorsed totals that also bothered participants across timepoints
pps_bother_totals = pps_counts_short[['subjectkey', 'interview_age', 'sex', 'visit', 'pqbc_bother_total_baseline', 
                                      'pqbc_bother_total_1', 'pqbc_bother_total_2', 'pqbc_bother_total_3']]

#Df containing only PC-BC endorsed screeer questions that also bothered participants across timepoints
pps_screener_bother_totals = pps_counts_short[['subjectkey', 'interview_age', 'sex', 'visit', 'screen_bother_total_baseline',
                                               'screen_bother_total_1', 'screen_bother_total_2', 'screen_bother_total_3']]
# %%
#Make dfs easier to plot

#df containing endorsed PQ-BC totals
pps_endorsed_totals_plt = pps_endorsed_totals
pps_endorsed_totals_plt = collapse_cols_plt(pps_endorsed_totals_plt, 'pqbc_endorsed_total_baseline', 'pqbc_endorsed_total_1', 'pqbc_endorsed_total_2', 'pqbc_endorsed_total_3', 'pqbc_endorsed_totals')

#df containing endorsed totals in PQ-BC seven-question screener
pps_screener_totals_plt = pps_screener_totals
pps_screener_totals_plt = collapse_cols_plt(pps_screener_totals_plt, 'screen_endorsed_total_baseline', 'screen_endorsed_total_1', 'screen_endorsed_total_2', 'screen_endorsed_total_3', 'screen_endorsed_totals')

#df containing endorsed PQ-BC totals that also bothered participants
pps_bother_totals_plt = pps_bother_totals
pps_bother_totals_plt = collapse_cols_plt(pps_bother_totals_plt, 'pqbc_bother_total_baseline', 'pqbc_bother_total_1', 'pqbc_bother_total_2', 'pqbc_bother_total_3', 'pqbc_bother_totals')

#df containing endorsed screener questions that also bothered participants
pps_screener_bother_totals_plt = pps_screener_bother_totals
pps_screener_bother_totals_plt = collapse_cols_plt(pps_screener_bother_totals_plt, 'screen_bother_total_baseline', 'screen_bother_total_1', 'screen_bother_total_2', 'screen_bother_total_3', 'screen_bother_totals')

# %%
#combine bother and endorsed totals into single dfs for PQ-BC ad 7-question screeer

#PQ-BC 
pqbc_bother = pps_bother_totals_plt[['subjectkey', 'visit', 'pqbc_bother_totals']]
pqbc_totals = pps_endorsed_totals_plt.merge(pqbc_bother, on = ['subjectkey', 'visit'])

#7 Question Screener
screener_bother = pps_screener_bother_totals_plt[['subjectkey', 'visit', 'screen_bother_totals']]
screener_totals = pps_screener_totals_plt.merge(screener_bother, on = ['subjectkey', 'visit'])
# %%
#7 Question screener df sanity check
for row in screener_totals.index:
    if screener_totals['screen_bother_totals'][row] > screener_totals['screen_endorsed_totals'][row]:
        print('error!')
        print(row)

#Data entry error for participant NDAR_INVAPLUW62L 2yr follow-up question 11--answer to initial question 
#recorded as "No", but bother indicated as "Yes" and rated a 2. Row therefore dropped from DFs
screener_totals = screener_totals.drop([0, 14571])
# %%
#PQ-BC df sanity check
for row in pqbc_totals.index:
    if pqbc_totals['pqbc_bother_totals'][row] > pqbc_totals['pqbc_endorsed_totals'][row]:
        print('error!')
        print(row)

#Data entry error for participant NDAR_INVAPLUW62L 2yr follow-up question 11--answer to initial question 
#recorded as "No", but bother indicated as "Yes" and rated a 2. Row therefore dropped from DFs
pqbc_totals = pqbc_totals.drop([0, 14571])
# %%
#Save CSVs 
screener_totals.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/screener_totals.csv')
pqbc_totals.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/pqbc_totals.csv')