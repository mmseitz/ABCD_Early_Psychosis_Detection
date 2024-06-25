
#This script investigates statistical differences in cognitive task performance and structural brain differences in LR and HR ABCD participants
#Preprocessing of csvs done in "structural_data_preprocessing.py" and "task_CSV_preprocessing"
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
#merge chosen PQ-BC df with other df
def df_merge(df1, pps_df, visit):
    pps_df = pps_df.loc[pps_df['visit'] == visit]
    pps_df = pps_df[['subjectkey', 'group']]
    new_df = df1.merge(pps_df, on = 'subjectkey')
    return new_df
# %%
####
#Count NaN values in df
def count_nans(df):
    l = list(df)
    for i in l:
        if df[i].isna().sum() >= 50:
            print(i, df[i].isna().sum())
# %%
####
#Find significant values in test dictionary outputs
def find_sigs(dic):
    sig_dic = {}
    for key in dic:
        st = str(dic[key])
        if 'e-' in st:
            sig_dic[key] = dic[key]
        else:
            s = str(dic[key]).find('e=')
            if float(str(dic[key])[s+2:s+6]) <= 0.05:
                sig_dic[key] = dic[key]
    return sig_dic

# %%
####
#Make dictionary of t-test outputs
def make_ttest_dic(df, col_list):
    ttest_dic = {}
    HR = df.loc[df['group'] == 1]
    LR = df.loc[df['group'] == 0]
    for item in col_list:
        ttest_dic[item] = stats.ttest_ind(HR[item], LR[item], equal_var = False)
    return ttest_dic
# %%
####
#Convert ttest dictionary into dataframe 
def make_ttest_df(dic):
    data_dic = {}
    for key in dic: 
        st = str(dic[key])
        l = st.split(',')
        data = []
        for s in l: 
            s = s.split('=')
            data.append(s[1])
        data[2] = data[2].replace(')', '')
        data_dic[key] = data
    ttest_df = pd.DataFrame.from_dict(data_dic, orient='index', columns=['t-statistic', 'p-value', 'DOF'])
    ttest_df['significant'] = 0
    for row in ttest_df.index:
        if float(ttest_df['p-value'][row]) <= 0.05:
            ttest_df['significant'][row] = 1
        else:
            pass
    return(ttest_df)
# %%
####
#import structural data and task csvs
#structural scans
anat_scan = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Structural_Scans/structural_data.csv')
#PQ-BC groupings
pps = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/PQ-BC/PPS_T2_endorsed_4.csv')
#Monetary Incentive Delay Task (behavioral and fMRI measures)
mid_behav = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/MID_files/mid_behav_reduced.csv')
mid_brain = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/MID_files/MID_brain/mid_brain_reduced.csv')
#Stop Signal Task (behavioral and fMRI measures)
sst_behav = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Stop_Signal_Task/sst_reduced.csv')
sst_brain = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Stop_Signal_Task/SST_brain/sst_brain_reduced.csv')
#Emotional N-Back (behavioral and fMRI measures)
enb_behav = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Emotional_N_Back/enb_behav_reduced.csv')
enb_brain = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Emotional_N_Back/N-Back_Brain/enb_brain_reduced.csv')
#Game of Dice Task (behavioral task)
gdt = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Game_of_Dice/gdt_reduced.csv')
#Social Influence Task (behavioral task)
sit = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Social_Influence/social_influence_reduced.csv')
#NIH Toolbox (behavioral tasks)
nihtb = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/NIH_Toolbox/nihtb_reduced.csv')
#Little Man Task (behavioral task)
lmt = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Little_Man_Task/lmt_reduced.csv')
# %%
####
#identify whether significant differences exist in anatomical data (structural scans; desikan parcellations)
anat_grouped = df_merge(anat_scan, pps, 2)
#make list of df columns to be tested
anat_list = []
for i in list(anat_grouped):
    if 'smri' in i: 
        anat_list.append(i)
#count NaN values across columns
count_nans(anat_grouped) #no more than 50 NaNs per column
#make dictionary containing ttest results
anat_dic = make_ttest_dic(anat_grouped, anat_list)
#convert ttest dictionary to df, including column signifying whther p-value is less than 0.05
anat_ttest_df = make_ttest_df(anat_dic)
#111 of 142 ROIs were significantly different between HR and LR groups
#save dataframe containing t-test results to folder
anat_ttest_df.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/spreadsheets/data_sig_test_results/anat_scan_ttest.csv')
# %%
####
#identify whether significant differences exist in MID data
#Behavioral data
mid_behav_grouped = df_merge(mid_behav, pps, 2)
#make list containing behavioral measures
mid_behav_list = []
for i in list(mid_behav):
    if 'tfmri' in i: 
        mid_behav_list.append(i)
#count NaN values in each column
count_nans(mid_behav_grouped) #2624 NaNs per column
#drop NaNs
mid_behav_nonan = mid_behav_grouped.dropna()
mid_behav_nonan['group'].sum() #971 of 1279 HR participants remaining
#make dictionary containing ttest results
mid_behav_dic = make_ttest_dic(mid_behav_nonan, mid_behav_list)
#convert ttest dictionary to df, including column signifying whether p-value is less than 0.05
mid_behav_ttest_df = make_ttest_df(mid_behav_dic) # 2 of 27 measures were significanlty different between HR and LR groups
#save df to folder
mid_behav_ttest_df.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/spreadsheets/data_sig_test_results/mid_behav_ttest.csv')
#Brain data
mid_brain_grouped = df_merge(mid_brain, pps, 2)
#count NaN values in each column
count_nans(mid_brain_grouped)#fewer than 50 NaN values per column
#drop NaNs
mid_brain_nonan = mid_brain_grouped.dropna()
#make list containing fMRI measures
mid_brain_list = []
for i in list(mid_brain_nonan):
    if 'tfmri' in i: 
        mid_brain_list.append(i)
#make dictionary containing ttest results
mid_brain_dic = make_ttest_dic(mid_brain_nonan, mid_brain_list)
#convert ttest dictionary to df, including column signifying whether p-value is less than 0.05
mid_brain_ttest_df = make_ttest_df(mid_brain_dic)
#save df to folder
mid_brain_ttest_df.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/spreadsheets/data_sig_test_results/mid_brain_ttest.csv')
# %%
####
#Identify whether significant differences exist in SST data
#Behavioral data
sst_behav_grouped = df_merge(sst_behav, pps, 2)
#count NaN values in each column
count_nans(sst_behav_grouped) #2692 NaNs per column
#drop NaNs
sst_behav_nonan = sst_behav_grouped.dropna() #947 of 1279 HR participants remaining
#make list containing behav measures
sst_behav_list = []
for i in list(sst_behav_nonan):
    if 'tfmri' in i: 
        sst_behav_list.append(i)
#make dictionary containing ttest results
sst_behav_dic = make_ttest_dic(sst_behav_nonan, sst_behav_list)
#convert dictionary to df, including column signifying whether p-value is less than 0.05
sst_behav_ttest_df = make_ttest_df(sst_behav_dic)
#count how many significant valeus
sst_behav_ttest_df['significant'].sum() #21 of 29 measures significant
#save df to folder
sst_behav_ttest_df.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/spreadsheets/data_sig_test_results/sst_behav_ttest.csv')
#Brain data
sst_brain_grouped = df_merge(sst_brain, pps, 2)
#count NaN values in each column
count_nans(sst_brain_grouped)#fewer than 50 per column
sst_brain_nonan = sst_brain_grouped.dropna()
#make list containing behav measures
sst_brain_list = []
for i in list(sst_brain_nonan):
    if 'tfmri' in i: 
        sst_brain_list.append(i)
#make dictionary containing ttest results
sst_brain_dic = make_ttest_dic(sst_brain_nonan, sst_brain_list)
#convert dictionary to df, including column signifying whether p-value is less than 0.05
sst_brain_ttest_df = make_ttest_df(sst_brain_dic)
sst_brain_ttest_df['significant'].sum() #3 significant measures
#save df to folder
sst_brain_ttest_df.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/spreadsheets/data_sig_test_results/sst_brain_ttest.csv')
# %%
####
#Identify whether significant differences exist in Emotional N-Back data
#Behavioral data
enb_behav_grouped = df_merge(enb_behav, pps, 2)
#count NaN values in each column
count_nans(enb_behav_grouped)#3009 NaNs per column
enb_behav_nonan = enb_behav_grouped.dropna() #914 HR of 1279 remaining
#make list containing behav measures
enb_behav_list = []
for i in list(enb_behav_nonan):
    if 'tfmri' in i: 
        enb_behav_list.append(i)
#make dictionary containing ttest results
enb_behav_dic = make_ttest_dic(enb_behav_nonan, enb_behav_list)
#convert dictionary to df, including column signifying whether p-value is less than 0.05
enb_behav_ttest_df = make_ttest_df(enb_behav_dic)
enb_behav_ttest_df['significant'].sum() #20 0f 22 measures significant
#save df to folder
enb_behav_ttest_df.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/spreadsheets/data_sig_test_results/enb_behav_ttest.csv')
#Brain data
enb_brain_grouped = df_merge(enb_brain, pps, 2)
#count NaNs in each column
count_nans(enb_brain_grouped)#less than 50 NaNs per column
enb_brain_nonan = enb_brain_grouped.dropna()
#make list containing behav measures
enb_brain_list = []
for i in list(enb_brain_nonan):
    if 'tfmri' in i: 
        enb_brain_list.append(i)
#make dictionary containing ttest results
enb_brain_dic = make_ttest_dic(enb_brain_nonan, enb_brain_list)
#convert dictionary to df
enb_brain_ttest_df = make_ttest_df(enb_brain_dic)
enb_brain_ttest_df['significant'].sum() #11 of 90 measures significant
#save df to folder
enb_brain_ttest_df.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/spreadsheets/data_sig_test_results/enb_brain_ttest.csv')
# %%
####
#Identify whether significant differences exist between groups in the Game of Dice Task
gdt_grouped = df_merge(gdt, pps, 2)
#count NaNs in each column
count_nans(gdt_grouped)#566 NaNs per column; 
gdt_nonan = gdt_grouped.dropna() #1191 HR remaining
#make list containing measures
gdt_list = []
for i in list(gdt_nonan):
    if 'gdt' in i: 
        gdt_list.append(i)
#make dictionary containing ttest results
gdt_dic = make_ttest_dic(gdt_nonan, gdt_list)
#convert dictionary to df
gdt_ttest_df = make_ttest_df(gdt_dic) #9 of 10 measures significant
#save df to folder
gdt_ttest_df.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/spreadsheets/data_sig_test_results/gdt_ttest.csv')
# %%
####
#Identify whether significant differences exist between groups in the Social Influence Task
sit_grouped = df_merge(sit, pps, 2)
#count NaNs in each column
count_nans(sit_grouped) #~700 NaNs per column
#drop NaNs
sit_nonan = sit_grouped.dropna() #1164 HR remaining
#make list containing measures
sit_list = []
for i in list(sit_nonan):
    if 'sit' in i: 
        sit_list.append(i)
#make dictionary containing ttest results
sit_dic = make_ttest_dic(sit_nonan, sit_list)
#convert dictionary to df
sit_ttest_df = make_ttest_df(sit_dic) #7 of 14 measures significant
#save df to folder
sit_ttest_df.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/spreadsheets/data_sig_test_results/sit_ttest.csv')
# %%
####
#Identify whether significant differences exist between groups in the NIH Toolbox tasks
nih_grouped = df_merge(nihtb, pps, 2)
#count NaNs in each column
count_nans(nih_grouped)#500-2000 NaNs per column
#drop NaN values
nih_nonan = nih_grouped.dropna() #893 HR participants remaining
#make list containing measures
nih_list = []
for i in list(nih_nonan):
    if 'nih' in i: 
        nih_list.append(i)
#make dictionary containing ttest results
nih_dic = make_ttest_dic(nih_nonan, nih_list)
#convert dictionary to df
nih_ttest_df = make_ttest_df(nih_dic) #11 of 12 measures significant (all but Flanker)
#save df to folder
nih_ttest_df.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/spreadsheets/data_sig_test_results/nihtb_ttest.csv')
# %%
####
#Identify whether significant differences exist between groups in the Little Man Task
lmt_grouped = df_merge(lmt, pps, 2)
#count NaNs in each column
count_nans(lmt_grouped)#485 NaNs per column
#drop NaNs
lmt_nonan = lmt_grouped.dropna() #1210 HR participants remaining
#make list containing measures
lmt_list = []
for i in list(lmt_nonan):
    if 'lmt' in i: 
        lmt_list.append(i)
#make dictionary containing ttest results
lmt_dic = make_ttest_dic(lmt_nonan, lmt_list)
#convert dictionary to df
lmt_ttest_df = make_ttest_df(lmt_dic) #0 of 4 measures significant
#save df to folder
lmt_ttest_df.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/spreadsheets/data_sig_test_results/lmt_ttest.csv')
# %%
