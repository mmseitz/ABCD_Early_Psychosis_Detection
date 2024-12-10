#This script performs minimal preprocessing on individual task CSVs and creates a master file of chosen 
#cognitive task brain and behavior parameters
# %%
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats as stats
import numpy as np

# %% 

#import csvs containing behavioral and brain data corresponding to ABCD cognitive battery tasks
#Note: Stop Signal Task, Monetary Incentive Delay Task, and Emotional N-Back Task have both brain and 
#behavioral data CSVs

#Little Man Task--behavioral data (data collected at follow-up 2)
lmt = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/cognitive_tasks/nc_y_lmt.csv')

#Game of Dice Task--behavioral data (data collected at follow-up 2)
gdt = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/cognitive_tasks/nc_y_gdt.csv')

#Monetary Incentive Delay Task--behavioral data (data collected at follow-up 2)
mid_behav = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/cognitive_tasks/mri_y_tfmr_mid_beh.csv')
#MID--brain data (collected at follow-up 2) desikan
mid_brain_1 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/mri_y_tfmr_mid_allvn_dsk.csv')
mid_brain_2 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/mri_y_tfmr_mid_alrvn_dsk.csv')
mid_brain_3 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/mri_y_tfmr_mid_alvn_dsk.csv')
mid_brain_4 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/mri_y_tfmr_mid_alvsl_dsk.csv')
mid_brain_5 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/mri_y_tfmr_mid_alvsr_dsk.csv')
mid_brain_6 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/mri_y_tfmr_mid_arvn_dsk.csv')
mid_brain_7 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/mri_y_tfmr_mid_aslvn_dsk.csv')
mid_brain_8 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/mri_y_tfmr_mid_asrvn_dsk.csv')
mid_brain_9 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/mri_y_tfmr_mid_lpvnf_dsk.csv')
mid_brain_10 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/mri_y_tfmr_mid_rpvnf_dsk.csv')
#MID--brain data (subcortical)
mid_sub_1 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/MID_aseg/mri_y_tfmr_mid_allvn_aseg.csv')
mid_sub_2 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/MID_aseg/mri_y_tfmr_mid_alrvn_aseg.csv')
mid_sub_3 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/MID_aseg/mri_y_tfmr_mid_alvn_aseg.csv')
mid_sub_4 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/MID_aseg/mri_y_tfmr_mid_alvsl_aseg.csv')
mid_sub_5 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/MID_aseg/mri_y_tfmr_mid_alvsr_aseg.csv')
mid_sub_6 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/MID_aseg/mri_y_tfmr_mid_arvn_aseg.csv')
mid_sub_7 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/MID_aseg/mri_y_tfmr_mid_aslvn_aseg.csv')
mid_sub_8 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/MID_aseg/mri_y_tfmr_mid_asrvn_aseg.csv')
mid_sub_9 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/MID_aseg/mri_y_tfmr_mid_lpvnf_aseg.csv')
mid_sub_10 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/MID/MID_aseg/mri_y_tfmr_mid_rpvnf_aseg.csv')

#Social Influence Task--behavioral data (collected at follow-up 2)
sit = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/cognitive_tasks/nc_y_sit.csv')

#Stop Signal Task--behavioral data (collected at follow-up 2)
sst_behav = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/cognitive_tasks/mri_y_tfmr_sst_beh.csv')
#SST--brain data (collected at follow-up 2)
sst_brain_1 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/SST/mri_y_tfmr_sst_igvis_dsk.csv')
sst_brain_2 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/SST/mri_y_tfmr_sst_isvcg_dsk.csv')
sst_brain_3 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/SST/mri_y_tfmr_sst_csvis_dsk.csv')
sst_brain_4 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/SST/mri_y_tfmr_sst_cgvfx_dsk.csv')
sst_brain_5 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/SST/mri_y_tfmr_sst_csvcg_dsk.csv')
sst_brain_6 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/SST/mri_y_tfmr_sst_asvcg_dsk.csv')
sst_brain_7 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/SST/mri_y_tfmr_sst_igvcg_dsk.csv')
#SST-aseg data
sst_sub_1 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/SST/sst_aseg/mri_y_tfmr_sst_asvcg_aseg.csv')
sst_sub_2 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/SST/sst_aseg/mri_y_tfmr_sst_cgvfx_aseg.csv')
sst_sub_3 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/SST/sst_aseg/mri_y_tfmr_sst_csvcg_aseg.csv')
sst_sub_4 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/SST/sst_aseg/mri_y_tfmr_sst_csvis_aseg.csv')
sst_sub_5 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/SST/sst_aseg/mri_y_tfmr_sst_igvcg_aseg.csv')
sst_sub_6 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/SST/sst_aseg/mri_y_tfmr_sst_igvis_aseg.csv')
sst_sub_7 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/SST/sst_aseg/mri_y_tfmr_sst_isvcg_aseg.csv')
#NIH Toolbox--behavioral data (collected at follow-up 2)
nihtb = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/cognitive_tasks/nc_y_nihtb.csv')

#Emotional N-back--behavioral data (collected at follow-up 2)
enb_behav = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/mri_y_tfmr_nback_rec_beh.csv')
#ENB--brain data (collected at follow-up 2)
enb_brain_1 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/mri_y_tfmr_nback_0b_dsk.csv')
enb_brain_2 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/mri_y_tfmr_nback_2b_dsk.csv')
enb_brain_3 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/mri_y_tfmr_nback_2bv0b_dsk.csv')
enb_brain_4 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/mri_y_tfmr_nback_emo_dsk.csv')
enb_brain_5 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/mri_y_tfmr_nback_emovntf_dsk.csv')
enb_brain_6 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/mri_y_tfmr_nback_fvplc_dsk.csv')
enb_brain_7 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/mri_y_tfmr_nback_ngfvntf_dsk.csv')
enb_brain_8 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/mri_y_tfmr_nback_plc_dsk.csv')
enb_brain_9 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/mri_y_tfmr_nback_psfvntf_dsk.csv')
#ENB-aseg
enb_sub_1 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/nback_aseg/mri_y_tfmr_nback_0b_aseg.csv')
enb_sub_2 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/nback_aseg/mri_y_tfmr_nback_2b_aseg.csv')
enb_sub_3 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/nback_aseg/mri_y_tfmr_nback_2bv0b_aseg.csv')
enb_sub_4 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/nback_aseg/mri_y_tfmr_nback_emo_aseg.csv')
enb_sub_5 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/nback_aseg/mri_y_tfmr_nback_emovntf_aseg.csv')
enb_sub_6 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/nback_aseg/mri_y_tfmr_nback_fvplc_aseg.csv')
enb_sub_7 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/nback_aseg/mri_y_tfmr_nback_ngfvntf_aseg.csv')
enb_sub_8 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/nback_aseg/mri_y_tfmr_nback_plc_aseg.csv')
enb_sub_9 = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/MRI_tasks/Emotional_N_Back/nback_aseg/mri_y_tfmr_nback_psfvntf_aseg.csv')
# %%
#Import fMRI parameters for SST, MID, and ENB brain data 
parameters = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/data/parameters/fMRI_paramters_csv.csv')
# %%
#function to check how many NaNs exist in each selected df column
def count_nans(df):
    l = list(df)
    for i in l:
        if df[i].isna().sum() >= 50:
            print(i, df[i].isna().sum())

# %%
#Preprocessing of Little Man Task CSV
#select salient columns (lmt_scr_perc_wrong, lmt_scr_rt_wrong, and lmt_scr_efficiency ommitted due to high # of NaN values)
lmt = lmt[['src_subject_id', 'eventname', 'lmt_scr_perc_correct', 'lmt_scr_num_timed_out', 'lmt_scr_avg_rt', 'lmt_scr_rt_correct']]
#isolate T2 and T4 data
lmt_2 = lmt.loc[lmt['eventname'] == '2_year_follow_up_y_arm_1'] 
lmt_4 = lmt.loc[lmt['eventname'] == '4_year_follow_up_y_arm_1']
#check NaNs
count_nans(lmt_2) #max 86 NaNs per column (shape: (10527, 6))
count_nans(lmt_4) #max 88 NaNs per column (shape: (4557, 6))
#save reduced csvs 
lmt_2.to_csv('/Users/madeleineseitz/Desktop/abcd5_T2_lmt_reduced.csv')
lmt_4.to_csv('/Users/madeleineseitz/Desktop/abcd5_T4_lmt_reduced.csv')

# %%
#preprocessing of Game of Dice Task CSV
#select salient columns
gdt = gdt[['src_subject_id', 'eventname', 'gdt_scr_values_account_balance', 'gdt_scr_values_single', 
           'gdt_scr_values_double', 'gdt_scr_values_triple', 'gdt_scr_values_quadruple', 'gdt_scr_values_safe', 'gdt_scr_values_risky', 'gdt_scr_expressions_net_score', 
           'gdt_scr_values_wins', 'gdt_scr_values_losses']]
#idolate T2 and T4 data
gdt_2 = gdt.loc[gdt['eventname'] == '2_year_follow_up_y_arm_1'] 
gdt_4 = gdt.loc[gdt['eventname'] == '4_year_follow_up_y_arm_1']
#check NaNs
count_nans(gdt_2) #126 NaNs per column (10473, 12)
# %%
count_nans(gdt_4) #135 NaNs per column (4515, 12)
#save reduced csvs
gdt_2.to_csv('/Users/madeleineseitz/Desktop/abcd5_T2_gdt_reduced.csv')
gdt_4.to_csv('/Users/madeleineseitz/Desktop/abcd5_T4_gdt_reduced.csv')

# %%
#Preprocessing MID behavioral data
#select salient columns
mid_behav = mid_behav[['src_subject_id', 'eventname', 'tfmri_mid_all_beh_srwpfb_nt', 'tfmri_mid_all_beh_srwpfb_rate', 'tfmri_mid_all_beh_srwnfb_nt', 
                       'tfmri_mid_all_beh_srwnfb_rate', 'tfmri_mid_all_beh_lrwpfb_nt', 'tfmri_mid_all_beh_lrwpfb_rate', 'tfmri_mid_all_beh_lrwnfb_nt', 'tfmri_mid_all_beh_lrwnfb_rate',
                       'tfmri_mid_all_beh_slpfb_nt', 'tfmri_mid_all_beh_slpfb_rate', 'tfmri_mid_all_beh_slnfb_nt', 'tfmri_mid_all_beh_slnfb_rate', 'tfmri_mid_all_beh_llpfb_nt', 
                       'tfmri_mid_all_beh_llpfb_rate', 'tfmri_mid_all_beh_llnfb_nt', 'tfmri_mid_all_beh_llnfb_rate', 'tfmri_mid_all_beh_ntpfb_nt', 
                       'tfmri_mid_all_beh_ntpfb_rate', 'tfmri_mid_all_beh_ntnfb_nt', 'tfmri_mid_all_beh_ntnfb_rate', 'tfmri_mid_all_beh_hrwpfb_nt', 'tfmri_mid_all_beh_hrwpfb_rate', 
                       'tfmri_mid_all_beh_hrwnfb_nt', 'tfmri_mid_all_beh_hrwnfb_rate', 'tfmri_mid_all_beh_hlpfb_nt', 'tfmri_mid_all_beh_hlpfb_rate', 'tfmri_mid_all_beh_hlnfb_nt', 
                       'tfmri_mid_all_beh_hlnfb_rate']]
#isolate follow-up 2 data
mid_behav_2 = mid_behav.loc[mid_behav['eventname'] == '2_year_follow_up_y_arm_1']
mid_behav_4 = mid_behav.loc[mid_behav['eventname'] == '4_year_follow_up_y_arm_1']
#check NaNs
count_nans(mid_behav_2) # < 50 NaNs per column (8057, 30)
count_nans(mid_behav_4)# < 50 NaNs per column (3049, 30)
#save reduced csvs
mid_behav_2.to_csv('/Users/madeleineseitz/Desktop/abcd5_T2_mid_behav_reduced.csv')
mid_behav_4.to_csv('/Users/madeleineseitz/Desktop/abcd5_T4_mid_behav_reduced.csv')

# %%
#merge MID brain csvs
mid_brain = mid_brain_1.merge(mid_brain_2, on = ['src_subject_id', 'eventname'])
mid_brain = mid_brain.merge(mid_brain_3, on = ['src_subject_id', 'eventname'])
mid_brain = mid_brain.merge(mid_brain_4, on = ['src_subject_id', 'eventname'])
mid_brain = mid_brain.merge(mid_brain_5, on = ['src_subject_id', 'eventname'])
mid_brain = mid_brain.merge(mid_brain_6, on = ['src_subject_id', 'eventname'])
mid_brain = mid_brain.merge(mid_brain_7, on = ['src_subject_id', 'eventname'])
mid_brain = mid_brain.merge(mid_brain_8, on = ['src_subject_id', 'eventname'])
mid_brain = mid_brain.merge(mid_brain_9, on = ['src_subject_id', 'eventname'])
mid_brain = mid_brain.merge(mid_brain_10, on = ['src_subject_id', 'eventname'])
# %%
#merge MID subcortical csvs
mid_sub = mid_sub_1.merge(mid_sub_2, on = ['src_subject_id', 'eventname'])
mid_sub = mid_sub.merge(mid_sub_3, on = ['src_subject_id', 'eventname'])
mid_sub = mid_sub.merge(mid_sub_4, on = ['src_subject_id', 'eventname'])
mid_sub = mid_sub.merge(mid_sub_5, on = ['src_subject_id', 'eventname'])
mid_sub = mid_sub.merge(mid_sub_6, on = ['src_subject_id', 'eventname'])
mid_sub = mid_sub.merge(mid_sub_7, on = ['src_subject_id', 'eventname'])
mid_sub = mid_sub.merge(mid_sub_8, on = ['src_subject_id', 'eventname'])
mid_sub = mid_sub.merge(mid_sub_9, on = ['src_subject_id', 'eventname'])
mid_sub = mid_sub.merge(mid_sub_10, on = ['src_subject_id', 'eventname'])
# %%
#Preprocessing MID brain data
#identify MID parameters most closely associated with psychosis risk (by keyword)
mid_2_list = []
mid_2_list.append('src_subject_id')
mid_2_list.append('eventname')
for column in parameters:
    if 'feedback' in str(parameters[column][3]):
        if 'loss' in str(parameters[column][3]):
            pass
        if 'anteriorcingulate' in str(parameters[column][3]):
            mid_2_list.append(parameters[column][2])
        if 'middlefrontal' in str(parameters[column][3]):
             mid_2_list.append(parameters[column][2])

#identify midaparcp203 parameters most closely associated with psychosis risk (by keyword)
mid_1_list = []
mid_1_list.append('src_subject_id')
mid_1_list.append('eventname')
for column in parameters:
    if 'anticipation' in str(parameters[column][5]):
        if 'amygdala' in str(parameters[column][5]):
            mid_1_list.append(parameters[column][4])
        if 'thalamus' in str(parameters[column][5]):
             mid_1_list.append(parameters[column][4])
# %%
#midaparc03
#select salient columns in MID df (using list generated above)
mid_brain_short = mid_brain[mid_2_list]
mid_sub_short = mid_sub[mid_1_list]
#merge MID fMRI data together
mid_fmri = mid_brain_short.merge(mid_sub_short, on = ['src_subject_id', 'eventname'])
#isolate follow-up-2 and follow-up-4 data
mid_mri_T2 = mid_fmri.loc[mid_fmri['eventname'] == '2_year_follow_up_y_arm_1'] # (7340, 50)
mid_mri_T4 = mid_fmri.loc[mid_fmri['eventname'] == '4_year_follow_up_y_arm_1'] # (2795, 50)
#save csvs
mid_mri_T2.to_csv('/Users/madeleineseitz/Desktop/abcd5_mid_mri_T2.csv')
mid_mri_T4.to_csv('/Users/madeleineseitz/Desktop/abcd5_mid_mri_T4.csv')
# %%
#Preprocessing the Social Influence Task 
#Select salient columns in abcd_siss01
sit = sit[['src_subject_id', 'eventname', 'sit_scr_expr_minitialrat', 'sit_scr_expr_mrt_initialrat', 'sit_scr_expr_mfinalrat',
           'sit_scr_expr_mrt_finalrat', 'sit_scr_expr_mratdiff1', 'sit_scr_expr_mratdiff2', 'sit_scr_expr_mratdiff3',
         'sit_scr_values_count1', 'sit_scr_values_count2', 'sit_scr_values_count3', 'sit_scr_values_count4', 
           'sit_scr_values_countflips', 'sit_scr_values_countnr_initial', 'sit_scr_values_countnr_final']]
#['sit_scr_expr_mratdiff4'] removed due to high volume of NaNs
sit_T2 = sit.loc[sit['eventname'] == '2_year_follow_up_y_arm_1']
sit_T4 = sit.loc[sit['eventname'] == '4_year_follow_up_y_arm_1']
#Check NaNs in df
count_nans(sit_T2) #highest count 159 per column (10394, 16)
count_nans(sit_T4) #highest count 159 per column (3334, 16)
#save csv 
sit_T2.to_csv('/Users/madeleineseitz/Desktop/abcd5_T2_social_influence_reduced.csv')
sit_T4.to_csv('/Users/madeleineseitz/Desktop/abcd5_T4_social_influence_reduced.csv')
# %%

#Preprocessing stop signal task behavioral data
#select salient columns in abcd_sst02
sst_behav = sst_behav[['src_subject_id', 'eventname', 'tfmri_sst_beh_performflag', 'tfmri_sst_nbeh_nruns', 
                       'tfmri_sst_all_beh_go_nt', 'tfmri_sst_all_beh_s_nt', 'tfmri_sst_all_beh_total_nt', 'tfmri_sst_all_beh_crgo_nt', 
                       'tfmri_sst_all_beh_crgo_rt', 'tfmri_sst_all_beh_crgo_mrt', 'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_sst_all_beh_crlg_nt',
                       'tfmri_sst_all_beh_crlg_rt', 'tfmri_sst_all_beh_crlg_mrt', 'tfmri_sst_all_beh_crlg_stdrt', 'tfmri_sst_all_beh_incrgo_nt', 
                       'tfmri_sst_all_beh_incrgo_rt', 'tfmri_sst_all_beh_incrgo_mrt', 'tfmri_sst_all_beh_incrgo_stdrt', 
                       'tfmri_sst_all_beh_incrlg_nt', 'tfmri_sst_all_beh_incrlg_rt', 'tfmri_sst_all_beh_incrlg_mrt', 'tfmri_sst_all_beh_incrlg_stdrt',
                       'tfmri_sst_all_beh_nrgo_nt', 'tfmri_sst_all_beh_nrgo_rt', 'tfmri_sst_all_beh_crs_nt', 'tfmri_sst_all_beh_crs_rt', 
                       'tfmri_sst_all_beh_incrs_nt', 'tfmri_sst_all_beh_incrs_rt', 'tfmri_sst_all_beh_incrs_mrt', 'tfmri_sst_all_beh_incrs_stdrt', 
                       'tfmri_sst_all_beh_ssds_nt', 'tfmri_sst_all_beh_ssds_rt', 'tfmri_sst_all_beh_tot_mssd', 
                       'tfmri_sst_all_beh_total_mssrt', 'tfmri_sst_all_beh_total_issrt']]
# ['tfmri_sst_all_beh_crlg_mrt', 'tfmri_sst_all_beh_crlg_stdrt', 'tfmri_sst_all_beh_incrlg_mrt', 'tfmri_sst_all_beh_incrlg_stdrt',
#   'tfmri_sst_all_beh_total_mssrt', 'tfmri_sst_all_beh_total_issrt'] removed due to high NaN volume
sst_behav = sst_behav.drop(columns= ['tfmri_sst_all_beh_crlg_mrt', 'tfmri_sst_all_beh_crlg_stdrt', 'tfmri_sst_all_beh_incrlg_mrt', 'tfmri_sst_all_beh_incrlg_stdrt', 
                                     'tfmri_sst_all_beh_total_mssrt', 'tfmri_sst_all_beh_total_issrt'])
#isolate follow-up-2 data
sst_behav_T2 = sst_behav.loc[sst_behav['eventname'] == '2_year_follow_up_y_arm_1']
sst_behav_T4 = sst_behav.loc[sst_behav['eventname'] == '4_year_follow_up_y_arm_1']
#check NaNs in df
count_nans(sst_behav_T2) #highest count 117  NaNs
count_nans(sst_behav_T4) #highest count 86 NaNs
#save csv
sst_behav_T2.to_csv('/Users/madeleineseitz/Desktop/abcd5_T2_sst_beh_reduced.csv') #(7964, 30)
sst_behav_T4.to_csv('/Users/madeleineseitz/Desktop/abcd5_T4_sst_beh_reduced.csv') #(2999, 30)
# %%
#Preprocessing SST brain data
#identify SST parameters most pertaining to psychosis risk (thalamus, cerebellum, putamen, pallidum, NAcc ROIs)
sst_list = []
sst_list.append('src_subject_id')
sst_list.append('eventname')
for column in parameters:
    if 'thalamus' in str(parameters[column][7]):
        sst_list.append(parameters[column][6])
    if 'cerebellum' in str(parameters[column][7]):
        sst_list.append(parameters[column][6])
    if 'caudate' in str(parameters[column][7]):
        sst_list.append(parameters[column][6])
    if 'putamen' in str(parameters[column][7]):
        sst_list.append(parameters[column][6])
    if 'pallidum' in str(parameters[column][7]):
        sst_list.append(parameters[column][6])
    if 'accumbens' in str(parameters[column][7]):
        sst_list.append(parameters[column][6])
print(sst_list)
# %%
sst_brain = sst_brain_1.merge(sst_brain_2, on = ['src_subject_id', 'eventname'])
sst_brain = sst_brain.merge(sst_brain_3, on = ['src_subject_id', 'eventname'])
sst_brain = sst_brain.merge(sst_brain_4, on = ['src_subject_id', 'eventname'])
sst_brain = sst_brain.merge(sst_brain_5, on = ['src_subject_id', 'eventname'])
sst_brain = sst_brain.merge(sst_brain_6, on = ['src_subject_id', 'eventname'])
sst_brain = sst_brain.merge(sst_brain_7, on = ['src_subject_id', 'eventname'])
sst_brain = sst_brain.merge(sst_sub_1, on = ['src_subject_id', 'eventname'])
sst_brain = sst_brain.merge(sst_sub_2, on = ['src_subject_id', 'eventname'])
sst_brain = sst_brain.merge(sst_sub_3, on = ['src_subject_id', 'eventname'])
sst_brain = sst_brain.merge(sst_sub_4, on = ['src_subject_id', 'eventname'])
sst_brain = sst_brain.merge(sst_sub_5, on = ['src_subject_id', 'eventname'])
sst_brain = sst_brain.merge(sst_sub_6, on = ['src_subject_id', 'eventname'])
sst_brain = sst_brain.merge(sst_sub_7, on = ['src_subject_id', 'eventname'])

#select salient columns in df
sst_brain = sst_brain[sst_list]
#isolate follow-up 2 data
sst_brain_T2 = sst_brain.loc[sst_brain['eventname'] == '2_year_follow_up_y_arm_1']
sst_brain_T4 = sst_brain.loc[sst_brain['eventname'] == '4_year_follow_up_y_arm_1']
#check NaNs
count_nans(sst_brain_2) #NaNs < 50 per column, shape (7222, 100)
count_nans(sst_brain_4) #NaNs < 50 per column, shape (2744, 100)
#save csv
sst_brain_T2.to_csv('/Users/madeleineseitz/Desktop/abcd5_T2_sst_brain_reduced.csv')
sst_brain_T4.to_csv('/Users/madeleineseitz/Desktop/abcd5_T4_sst_brain_reduced.csv')
# %%

#Preprocessing NIH Toolbox Data 
#Select salient columns in abcd_tbss01
nihtb = nihtb[['src_subject_id', 'eventname', 'nihtbx_picvocab_agecorrected',  
               'nihtbx_flanker_agecorrected', 'nihtbx_list_agecorrected', 'nihtbx_cardsort_agecorrected', 'nihtbx_pattern_agecorrected', 
               'nihtbx_picture_agecorrected', 'nihtbx_reading_agecorrected', 'nihtbx_fluidcomp_agecorrected', 
               'nihtbx_totalcomp_agecorrected', 'nihtbx_picvocab_fc', 'nihtbx_flanker_fc', 'nihtbx_list_fc', 'nihtbx_cardsort_fc', 
               'nihtbx_pattern_fc', 'nihtbx_picture_fc', 'nihtbx_reading_fc', 'nihtbx_fluidcomp_fc', 
               'nihtbx_totalcomp_fc']]
#Columns ['nihtbx_list_agecorrected', 'nihtbx_cardsort_agecorrected', 'nihtbx_fluidcomp_agecorrected', 'nihtbx_totalcomp_agecorrected',
#           'nihtbx_list_fc', 'nihtbx_cardsort_fc', 'nihtbx_fluidcomp_fc', 'nihtbx_totalcomp_fc'] dropped due to high NaN frequency
nihtb = nihtb.drop(columns= ['nihtbx_list_agecorrected', 'nihtbx_cardsort_agecorrected', 'nihtbx_fluidcomp_agecorrected', 'nihtbx_totalcomp_agecorrected', 
                             'nihtbx_list_fc', 'nihtbx_cardsort_fc', 'nihtbx_fluidcomp_fc', 'nihtbx_totalcomp_fc'])
#isolate follow-up 2 data
nihtb_T2 = nihtb.loc[nihtb['eventname'] == '2_year_follow_up_y_arm_1']
nihtb_T4 = nihtb.loc[nihtb['eventname'] == '4_year_follow_up_y_arm_1']
#check NaNs
count_nans(nihtb_T2) #highest NaN 1434 (NaN range 452-1434) (10649, 12)
count_nans(nihtb_T4) #highest NaN 3984 (range 163-3984) (4652, 12)
#save csvs
nihtb_T2.to_csv('/Users/madeleineseitz/Desktop/abcd5_T2_nihtb_reduced.csv')
nihtb_T4.to_csv('/Users/madeleineseitz/Desktop/abcd5_T4_nihtb_reduced.csv')
# %%

#Preprocessing Emotional N-Back Data (behav)
#Select salient columns
enb_behav_short = enb_behav[['src_subject_id', 'eventname', 'tfmri_rec_all_beh_newnf_hr', 'tfmri_rec_all_beh_newnf_fa', 'tfmri_rec_all_beh_newpl_hr', 'tfmri_rec_all_beh_newpl_fa', 
                       'tfmri_rec_all_beh_newposf_hr', 'tfmri_rec_all_beh_newposf_fa', 'tfmri_rec_all_beh_newnegf_hr', 'tfmri_rec_all_beh_newnegf_fa', 
                       'tfmri_rec_all_beh_oldposf_hr', 'tfmri_rec_all_beh_oldposf_fa', 'tfmri_rec_all_beh_posface_pr', 'tfmri_rec_all_beh_posface_br', 
                       'tfmri_rec_all_beh_posf_dpr', 'tfmri_rec_all_beh_neutface_pr', 'tfmri_rec_all_beh_neutface_br', 'tfmri_rec_all_beh_neutf_dp', 
                       'tfmri_rec_all_beh_negface_pr', 'tfmri_rec_all_beh_negface_br', 'tfmri_rec_all_beh_negf_dp', 'tfmri_rec_all_beh_place_pr', 
                       'tfmri_rec_all_beh_place_br', 'tfmri_rec_all_beh_place_dp']]
#Isolate follow-up 2 data
enb_behav_T2 = enb_behav_short.loc[enb_behav['eventname'] == '2_year_follow_up_y_arm_1']
enb_behav_T4 = enb_behav_short.loc[enb_behav['eventname'] == '4_year_follow_up_y_arm_1']
#Check NaNs
count_nans(enb_behav_T2) # <50 NaNs per col (7004, 24)
count_nans(enb_behav_T4) # <50 NaNs per col (2670, 24)
#save csv
enb_behav_T2.to_csv('/Users/madeleineseitz/Desktop/abcd5_T2_enb_behav_reduced.csv')
enb_behav_T4.to_csv('/Users/madeleineseitz/Desktop/abcd5_T4_enb_behav_reduced.csv')
# %%
#Preprocessing Emotion N-Back Brain Data
#Identify ENB parameters pertaining to psychosis risk (cerebellum, thalamus, hippocampus, amygdala ROIs)
nb_list = []
nb_list.append('src_subject_id')
nb_list.append('eventname')
for column in parameters:
    if 'cerebellum' in str(parameters[column][1]):
        nb_list.append(parameters[column][0])
    if 'thalamus' in str(parameters[column][1]):
        nb_list.append(parameters[column][0])
    if 'hippocampus' in str(parameters[column][1]):
        nb_list.append(parameters[column][0])
    if 'amygdala' in str(parameters[column][1]):
        nb_list.append(parameters[column][0])
print(nb_list)
# %%
enb_brain = enb_brain_1.merge(enb_brain_2, on = ['src_subject_id', 'eventname'])
enb_brain = enb_brain.merge(enb_brain_3, on = ['src_subject_id', 'eventname'])
enb_brain = enb_brain.merge(enb_brain_4, on = ['src_subject_id', 'eventname'])
enb_brain = enb_brain.merge(enb_brain_5, on = ['src_subject_id', 'eventname'])
enb_brain = enb_brain.merge(enb_brain_6, on = ['src_subject_id', 'eventname'])
enb_brain = enb_brain.merge(enb_brain_7, on = ['src_subject_id', 'eventname'])
enb_brain = enb_brain.merge(enb_brain_8, on = ['src_subject_id', 'eventname'])
enb_brain = enb_brain.merge(enb_brain_9, on = ['src_subject_id', 'eventname'])
enb_brain = enb_brain.merge(enb_sub_1, on = ['src_subject_id', 'eventname'])
enb_brain = enb_brain.merge(enb_sub_2, on = ['src_subject_id', 'eventname'])
enb_brain = enb_brain.merge(enb_sub_3, on = ['src_subject_id', 'eventname'])
enb_brain = enb_brain.merge(enb_sub_4, on = ['src_subject_id', 'eventname'])
enb_brain = enb_brain.merge(enb_sub_5, on = ['src_subject_id', 'eventname'])
enb_brain = enb_brain.merge(enb_sub_6, on = ['src_subject_id', 'eventname'])
enb_brain = enb_brain.merge(enb_sub_7, on = ['src_subject_id', 'eventname'])
enb_brain = enb_brain.merge(enb_sub_8, on = ['src_subject_id', 'eventname'])
enb_brain = enb_brain.merge(enb_sub_9, on = ['src_subject_id', 'eventname'])
# %%
#select salient columns in enb_brain df
enb_brain_short = enb_brain[nb_list]
#isolate follow-up 2 and 4 data
enb_brain_T2 = enb_brain_short.loc[enb_brain_short['eventname'] == '2_year_follow_up_y_arm_1']
enb_brain_T4 = enb_brain_short.loc[enb_brain_short['eventname'] == '4_year_follow_up_y_arm_1']
#check NaNs
count_nans(enb_brain_T2) #shape (7212, 92)
count_nans(enb_brain_T4) #shape (2734, 92)
#save csv
enb_brain_T2.to_csv('/Users/madeleineseitz/Desktop/abcd5_T2_enb_brain_reduced.csv')
enb_brain_T4.to_csv('/Users/madeleineseitz/Desktop/abcd5_T4_enb_brain_reduced.csv')
# %%
