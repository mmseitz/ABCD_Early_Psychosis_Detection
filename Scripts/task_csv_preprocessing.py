#This script performs minimal preprocessing on individual task CSVs and creates master files of chosen 
#cognitive task brain and behavior parameters
# %%
#import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import numpy as np

# %% 

#import csvs containing behavioral and brain data corresponding to ABCD cognitive battery tasks
#Note: Stop Signal Task, Monetary Incentive Delay Task, and Emotional N-Back Task have both brain and 
#behavioral data CSVs

#Little Man Task--behavioral data (data collected at follow-up 2)
lmt = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Little_Man_Task/lmtp201.csv')

#Game of Dice Task--behavioral data (data collected at follow-up 2)
gdt = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Game_of_Dice/abcd_gdss01 copy.csv')

#Monetary Incentive Delay Task--behavioral data (data collected at follow-up 2)
mid_behav = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/MID_files/abcd_mid02.csv')
#MID--brain data (collected at follow-up 2)
mid_brain_1 = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/MID_files/MID_brain/midaparc03.csv')
mid_brain_2 = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/MID_files/MID_brain/midaparcp203.csv')

#Social Influence Task--behavioral data (collected at follow-up 2)
sit = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Social_Influence/abcd_siss01.csv')

#Stop Signal Task--behavioral data (collected at follow-up 2)
sst_behav = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Stop_Signal_Task/abcd_sst02.csv')
#SST--brain data (collected at follow-up 2)
sst_brain = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Stop_Signal_Task/SST_brain/mrisst02.csv')

#NIH Toolbox--behavioral data (collected at follow-up 2)
nihtb = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/NIH_Toolbox/abcd_tbss01.csv')

#Emotional N-back--behavioral data (collected at follow-up 2)
enb_behav = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Emotional_N_Back/mribrec02.csv')
#ENB--brain data (collected at follow-up 2)
enb_brain = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Emotional_N_Back/N-Back_Brain/nback_bwroi02.csv')

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
lmt = lmt.loc[lmt['eventname'] == '2_year_follow_up_y_arm_1'] #removing data collected at baseline from df
#select salient columns (lmt_scr_perc_wrong, lmt_scr_rt_wrong, and lmt_scr_efficiency ommitted due to high # of NaN values)
lmt = lmt[['subjectkey', 'eventname', 'interview_age', 'lmt_scr_perc_correct', 'lmt_scr_num_timed_out', 'lmt_scr_avg_rt', 'lmt_scr_rt_correct']]
#check NaNs
count_nans(lmt) #max 485 NaNs per column
#save reduced csv
lmt.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Little_Man_Task/lmt_reduced.csv')

# %%

#preprocessing of Game of Dice Task CSV
#select salient columns
gdt = gdt[['subjectkey', 'eventname', 'interview_age', 'gdt_scr_values_account_balance', 'gdt_scr_values_single', 
           'gdt_scr_values_double', 'gdt_scr_values_triple', 'gdt_scr_values_quadruple', 'gdt_scr_values_safe', 'gdt_scr_values_risky', 'gdt_scr_expressions_net_score', 
           'gdt_scr_values_wins', 'gdt_scr_values_losses']]
#drop first row
gdt = gdt.drop([0], axis = 0)
#check NaNs
count_nans(gdt) #566 NaNs per column
#save reduced csv
gdt.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Game_of_Dice/gdt_reduced.csv')

# %%
#Preprocessing MID behavioral data
#select salient columns
mid_behav = mid_behav[['subjectkey', 'eventname', 'interview_age', 'tfmri_mid_all_beh_srwpfb_nt', 'tfmri_mid_all_beh_srwpfb_rate', 'tfmri_mid_all_beh_srwnfb_nt', 
                       'tfmri_mid_all_beh_srwnfb_rate', 'tfmri_mid_all_beh_lrwpfb_nt', 'tfmri_mid_all_beh_lrwpfb_rate', 'tfmri_mid_all_beh_lrwnfb_nt', 'tfmri_mid_all_beh_lrwnfb_rate',
                       'tfmri_mid_all_beh_slpfb_nt', 'tfmri_mid_all_beh_slpfb_rate', 'tfmri_mid_all_beh_slnfb_nt', 'tfmri_mid_all_beh_slnfb_rate', 'tfmri_mid_all_beh_llpfb_nt', 
                       'tfmri_mid_all_beh_llpfb_rate', 'tfmri_mid_all_beh_llnfb_nt', 'tfmri_mid_all_beh_llnfb_rate', 'tfmri_mid_all_beh_ntpfb_nt', 
                       'tfmri_mid_all_beh_ntpfb_rate', 'tfmri_mid_all_beh_ntnfb_nt', 'tfmri_mid_all_beh_ntnfb_rate', 'tfmri_mid_all_beh_hrwpfb_nt', 'tfmri_mid_all_beh_hrwpfb_rate', 
                       'tfmri_mid_all_beh_hrwnfb_nt', 'tfmri_mid_all_beh_hrwnfb_rate', 'tfmri_mid_all_beh_hlpfb_nt', 'tfmri_mid_all_beh_hlpfb_rate', 'tfmri_mid_all_beh_hlnfb_nt', 
                       'tfmri_mid_all_beh_hlnfb_rate']]
#isolate follow-up 2 data
mid_behav = mid_behav.loc[mid_behav['eventname'] == '2_year_follow_up_y_arm_1']
#check NaNs
count_nans(mid_behav) #2625 NaNs per column
#save reduced csv
mid_behav.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/MID_files/mid_behav_reduced.csv')

# %%

#Preprocessing MID brain data
#identify midaparc03 parameters most closely associated with psychosis risk  (by keyword)
mid_2_list = []
mid_2_list.append('subjectkey')
mid_2_list.append('eventname')
mid_2_list.append('interview_age')
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
mid_1_list.append('subjectkey')
mid_1_list.append('eventname')
mid_1_list.append('interview_age')
for column in parameters:
    if 'anticipation' in str(parameters[column][5]):
        if 'amygdala' in str(parameters[column][5]):
            mid_1_list.append(parameters[column][4])
        if 'thalamus' in str(parameters[column][5]):
             mid_1_list.append(parameters[column][4])
#midaparc03
#select salient columns in midaparc03 df (using list generated above)
mid_brain_1 = mid_brain_1[mid_1_list]
#isolate follow-up-2 data
mid_brain_1 = mid_brain_1.loc[mid_brain_1['eventname'] == '2_year_follow_up_y_arm_1']
#check NaNs in dataset
mid_brain_1.shape # (7069, 35)--reduced df size, NaNs comparable to behav csv
#midaparcp203
#select salient columns in midaparcp203 df (using list generated above)
mid_brain_2 = mid_brain_2[mid_2_list]
#isolate follow-up-2 data
mid_brain_2 = mid_brain_2.loc[mid_brain_2['eventname'] == '2_year_follow_up_y_arm_1']
#check NaNs in dataset
mid_brain_2.shape #(7069, 19)--reduced df size, NaNs comparable to behav csv
#merge mid_brain_1 and mid_brain_2
mid_brain_2 = mid_brain_2.drop(columns = ['eventname', 'interview_age'])
mid_brain_data = mid_brain_1.merge(mid_brain_2, on = 'subjectkey')
#save csv
mid_brain_data.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/MID_files/MID_brain/mid_brain_reduced.csv')
# %%

#Preprocessing the Social Influence Task 
#Select salient columns in abcd_siss01
sit = sit[['subjectkey', 'eventname', 'interview_age', 'sit_scr_expr_minitialrat', 'sit_scr_expr_mrt_initialrat', 'sit_scr_expr_mfinalrat',
           'sit_scr_expr_mrt_finalrat', 'sit_scr_expr_mratdiff1', 'sit_scr_expr_mratdiff2', 'sit_scr_expr_mratdiff3',
         'sit_scr_values_count1', 'sit_scr_values_count2', 'sit_scr_values_count3', 'sit_scr_values_count4', 
           'sit_scr_values_countflips', 'sit_scr_values_countnr_initial', 'sit_scr_values_countnr_final']]
#['sit_scr_expr_mratdiff4'] removed due to high volume of NaNs
#drop first row of df
sit = sit.drop([0], axis=0)
#Check NaNs in df
count_nans(sit) #highest count 716
#save csv
sit.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Social_Influence/social_influence_reduced.csv')
# %%

#Preprocessing stop signal task behavioral data
#select salient columns in abcd_sst02
sst_behav = sst_behav[['subjectkey', 'eventname', 'interview_age', 'tfmri_sst_beh_performflag', 'tfmri_sst_nbeh_nruns', 
                       'tfmri_sst_all_beh_go_nt', 'tfmri_sst_all_beh_s_nt', 'tfmri_sst_all_beh_total_nt', 'tfmri_sst_all_beh_crgo_nt', 
                       'tfmri_sst_all_beh_crgo_rt', 'tfmri_sst_all_beh_crgo_mrt', 'tfmri_sst_all_beh_crgo_stdrt', 'tfmri_sst_all_beh_crlg_nt',
                       'tfmri_sst_all_beh_crlg_rt', 'tfmri_sst_all_beh_crlg_mrt', 'tfmri_sst_all_beh_crlg_stdrt', 'tfmri_sst_all_beh_incrgo_nt', 
                       'tfmri_sst_all_beh_incrgo_rt', 'tfmri_sst_all_beh_incrgo_mrt', 'tfmri_sst_all_beh_incrgo_stdrt', 
                       'tfmri_sst_all_beh_incrlg_nt', 'tfmri_sst_all_beh_incrlg_rt', 'tfmri_sst_all_beh_incrlg_mrt', 'tfmri_sst_all_beh_incrlg_stdrt',
                       'tfmri_sst_all_beh_nrgo_nt', 'tfmri_sst_all_beh_nrgo_rt', 'tfmri_sst_all_beh_crs_nt', 'tfmri_sst_all_beh_crs_rt', 
                       'tfmri_sst_all_beh_incrs_nt', 'tfmri_sst_all_beh_incrs_rt', 'tfmri_sst_all_beh_incrs_mrt', 'tfmri_sst_all_beh_incrs_stdrt', 
                       'tfmri_sst_all_beh_ssds_nt', 'tfmri_sst_all_beh_ssds_rt', 'tfmri_sst_all_beh_tot_mssd', 'tfmri_sst_all_beh_total_meanrt', 
                       'tfmri_sst_all_beh_total_mssrt', 'tfmri_sst_all_beh_total_issrt']]
# ['tfmri_sst_all_beh_crlg_mrt', 'tfmri_sst_all_beh_crlg_stdrt', 'tfmri_sst_all_beh_incrlg_mrt', 'tfmri_sst_all_beh_incrlg_stdrt',
#   'tfmri_sst_all_beh_total_mssrt', 'tfmri_sst_all_beh_total_issrt'] removed due to high NaN volume
sst_behav = sst_behav.drop(columns= ['tfmri_sst_all_beh_crlg_mrt', 'tfmri_sst_all_beh_crlg_stdrt', 'tfmri_sst_all_beh_incrlg_mrt', 'tfmri_sst_all_beh_incrlg_stdrt', 
                                     'tfmri_sst_all_beh_total_mssrt', 'tfmri_sst_all_beh_total_issrt'])
#isolate follow-up-2 data
sst_behav = sst_behav.loc[sst_behav['eventname'] == '2_year_follow_up_y_arm_1']
#check NaNs in df
count_nans(sst_behav) #highest count 2830 NaNs
#save csv
sst_behav.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Stop_Signal_Task/sst_reduced.csv')
# %%
#Preprocessing SST brain data
#identify SST parameters most pertaining to psychosis risk (thalamus, cerebellum, putamen, pallidum, NAcc ROIs)
sst_list = []
sst_list.append('subjectkey')
sst_list.append('eventname')
sst_list.append('interview_age')
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
#select salient columns in df
sst_brain = sst_brain[sst_list]
#isolate follow-up 2 data
sst_brain = sst_brain.loc[sst_brain['eventname'] == '2_year_follow_up_y_arm_1']
#check NaNs
count_nans(sst_brain) #NaNs < 50 per column, shape (6960, 101)
#save csv
sst_brain.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Stop_Signal_Task/SST_brain/sst_brain_reduced.csv')
# %%

#Preprocessing NIH Toolbox Data 
#Select salient columns in abcd_tbss01
nihtb = nihtb[['subjectkey', 'eventname', 'interview_age', 'nihtbx_picvocab_agecorrected',  
               'nihtbx_flanker_agecorrected', 'nihtbx_list_agecorrected', 'nihtbx_cardsort_agecorrected', 'nihtbx_pattern_agecorrected', 
               'nihtbx_picture_agecorrected', 'nihtbx_reading_agecorrected', 'nihtbx_fluidcomp_agecorrected', 'nihtbx_cryst_agecorrected', 
               'nihtbx_totalcomp_agecorrected', 'nihtbx_picvocab_fc', 'nihtbx_flanker_fc', 'nihtbx_list_fc', 'nihtbx_cardsort_fc', 
               'nihtbx_pattern_fc', 'nihtbx_picture_fc', 'nihtbx_reading_fc', 'nihtbx_fluidcomp_fc', 'nihtbx_cryst_fc', 
               'nihtbx_totalcomp_fc']]
#Columns ['nihtbx_list_agecorrected', 'nihtbx_cardsort_agecorrected', 'nihtbx_fluidcomp_agecorrected', 'nihtbx_totalcomp_agecorrected',
#           'nihtbx_list_fc', 'nihtbx_cardsort_fc', 'nihtbx_fluidcomp_fc', 'nihtbx_totalcomp_fc'] dropped due to high NaN frequency
nihtb = nihtb.drop(columns= ['nihtbx_list_agecorrected', 'nihtbx_cardsort_agecorrected', 'nihtbx_fluidcomp_agecorrected', 'nihtbx_totalcomp_agecorrected', 
                             'nihtbx_list_fc', 'nihtbx_cardsort_fc', 'nihtbx_fluidcomp_fc', 'nihtbx_totalcomp_fc'])
#isolate follow-up 2 data
nihtb = nihtb.loc[nihtb['eventname'] == '2_year_follow_up_y_arm_1']
#check NaNs
count_nans(nihtb) #highest NaN 3046 (NaN range 563-3046)
#save csv
nihtb.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/NIH_Toolbox/nihtb_reduced.csv')
# %%

#Preprocessing Emotional N-Back Data
#Behavioral data--mribrec02
#Select salient columns
enb_behav = enb_behav[['subjectkey', 'eventname', 'interview_age', 'tfmri_rec_all_beh_newnf_hr', 'tfmri_rec_all_beh_newnf_fa', 'tfmri_rec_all_beh_newpl_hr', 'tfmri_rec_all_beh_newpl_fa', 
                       'tfmri_rec_all_beh_newposf_hr', 'tfmri_rec_all_beh_newposf_fa', 'tfmri_rec_all_beh_newnegf_hr', 'tfmri_rec_all_beh_newnegf_fa', 
                       'tfmri_rec_all_beh_oldposf_hr', 'tfmri_rec_all_beh_oldposf_fa', 'tfmri_rec_all_beh_posface_pr', 'tfmri_rec_all_beh_posface_br', 
                       'tfmri_rec_all_beh_posf_dpr', 'tfmri_rec_all_beh_neutface_pr', 'tfmri_rec_all_beh_neutface_br', 'tfmri_rec_all_beh_neutf_dp', 
                       'tfmri_rec_all_beh_negface_pr', 'tfmri_rec_all_beh_negface_br', 'tfmri_rec_all_beh_negf_dp', 'tfmri_rec_all_beh_place_pr', 
                       'tfmri_rec_all_beh_place_br', 'tfmri_rec_all_beh_place_dp']]
#Isolate follow-up 2 data
enb_behav = enb_behav.loc[enb_behav['eventname'] == '2_year_follow_up_y_arm_1']
#Check NaNs
count_nans(enb_behav) #(max NaN 3043, min 3010)
#save csv
enb_behav.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Emotional_N_Back/enb_behav_reduced.csv')
# %%
#Preprocessing Emotion N-Back Brain Data
#Identify ENB parameters pertaining to psychosis risk (cerebellum, thalamus, hippocampus, amygdala ROIs)
nb_list = []
nb_list.append('subjectkey')
nb_list.append('eventname')
nb_list.append('interview_age')
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
#select salient columns in enb_brain df
enb_brain = enb_brain[nb_list]
#isolate follow-up 2 data
enb_brain = enb_brain.loc[enb_brain['eventname'] == '2_year_follow_up_y_arm_1']
#check NaNs
count_nans(enb_brain) #shape (6951, 93)
#save csv
enb_brain.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Emotional_N_Back/N-Back_Brain/enb_brain_reduced.csv')
# %%

#create master file of behavioral tasks 
behav_master = lmt.merge(gdt, on = ['subjectkey', 'eventname', 'interview_age'])
behav_master = behav_master.merge(mid_behav, on = ['subjectkey', 'eventname', 'interview_age'])
behav_master = behav_master.merge(sit, on = ['subjectkey', 'eventname', 'interview_age'])
behav_master = behav_master.merge(sst_behav, on = ['subjectkey', 'eventname', 'interview_age'])
behav_master = behav_master.merge(nihtb, on = ['subjectkey', 'eventname', 'interview_age'])
behav_master = behav_master.merge(enb_behav, on = ['subjectkey', 'eventname', 'interview_age'])
#save behav_master as csv
behav_master.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Master_Files/behav_master.csv')
# %%

#create master of task fMRI beta weights
brain_master = mid_brain_data.merge(sst_brain, on = ['subjectkey', 'eventname', 'interview_age'])
brain_master = brain_master.merge(enb_brain, on = ['subjectkey', 'eventname', 'interview_age'])
#save brain_master as csv
brain_master.to_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Master_Files/brain_master.csv')
# %%