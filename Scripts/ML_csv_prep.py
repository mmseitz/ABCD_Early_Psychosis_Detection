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
#Import raw data
#Demographics
grouped_demos = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/matched_demos.csv')
grouped_demos = pd.DataFrame(grouped_demos)
grouped_demos = grouped_demos.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
#Mental health summary scores
mhss = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/mhss_reduced.csv')
mhss = pd.DataFrame(mhss)
mhss = mhss.drop(['Unnamed: 0'], axis=1)
#Developmental History
devhx = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/devhx_reduced.csv')
devhx = pd.DataFrame(devhx)
devhx = devhx.drop(['Unnamed: 0'], axis=1)
#Medical History
medhx = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Demos_and_Env/med_hx_reduced.csv')
medhx = pd.DataFrame(medhx)
medhx = medhx.drop(['Unnamed: 0'], axis=1)
#Structural data
anat = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Structural_Scans/structural_data.csv')
anat = pd.DataFrame(anat)
anat = anat.drop(['Unnamed: 0', 'sex'], axis=1)
#Monetary Incentive Delay data
#Behavioral data
mid_behav = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/MID_files/mid_behav_reduced.csv')
mid_behav = pd.DataFrame(mid_behav)
mid_behav = mid_behav.drop(['Unnamed: 0'], axis=1)
#Brain data
mid_brain = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/MID_files/MID_brain/mid_brain_reduced.csv')
mid_brain = pd.DataFrame(mid_brain)
mid_brain = mid_brain.drop(['Unnamed: 0'], axis=1)
#Stop Signal Task data
#Behavioral data
sst_behav = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Stop_Signal_Task/sst_reduced.csv')
sst_behav = pd.DataFrame(sst_behav)
sst_behav = sst_behav.drop(['Unnamed: 0'], axis=1)
#Brain data
sst_brain = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Stop_Signal_Task/SST_brain/sst_brain_reduced.csv')
sst_brain = pd.DataFrame(sst_brain)
sst_brain = sst_brain.drop(['Unnamed: 0'], axis=1)
#Emotional N-Back data
#behavioral data
enb_behav = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Emotional_N_Back/enb_behav_reduced.csv')
enb_behav = pd.DataFrame(enb_behav)
enb_behav = enb_behav.drop(['Unnamed: 0'], axis=1)
#brain data
enb_brain = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Emotional_N_Back/N-Back_Brain/enb_brain_reduced.csv')
enb_brain = pd.DataFrame(enb_brain)
enb_brain = enb_brain.drop(['Unnamed: 0'], axis=1)
#Game of Dice task
gdt = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Game_of_Dice/gdt_reduced.csv')
gdt = pd.DataFrame(gdt)
gdt = gdt.drop(['Unnamed: 0'], axis=1)
#Social Influence Task
sit = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Social_Influence/social_influence_reduced.csv')
sit = pd.DataFrame(sit)
sit = sit.drop(['Unnamed: 0'], axis=1)
#NIH Toolbox
nihtb = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/NIH_Toolbox/nihtb_reduced.csv')
nihtb = pd.DataFrame(nihtb)
nihtb = nihtb.drop(['Unnamed: 0'], axis=1)
#Little Man Task
lmt = pd.read_csv('/Users/madeleineseitz/Desktop/thesis/ABCD_Project/csvs/Little_Man_Task/lmt_reduced.csv')
lmt = pd.DataFrame(lmt)
lmt = lmt.drop(['Unnamed: 0'], axis=1)
# %%
#merge environmental dfs into one master df
env = mhss.merge(devhx, on= ['subjectkey'])
env = env.merge(medhx, on= ['subjectkey'])
env = env.merge(grouped_demos, on= ['subjectkey'])
env = env[['subjectkey', 'ple_y_ss_total_bad', 'ple_y_ss_affected_bad_sum', 'peq_ss_overt_aggression', 'peq_ss_relational_aggs', 
             'peq_ss_overt_victim', 'total_substances_used', 'weeks_preterm', 'motor_dev_comparison', 'speech_dev_comparison', 'injury_totals', 'sex', 'income',
              'p_edu', 'race', 'group']]
#merge anat and cog dfs into one master df
abcd = mid_behav.merge(mid_brain, on= ['subjectkey', 'eventname', 'interview_age'])
abcd = abcd.merge(sst_behav, on= ['subjectkey', 'eventname', 'interview_age'])
abcd = abcd.merge(sst_brain, on= ['subjectkey', 'eventname', 'interview_age'])
abcd = abcd.merge(enb_brain, on= ['subjectkey', 'eventname', 'interview_age'])
abcd = abcd.merge(enb_behav, on= ['subjectkey', 'eventname', 'interview_age'])
abcd = abcd.merge(anat, on= ['subjectkey', 'eventname', 'interview_age'])
abcd = abcd.merge(gdt, on= ['subjectkey', 'eventname', 'interview_age'])
abcd = abcd.merge(sit, on= ['subjectkey', 'eventname', 'interview_age'])
abcd = abcd.merge(nihtb, on= ['subjectkey', 'eventname', 'interview_age'])
abcd = abcd.merge(lmt, on= ['subjectkey', 'eventname', 'interview_age'])
abcd = abcd.drop (columns= ['eventname'])
abcd = abcd.merge(env, on = ['subjectkey'])
abcd.shape #(2001, 514)
abcd['group'].sum() #HR group = 968
# %%
####
#Drop NaN values from df
abcd = abcd.dropna()
abcd.shape # (1347, 514)
abcd['group'].sum() #HR group = 647
abcd.to_csv('/Users/madeleineseitz/Desktop/uniform_abcd.csv')
# %%
####
#Separate full df into separate dfs for sub-models
####
#Environmental model
env_abcd = abcd[['subjectkey', 'interview_age', 'ple_y_ss_total_bad', 'ple_y_ss_affected_bad_sum', 'peq_ss_overt_aggression', 'peq_ss_relational_aggs', 
             'peq_ss_overt_victim', 'total_substances_used', 'weeks_preterm', 'motor_dev_comparison', 'speech_dev_comparison', 'injury_totals', 'sex', 'income',
              'p_edu', 'race', 'group']]
# %%
#save csv of uniform environmental data
env_abcd.to_csv('/Users/madeleineseitz/Desktop/env_abcd.csv')
# %%
####
#Anatomical Model
anat_list = list(anat)
anat_list.pop(2)
dem_list = ['sex', 'p_edu', 'income', 'race', 'group']
anat_list = anat_list + dem_list
anat_abcd = abcd[anat_list]
#save csv of uniform sMRI data
anat_abcd.to_csv('/Users/madeleineseitz/Desktop/anat_abcd.csv')
#Cognitive task model
mid_brain_list = list(mid_brain)
mid_brain_list.pop(1)
mid_behav_list = list(mid_behav)
mid_behav_list.pop(2)
mid_behav_list.pop(1)
mid_behav_list.pop(0)
sst_brain_list = list(sst_brain)
sst_brain_list.pop(2)
sst_brain_list.pop(1)
sst_brain_list.pop(0)
sst_behav_list = list(sst_behav)
sst_behav_list.pop(2)
sst_behav_list.pop(1)
sst_behav_list.pop(0)
enb_brain_list = list(enb_brain)
enb_brain_list.pop(2)
enb_brain_list.pop(1)
enb_brain_list.pop(0)
enb_behav_list = list(enb_behav)
enb_behav_list.pop(2)
enb_behav_list.pop(1)
enb_behav_list.pop(0)
gdt_list = list(gdt)
gdt_list.pop(2)
gdt_list.pop(1)
gdt_list.pop(0)
sit_list = list(sit)
sit_list.pop(2)
sit_list.pop(1)
sit_list.pop(0)
nihtb_list = list(nihtb)
nihtb_list.pop(2)
nihtb_list.pop(1)
nihtb_list.pop(0)
lmt_list = list(lmt)
lmt_list.pop(2)
lmt_list.pop(1)
lmt_list.pop(0)
cog_list = mid_brain_list + mid_behav_list + sst_brain_list + sst_behav_list + enb_brain_list + enb_behav_list + gdt_list + sit_list + nihtb_list + lmt_list + dem_list
cog_abcd = abcd[cog_list]
cog_abcd.to_csv('/Users/madeleineseitz/Desktop/cog_abcd.csv')
# %%
####
#Cogntiive and Anatomical Model
anat_list.pop(1)
anat_list.pop(0)
anat_cog_list = mid_brain_list + mid_behav_list + sst_brain_list + sst_behav_list + enb_brain_list + enb_behav_list + gdt_list + sit_list + nihtb_list + lmt_list + anat_list
anat_cog_abcd = abcd[anat_cog_list]
anat_cog_abcd.to_csv('/Users/madeleineseitz/Desktop/anat_cog_abcd.csv')
# %%
