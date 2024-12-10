#This script performs minimal preprocessing on ABCD structural brain data (abcd_smrip10201) and creates a master file of participant cortical
#thickness and cortical volume
# %%
#import libraries
import math
import pandas as pd
import scipy.stats as stats
import numpy as np

# %%
#import csv
s_cdk = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/sMRI/mri_y_smr_vol_dsk.csv')
s_vol = pd.read_csv('/Users/madeleineseitz/Desktop/ABCD_5/sMRI/mri_y_smr_thk_dsk.csv')

# %%
#merge csvs
smri = s_cdk.merge(s_vol, on = ['src_subject_id', 'eventname'])
# %%

#select subject data from follow-up 2
T2_smri = smri.loc[smri['eventname'] == '2_year_follow_up_y_arm_1']
T4_smri = smri.loc[smri['eventname'] == '4_year_follow_up_y_arm_1']
# %%
#save csvs 
T2_smri.to_csv('/Users/madeleineseitz/Desktop/abcd5_T2_structural_data.csv')
T4_smri.to_csv('/Users/madeleineseitz/Desktop/abcd5_T4_structural_data.csv')
# %%
