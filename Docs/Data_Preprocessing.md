# Preprocessing Brain and Behavioral Cognitive Task Data

Selected data from the ABCD Cognitive task battery are preprocessed in preparation for ML pipeline and performing statistical analysis to determine presence of significant differences in performance between high and low psychosis risk groups (predetermined in pps_hr_lr_groups script). 

Tasks selected are:
- Little Man Task (follow-up 2 data only)
- Game of Dice Task
- Monetary Incentive Delay Task -- behavioral and fMRI data (follow-up 2 data only)
- Social Influence Task
- NIH Toolbox (follow-up 2 data only)
- Emotional N-Back Task -- behavioral and fMRI data (follow-up 2 data only)
- Stop Signal Task -- behavioral and fMRI data (follow-up 2 data only)

Beta weight parameters were selected based on brain regions relevant to psychosis risk. The most descriptive behavioral parameters were selected for each task. DFs were filtered to include only data from follow-up 2. Each DF column was evaluated for frequency of NaN values, and columns with particularly high NaN values were dropped from the DF (noted in script).