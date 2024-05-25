# High- and Low-Risk Group Differences in Demographics and Environment

In order to provide some context to participant Prodromal Psychosis Scores and cognitive task performance, we examined differences between groups within an abridged collection of demographic and mental health-related data. 

CSVs used: pdem02 (demographics), abcd_cbcls01 (child behavior checklist) and abcd_mhy02 (mental health summary scores--youth surveys)

Script: demo_mh_stats.py

Results: see PQBC_demos_MH_stats.xls in 'Spreadsheets'

### Demographics
Within demographics, we examine highest level of parental education, household income, participant race, and participant assigned sex at birth

Income has been simplified into three categories: Lower, Middle, and Higher Income. Annual household income below $35K is considered lower income, between $35K and $100K is middle income, and above $100K is considered higher income. (numerically coded: lower = 1, middle = 2, higher = 3)

Race has been simplified into four categories: White, Black, Hispanic, and Other (numerically coded: White = 1, Black = 2, Hispanic = 3, Other = 4)

Highest level of parental education has been simplified into three categories: High School Education (meaning that the parents completed high school or less), college (meaning that the parents completed some or all of college, but not beyond a bachelor's degree), post-grad (meaning that the parents completed at least some education beyond a bachelor's degree). (numerically coded: high school = 1, college = 2, post grad = 3)

Sex (as assigned at birth) has been coded as male = 1, female = 2, other = 3

These columns were tested for signficant differences between psychosis high- and low-risk groups

### Mental Health Summary Score
Within the mental health summary scores, we selected columns closely associated with psychosis risk factors: ['ple_y_ss_total_bad' (total number of bad events), 'ple_y_ss_affected_bad_sum' (how much affected by bad events), 'peq_ss_overt_aggression' (overt aggression summary score), 'peq_ss_relational_aggs (relational aggression summary score)']. 

These columns were tested for significant differences between psychosis high- and low-risk groups 

### Child Behavior Checklist (CBCL)
WIthin the CBCL, we selected columns most closely associated with psychosis risk: ['cbcl_scr_syn_thought_t' (thought CBCL syndrome scale t-score), 'cbcl_scr_syn_totprob_t' (TotProb CBCL syndrome scale t-score)]. Thought is the HiTOP spectrum associated with psychotic features. The total score contains data across HiTOP spectra. 

These columns were tested for significant differences between psychosis high- and low-risk groups. 

### Evaluating Familial Psychosis Risk
Elevated familiar risk for psychosis determined by a participant having at least 1 affected first degree relative or at least 2 affected second degree relatives (Sheffield et al., 2018)