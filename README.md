## ABCD_Early_Psychosis_Detection

This repository contains documentation and scripts for early detection of psychosis in the ABCD cohort research project. 

### Project scope:

#### Introduction
Sub-threshold psychotic symptoms in adolescence are a risk factor for psychotic disorders and other diverse psychopathology later in life (Calkins, 2014). Early identification of psychosis risk can allow for the deployment of life-changing interventions that may ameliorate the trajectory of an individual’s prognosis (Gur, 2014). Psychosis is typically preceded by a prodromal phase characterized by social and cognitive deficits and structural brain abnormalities, and significantly associated with environmental factors such as socioeconomic status and exposure to adverse life events (Satterthwaite, 2015; Sheffield, 2018; Bhasvar, 2017). Here we apply a series of supervised machine learning models in a multisite large-scale dataset to comprehensively evaluate key indicators of a psychosis high-risk state and compare their performances on different classes of psychosis-relevant features: cognition, MRI, and early stress and environment.

#### Methods: 
Data from ABCD follow-up 2 (n = 10,973) and ABCD follow-up 4 (n = 4,754) were analyzed to evaluate the efficacy of cognitive, structural, and environmental features in indicating psychosis risk in adolescents (mean age = 11.71 years). Participants who endorsed 4+ items on the Prodromal Psychosis Scale (PPS) were identified as high-risk for developing psychosis and matched by sex, age, and ethnicity with a low-risk cohort who endorsed no PPS questions and had no family history of psychosis at both T2 and T4 (T2 high-risk n = 566, T2 low-risk n = 614; T4 high-risk n = 109, T4 low-risk n = 100). Before participant classification, data were preprocessed using a range of feature selection algorithms (sequential forward and backward logistic regression and random forest and recursive feature selection). In total we compared the accuracy of four supervised machine learning models with different subsets of ABCD features: cognitive (trained on cognitive task battery behavioral scores and fMRI beta weights), structural MRI (cortical thickness and volume of right and left Desikan-Killiany regions), environment (environmental features relevant to psychosis risk), and composite models with all features: cognitive, structural MRI, and environment. The following models were tested for each set of features: logistic regression, linear and quadratic discriminant analysis, Gaussian Naïve Bayes, K-nearest neighbor, and decision tree, random forest, extra trees, and XG boost classifiers. 

Models were run twice: initially with a T2 training set and a T2 test set. Models were run a second time with the same T2 training set and tested on T4 data (omitting any participants presents in T2 train set; test HR n = 80, test LR n = 59).

### Scripts

"Scripts" folder contains all preprocessing and ML scripts used in this project, organized by type and ABCD wave

### Docs

"Docs" folder contains documentation and complementary information to scripts

### Spreadsheeets

"Spreadsheets" contain various data breakdowns at different points in analysis 

