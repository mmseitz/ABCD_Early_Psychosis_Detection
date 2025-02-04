## ABCD_Early_Psychosis_Detection

This repository contains documentation and scripts for classification of participants at high- and low-risk for developing psychosis in the ABCD dataset. Below is the necessary information to calculate predictions. While some pipelines have been tested to work out of hte box, please note that this is not supposed to be a toolbox and is not polished or maintained as one. I am happy to answer any additional questions at: madeleinemseitz@gmail.com.

The scripts in this repository take data from the ABCD 5.0 release and classify particpants at both follow-up 2 and follow-up 4. No data are stored in this repository. Scripts are broken into four sub-folders: 

csv_preprocessing:
- This folder contains all scripts for preprocessing ABCD csvs and conducting preliminary statistical analyses
- high- and low-risk groups are determined by participant scores on the Prodromal Psychosis Scale (pps_breakdown and refinement)
- High-risk groups are matched to a demographically similar low-risk group of equal size usign R-Matchit 

T2_feature_selection: 
- T2 feature selection scripts determine the features used in both T2 adn T4 models
- feature selection is conducted for each of the four sets of features evaluated in each dataset: MRI, cognition, environment, and a composite model

T2_ML:
- This folder preprocesses and runs data at follow-up 2 (models are trained and tested on follow-up 2 data)
- This folder contains preprocessing scripts for the four sets of features evaluated and supervised ML scripts
- each ML script contains 9 supervised machine learning models: logistic regression, linear discriminant analysis, quadratic discriminant analysis, Gaussian Naive Bayes, K nearest Neighbor, Descision Tree, Random Forest, Extra Trees, and XGBoost

T4_ML:
- This folder preprocesses data at follow-up 4 and runs models trained on follow-up 2 and tested on data from follow-up 4
- This folder contains preprocessing scripts for the four sets of features evaluated and supervised ML scripts
- each ML script contains 9 supervised machine learning models: logistic regression, linear discriminant analysis, quadratic discriminant analysis, Gaussian Naive Bayes, K nearest Neighbor, Descision Tree, Random Forest, Extra Trees, and XGBoost
  
### Environment

These scripts require a number of python modules to run. ML scripts rely primariy on scikit-learn and scipy-stats.


