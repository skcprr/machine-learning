# Classification Algorithms on the Adult Dataset

## Introduction

This report summarizes the analysis conducted in the Jupyter notebook `adult_income_classification.ipynb`, which explores binary classification techniques using the Adult dataset. The goal is to predict whether an individual's annual income exceeds $50,000 based on demographic and employment features.

## Data Description 

The dataset used is the **Adult** (Census Income) dataset from OpenML, version 2. Also available on UC Irvine ML Repository on https://archive.ics.uci.edu/dataset/2/adult. It consists of 48,842 instances with 14 features and a binary target variable:
| Variable Name  | Variable Type |
| :-------------: | :-------------: |
| **age**           | *Integer*  |
| **workclass**     | *Categorical*  | 
| **fnlwgt**     | *Integer*  |  
| **education**     | *Categorical*  |
| **education-num**     | *Integer*  |
| **marital-status**     | *Categorical*  |
| **occupation**     | *Categorical*  |
| **relationship**     | *Categorical*  |
| **race**     | *Categorical*  |
| **sex**     | *Categorical*  |
| **capital-gain**           | *Integer*  |
| **capital-loss**           | *Integer*  |
| **hours-per-week**           | *Integer*  |
| **native-country**     | *Categorical*  |
| **income**     | *Binary*  |


 

The target variable indicates income levels: `<=50K` or `>50K`. The dataset is imbalanced, with more instances in the lower income category.


## Algorithms Explored

The notebook implements and compares the following classification algorithms:

1. **SGD Classifier**


2. **Random Forest**


3. **K-Nearest Neighbors (KNN)**


4. **Histogram-based Gradient Boosting Classification Tree**
