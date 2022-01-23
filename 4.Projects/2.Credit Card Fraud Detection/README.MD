# Credit Card Fraud Detection
With the online transactions we see that the frauds being committed using the credit cards are also on rise.

Manual verification on each and every transaction is very hard, so we can take help of Machine Learning where we train models to predict what transactions could be fraud.

In this project we will predict fraudulent credit card transactions with the help of Machine learning models

This dataset contains transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for0.172% of all transactions.

Due to confidentiality input variables are transformed into numerical using PCA transformations.

## Steps Taken


### EDA:-
Performing Univariate Analysis, Bi-Variate Analysis, New Feature preparation, Outlier Treatment, Data Imbalance Check, Check corelation.
### Data Preparation:-
Convert features into gaussian,transformations, dropping columns which doesnot add value, Scaling the data.
### Test -Train split-:-
Perform train test split for the dataset to work that would be used for model training and testing.
### Building models:-

a. Building models on Imabalanced Dataset:

     - Logistic Regression
     - KNN
     - SVM
     - Decision Tree
     - Random Forest            - 
b. Model building with balancing Classes using Below techniques:-

Random Oversampling --> Logistic Regression,Decision Tree, XGBoost
SMOTEXGBoost --> Logistic Regression,Decision Tree, XGBoost
ADASYN --> Logistic Regression,Decision Tree, XGBoost
### Conclusions:- All the Compared models based on the scores of the Recall and AUC metric and accordingly consider the best model that can be recommended to the Buisness to opt for, that would be beneficial to the overall Business.

### Cost-benefit analysis:- It is necessary to determine how much profit or dollar/rupee value we would be saving with the best-selected model that is being built and considered.
