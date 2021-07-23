# Naive-Bayes-Classifier_with_Normality_Tests
Gaussian Navie Bayes Classifier was applied on IRIS dataset. Different types of normality tests were used to introduce the normality concepts. 
Implementation of Gaussian Naive Bayes Model is done to demonstrate its working. 

**First, we need to import the libraries**

Pandas,	Numpy,	Seaborn, Matplotlib,	Sklearn,	Scipy
  
**Second, need to prepare the data**

1. Exploratory Data Analysis

   `profile = ProfileReport(df, title='Profile Reports', explorative=True)`
   
2. Standardization : Through StandardScaler   
3. Checking Null cells
4. Train/Test Split

**Third, Normality Test**
*check p value according to tests
1. Skewness - Kurtosis Test
   
   `print(stats.kurtosis(X))`
   
   `print(stats.skew(X))`
2. Shapiro - Wilk Test

    `stat, p = shapiro(X[i])`
    
3. Kolmogorov-Smirnov Test

    `stat1, p1 = kstest(X[i], 'norm')`


**Fourth, Applying Gaussian Naive-Bayes Model**

*Model accuracy was found to be 97.7%*

**Fifth, Confusion Matrix & Classification Report**
1. Confusion Matrix Plot (heatmap)

   `sns.heatmap(confusion_mat, annot = True)`
3. CLassification Report

**Finally, Predict the outcomes**



