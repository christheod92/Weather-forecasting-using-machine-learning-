## Weather forecasting using machine learning 

**Dataset Information**

We use and compare various different methods for weather forecasting on meteo dataset. The training and test datasets are expected to be csv files. Please note that there are not csv headers in both training and test datasets.

**Data Description**

Dataset: athens_09-16

- month: [1-12]
- maxtemp: daily maximum air temperature
- mintemp: daily minimum air temperature
- heat: heating degree days
- cool: cooling degree days
- rainmm: rainfall measured in millimeters
- meanspeed: daily mean wind speed
- maxspeed: daily maximum wind speed

Dataset: athens_2017

- month: [1-12]
- maxtemp: daily maximum air temperature
- mintemp: daily minimum air temperature
- heat: heating degree days
- cool: cooling degree days
- rainmm: rainfall measured in millimeters
- meanspeed: daily mean wind speed
- maxspeed: daily maximum wind speed

Data Source: Greece, Athens, meteo.gr, 2009-2016

**Machine learning methods**

1.  Predicted variable: Daily maximum air temperature (maxtemp)
    - Machine learning model: Multiple linear regression, Random Forest
    - Evaluation Metrics: Root mean square error
    
2.  Predicted variable: Rainfall measured in millimeters (rainmm)  
    - Machine learning model: Multiple linear regression, Random Forest
    - Evaluation Metrics: Root mean square error
    
3.  Classification in autumn, winter, spring, summer
    - Machine learning model: Gaussian process, SVC
    - Evaluation Metrics: Accuracy
    
    
**Requirements**

There are some general library requirements for the project. These requirements are as follows.
- nltk
- numpy
- matplotlib.pyplot 
- (math) sqrt
- pandas 
- scipy
- pearsonr
- seaborn 
- (sklearn.model_selection)  train_test_split
- (sklearn.metrics) mean_squared_error
- (scikit-learn) linear_model, train_test_split
- (sklearn.ensemble) RandomForestRegressor
- (sklearn.naive_bayes) GaussianNB
- (sklearn.metrics) accuracy_score
- (sklearn.svm) SVC

Note: It is recommended to use Anaconda distribution of Python.
