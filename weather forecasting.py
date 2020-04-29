# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 18:37:23 2019

@author: Chris
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:36:00 2019

@author: Chris
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd 
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def a_scatterplot(dt):
    
    plt.scatter(dt['heat'], dt['maxtemp'], color='red')
    #plt.scatter(dt['cool'], dt['maxtemp'], color='blue')
    #plt.scatter(df['rainmm'], df['maxtemp'], color='green')
    #plt.scatter(df['meanspeed'], df['maxtemp'], color='black')
    #plt.scatter(df['maxspeed'], df['maxtemp'], color='red')
    plt.title('heat, cool Vs maxtemp', fontsize=14)
    plt.xlabel('heat , cool', fontsize=14)
    plt.ylabel('maxtemp', fontsize=14)
    plt.show()
    #correlation
    sns.heatmap(dt.corr(),annot=True)
    plt.show()
    # παρατηρω οτι υπαρχει γραμμικη σχεση μεταξυ των μεταβλητων 
    
    plt.scatter(dt['maxtemp'], dt['rainmm'], color='red')
    #plt.scatter(dt['mintemp'], dt['rainmm'], color='red')
    #plt.scatter(dt['heat'], dt['rainmm'], color='red')
    #plt.scatter(dt['cool'], dt['rainmm'], color='blue')
    #plt.scatter(df['meanspeed'], df['rainmm'], color='black')
    plt.scatter(dt['maxspeed'], dt['rainmm'], color='red')
    plt.title('heat, cool Vs rainmm', fontsize=14)
    plt.xlabel('heat , cool', fontsize=14)
    plt.ylabel('rainmm', fontsize=14)
    plt.show()
    #correlation
    sns.heatmap(dt.corr(),annot=True)
    plt.show()
    # παρατηρω απο τα scatterplot αλλα και το heatmap οτι δεν  υπαρχει γραμμικη σχεση μεταξυ των μεταβλητων 
      

def a_erotima(fin2016,fin2017):
    #multiple linear regression
    #keep the right columns
    print('/////////ΕΡΩΤΗΜΑ Α/////////')
    print('/////////MULTIPLE LINEAR REGRESSION/////////')
    X=fin2016.drop(['month','maxtemp','mintemp'],axis=1)
    Y=fin2016['maxtemp']
    Z=fin2017.drop(['month','maxtemp','mintemp'],axis=1)
    W=fin2017['maxtemp']
    X_train, X_test,y_train,y_test=train_test_split(X, Y, test_size=0.3, random_state=101)
    regr= linear_model.LinearRegression()
    #fit model
    regr.fit(X_train, y_train)
    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)
    #predict values
    predictions16=regr.predict(X_test)
    predictions17=regr.predict(Z)
    #make dataframes with predictions-real
    df1=pd.DataFrame({'Real 2016':y_test, 'Predicted 2016':predictions16})
    df2=pd.DataFrame({'Real 2017':W, 'Predicted 2017':predictions17})
    print(df1)
    print(df2)
    #calculate rmse
    rmse16 = sqrt(mean_squared_error(y_test,predictions16))
    rmse17 = sqrt(mean_squared_error(W,predictions17))
    print('RMSE 2016 :',rmse16, 'RMSE 2017 :', rmse17)
    #plot real vs predicted
    plt.plot(df2['Real 2017'],color = 'red' )
    plt.plot(df2['Predicted 2017'],color = 'blue')
    plt.xlabel('days',fontsize=18)
    plt.ylabel('temps', fontsize=18)
    plt.title('Real maxtemps 2017 vs Predicted maxtemps 2017 linear regr', fontsize=18)
    plt.show()
    #δεν υπαρχει ευθεια παλινδρομησης αλλα επιπεδο πολλων διαστασεων
    
    # Randomforest 
    print('/////////RANDOM FOREST/////////')
    #built randomforestregress model with 20 number of trees
    regrr= RandomForestRegressor(n_estimators=20, random_state=0)
    #fit model
    regrr.fit(X_train, y_train)
    pred16=regrr.predict(X_test)
    pred17=regrr.predict(Z)
    #make dataframes with predictions-real
    df116=pd.DataFrame({'Real 2016':y_test, 'Predicted 2016':pred16})
    df117=pd.DataFrame({'Real 2017':W, 'Predicted 2017':pred17})
    print(df116)
    print(df117)
    #calculate rmse
    rmse116 = sqrt(mean_squared_error(y_test,pred16))
    rmse117 = sqrt(mean_squared_error(W,pred17)) 
    print('RMSE 2016 :',rmse116, 'RMSE 2017 :', rmse117)
    #plot real vs predicted
    plt.plot(df117['Real 2017'],color = 'red' )
    plt.plot(df117['Predicted 2017'],color = 'blue')
    plt.xlabel('days',fontsize=18)
    plt.ylabel('temps', fontsize=18)
    plt.title('Real maxtemps 2017 vs Predicted maxtemps 2017 randomforest', fontsize=18)
    plt.show()
    #compare rmse linear-randomfoerst  
    # ftiakse to barplot
    plt.bar('regr rmse 16',rmse16,color='red')
    plt.bar('regr rmse 17',rmse17,color='blue')
    plt.bar('randfor rmse 16',rmse116,color='black')
    plt.bar('randfror 17',rmse117,color='green')
    plt.title('Bar Graph RMSE')
    plt.xlabel('MODELS rmse')
    plt.ylabel('RMSE values')
    plt.show()
    
    
   
def b_erotima(fin2016,fin2017):
    #multiple linear regression
    #keep the right columns
    print('/////////ΕΡΩΤΗΜΑ Β/////////')
    print('/////////MULTIPLE LINEAR REGRESSION/////////')
    X=fin2016.drop(['month'],axis=1)
    Y=fin2016['rainmm']
    Z=fin2017.drop(['month'],axis=1)
    W=fin2017['rainmm']
    X_train, X_test,y_train,y_test=train_test_split(X, Y, test_size=0.3, random_state=101)
    regr= linear_model.LinearRegression()
    #fit model
    regr.fit(X_train, y_train)
    print('Intercept: \n', regr.intercept_)
    print('Coefficients: \n', regr.coef_)
    #predict values
    predictions16=regr.predict(X_test)
    predictions17=regr.predict(Z)
    #make dataframes with predictions-real
    df1=pd.DataFrame({'Real 2016':y_test, 'Predicted 2016':predictions16})
    df2=pd.DataFrame({'Real 2017':W, 'Predicted 2017':predictions17})
    print(df1)
    print(df2)
    #calculate rmse
    rmse16 = sqrt(mean_squared_error(y_test,predictions16))
    rmse17 = sqrt(mean_squared_error(W,predictions17))
    print('RMSE 2016 :',rmse16, 'RMSE 2017 :', rmse17)
    #plot real vs predicted
    plt.plot(df2['Real 2017'],color = 'red' )
    plt.plot(df2['Predicted 2017'],color = 'blue')
    plt.xlabel('days',fontsize=18)
    plt.ylabel('rainmm', fontsize=18)
    plt.title('Real rainmm 2017 vs Predicted rainmm 2017 linear regr', fontsize=18)
    plt.show()
    #δεν υπαρχει ευθεια παλινδρομησης αλλα επιπεδο πολλων διαστασεων
    #ασχημα αποτελεσματα, μη γραμμικα
    
    

    
    # Randomforest 
    print('/////////RANDOM FOREST/////////')
    #built randomforestregress model with 20 number of trees
    regrr= RandomForestRegressor(n_estimators=20, random_state=0)
    #fit model
    regrr.fit(X_train, y_train)
    pred16=regrr.predict(X_test)
    pred17=regrr.predict(Z)
    #make dataframes with predictions-real
    df116=pd.DataFrame({'Real 2016':y_test, 'Predicted 2016':pred16})
    df117=pd.DataFrame({'Real 2017':W, 'Predicted 2017':pred17})
    print(df116)
    print(df117)
    #calculate rmse
    rmse116 = sqrt(mean_squared_error(y_test,pred16))
    rmse117 = sqrt(mean_squared_error(W,pred17)) 
    print('RMSE 2016 :',rmse116, 'RMSE 2017 :', rmse117)
    #plot real vs predicted
    plt.plot(df117['Real 2017'],color = 'red' )
    plt.plot(df117['Predicted 2017'],color = 'blue')
    plt.xlabel('days',fontsize=18)
    plt.ylabel('temps', fontsize=18)
    plt.title('Real rainmm 2017 vs Predicted rainmm 2017 randomforest', fontsize=18)
    plt.show()
    #compare rmse linear-randomfoerst  
    # ftiakse to barplot
    plt.bar('regr rmse 16',rmse16,color='red')
    plt.bar('regr rmse 17',rmse17,color='blue')
    plt.bar('randfor rmse 16',rmse116,color='black')
    plt.bar('randfror 17',rmse117,color='green')
    plt.title('Bar Graph 1 RMSE')
    plt.xlabel('MODELS rmse')
    plt.ylabel('RMSE values')
    plt.show()
   
    

def c_erotima(fin2016,fin2017):
    print('/////////ΕΡΩΤΗΜΑ Γ/////////')
    print('/////////GAUSSIANNB////////')
    #GAUSSIANNB
    #keep the right columns
    fin2016['season']=['w' if m==1 or m==2 or m==12 else 'sp' if m==3 or m==4 or m==5 else 'sm' if m==6 or m==7 or m==8 else 'a' for m in fin2016['month']]
    Y=fin2016['season']
    X=fin2016.drop(['month','season'],axis=1)
    fin2017['season']=['w' if m==1 or m==2 or m==12 else 'sp' if m==3 or m==4 or m==5 else 'sm' if m==6 or m==7 or m==8 else 'a' for m in fin2017['month']]
    W=fin2017['season']
    Z=fin2017.drop(['month','season'],axis=1)
    X_train, X_test,y_train,y_test=train_test_split(X, Y, test_size=0.3, random_state=101)
    gnb = GaussianNB()
    #fit model
    gnb.fit(X_train, y_train)
    #predict
    pred16 = gnb.predict(X_test)
    pred17 = gnb.predict(Z)
    #accuracy
    df16=pd.DataFrame({'Real 2016':y_test, 'Predicted 2016':pred16})
    df17=pd.DataFrame({'Real 2017':W, 'Predicted 2017':pred17})
    a=accuracy_score(pred16, y_test)
    b=accuracy_score(pred17, W)
    print(df17)
    print(df16)
    print("Accuracy of GNB classifier 2016: ", a)
    print("Accuracy of GNB classifier 2017: ", b)
    #compare accuracy
    plt.bar('Accuracy gaussiannb 16',a,color='red')
    plt.bar('Accuracy gaussiannb 17',b,color='blue')
    plt.title('Bar Graph 1 accuracy')
    plt.xlabel('MODELS accuracy')
    plt.ylabel('accuracy values')
    plt.show()
    #plot real vs predicted
    plt.plot(df17['Real 2017'],color = 'red' )
    plt.plot(df17['Predicted 2017'],color = 'blue')
    plt.xlabel('days',fontsize=18)
    plt.ylabel('season', fontsize=18)
    plt.title('Real season 2017 vs Predicted season 2017 GAUSSIANNB', fontsize=18)
    plt.show()
    
    
    print('/////////Support Vector Machine////////')
    #Support Vector Machine
    svm = SVC()
    #fit model
    svm.fit(X_train, y_train)
    #predict
    pred116 = gnb.predict(X_test)
    pred117 = gnb.predict(Z)
    #accuracy
    df116=pd.DataFrame({'Real 2016':y_test, 'Predicted 2016':pred116})
    df117=pd.DataFrame({'Real 2017':W, 'Predicted 2017':pred117})
    c=accuracy_score(pred116, y_test)
    d=accuracy_score(pred117, W)
    print(df117)
    print(df116)
    print("Accuracy of Support Vector Machine classifier 2016: ", c)
    print("Accuracy of Support Vector Machine classifier 2017: ", d)
    #compare accuracy
    plt.bar('Accuracy SVM 16',c,color='red')
    plt.bar('Accuracy SVM 17',d,color='blue')
    plt.title('Bar Graph 1 accuracy')
    plt.xlabel('MODELS accuracy')
    plt.ylabel('accuracy values')
    plt.show()
    #plot real vs predicted
    plt.plot(df117['Real 2017'],color = 'red' )
    plt.plot(df117['Predicted 2017'],color = 'blue')
    plt.xlabel('days',fontsize=18)
    plt.ylabel('season', fontsize=18)
    plt.title('Real season 2017 vs Predicted season 2017 Support Vector Machine', fontsize=18)
    plt.show()

#read dat as table
df16=pd.read_table('athens_09-16.dat',header=None, sep = ' ',names=['month','maxtemp','mintemp','heat','cool','rainmm','meanspeed','maxspeed'])
df17=pd.read_table('athens_2017.dat',header=None, sep = ' ',names=['month','maxtemp','mintemp','heat','cool','rainmm','meanspeed','maxspeed'])
#convert table to df
fin2016=pd.DataFrame(df16)
fin2017=pd.DataFrame(df17)
#check for null
print(fin2016.isnull().sum(),fin2017.isnull().sum())
#scatterplot and correlation
a_scatterplot(fin2016)
a_erotima(fin2016,fin2017)
b_erotima(fin2016,fin2017)
c_erotima(fin2016,fin2017)





