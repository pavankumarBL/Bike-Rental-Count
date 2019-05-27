#!/usr/bin/env python
# coding: utf-8

# In[64]:


#Importing all necessary libraries 
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.model_selection import train_test_split,RandomizedSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor


# In[2]:


os.getcwd()


# In[4]:


os.chdir("C:\\Users\pavankumar.bl\\Documents\\datascience\\Edwisor\\Project_3\\Training_data")


# # Read Data

# In[5]:


data_df=pd.read_csv('day.csv')


# In[6]:


data_df.head(5)


# In[7]:


data_df.dtypes


# In[8]:


data_df.describe(include='all')


# # Feature Engineering

# In[9]:


data_df['dteday'] = pd.to_datetime(data_df['dteday'])
data_df['day']=data_df['dteday'].dt.day


# In[10]:


#removing the Date variable since we have extracted the day from it which is needed for analysis.
data_df=data_df.drop('dteday',axis=1)


# In[11]:


#converting the attributes to valid datatype
data_df['season'] = data_df.season.astype('category')
data_df['yr'] = data_df.yr.astype('category')
data_df['mnth'] = data_df.mnth.astype('category')
data_df['weekday'] = data_df.weekday.astype('category')
data_df['holiday'] = data_df.holiday.astype('category')
data_df['workingday'] = data_df.workingday.astype('category')
data_df['weathersit'] = data_df.weathersit.astype('category')


# In[12]:


data_df.dtypes


# In[13]:


#calculating number of unique values for all df columns
data_df.nunique()


# # Missing Values

# In[14]:


#checking if there any Missing Values
data_df.isnull().sum()


# # Exploratory Analysis

# # Number Summary of the Bike Rental Count 'cnt' Feature

# In[15]:


fig, ax = plt.subplots(1)
ax.plot(sorted(data_df['cnt']), color = 'blue', marker = '*', label='cnt')
ax.legend(loc= 'upper left')
ax.set_ylabel('Sorted Rental Counts', fontsize = 10)
fig.suptitle('Recorded Bike Rental Counts', fontsize = 10)


# # Quantitative Features vs. Rental Counts

# In[16]:


plt.scatter(data_df['temp'], data_df['cnt'])
plt.suptitle('Numerical Feature: Cnt v/s temp')
plt.xlabel('temp')
plt.ylabel('Count of all Biks Rented')


# In[17]:


#from above plot we can get to know as the temperature increase rented bike count also increased.


# In[18]:


plt.scatter(data_df['atemp'], data_df['cnt'])
plt.suptitle('Numerical Feature: Cnt v/s atemp')
plt.xlabel('atemp')
plt.ylabel('Count of all Biks Rented')


# In[19]:


#from above plot atemp vs cnt we can say that both looks similar and chances of Multicollinearity .
#let we will check correlation and then we can decide.


# In[20]:


plt.scatter(data_df['hum'], data_df['cnt'])
plt.suptitle('Numerical Feature: Cnt v/s hum')
plt.xlabel('hum')
plt.ylabel('Count of all Biks Rented')


# In[21]:


plt.scatter(data_df['windspeed'], data_df['cnt'])
plt.suptitle('Numerical Feature: Cnt v/s windspeed')
plt.xlabel('windspeed')
plt.ylabel('Count of all Biks Rented')


# In[22]:


# feature 'windspeed' shows inverse relationship with rentals


# # Lets Explore on Categorical Veriable

# In[23]:


f,  (ax1, ax2)  =  plt.subplots(nrows=1, ncols=2, figsize=(13, 6))

ax1 = data_df[['season','cnt']].groupby(['season']).sum().reset_index().plot(kind='bar',
                                       legend = False, title ="Counts of Bike Rentals by season",color='Green',
                                         stacked=True, fontsize=12, ax=ax1)
ax1.set_xlabel("season", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)
ax1.set_xticklabels(['spring','summer','fall','winter'])

 
ax2 = data_df[['weathersit','cnt']].groupby(['weathersit']).sum().reset_index().plot(kind='bar',  
      legend = False, stacked=True, title ="Counts of Bike Rentals by weathersit", fontsize=12, ax=ax2)

ax2.set_xlabel("weathersit", fontsize=12)
ax2.set_ylabel("Count", fontsize=12)
ax2.set_xticklabels(['1: Clear','2: Mist','3: Light Snow','4: Heavy Rain'])

f.tight_layout()


# In[24]:


#its observerd from above plot that Bike rent count is high in Fall season and clear weather


# In[25]:


ax = data_df[['day','cnt']].groupby(['day']).sum().reset_index().plot(kind='bar', figsize=(8, 6),
                                       legend = False, title ="Total Bike Rentals by day", 
                                       color='orange', fontsize=12)
ax.set_xlabel("day", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
plt.show()


# In[26]:


fig,(ax1,ax2,ax3)= plt.subplots(nrows=3)
fig.set_size_inches(12,20)
hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]


ax1 = data_df[['mnth','cnt']].groupby(['mnth']).mean().reset_index().plot(kind='bar',
                                        title ="Counts of Bike Rentals by month",
                                         stacked=True, fontsize=12, ax=ax1)
ax1.set_xlabel("season", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)
ax1.set_xticklabels(["January","February","March","April","May","June","July","August","September","October","November","December"])

dayAggregated = pd.DataFrame(data_df.groupby(["day","season"],sort=True)["cnt"].mean()).reset_index()
sns.pointplot(x=dayAggregated["day"], y=dayAggregated["cnt"],hue=dayAggregated["season"], data=dayAggregated, join=True,ax=ax2)
ax2.set(xlabel='Day', ylabel='Users Count',title="Average Users Count By day  Across Season",label='big')


hourAggregated = pd.DataFrame(data_df.groupby(["day","weekday"],sort=True)["cnt"].mean()).reset_index()
sns.pointplot(x=hourAggregated["day"], y=hourAggregated["cnt"],hue=hourAggregated["weekday"], data=hourAggregated, ax=ax3)
ax3.set(xlabel='The Day', ylabel='Users Count',title="Average Users Count By The Day Across Weekdays",label='big')


# In[27]:


#holiday
data_df.holiday.value_counts()
sns.catplot(x='holiday',data=data_df,kind='count',height=5,aspect=1) # majority of data is for non holiday days.


# In[28]:


# obtaining the List of Variables that are Continuous and Categorical
continuous = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']

categorical = ['season','yr','mnth',
                     'holiday','weekday', 'workingday', 'weathersit']


# In[29]:


bike_piplot=data_df[categorical]
plt.figure(figsize=(15,12))
plt.suptitle('pie distribution of categorical features', fontsize=20)
for i in range(1,bike_piplot.shape[1]+1):
    plt.subplot(3,3,i)
    f=plt.gca()
    f.set_title(bike_piplot.columns.values[i-1])
    values=bike_piplot.iloc[:,i-1].value_counts(normalize=True).values
    index=bike_piplot.iloc[:,i-1].value_counts(normalize=True).index
    plt.pie(values,labels=index,autopct='%1.1f%%')
#plt.tight_layout()


# In[30]:


#What we can infer from above piplot:
#Most of the categorical variables are uniformally distributed, except 'holiday','weathersit','workingday'
# This makes sense for 'weathersit', as extreme weather is rare and hence %percentage of extreme weather in whole dataset is low
# This makes sense for 'holiday', as number of holidays are less in comparison to working days
# This makes sense for 'workingday' for the same reason as above
# So, categorical data seems o be pretty much uniformly distributed


# In[31]:


#graph individual categorical features by count
fig, saxis = plt.subplots(3, 3,figsize=(16,12))

sns.barplot(x = 'season', y = 'cnt',hue= 'yr', data=data_df, ax = saxis[0,0], palette ="Blues_d")
sns.barplot(x = 'yr', y = 'cnt', order=[0,1,2,3], data=data_df, ax = saxis[0,1], palette ="Blues_d")
sns.barplot(x = 'mnth', y = 'cnt', data=data_df, ax = saxis[0,2])
sns.barplot(x = 'holiday', y = 'cnt',  data=data_df, ax = saxis[1,0])
sns.barplot(x = 'weekday', y = 'cnt',  data=data_df, ax = saxis[1,1])
sns.barplot(x = 'workingday', y = 'cnt', data=data_df, ax = saxis[1,2])
#sns.barplot(x = 'weather', y = 'cnt', data=data_df, ax = saxis[2,0])
sns.barplot(x = 'day', y = 'cnt' , data=data_df, ax = saxis[2,1])
sns.pointplot(x = 'weathersit', y = 'cnt', data=data_df, ax = saxis[2,0])
sns.pointplot(x='day', y='cnt', hue='yr', data=data_df, markers='o', linestyles='-', dodge=False, join=True,ax = saxis[2,2])
#sn.pointplot()


# # Outlier Analysis

# In[32]:


## -- Lets do the outlier analysis ----
## -- Visualize continous variables(cnt,temp,atemp,humidity,windspeed) and 
##  count with respect to categorical variables("season", "yr","mnth","holiday","weekday","workingday","weathersit","date")with boxplots ---
fig, axes = plt.subplots(nrows=3,ncols=4)
fig.set_size_inches(20,15)

#-- Plot total counts on y bar
sns.boxplot(data=data_df, y="cnt",ax=axes[0][0])

#-- Plot temp on y bar
sns.boxplot(data=data_df, y="temp",ax=axes[0][1])

#-- Plot atemp on y bar
sns.boxplot(data=data_df, y="atemp",ax=axes[0][2])

#-- Plot hum on y bar
sns.boxplot(data=data_df, y="hum",ax=axes[0][3])

#-- Plot windspeed on y bar
sns.boxplot(data=data_df, y="windspeed",ax=axes[1][0])

#-- Plot total counts on y-bar and 'yr' on x-bar
sns.boxplot(data=data_df,y="cnt",x="yr",ax=axes[1][1])

#-- Plot total counts on y-bar and 'mnth' on x-bar
sns.boxplot(data=data_df,y="cnt",x="mnth",ax=axes[1][2])

#-- Plot total counts on y-bar and 'date' on x-bar
sns.boxplot(data=data_df,y="cnt",x="day",ax=axes[1][3])

#-- Plot total counts on y-bar and 'season' on x-bar
sns.boxplot(data=data_df,y="cnt",x="season",ax=axes[2][0])

#-- Plot total counts on y-bar and 'weekday' on x-bar
sns.boxplot(data=data_df,y="cnt",x="weekday",ax=axes[2][1])

#-- Plot total counts on y-bar and 'workingday' on x-bar
sns.boxplot(data=data_df,y="cnt",x="workingday",ax=axes[2][2])

#-- Plot total counts on y-bar and 'weathersit' on x-bar
sns.boxplot(data=data_df,y="cnt",x="weathersit",ax=axes[2][3])


# In[33]:


# just to visualize.
sns.boxplot(data=data_df[['temp',
       'atemp', 'hum', 'windspeed']])
fig=plt.gcf()
fig.set_size_inches(10,10)


# In[34]:


# just to visualize.
sns.boxplot(data=data_df[['casual', 'registered', 'cnt']])
fig=plt.gcf()
fig.set_size_inches(10,10)


# In[35]:


#Outliers Treatment


# In[36]:


# Getting 75 and 25 percentile of variable "windspeed"
q75, q25 = np.percentile(data_df['windspeed'], [75,25])
# Calculating Interquartile range
iqr = q75 - q25
    
# Calculating upper extream and lower extream
minimum_wind = q25 - (iqr*1.5)
maximum_wind = q75 + (iqr*1.5)
    
# Replacing all the outliers value to NA
data_df.loc[data_df['windspeed']< minimum_wind,'windspeed'] = minimum_wind
data_df.loc[data_df['windspeed']> maximum_wind,'windspeed'] = maximum_wind
#-------------------------------##------------------------------------------##-----------#
# Getting 75 and 25 percentile of variable "Humidity"
q75, q25 = np.percentile(data_df['hum'], [75,25])
# Calculating Interquartile range
iqr = q75 - q25
    
# Calculating upper extream and lower extream
minimum_hum = q25 - (iqr*1.5)
maximum_hum = q75 + (iqr*1.5)
    
# Replacing all the outliers value to NA
data_df.loc[data_df['hum']< minimum_hum,'hum'] = minimum_hum
data_df.loc[data_df['hum']> maximum_hum,'hum'] = maximum_hum
##-----------------------------------##---------------------------------##----------------#

# Getting 75 and 25 percentile of variable "Casual"
q75, q25 = np.percentile(data_df['casual'], [75,25])
# Calculating Interquartile range
iqr = q75 - q25
    
# Calculating upper extream and lower extream
minimum_cas = q25 - (iqr*1.5)
maximum_cas = q75 + (iqr*1.5)
    
# Replacing all the outliers value to NA
data_df.loc[data_df['casual']< minimum_cas,'casual'] = minimum_cas
data_df.loc[data_df['casual']> maximum_cas,'casual'] = maximum_cas


# # Feature Selection

# In[37]:


#Code for plotting pairplot
sns_plot = sns.pairplot(data=data_df[continuous])
plt.plot()


# In[38]:


#implementing VIF


# In[39]:


class ReduceVIF(BaseEstimator, TransformerMixin):
    def __init__(self, thresh=10.0, impute=True, impute_strategy='median'):
        # From looking at documentation, values between 5 and 10 are "okay".
        # Above 10 is too high and so should be removed.
        self.thresh = thresh
        
        # The statsmodel function will fail with NaN values, as such we have to impute them.
        # By default we impute using the median value.
        # This imputation could be taken out and added as part of an sklearn Pipeline.
        if impute:
            self.imputer = SimpleImputer(strategy=impute_strategy)

    def fit(self, X, y=None):
        print('ReduceVIF fit')
        if hasattr(self, 'imputer'):
            self.imputer.fit(X)
        return self

    def transform(self, X, y=None):
        print('ReduceVIF transform')
        columns = X.columns.tolist()
        if hasattr(self, 'imputer'):
            X = pd.DataFrame(self.imputer.transform(X), columns=columns)
        return ReduceVIF.calculate_vif(X, self.thresh)

    @staticmethod
    def calculate_vif(X, thresh=10.0):
        # Taken from https://stats.stackexchange.com/a/253620/53565 and modified
        dropped=True
        while dropped:
            variables = X.columns
            dropped = False
            vif = [variance_inflation_factor(X[variables].values, X.columns.get_loc(var)) for var in X.columns]
            
            max_vif = max(vif)
            if max_vif > thresh:
                maxloc = vif.index(max_vif)
                print(f'Dropping {X.columns[maxloc]} with vif={max_vif}')
                X = X.drop([X.columns.tolist()[maxloc]], axis=1)
                dropped=True
        return X


# In[40]:


test_data=data_df.copy()


# In[41]:


y = test_data.pop('cnt')


# In[42]:


continuous1 = ['temp', 'atemp', 'hum', 'windspeed']


# In[43]:


transformer = ReduceVIF()

# Only use 10 columns for speed in this example
test_data = transformer.fit_transform(test_data[continuous1], y)


# In[44]:


#From VIF test on continous variable reveals we can remove the column atemp 


# # Method 2 Correlation Analysis

# In[45]:


##Correlation analysis for continuous variables
#Correlation plot
data_corr = data_df.loc[:,continuous]

#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(10, 10))

#Generate correlation matrix
corr = data_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 50, as_cmap=True),
            square=True, ax=ax, annot = True)
plt.plot()


# # Anova Test for Correlation check on Categorical variables

# In[46]:


#Initializing the Variables
label = 'cnt'
obj_dtype = categorical
drop_feat = []

## ANOVA TEST FOR P VALUES
import statsmodels.api as sm
from statsmodels.formula.api import ols

anova_p = []
for  i in obj_dtype:
    buf = label + ' ~ ' + i
    mod = ols(buf,data=data_df).fit()
    anova_op = sm.stats.anova_lm(mod, typ=2)
    print(anova_op)
    anova_p.append(anova_op.iloc[0:1,3:4])
    p = anova_op.loc[i,'PR(>F)']
    if p >= 0.05:
        drop_feat.append(i)


# In[47]:


drop_feat


# In[48]:


#As a result of correlation analysis and ANOVA, we have concluded that we should remove 6 columns
#Based on VIF test 'temp' and 'atemp' are correlated and hence atemp of them should be removed
#'holiday', 'weekday' and 'workingday' have p>0.05 and hence should be removed


# In[49]:


# Droping the variables which has redundant information
to_drop = ['atemp', 'holiday', 'weekday', 'workingday','instant']
data_df = data_df.drop(to_drop, axis = 1)


# In[50]:


data_df.info()


# # Feature Scaling

# In[51]:


# Updating the Continuous and Categorical Variables after droping correlated variables
continuous = [i for i in continuous if i not in to_drop]
categorical = [i for i in categorical if i not in to_drop]


# In[52]:


# Checking the distribution of values for variables in data_df

for i in continuous:
    if i == 'data_df':
        continue
    plt.figure(figsize=(10,4))
    ax=sns.distplot(data_df[i],bins = 'auto')
    plt.title("Checking Distribution for Variable "+str(i))
    plt.ylabel("Density")
    plt.savefig('{i}_Vs_Density.png'.format(i=i))
    plt.show()


# In[53]:


#Data before scaling
data_df.head()


# In[54]:


# Since our data is normally distributed, we will use Standardization for Feature Scalling
# #Standardization
for i in continuous:
    if i == 'cnt':
        continue
    data_df[i] = (data_df[i] - data_df[i].mean())/(data_df[i].std())


# In[55]:


#Data after scaling
data_df.head()


# In[56]:


#Before going for modelling algorithms, we will create dummy variables for our categorical variables


# In[57]:


dummy_data = pd.get_dummies(data = data_df, columns = categorical)

#Copying dataframe
bike_data = dummy_data.copy()


# In[58]:


bike_data.head()


# # Machine Learning algorithms

# In[61]:


#Using train test split functionality for creating sampling
X_train, X_test, y_train, y_test = train_test_split(bike_data.iloc[:, bike_data.columns != 'cnt'], 
                         bike_data.iloc[:, 3], test_size = 0.33, random_state=101)


# In[62]:


(X_train.shape),(y_train.shape)


# # Decision Tree Regressor

# In[65]:


# Building model on top of training dataset
fit_DT = DecisionTreeRegressor(max_depth = 2).fit(X_train,y_train)

# Calculating RMSE for test data to check accuracy
pred_test = fit_DT.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,pred_test))

def MAPE(y_true,y_pred):
    mape = np.mean(np.abs((y_true-y_pred)/y_true))*100
    return mape

DT_rmse = rmse_for_test
DT_mape = MAPE(y_test,pred_test)
DT_r2 = r2_score(y_test,pred_test)

print('Decision Tree Regressor Model Performance:')
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,pred_test)))
print("MAPE(Mean Absolute Percentage Error) = "+str(DT_mape))


# # Random Forest

# In[67]:


# Building model on top of training dataset
fit_RF = RandomForestRegressor(n_estimators = 500).fit(X_train,y_train)

# Calculating RMSE for test data to check accuracy
pred_test = fit_RF.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,pred_test))

RF_rmse = rmse_for_test
RF_mape = MAPE(y_test,pred_test)
RF_r2 = r2_score(y_test,pred_test)

print('Random Forest Regressor Model Performance:')
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,pred_test)))
print("MAPE(Mean Absolute Percentage Error) = "+str(RF_mape))


# # Linear Regression

# In[68]:


# Building model on top of training dataset
fit_LR = LinearRegression().fit(X_train , y_train)

# Calculating RMSE for test data to check accuracy
pred_test = fit_LR.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,pred_test))

LR_rmse = rmse_for_test
LR_mape = MAPE(y_test,pred_test)
LR_r2 = r2_score(y_test,pred_test)

print('Linear Regression Model Performance:')
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,pred_test)))
print("MAPE(Mean Absolute Percentage Error) = "+str(LR_mape))


# # Gradient Boosting Regressor

# In[69]:


# Building model on top of training dataset
fit_GB = GradientBoostingRegressor().fit(X_train, y_train)

# Calculating RMSE for test data to check accuracy
pred_test = fit_GB.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,pred_test))

GBR_rmse = rmse_for_test
GBR_mape = MAPE(y_test,pred_test)
GBR_r2 = r2_score(y_test,pred_test)

print('Gradient Boosting Regressor Model Performance:')
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,pred_test)))
print("MAPE(Mean Absolute Percentage Error) = "+str(GBR_mape))


# In[70]:


dat = {'Model_name': ['Decision tree default', 'Random Forest Default', 'Linear Regression',
                   'Gradient Boosting Default'], 
          'RMSE': [DT_rmse, RF_rmse, LR_rmse, GBR_rmse], 
         'MAPE':[DT_mape, RF_mape, LR_mape, GBR_mape],
        'R^2':[DT_r2, RF_r2, LR_r2, GBR_r2]}
results = pd.DataFrame(data=dat)


# In[71]:


results


# In[ ]:




