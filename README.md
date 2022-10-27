# Ex-06-Feature-Transformation

## AIM
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM
### STEP 1
Read the given Data

### STEP 2
Clean the Data Set using Data Cleaning Process

### STEP 3
Apply Feature Transformation techniques to all the features of the data set

### STEP 4
Save the data to the file

## CODE
```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.head()

df.isnull().sum()

df.info()

df.describe()

df1 = df.copy()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()


``` 

# OUPUT
### Dataset:
![o](https://github.com/dharanielango/Ex-06-Feature-Transformation/blob/main/1.png)
### Head:
![o](https://github.com/dharanielango/Ex-06-Feature-Transformation/blob/main/2.png)
### Null data:
![o](https://github.com/dharanielango/Ex-06-Feature-Transformation/blob/main/3.png)
### Information:
![o](https://github.com/dharanielango/Ex-06-Feature-Transformation/blob/main/4.png)
### Description:
![o](https://github.com/dharanielango/Ex-06-Feature-Transformation/blob/main/5.png)
### Highly Positive Skew:
![o](https://github.com/dharanielango/Ex-06-Feature-Transformation/blob/main/6.png)
### Highly Negative Skew:
![o](https://github.com/dharanielango/Ex-06-Feature-Transformation/blob/main/7.png)
### Moderate Positive Skew:
![o](https://github.com/dharanielango/Ex-06-Feature-Transformation/blob/main/8.png)
### Moderate Negative Skew:
![o](https://github.com/dharanielango/Ex-06-Feature-Transformation/blob/main/9.png)
### Log of Highly Positive Skew:
![o](https://github.com/dharanielango/Ex-06-Feature-Transformation/blob/main/10.png)
### Log of Moderate Positive Skew:
![o](https://github.com/dharanielango/Ex-06-Feature-Transformation/blob/main/11.png)
### Reciprocal of Highly Positive Skew:
![o](https://github.com/dharanielango/Ex-06-Feature-Transformation/blob/main/12.png)
### Square root tranformation:
![o](https://github.com/dharanielango/Ex-06-Feature-Transformation/blob/main/13.png)
### Power transformation of Moderate Positive Skew:
![o](https://github.com/dharanielango/Ex-06-Feature-Transformation/blob/main/14.png)
### Power transformation of Moderate Negative Skew:
![o](https://github.com/dharanielango/Ex-06-Feature-Transformation/blob/main/15.png)
### Quantile transformation:
![o](https://github.com/dharanielango/Ex-06-Feature-Transformation/blob/main/16.png)

# Result
Thus, Feature transformation is performed and executed successfully for the given dataset
