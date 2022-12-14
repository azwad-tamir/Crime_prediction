# -*- coding: utf-8 -*-
"""dataset_preprocess_part_1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JRzw8fkuqSE5IBR0kW-gQqteMC0Zv36C

##Import NumPy, Pandas
"""

import numpy as np
import pandas as pd

from google.colab import files
from google.colab import drive

"""##Download and extract the Chicago Crime dataset"""

drive.mount('/content/drive')

crimes1 = pd.read_csv('drive/My Drive/Colab Notebooks/Chicago_Crimes_2001_to_2004.csv', 
                      error_bad_lines=False)
crimes2 = pd.read_csv('drive/My Drive/Colab Notebooks/Chicago_Crimes_2005_to_2007.csv', 
                      error_bad_lines=False)
crimes3 = pd.read_csv('drive/My Drive/Colab Notebooks/Chicago_Crimes_2008_to_2011.csv', 
                      error_bad_lines=False)
crimes4 = pd.read_csv('drive/My Drive/Colab Notebooks/Chicago_Crimes_2012_to_2017.csv', 
                      error_bad_lines=False)

crimes = pd.concat([crimes1, crimes2, crimes3, crimes4])

del crimes1
del crimes2
del crimes3
del crimes4

"""##Drop the duplicates of the dataset"""

print('Before Dropping Duplicates: ' + str(crimes.shape))
crimes.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)
print('After Dropping Duplicates: ' + str(crimes.shape))

"""##Drop the features that won't be used"""

crimes.drop(columns=['Unnamed: 0', 'ID', 'Case Number', 'IUCR', 'FBI Code', 
                     'Location', 'X Coordinate', 'Y Coordinate', 'Updated On'], 
            axis=1, inplace=True, errors='ignore')
crimes.head(10)

"""##Display the columns that have null values"""

crimes.isnull().sum(axis=0)

"""##Drop the null values in the column of 'Location Description'"""

crimes.dropna(subset=['Location Description'], inplace=True)
crimes.isnull().sum(axis=0)

"""##Convert date to match format for Pandas and create an index using the 'Date' feature"""

crimes['Date'] = pd.to_datetime(crimes['Date'], format='%m/%d/%Y %I:%M:%S %p')
crimes.index = pd.DatetimeIndex(crimes['Date'])

"""##Add the features of 'Month' and 'Weekday' using the date index"""

crimes['Month'] = crimes.index.month.astype(int)
crimes['Weekday'] = crimes.index.weekday.astype(int)
crimes.head(10)

"""##Display the Date, Primary Type, Location Description, Month, and Weekday Features"""

crimes_date = crimes[['Primary Type', 'Location Description', 'Year', 'Month', 'Weekday']]
crimes_date.head(5)

"""##Drop the 'Date' feature"""

crimes.drop(columns=['Date'], axis=1, inplace=True, errors='ignore')
crimes.head(10)

"""##Display the value counts for the feature of 'Year'"""

value_counts = crimes['Year'].value_counts()
print(value_counts)

"""##Drop the 'Year' feature value of 2017"""

crimes.drop(crimes[crimes['Year'] == 2017].index, inplace=True, errors='ignore')
value_counts = crimes['Year'].value_counts()
print(value_counts)

"""##Display the value counts for feature of 'Primary Type'"""

value_counts = crimes['Primary Type'].value_counts()
print(value_counts)

"""##Drop the 'Primary Type' categories that have a value count less than 1000"""

value_counts = crimes['Primary Type'].value_counts()
remove_values = value_counts[value_counts < 1000].index
crimes_final = crimes[~crimes['Primary Type'].isin(remove_values)]

"""##Display the value counts after dropping the 'Primary Type' categories that have a value count less than 1000"""

value_counts = crimes_final['Primary Type'].value_counts()
print(value_counts)

"""##Display Primary Type of Domestic Crimes"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

domestic = crimes[crimes['Domestic'] == True]

plt.figure(figsize=(8, 8))
domestic.groupby([domestic['Primary Type']]).size().sort_values(ascending=False)[:15].plot(kind='barh')
plt.xlabel('Number of Crimes')
plt.ylabel('Primary Type')
plt.show()

"""##Create the new .csv file from the pre-processed dataset"""

file_name = 'Chicago_Crimes_2001_to_2016.csv'
crimes_final.to_csv(file_name, encoding='utf-8', index=False)
files.download(file_name)