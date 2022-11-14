# importing required packages:
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder

########################################################################################################################
########################################################################################################################

final_df = pd.read_csv('./Chicago_dataset/Chicago_Crimes_2001_to_2016.csv')
total_df = final_df[['Block', 'Primary_Type', 'Description',
                      'Location_Description', 'Arrest', 'Domestic', 'Beat',
                      'District', 'Year', 'Latitude', 'Longitude', 'Month', 'Weekday']]

# Dropping rows for district, Latitude and Longitude NaN values
total_df = total_df.dropna(subset = ['District'])
total_df = total_df.dropna(subset = ['Latitude'])
total_df = total_df.dropna(subset = ['Longitude'])

Block_temp = []
Block_num = []
Block_street = []
temp_df = total_df[['Block']]
for index, row in temp_df.iterrows():
    Block_temp.append(row['Block'])

pattern = r'\((.+)'
pattern1 = "X (.+)"
pattern2 = "(.+)X "
pattern3 = r'\d+'

i=0
temp=[]
for data in Block_num:
    if data == 0:
        temp.append(i)
    i+=1

Block_temp1 = [s for s in Block_temp if s != 'XX  UNKNOWN'] #3176540,3186101
i=0
for data in Block_temp1:
    #print(data)
    Block_street.append(re.findall(pattern1, data)[0])
    Block_num.append(int(re.findall(pattern3, data)[0]))
    i+=1

# Get names of indexes for which column Block has value XX UNKNOWN
indexNames = total_df[ total_df['Block'] == 'XX  UNKNOWN'].index
# Delete these row indexes from dataFrame
total_df.drop(indexNames , inplace=True)

# Adding new Block features to the dataframe
total_df["Block_Street"] = Block_street
total_df["Block_Num"] = Block_num

# Encoding string values:
le = LabelEncoder()
Block_array = le.fit_transform(total_df['Block'])
Block_street_array = le.fit_transform(total_df['Block_Street'])
total_df = total_df.drop(columns=['Block', 'Block_Street'])
total_df['Block'] = Block_array
total_df['Block_Street'] = Block_street_array

total_df['Block_Num'] = total_df.Block_Num.astype(int)
total_df['Beat'] = total_df.Beat.astype(int)
total_df['District'] = total_df.District.astype(int)
# total_df['Ward'] = total_df.Ward.astype(int)
# total_df['Community_Area'] = total_df.Community_Area.astype(int)


# location_df = total_df[['Block', 'Block_Num', 'Block_Street', 'Beat', 'District', 'Ward', 'Community_Area', 'Latitude', 'Longitude']]
# imputer = KNNImputer(n_neighbors=5)
# temp = imputer.fit_transform(location_df)
# location_df1 = pd.DataFrame(imputer.fit_transform(location_df), columns=Latitude)
# location_df = pd.DataFrame(imputer.fit_transform(location_df), columns=Longitude)
# location_df = pd.DataFrame(imputer.fit_transform(location_df), columns=Ward)
# location_df = pd.DataFrame(imputer.fit_transform(location_df), columns=Community_Area)
# # Extracting list from total_df
# temp_df = total_df[['Ward', 'Community_Area', 'Latitude', 'Longitude']]
# for index, row in temp_df.iterrows():
#     Latitude.append(row['Latitude'])
#     Longitude.append(row['Longitude'])
#     Ward.append(row['Ward'])
#     Community_Area.append(row['Community_Area'])

# Encoding all remaining columns:
le = LabelEncoder()
Primary_Type_array = le.fit_transform(total_df['Primary_Type'])
Description_array = le.fit_transform(total_df['Description'])
Location_Description_array = le.fit_transform(total_df['Location_Description'])
Arrest_array = le.fit_transform(total_df['Arrest'])
Domestic_array = le.fit_transform(total_df['Domestic'])
Year_array = le.fit_transform(total_df['Year'])

total_df = total_df.drop(columns=['Primary_Type', 'Description', 'Location_Description', 'Arrest', 'Domestic'])
total_df = total_df.drop(columns=['Year'])
total_df['Primary_Type'] = Primary_Type_array
total_df['Description'] = Description_array
total_df['Location_Description'] = Location_Description_array
total_df['Arrest'] = Arrest_array
total_df['Domestic'] = Domestic_array
total_df['Year'] = Year_array

# Observing Maximum of each column:
print(total_df['Primary_Type'].max())
print(total_df['Description'].max())
print(total_df['Location_Description'].max())
print(total_df['Domestic'].max())

# Saving final dataframe:
total_df.to_csv('./Chicago_dataset/total_df.csv')

# Analyzing Arrest Skew:
indexNames = final_df[ final_df['Arrest'] == 'False'].index
indexNames = total_df[ total_df['Arrest'] == 1].index