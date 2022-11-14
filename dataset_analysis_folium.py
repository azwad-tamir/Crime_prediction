# -*- coding: utf-8 -*-
"""ChicagoMap.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iL2vj7AW17Pa0WrnWdXGrgvKayzTQGRu
"""

#Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime
import folium
import json
from folium.plugins import MarkerCluster

#Import Graph Libraries
from matplotlib import style
from collections import Counter

# Import PyDrive and associated libraries.
# This only needs to be done once per notebook.
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Download a file based on its file ID.
#
# A file ID looks like: laggVyWshwcyP6kEI-y_W3P8D26sz
file_id = 'REPLACE_WITH_YOUR_FILE_ID'
downloaded = drive.CreateFile({'id': file_id})
print('Downloaded content "{}"'.format(downloaded.GetContentString()))

# Import PyDrive and associated libraries.
# This only needs to be done once per notebook.
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Download a file based on its file ID.
#
# A file ID looks like: laggVyWshwcyP6kEI-y_W3P8D26sz
file_id = 'REPLACE_WITH_YOUR_FILE_ID'
downloaded = drive.CreateFile({'id': file_id})
print('Downloaded content "{}"'.format(downloaded.GetContentString()))

# Import PyDrive and associated libraries.
# This only needs to be done once per notebook.
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Download a file based on its file ID.
#
# A file ID looks like: laggVyWshwcyP6kEI-y_W3P8D26sz
file_id = 'REPLACE_WITH_YOUR_FILE_ID'
downloaded = drive.CreateFile({'id': file_id})
print('Downloaded content "{}"'.format(downloaded.GetContentString()))

#Load Data
df = pd.read_csv('drive/My Drive/CAP5610/Group Project/Data/total_df.csv', error_bad_lines=False)
df = df.drop(['Unnamed: 0'], axis=1)

print(df.columns)

X = df.drop(['Arrest'], axis=1)
Y = df['Arrest']

BBox = ((-87.9846, -87.5050, 41.6165, 42.0538)) # These are the bounds of Chicago's area

print(df.Latitude.max())
print(df.Latitude.min())
print(df.Longitude.max())
print(df.Longitude.min())

df = df[df.Latitude > BBox[2]]

print()
print(df.Latitude.max())
print(df.Latitude.min())
print(df.Longitude.max())
print(df.Longitude.min())

def color_picker(val):
  if val == 1:
    return '#00ffff'
  else:
    return '#FF0000'

def get_arrest_rate(total, count):
  return float((count / total) * 100.0)

df['Arrest Rate'] = 0
df['Number of Incidents'] = 0

for dist in np.unique(df['District']):
  district_data = df[df['District'] == dist]
  total_arrests = district_data[district_data['Arrest'] == 1]

  df.loc[(df.District == dist),'Arrest Rate'] = get_arrest_rate(len(district_data), len(total_arrests))
  df.loc[(df.District == dist),'Number of Incidents'] = len(district_data)

map = folium.Map(location=[41.8500300,-87.6500500], zoom_start=11.4, tiles='cartodbdark_matter')

#Setup the plugins
map.add_child(folium.plugins.MiniMap(toggle_display=True))
folium.plugins.Fullscreen(position="topright").add_to(map)

geo_data = r'drive/My Drive/CAP5610/Group Project/Data/Boundaries - Police Districts (current).geojson'

#Geo District display Simple
#folium.GeoJson(geo_data).add_to(map)

folium.Choropleth(
    geo_data = geo_data,
    name = 'Arrest Rate Per District',
    data = df,
    columns = ['District', 'Arrest Rate'],
    legend_name = 'Arrest Rate',
    highlight = True,
    key_on = 'feature.properties.dist_num',
    fill_color = 'YlGnBu',
    fill_opacity = 0.6,
    line_color = 'Black',
    line_opacity = 0.8,
    nan_fill_color = 'grey'
).add_to(map)

folium.LayerControl().add_to(map)

map

incident_map = folium.Map(location=[41.8500300,-87.6500500], zoom_start=11.4, tiles='cartodbdark_matter')

#Setup the plugins
map.add_child(folium.plugins.MiniMap(toggle_display=True))
folium.plugins.Fullscreen(position="topright").add_to(incident_map)

geo_data = r'drive/My Drive/CAP5610/Group Project/Data/Boundaries - Police Districts (current).geojson'

#Geo District display Simple
#folium.GeoJson(geo_data).add_to(map)

folium.Choropleth(
    geo_data = geo_data,
    name = 'Number of Incidents Per District',
    data = df,
    columns = ['District', 'Number of Incidents'],
    legend_name = 'Number of Incidents',
    highlight = True,
    key_on = 'feature.properties.dist_num',
    fill_color = 'YlGnBu',
    fill_opacity = 0.6,
    line_color = 'Black',
    line_opacity = 0.8,
    nan_fill_color = 'grey'
).add_to(incident_map)

folium.LayerControl().add_to(incident_map)

incident_map

cluster_map = folium.Map(location=[41.8500300,-87.6500500], zoom_start=11.4, tiles='cartodbdark_matter')

#Geo District display Simple
folium.GeoJson(geo_data).add_to(cluster_map)

for dist in np.unique(df['District']):
  district_data = df[df['District']==dist]

  # Clustering the Markers
  marker_cluster = MarkerCluster().add_to(cluster_map)

  for idx, row in district_data[0:1000].iterrows():
    folium.Marker(location=[row['Latitude'], row['Longitude']],icon=folium.Icon(color='black',icon_color=color_picker(row['Arrest']))).add_to(marker_cluster)

  # Plots circles
  # for idx, row in district_data[0:100].iterrows():
  #   folium.CircleMarker(location=[row['Latitude'], row['Longitude']],
  #                       radius=5,
  #                       fill=True,
  #                       fill_color=color_picker(row['Arrest']),
  #                       color=color_picker(row['Arrest']),
  #                       fill_opacity=0.7).add_to(map)

  # Plots Markers
  # for idx, row in year[0:100].iterrows():
  #   folium.Marker(location=[row['Latitude'], row['Longitude']],icon=folium.Icon(color='black',icon_color=color_picker(row['Arrest']))).add_to(m)

folium.LayerControl().add_to(cluster_map)

cluster_map

heat_map = folium.Map(location=[41.8500300,-87.6500500], zoom_start=11.4, tiles='cartodbdark_matter')

crimeArr = []

for data in df[df.Year == 0][0:10000].itertuples():
  crimeArr.append([data.Latitude, data.Longitude])

# plot heatmap
heat_map.add_child(folium.plugins.HeatMap(crimeArr))

heat_map

heat_map = folium.Map(location=[41.8500300,-87.6500500], zoom_start=11.4, tiles='cartodbdark_matter')

# dt = datetime.date(2019, 10, 20)

# yearArr = np.unique(df.Year)
# monthArr = np.unique(df.Month)

# dates = []

# for year in yearArr:
#   for month in monthArr:
#     dates.append(datetime.date(2000 + year + 1, month, 1))

# heat_dates = [dt.strftime('%Y-%m-%d') for dt in dates]

crimeArr = []

for data in df[df.Year == 15][0:10000].itertuples():
  crimeArr.append([data.Latitude, data.Longitude])

# plot heatmap
#heat_map.add_child(folium.plugins.HeatMapWithTime(crimeArr, index=heat_dates))
heat_map.add_child(folium.plugins.HeatMap(crimeArr))

heat_map