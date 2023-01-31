import streamlit as st

import pandas as pd
import numpy as np
import re

import geopandas as gpd
from streamlit_folium import folium_static

from helpers import plot_map

import warnings

warnings.simplefilter('ignore', FutureWarning)

st.title('The Right Price')

IDF_sample = pd.read_csv('data_right_price/Data localisée/sample_mutations_IDF_train_localized.csv')

geo_IDF_sample = gpd.GeoDataFrame(
    IDF_sample, geometry=gpd.points_from_xy(IDF_sample.longitude, IDF_sample.latitude))

# Parameters
st.subheader('Map of all mutations')

dept_slider = st.selectbox(
    'Select a department',
    (None, 75, 77, 78, 91, 92, 93, 94, 95))

insee_list = IDF_sample[IDF_sample['Dept'] == dept_slider].l_codinsee.apply(lambda x: int(re.findall(r'[0-9]+', x)[0])).unique().tolist() if dept_slider is not None else IDF_sample.l_codinsee.apply(lambda x: int(re.findall(r'[0-9]+', x)[0])).unique().tolist()
insee_list.sort()
insee_slider = st.selectbox(
    'Select a municipality',
    [None] + insee_list)

section_list = IDF_sample[IDF_sample['Dept'] == dept_slider].l_section.apply(lambda x: re.findall(r'[0-9A-Z]+', x)[0]).unique().tolist() if dept_slider is not None else IDF_sample.l_section.apply(lambda x: re.findall(r'[0-9A-Z]+', x)[0]).unique().tolist()
section_list.sort()
section_slider = st.selectbox(
    'Select a section',
    [None] + section_list)

property_list = IDF_sample[IDF_sample['Dept'] == dept_slider].libtypbien.unique().tolist() if dept_slider is not None else IDF_sample.libtypbien.unique().tolist()
property_slider = st.selectbox(
    'Select a type of property',
    [None] + property_list)


i = IDF_sample['valeurfonc'].min()
a = 10**(len(str(int(i)))-1)
min_value = a*np.floor(i/a) if i > 1 else 0.0
j = IDF_sample['valeurfonc'].max()
b = 10**(len(str(int(j)))-1)
max_value = b*np.ceil(j/b) if j > 1 else 0.0

luxury = st.checkbox('Check this box to see the most expansive properties')

if luxury:
    min_value_slider, max_value_slider = st.select_slider(
        'Select a property value', 
        np.arange(1500000, max_value+1, 1000), 
        value=(1500000, max_value))
else:
    min_value_slider, max_value_slider = st.select_slider(
        'Select a property value', 
        np.arange(min_value, 1500001, 1000), 
        value=(min_value, 1500000))
    
    
k = IDF_sample['sbati'].min()
c = 10**(len(str(int(k)))-1)
min_surface = c*np.floor(k/c) if k > 1 else 0.0
l = IDF_sample['sbati'].max()
d = 10**(len(str(int(l)))-1)
max_surface = d*np.ceil(l/d) if l > 1 else 0.0

if luxury:
    min_surface_slider, max_surface_slider = st.select_slider(
        'Select a surface', 
        np.arange(min_surface, max_surface+1, 10), 
        value=(min_surface, max_surface))
else:
    min_surface_slider, max_surface_slider = st.select_slider(
        'Select a surface', 
        np.arange(min_surface, 1501, 10), 
        value=(min_surface, 1500))



year_list = IDF_sample['year'].unique().tolist()
year_list.sort()
year_slider = st.selectbox(
    'Select a year',
    [None] + year_list)


folium_map = plot_map(geo_IDF_sample, 
                      dept=dept_slider, 
                      insee=insee_slider, 
                      section=section_slider, 
                      type_of_property=property_slider, 
                      year_of_mutation=year_slider, 
                      min_value=min_value_slider, max_value=max_value_slider,
                      min_surface=min_surface_slider, max_surface=max_surface_slider)

if folium_map is None:
    st.subheader('There are no mutations that meet your criterias')
else:
    folium_static(folium_map)

# hour_to_filter = st.slider('hour', 0, 31, 17)  # min: 0h, max: 23h, default: 17h
# filtered_data = sample_IDF[sample_IDF['day'].dt.hour == hour_to_filter]
# st.subheader(f'Map of all pickups at {hour_to_filter}:00')
# plot_map(dept=None, insee=None, section=None, type_of_property=None, year_of_mutation=None):